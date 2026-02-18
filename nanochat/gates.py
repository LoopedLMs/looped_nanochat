"""
Early-exit gate functions for looped transformer inference and training.

Provides incremental gate checkers that evaluate exit conditions at each
recurrence iteration, enabling actual early exit during the forward pass.

Gate types:
  Training-free (no learned parameters):
    1. acceleration: second-order convergence (Pappone et al.), two-hit
    2. relative_l2: relative state change magnitude
    3. kl_divergence: KL between consecutive logit distributions
    4. entropy: Shannon entropy of output distribution (LoopViT-style crystallization)

  Learned:
    5. LearnedGate: linear probe on recurrent state, trained with soft mixture loss

Joint gate training (Stage I) is integrated into GPT._forward_joint_gate().
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GateConfig:
    """Configuration for early-exit gating."""

    gate_type: Literal["acceleration", "relative_l2", "kl_divergence", "entropy"]
    threshold: float
    normalized: bool = True  # acceleration gate only


@dataclass
class GateStats:
    """Statistics from a gated forward pass."""

    exit_depths: torch.Tensor  # (B, T) per-position exit depth, 1-indexed
    num_recur: int  # max recurrence depth used

    @property
    def mean_depth(self) -> float:
        return self.exit_depths.float().mean().item()

    @property
    def fraction_early(self) -> float:
        return (self.exit_depths < self.num_recur).float().mean().item()

    def flops_fraction(self, fixed_params: int, recur_params: int) -> float:
        """Approximate FLOPs fraction vs full recurrence (linear cost model)."""
        full_cost = fixed_params + self.num_recur * recur_params
        mean_cost = fixed_params + self.mean_depth * recur_params
        return mean_cost / full_cost


class GateChecker:
    """
    Incremental gate condition checker for the recurrence loop.

    Call step() after each recurrence iteration with the new state (and logits
    for KL gate). The checker tracks which (B, T) positions have exited and
    at what depth. Call finalize() after the loop to get GateStats.
    """

    def __init__(self, config: GateConfig, B: int, T: int, device: torch.device):
        self.config = config
        self.threshold = config.threshold
        self.exited = torch.zeros(B, T, dtype=torch.bool, device=device)
        self.exit_depths = torch.zeros(B, T, dtype=torch.long, device=device)

        # Gate-specific running state
        self._prev_state: torch.Tensor | None = None
        self._prev_delta: torch.Tensor | None = None
        self._prev_accel_small: torch.Tensor | None = None
        self._prev_log_probs: torch.Tensor | None = None

    def step(
        self,
        recurrence_idx: int,
        state: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Check gate condition after a recurrence iteration.

        Args:
            recurrence_idx: 0-indexed iteration number
            state: recurrent state after norm_recur, shape (B, T, D)
            logits: intermediate logits (B, T, V) — required for kl_divergence

        Returns:
            newly_exited: (B, T) bool mask of positions that just exited
        """
        if self.config.gate_type == "acceleration":
            fires = self._check_acceleration(state)
        elif self.config.gate_type == "relative_l2":
            fires = self._check_relative_l2(state)
        elif self.config.gate_type == "entropy":
            assert logits is not None, "entropy gate requires logits"
            fires = self._check_entropy(logits)
        else:
            assert logits is not None, "kl_divergence gate requires logits"
            fires = self._check_kl(logits)

        # Only count positions not already exited
        newly_exited = fires & ~self.exited
        self.exited = self.exited | newly_exited
        self.exit_depths[newly_exited] = recurrence_idx + 1  # 1-indexed

        self._prev_state = state
        return newly_exited

    # ------------------------------------------------------------------
    # Gate implementations (match dev/analysis/gating_analysis.py exactly)

    def _check_acceleration(self, state: torch.Tensor) -> torch.Tensor:
        """Normalized acceleration with two-hit rule."""
        false = torch.zeros_like(self.exited)

        if self._prev_state is None:
            return false

        delta = state - self._prev_state

        if self._prev_delta is None:
            self._prev_delta = delta
            return false

        # Acceleration norm
        accel = torch.norm(delta - self._prev_delta, dim=-1)
        if self.config.normalized:
            denom = torch.norm(delta, dim=-1) + torch.norm(self._prev_delta, dim=-1) + 1e-8
            accel = accel / denom

        small = accel < self.threshold

        # Two-hit: both current and previous must be small
        if self._prev_accel_small is None:
            self._prev_accel_small = small
            self._prev_delta = delta
            return false

        two_hit = small & self._prev_accel_small
        self._prev_accel_small = small
        self._prev_delta = delta
        return two_hit

    def _check_relative_l2(self, state: torch.Tensor) -> torch.Tensor:
        """Relative L2 change: ||s_i - s_{i-1}|| / (||s_i|| + eps)."""
        if self._prev_state is None:
            return torch.zeros_like(self.exited)

        diff_norm = torch.norm(state - self._prev_state, dim=-1)
        state_norm = torch.norm(state, dim=-1)
        rel_l2 = diff_norm / (state_norm + 1e-10)
        return rel_l2 < self.threshold

    def _check_kl(self, logits: torch.Tensor) -> torch.Tensor:
        """KL(p_i || p_{i-1}) summed over vocab."""
        log_probs = F.log_softmax(logits, dim=-1)

        if self._prev_log_probs is None:
            self._prev_log_probs = log_probs
            return torch.zeros_like(self.exited)

        kl = F.kl_div(
            self._prev_log_probs, log_probs, reduction="none", log_target=True
        ).sum(dim=-1)
        self._prev_log_probs = log_probs
        return kl < self.threshold

    def _check_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Shannon entropy of output distribution: H = -sum p log p. Fires when H < threshold."""
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
        return entropy < self.threshold

    # ------------------------------------------------------------------

    def finalize(self, num_recur: int) -> GateStats:
        """Set non-exited positions to num_recur and return stats."""
        self.exit_depths = torch.where(self.exited, self.exit_depths, num_recur)
        return GateStats(exit_depths=self.exit_depths, num_recur=num_recur)


# ---------------------------------------------------------------------------
# Learned gate


class LearnedGate(nn.Module):
    """
    Learned exit gate: a linear probe on the recurrent state.

    At each recurrence depth, outputs an exit probability per token.
    Trained with soft mixture loss (model frozen, only gate params updated).

    Input modes:
      - "state": probe sees s_i only (d_model inputs)
      - "state_delta": probe sees [s_i, s_i - s_{i-1}] (2 * d_model inputs).
        At depth 0 (no previous state), delta is zero.
    """

    def __init__(self, d_model: int, input_mode: Literal["state", "state_delta"] = "state"):
        super().__init__()
        self.input_mode = input_mode
        self.d_model = d_model
        input_dim = 2 * d_model if input_mode == "state_delta" else d_model
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, s: torch.Tensor, s_prev: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            s: recurrent state (B, T, D) after norm_recur
            s_prev: previous depth's state (B, T, D), or None at depth 0

        Returns:
            exit probability (B, T) in [0, 1]
        """
        if self.input_mode == "state_delta":
            delta = s - s_prev if s_prev is not None else torch.zeros_like(s)
            x = torch.cat([s, delta], dim=-1)
        else:
            x = s
        return torch.sigmoid(self.linear(x).squeeze(-1))


def compute_depth_weights(
    exit_probs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """
    Compute per-depth exit weights from gate exit probabilities.

    Models exit as a sequential decision process:
    - At depth i, the gate decides to exit with probability p_i
    - Probability of reaching depth i: alpha_i = prod_{j<i} (1 - p_j)
    - Probability of exiting at depth i: e_i = p_i * alpha_i
    - Remainder at max depth: e_R = alpha_R (forced exit)

    The returned weights sum to 1 per (B, T) position.

    Args:
        exit_probs: list of num_recur tensors, each (B, T) in [0, 1]

    Returns:
        list of num_recur tensors, each (B, T), summing to 1
    """
    num_depths = len(exit_probs)
    assert num_depths >= 2, (
        f"compute_depth_weights requires num_depths >= 2 (got {num_depths}). "
        "With num_depths=1, exit probs are unused and gate receives zero gradients."
    )
    alpha = torch.ones_like(exit_probs[0])
    weights = []

    for i in range(num_depths - 1):
        w_i = exit_probs[i] * alpha
        weights.append(w_i)
        alpha = alpha * (1 - exit_probs[i])

    weights.append(alpha)  # remainder at max depth
    return weights


def geometric_target(num_depths: int, rate: float, device: torch.device) -> torch.Tensor:
    """
    Geometric exit distribution target for KL regularization.

    P(exit at depth k) = rate * (1 - rate)^(k-1)  for k = 1, ..., N-1
    P(exit at depth N) = (1 - rate)^(N-1)          (remaining mass)

    Higher rate = more weight on early exits.

    Args:
        num_depths: number of recurrence depths (N)
        rate: per-depth exit probability in (0, 1)
        device: torch device

    Returns:
        (num_depths,) tensor of target probabilities summing to 1
    """
    target = torch.zeros(num_depths, device=device)
    for k in range(num_depths - 1):
        target[k] = rate * (1 - rate) ** k
    target[-1] = (1 - rate) ** (num_depths - 1)
    return target


def kl_to_target(weights: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """
    KL(weights || target) per position.

    Args:
        weights: list of num_depths (B, T) tensors from compute_depth_weights
        target: (num_depths,) target distribution

    Returns:
        Per-position KL divergence (B, T). Lower = closer to target.
    """
    kl = torch.zeros_like(weights[0])
    for i, w in enumerate(weights):
        log_w = torch.log(w.clamp(min=1e-12))
        log_t = torch.log(target[i].clamp(min=1e-12))
        kl = kl + w * (log_w - log_t)
    return kl


def exit_distribution_entropy(weights: list[torch.Tensor]) -> torch.Tensor:
    """
    Entropy of the exit-at-depth distribution: H = -sum_t w_t * log(w_t).

    Args:
        weights: list of (B, T) tensors from compute_depth_weights, summing to 1.

    Returns:
        Per-position entropy (B, T). Higher = more uniform across depths.
    """
    entropy = torch.zeros_like(weights[0])
    for w in weights:
        entropy = entropy - w * torch.log(w.clamp(min=1e-12))
    return entropy
