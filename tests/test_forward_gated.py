"""Verify forward_gated produces identical results to forward when gate never fires."""

import torch

import pytest

from nanochat.gates import GateConfig, LearnedGate, compute_depth_weights, exit_distribution_entropy
from nanochat.gpt import GPT, GPTConfig


def test_forward_gated_matches_forward():
    """With negative threshold (never fires), forward_gated should match forward exactly."""
    torch.manual_seed(42)
    config = GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_embd=128,
        n_head=4,
        n_kv_head=4,
        n_prelude=1,
        n_recur_block=2,
        n_coda=1,
        train_recur_mean=4.0,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()
    model.eval()

    idx = torch.randint(0, config.vocab_size, (2, 16))
    num_recur = 6

    # Reference: standard forward
    with torch.no_grad():
        logits_ref, s_ref = model(idx, num_recur=num_recur)

    # Gated forward with negative threshold — gate metrics are non-negative, so never fires
    for gate_type in ["acceleration", "relative_l2", "kl_divergence"]:
        gate_config = GateConfig(gate_type=gate_type, threshold=-1.0)
        with torch.no_grad():
            logits_g, s_g, stats = model.forward_gated(idx, gate_config, num_recur=num_recur)

        assert torch.allclose(logits_ref, logits_g, atol=1e-5), (
            f"{gate_type}: logits mismatch, max diff={torch.max(torch.abs(logits_ref - logits_g))}"
        )
        assert torch.allclose(s_ref, s_g, atol=1e-5), (
            f"{gate_type}: state mismatch, max diff={torch.max(torch.abs(s_ref - s_g))}"
        )
        assert stats.mean_depth == num_recur, f"{gate_type}: expected mean_depth={num_recur}, got {stats.mean_depth}"
        assert stats.fraction_early == 0.0, f"{gate_type}: expected no early exits"
        print(f"  {gate_type}: OK (mean_depth={stats.mean_depth}, fraction_early={stats.fraction_early})")


def test_forward_gated_early_exit():
    """With threshold=0, all gates should fire as early as possible."""
    torch.manual_seed(42)
    config = GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_embd=128,
        n_head=4,
        n_kv_head=4,
        n_prelude=1,
        n_recur_block=2,
        n_coda=1,
        train_recur_mean=4.0,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()
    model.eval()

    idx = torch.randint(0, config.vocab_size, (2, 16))
    num_recur = 8

    # Very high threshold = exit ASAP
    for gate_type, max_exit_depth in [("relative_l2", 2), ("kl_divergence", 2), ("acceleration", 4)]:
        gate_config = GateConfig(gate_type=gate_type, threshold=1e10)
        with torch.no_grad():
            logits_g, s_g, stats = model.forward_gated(idx, gate_config, num_recur=num_recur)

        # With threshold=inf, everything exits at earliest possible depth
        print(f"  {gate_type}: mean_depth={stats.mean_depth:.2f}, "
              f"fraction_early={stats.fraction_early:.2f}, "
              f"exit_depths_range=[{stats.exit_depths.min()}, {stats.exit_depths.max()}]")

        # With threshold=inf all should exit early
        assert stats.exit_depths.max() <= max_exit_depth, (
            f"{gate_type}: expected max exit depth <= {max_exit_depth}, got {stats.exit_depths.max()}"
        )


def test_gate_stats_flops():
    """Verify FLOPs fraction computation."""
    from nanochat.gates import GateStats

    exit_depths = torch.tensor([[4, 4], [4, 4]])
    stats = GateStats(exit_depths=exit_depths, num_recur=8)
    assert stats.mean_depth == 4.0
    assert stats.fraction_early == 1.0

    # fixed=100, recur=50: exit_cost=100+4*50=300, full=100+8*50=500
    frac = stats.flops_fraction(fixed_params=100, recur_params=50)
    assert abs(frac - 300 / 500) < 1e-6, f"Expected 0.6, got {frac}"
    print(f"  flops_fraction: {frac:.4f} (expected 0.6)")


def test_compute_depth_weights_sums_to_one():
    """Depth weights must form a valid probability distribution (sum to 1)."""
    torch.manual_seed(42)
    B, T = 2, 8
    num_depths = 6

    # Random exit probabilities
    exit_probs = [torch.rand(B, T) for _ in range(num_depths)]
    weights = compute_depth_weights(exit_probs)

    assert len(weights) == num_depths
    total = sum(weights)
    assert torch.allclose(total, torch.ones(B, T), atol=1e-6), (
        f"Depth weights don't sum to 1, max deviation={torch.max(torch.abs(total - 1))}"
    )
    # All weights should be non-negative
    for i, w in enumerate(weights):
        assert (w >= 0).all(), f"Negative weight at depth {i}"
    print(f"  sum={total.mean():.6f}, all non-negative: OK")


def test_compute_depth_weights_extremes():
    """Test depth weights at extreme exit probabilities."""
    B, T = 1, 1
    num_depths = 4

    # All exit at depth 0 (p=1 everywhere)
    exit_probs = [torch.ones(B, T)] * num_depths
    weights = compute_depth_weights(exit_probs)
    assert torch.allclose(weights[0], torch.ones(B, T)), "p=1: all weight should be at depth 0"
    for i in range(1, num_depths):
        assert torch.allclose(weights[i], torch.zeros(B, T), atol=1e-7), f"p=1: weight at depth {i} should be 0"
    print("  p=1 (exit immediately): OK")

    # Never exit (p=0 everywhere) — all weight at last depth
    exit_probs = [torch.zeros(B, T)] * num_depths
    weights = compute_depth_weights(exit_probs)
    for i in range(num_depths - 1):
        assert torch.allclose(weights[i], torch.zeros(B, T), atol=1e-7), f"p=0: weight at depth {i} should be 0"
    assert torch.allclose(weights[-1], torch.ones(B, T)), "p=0: all weight should be at last depth"
    print("  p=0 (never exit): OK")


@pytest.mark.parametrize("input_mode", ["state", "state_delta"])
def test_compute_depth_weights_gradient(input_mode):
    """Verify gradients flow through depth weights to gate parameters."""
    torch.manual_seed(42)
    B, T, D = 2, 4, 32
    num_depths = 3

    gate = LearnedGate(D, input_mode=input_mode)
    states = [torch.randn(B, T, D) for _ in range(num_depths)]

    exit_probs = []
    for i, s in enumerate(states):
        s_prev = states[i - 1] if i > 0 else None
        exit_probs.append(gate(s, s_prev))
    weights = compute_depth_weights(exit_probs)

    # Simulate loss: weighted sum of dummy per-depth losses
    per_depth_losses = [torch.randn(B, T) for _ in range(num_depths)]
    loss = sum(w * l for w, l in zip(weights, per_depth_losses)).sum()
    loss.backward()

    # Gate parameters should have gradients
    assert gate.linear.weight.grad is not None, "Gate weight has no gradient"
    assert gate.linear.bias.grad is not None, "Gate bias has no gradient"
    assert gate.linear.weight.grad.abs().sum() > 0, "Gate weight gradient is zero"
    print(f"  [{input_mode}] grad norm: weight={gate.linear.weight.grad.norm():.6f}, bias={gate.linear.bias.grad.norm():.6f}")


@pytest.mark.parametrize("input_mode", ["state", "state_delta"])
def test_learned_gate_output_shape(input_mode):
    """LearnedGate output shape and range."""
    B, T, D = 3, 16, 64
    gate = LearnedGate(D, input_mode=input_mode)
    s = torch.randn(B, T, D)
    s_prev = torch.randn(B, T, D)
    p = gate(s, s_prev)
    assert p.shape == (B, T), f"Expected shape ({B}, {T}), got {p.shape}"
    assert (p >= 0).all() and (p <= 1).all(), f"Exit probs out of [0,1] range: min={p.min()}, max={p.max()}"
    # Also test with s_prev=None (depth 0)
    p0 = gate(s, None)
    assert p0.shape == (B, T), f"s_prev=None: Expected shape ({B}, {T}), got {p0.shape}"
    print(f"  [{input_mode}] shape={p.shape}, range=[{p.min():.4f}, {p.max():.4f}]: OK")


@pytest.mark.parametrize("input_mode", ["state", "state_delta"])
def test_learned_gate_with_model(input_mode):
    """Test learned gate with actual model intermediate states."""
    torch.manual_seed(42)
    config = GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_embd=128,
        n_head=4,
        n_kv_head=4,
        n_prelude=1,
        n_recur_block=2,
        n_coda=1,
        train_recur_mean=4.0,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()
    model.eval()

    gate = LearnedGate(config.n_embd, input_mode=input_mode)
    idx = torch.randint(0, config.vocab_size, (2, 16))
    num_recur = 4

    # Get intermediate states
    with torch.no_grad():
        _, _, _, intermediate_states = model(idx, num_recur=num_recur, return_intermediate_states=True)

    assert len(intermediate_states) == num_recur, f"Expected {num_recur} states, got {len(intermediate_states)}"

    # Run gate on each state (passing s_prev for state_delta mode)
    exit_probs = []
    for i, s in enumerate(intermediate_states):
        s_prev = intermediate_states[i - 1] if i > 0 else None
        exit_probs.append(gate(s, s_prev))
    weights = compute_depth_weights(exit_probs)

    # Verify weights sum to 1
    total = sum(weights)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)

    # Compute expected depth
    expected_depth = sum((i + 1) * w.mean() for i, w in enumerate(weights))
    assert 1 <= expected_depth.item() <= num_recur, f"Expected depth {expected_depth} out of range [1, {num_recur}]"
    print(f"  [{input_mode}] expected_depth={expected_depth.item():.2f}, weights_sum={total.mean():.6f}: OK")


def test_compute_depth_weights_rejects_single_depth():
    """compute_depth_weights must reject num_depths=1 (zero gate gradients)."""
    with pytest.raises(AssertionError, match="num_depths >= 2"):
        compute_depth_weights([torch.rand(2, 4)])
    print("  num_depths=1 correctly rejected: OK")


def test_exit_distribution_entropy():
    """Verify entropy computation for known distributions."""
    import math

    B, T = 1, 1

    # Uniform over 4 depths: each weight = 0.25, H = log(4)
    uniform_weights = [torch.full((B, T), 0.25) for _ in range(4)]
    h = exit_distribution_entropy(uniform_weights)
    expected_h = math.log(4)
    assert abs(h.item() - expected_h) < 1e-5, f"Expected H={expected_h:.4f}, got {h.item():.4f}"
    print(f"  uniform H={h.item():.4f} (expected {expected_h:.4f}): OK")

    # Degenerate (all mass at depth 0): H = 0
    degenerate_weights = [torch.ones(B, T)] + [torch.zeros(B, T) for _ in range(3)]
    h = exit_distribution_entropy(degenerate_weights)
    assert abs(h.item()) < 1e-5, f"Expected H=0, got {h.item():.4f}"
    print(f"  degenerate H={h.item():.6f} (expected 0): OK")


@pytest.mark.parametrize("input_mode", ["state", "state_delta"])
def test_entropy_gradient_flow(input_mode):
    """Verify gradients flow through entropy to gate parameters."""
    torch.manual_seed(42)
    B, T, D = 2, 4, 32
    num_depths = 3

    gate = LearnedGate(D, input_mode=input_mode)
    states = [torch.randn(B, T, D) for _ in range(num_depths)]

    exit_probs = []
    for i, s in enumerate(states):
        s_prev = states[i - 1] if i > 0 else None
        exit_probs.append(gate(s, s_prev))
    weights = compute_depth_weights(exit_probs)
    entropy = exit_distribution_entropy(weights)
    loss = -entropy.mean()  # maximize entropy
    loss.backward()

    assert gate.linear.weight.grad is not None, "Gate weight has no gradient"
    assert gate.linear.weight.grad.abs().sum() > 0, "Gate weight gradient is zero"
    print(f"  [{input_mode}] entropy grad norm: weight={gate.linear.weight.grad.norm():.6f}: OK")


if __name__ == "__main__":
    print("Test 1: forward_gated matches forward (no early exit)")
    test_forward_gated_matches_forward()
    print("\nTest 2: forward_gated early exit")
    test_forward_gated_early_exit()
    print("\nTest 3: GateStats flops fraction")
    test_gate_stats_flops()
    print("\nTest 4: depth weights sum to 1")
    test_compute_depth_weights_sums_to_one()
    print("\nTest 5: depth weights extremes")
    test_compute_depth_weights_extremes()
    for mode in ["state", "state_delta"]:
        print(f"\nTest 6: depth weights gradient flow ({mode})")
        test_compute_depth_weights_gradient(mode)
        print(f"\nTest 7: learned gate output shape ({mode})")
        test_learned_gate_output_shape(mode)
        print(f"\nTest 8: learned gate with model ({mode})")
        test_learned_gate_with_model(mode)
    print("\nTest 9: num_depths=1 rejected")
    test_compute_depth_weights_rejects_single_depth()
    print("\nTest 10: exit distribution entropy")
    test_exit_distribution_entropy()
    for mode in ["state", "state_delta"]:
        print(f"\nTest 11: entropy gradient flow ({mode})")
        test_entropy_gradient_flow(mode)
    print("\nAll tests passed!")
