"""
Common utilities for nanochat.
"""

import logging
import math
import os
import re
import urllib.request

import numpy as np
import torch
import torch.distributed as dist
from filelock import FileLock


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == "INFO":
            # Highlight numbers and percentages
            message = re.sub(r"(\d+\.?\d*\s*(?:GB|MB|%|docs))", rf"{self.BOLD}\1{self.RESET}", message)
            message = re.sub(r"(Shard \d+)", rf"{self.COLORS['INFO']}{self.BOLD}\1{self.RESET}", message)
        return message


def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


setup_default_logging()
logger = logging.getLogger(__name__)


def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()  # bytes

        # Write to local file
        with open(file_path, "wb") as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path


def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get("RANK", 0))
    if ddp_rank == 0:
        print(s, **kwargs)


def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
     ████                                           █████                            
    ░░███                                          ░░███                             
     ░███   ██████   ██████  ████████   ██████   ███████                             
     ░███  ███░░███ ███░░███░░███░░███ ███░░███ ███░░███                             
     ░███ ░███ ░███░███ ░███ ░███ ░███░███████ ░███ ░███                             
     ░███ ░███ ░███░███ ░███ ░███ ░███░███░░░  ░███ ░███                             
     █████░░██████ ░░██████  ░███████ ░░██████ ░░████████                            
    ░░░░░  ░░░░░░   ░░░░░░   ░███░░░   ░░░░░░   ░░░░░░░░                             
                             ░███                                                    
                             █████                                                   
                            ░░░░░                                                    
                                                       █████                 █████   
                                                       ░░███                 ░░███    
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████   ███████  
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███ ░░░███░   
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████   ░███    
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███   ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░████████  ░░█████ 
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░    ░░░░░  
    """
    print0(banner)


def sample_poisson_lognormal_recurrence(mean_recur: float, sigma: float = 0.5, max_recur: int | None = None) -> int:
    """
    Sample number of recurrences from Poisson log-normal distribution.

    This distribution creates a heavy-tailed sampling scheme for recurrence depth,
    as described in Huginn paper (arXiv:2502.05171, Section 3.3).

    The distribution is: τ ~ N(log(r̄) - ½σ², σ), then r ~ Poisson(e^τ) + 1

    Args:
        mean_recur: Mean number of recurrences (r̄)
        sigma: Standard deviation of the log-normal component (default: 0.5)
        max_recur: Optional maximum recurrence value for clamping

    Returns:
        Sampled number of recurrences, clamped to [1, max_recur] if max_recur is provided
    """
    tau = np.random.normal(math.log(mean_recur) - 0.5 * sigma**2, sigma)
    num_recur = np.random.poisson(math.exp(tau)) + 1
    num_recur = max(1, min(num_recur, max_recur)) if max_recur is not None else max(1, num_recur)
    return int(num_recur)


def sample_num_recurs_for_step(
    recur_samples_per_step: int | None,
    mean_recur: float,
    sigma: float,
    max_recur: int | None,
    ddp: bool,
    master_process: bool,
    device: torch.device,
) -> list[int] | None:
    """
    Pre-sample all num_recur values for an optimization step.

    Samples are drawn on rank 0 and broadcast to all ranks for synchronization.
    Returns None if recur_samples_per_step is None/0 (fixed mode).

    Args:
        recur_samples_per_step: Number of different num_recur values to sample (global across all ranks)
        mean_recur: Mean number of recurrences for sampling
        sigma: Standard deviation for log-normal component
        max_recur: Maximum recurrence value
        ddp: Whether DDP is enabled
        master_process: Whether this is the master process (rank 0)
        device: Device for tensor operations

    Returns:
        List of sampled num_recur values, or None for fixed mode
    """
    if not recur_samples_per_step:
        return None

    # Rank 0 samples all values
    if master_process:
        sampled_num_recurs = [
            sample_poisson_lognormal_recurrence(mean_recur=mean_recur, sigma=sigma, max_recur=max_recur)
            for _ in range(recur_samples_per_step)
        ]
    else:
        sampled_num_recurs = [0] * recur_samples_per_step  # placeholder

    # Broadcast to all ranks
    if ddp:
        num_recur_tensor = torch.tensor(sampled_num_recurs, dtype=torch.int32, device=device)
        dist.broadcast(num_recur_tensor, src=0)
        sampled_num_recurs = num_recur_tensor.tolist()

    return sampled_num_recurs


def get_num_recur_for_microstep(
    sampled_num_recurs: list[int] | None,
    micro_step: int,
    ddp_rank: int,
    ddp_world_size: int,
    grad_accum_steps: int,
    recur_samples_per_step: int,
) -> int | None:
    """
    Get the num_recur value for a specific micro-step.

    Uses global micro-step indexing across all ranks to determine which
    sampled value to use for this micro-step.

    Args:
        sampled_num_recurs: Pre-sampled values from sample_num_recurs_for_step()
        micro_step: Local micro-step index (0 to grad_accum_steps-1)
        ddp_rank: Rank of this process in DDP
        ddp_world_size: Total number of ranks
        grad_accum_steps: Number of gradient accumulation steps per rank
        recur_samples_per_step: Number of samples per step (global)

    Returns:
        num_recur value for this micro-step, or None for fixed mode (use model default)
    """
    if sampled_num_recurs is None:
        return None

    # Compute global micro-step index across all ranks
    total_micro_steps = ddp_world_size * grad_accum_steps
    global_micro_step = ddp_rank * grad_accum_steps + micro_step
    microsteps_per_sample = total_micro_steps // recur_samples_per_step
    sample_idx = global_micro_step // microsteps_per_sample

    return sampled_num_recurs[sample_idx]


# ---------------------------------------------------------------------------
# Recurrence curriculum scheduling (arXiv:2511.07384)
# ---------------------------------------------------------------------------

def get_scheduled_recur_mean(
    step: int,
    num_iterations: int,
    target_mean: float,
    recur_warmup_ratio: float,
) -> float:
    """
    Compute the scheduled recurrence mean for a given training step.

    Uses a 1-sqrt schedule during warmup: mean = ceil(target * (1 - sqrt(1 - t/W))),
    clamped to [1.0, target_mean]. After warmup completes, returns target_mean.

    Based on curriculum learning for recurrence depth (McLeish et al., arXiv:2511.07384).

    Args:
        step: Current optimization step (0-indexed)
        num_iterations: Total number of optimization steps
        target_mean: Target (final) mean recurrence depth
        recur_warmup_ratio: Fraction of training for recurrence warmup (0.0 = no ramp)

    Returns:
        Scheduled mean recurrence for this step (float, >= 1.0)
    """
    if recur_warmup_ratio <= 0.0 or num_iterations <= 0:
        return target_mean

    warmup_steps = round(recur_warmup_ratio * num_iterations)
    if step >= warmup_steps or warmup_steps == 0:
        return target_mean

    progress = step / warmup_steps
    raw = target_mean * (1.0 - math.sqrt(1.0 - progress))
    return max(1.0, math.ceil(raw))


def compute_cumulative_flops(
    estimate_flops_fn,
    num_steps: int,
    num_iterations: int,
    target_mean: float,
    recur_warmup_ratio: float,
    tokens_per_step: int,
) -> float:
    """
    Compute total training FLOPs for steps [0, num_steps), accounting for
    recurrence curriculum ramping.

    Caches estimate_flops_fn results since there are only ~ceil(target_mean)
    distinct integer recurrence values across the schedule.

    Args:
        estimate_flops_fn: Callable(num_recur: int) -> flops_per_token
        num_steps: Number of steps to sum over
        num_iterations: Total training iterations (for schedule computation)
        target_mean: Target mean recurrence
        recur_warmup_ratio: Warmup ratio for recurrence ramp
        tokens_per_step: Tokens per optimization step (total_batch_size)

    Returns:
        Total FLOPs across the given steps
    """
    cache: dict[int, float] = {}
    total_flops = 0.0
    for s in range(num_steps):
        mean_r = int(get_scheduled_recur_mean(s, num_iterations, target_mean, recur_warmup_ratio))
        if mean_r not in cache:
            cache[mean_r] = estimate_flops_fn(num_recur=mean_r)
        total_flops += cache[mean_r] * tokens_per_step
    return total_flops


def solve_iterations_for_target_flops(
    estimate_flops_fn,
    target_flops: float,
    target_mean: float,
    recur_warmup_ratio: float,
    tokens_per_step: int,
) -> int:
    """
    Binary search for the number of iterations that achieves target_flops,
    accounting for the recurrence curriculum.

    Since cumulative FLOPs is monotonically increasing in N and the schedule
    (warmup fraction) scales with N, binary search converges cleanly.

    Args:
        estimate_flops_fn: Callable(num_recur: int) -> flops_per_token
        target_flops: Desired total training FLOPs
        target_mean: Target mean recurrence depth
        recur_warmup_ratio: Warmup ratio for recurrence ramp
        tokens_per_step: Tokens per optimization step (total_batch_size)

    Returns:
        Number of iterations to reach target_flops
    """
    # Upper bound: all steps at cheapest cost (recur=1)
    min_flops_per_step = estimate_flops_fn(num_recur=1) * tokens_per_step
    hi = int(target_flops / min_flops_per_step) + 2
    lo = 1

    while lo < hi:
        mid = (lo + hi) // 2
        total = compute_cumulative_flops(
            estimate_flops_fn, mid, mid, target_mean, recur_warmup_ratio, tokens_per_step,
        )
        if total < target_flops:
            lo = mid + 1
        else:
            hi = mid
    return lo


def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    return dist.is_available() and dist.is_initialized()


def get_dist_info():
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def autodetect_device_type():
    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type


def compute_init(device_type="cuda"):  # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    if device_type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"  # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)  # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp_initialized():
        dist.destroy_process_group()


class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""

    def __init__(self):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


# hardcoded BF16 peak flops for various GPUs
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float("inf")


def compute_gradient_stats(model: torch.nn.Module, track_level: str) -> dict[str, float]:
    """
    Compute gradient and parameter statistics for tracking optimization dynamics and model health.

    Args:
        model: PyTorch model with computed gradients
        track_level: "none" (disabled), "basic" (global norm), or "detailed" (per-component)

    Returns:
        Dictionary of gradient and parameter statistics for logging
    """
    if track_level == "none":
        return {}

    stats = {}

    # Basic: global gradient and parameter norms (L1 and L2)
    if track_level in ["basic", "detailed"]:
        global_grad_l1 = 0.0
        global_grad_norm_sq = 0.0
        global_param_l1 = 0.0
        global_param_norm_sq = 0.0

        for p in model.parameters():
            # Parameter norms
            global_param_l1 += p.data.abs().sum().item()
            global_param_norm_sq += p.data.pow(2).sum().item()

            # Gradient norms (if available)
            if p.grad is not None:
                global_grad_l1 += p.grad.data.abs().sum().item()
                global_grad_norm_sq += p.grad.data.norm(2).item() ** 2

        stats["grad_norm/l2"] = global_grad_norm_sq**0.5
        stats["grad_norm/l1"] = global_grad_l1
        stats["param_norm/l2"] = global_param_norm_sq**0.5
        stats["param_norm/l1"] = global_param_l1

    # Detailed: per-component and per-optimizer-type breakdown
    if track_level == "detailed":
        # Group by architecture component (prelude/recur_block/coda)
        prelude_grad_l1 = 0.0
        prelude_grad_norm_sq = 0.0
        prelude_param_l1 = 0.0
        prelude_param_norm_sq = 0.0

        recur_grad_l1 = 0.0
        recur_grad_norm_sq = 0.0
        recur_param_l1 = 0.0
        recur_param_norm_sq = 0.0

        coda_grad_l1 = 0.0
        coda_grad_norm_sq = 0.0
        coda_param_l1 = 0.0
        coda_param_norm_sq = 0.0

        other_grad_l1 = 0.0
        other_grad_norm_sq = 0.0
        other_param_l1 = 0.0
        other_param_norm_sq = 0.0

        # Group by optimizer type (muon vs adam)
        muon_grad_l1 = 0.0
        muon_grad_norm_sq = 0.0
        muon_param_l1 = 0.0
        muon_param_norm_sq = 0.0

        adam_grad_l1 = 0.0
        adam_grad_norm_sq = 0.0
        adam_param_l1 = 0.0
        adam_param_norm_sq = 0.0

        for name, p in model.named_parameters():
            # Parameter stats
            param_l1 = p.data.abs().sum().item()
            param_norm_sq = p.data.pow(2).sum().item()

            # Gradient stats (if available)
            if p.grad is not None:
                grad_l1 = p.grad.data.abs().sum().item()
                grad_norm_sq = p.grad.data.norm(2).item() ** 2
            else:
                grad_l1 = 0.0
                grad_norm_sq = 0.0

            # Classify by architecture component
            if "prelude" in name:
                prelude_grad_l1 += grad_l1
                prelude_grad_norm_sq += grad_norm_sq
                prelude_param_l1 += param_l1
                prelude_param_norm_sq += param_norm_sq
            elif ".recur." in name or "transformer.recur" in name:
                recur_grad_l1 += grad_l1
                recur_grad_norm_sq += grad_norm_sq
                recur_param_l1 += param_l1
                recur_param_norm_sq += param_norm_sq
            elif "coda" in name:
                coda_grad_l1 += grad_l1
                coda_grad_norm_sq += grad_norm_sq
                coda_param_l1 += param_l1
                coda_param_norm_sq += param_norm_sq
            else:
                other_grad_l1 += grad_l1
                other_grad_norm_sq += grad_norm_sq
                other_param_l1 += param_l1
                other_param_norm_sq += param_norm_sq

            # Classify by optimizer type
            # Muon: matrix parameters (weights with ndim >= 2)
            # Adam: embeddings and other parameters (biases, norms, etc.)
            if "embed" in name.lower():
                adam_grad_l1 += grad_l1
                adam_grad_norm_sq += grad_norm_sq
                adam_param_l1 += param_l1
                adam_param_norm_sq += param_norm_sq
            elif "weight" in name and p.ndim >= 2:
                muon_grad_l1 += grad_l1
                muon_grad_norm_sq += grad_norm_sq
                muon_param_l1 += param_l1
                muon_param_norm_sq += param_norm_sq
            else:
                adam_grad_l1 += grad_l1
                adam_grad_norm_sq += grad_norm_sq
                adam_param_l1 += param_l1
                adam_param_norm_sq += param_norm_sq

        # Gradient stats
        stats["grad_norm/prelude/l2"] = prelude_grad_norm_sq**0.5
        stats["grad_norm/prelude/l1"] = prelude_grad_l1
        stats["grad_norm/recur_block/l2"] = recur_grad_norm_sq**0.5
        stats["grad_norm/recur_block/l1"] = recur_grad_l1
        stats["grad_norm/coda/l2"] = coda_grad_norm_sq**0.5
        stats["grad_norm/coda/l1"] = coda_grad_l1
        stats["grad_norm/other/l2"] = other_grad_norm_sq**0.5
        stats["grad_norm/other/l1"] = other_grad_l1
        stats["grad_norm/muon_params/l2"] = muon_grad_norm_sq**0.5
        stats["grad_norm/muon_params/l1"] = muon_grad_l1
        stats["grad_norm/adam_params/l2"] = adam_grad_norm_sq**0.5
        stats["grad_norm/adam_params/l1"] = adam_grad_l1

        # Parameter stats
        stats["param_norm/prelude/l2"] = prelude_param_norm_sq**0.5
        stats["param_norm/prelude/l1"] = prelude_param_l1
        stats["param_norm/recur_block/l2"] = recur_param_norm_sq**0.5
        stats["param_norm/recur_block/l1"] = recur_param_l1
        stats["param_norm/coda/l2"] = coda_param_norm_sq**0.5
        stats["param_norm/coda/l1"] = coda_param_l1
        stats["param_norm/other/l2"] = other_param_norm_sq**0.5
        stats["param_norm/other/l1"] = other_param_l1
        stats["param_norm/muon_params/l2"] = muon_param_norm_sq**0.5
        stats["param_norm/muon_params/l1"] = muon_param_l1
        stats["param_norm/adam_params/l2"] = adam_param_norm_sq**0.5
        stats["param_norm/adam_params/l1"] = adam_param_l1

    return stats
