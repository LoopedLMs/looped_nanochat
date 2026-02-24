# Research Code - Looped LLMs based on nanochat

## Core Idea

This project develops a **looped (depth-recurrent) transformer architecture** that scales test-time compute by iterating a recurrent block in latent space, rather than producing more tokens (like Chain-of-Thought). This allows the model to "think" in continuous high-dimensional space before emitting each token.

## Repository
Forked from [karpathy/nanochat](https://github.com/karpathy/nanochat). Note: Some code comments or variable names may reference the original non-looped architecture.

## Philosophy
Research code — simple, correct, and efficient:
- Simple, hackable implementations > frameworks
- Correctness is non-negotiable — write pytest tests for non-trivial functions
- Tests should cover behavior and edge cases, not implementation details — keep them maintainable so refactors don't require rewriting every test
- Compute is scarce (2×A100-SXM4-80GB) — always consider memory, FLOPs, and throughput implications

## Code Standards
- Before writing new functions, check existing modules for code that can be extended or reused
- Type hints on all signatures (modern syntax: `str | None`, `list[int]`)
- Run ruff after changes: `uv run ruff format . && uv run ruff check --fix .`

## Package Management (CRITICAL)
- ALWAYS: `uv add <package>`
- NEVER: manually edit pyproject.toml
- NEVER: `pip install` or `uv pip install`

## Running Code
Python scripts must be run within the uv environment:
- **Option 1**: `uv run python script.py` (recommended for one-off commands)
- **Option 2**: Activate environment first with `source .venv/bin/activate`, then run normally

## Debugging
Check `.venv` source code directly for library implementation details

## Project Paths
`NANOCHAT_BASE_DIR` (where results, checkpoints, and plots are saved) is configured in `slurm/machine_config.sh`. Check there for the current value.

## Background Knowledge
Paper summaries and research notes live in `./knowledge/`. Check there for context on relevant prior work (e.g. layer redundancy, recurrence retrofitting). The paper behind this repository is summarized in knowledge/summary_retrofitting_recurrence.md.

## Research Stack
- Framework: PyTorch
- Testing: pytest for core components only (skip for exploratory code)
