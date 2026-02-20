# Research Code - Looped LLMs based on nanochat

## Core Idea

This project develops a **looped (depth-recurrent) transformer architecture** that scales test-time compute by iterating a recurrent block in latent space, rather than producing more tokens (like Chain-of-Thought). This allows the model to "think" in continuous high-dimensional space before emitting each token.

## Repository
Forked from [karpathy/nanochat](https://github.com/karpathy/nanochat). Note: Some code comments or variable names may reference the original non-looped architecture.

## Philosophy
Research code optimized for rapid iteration and debugging:
- Simple, hackable implementations > frameworks
- Missing error handling is GOOD (faster bug discovery)
- Understand every component > black-box abstractions

## Code Standards
- Type hints on all signatures (modern syntax: `str | None`, `list[int]`)
- Self-documenting names > comments
- Use `pathlib.Path` for file paths (not `os.path`)
- Save plots to `get_base_dir() + "/plots"` (not `plt.show()`)
- Plots must be colorblind-accessible (use colorblind-safe palettes, distinct markers/linestyles)

## Conventions
- Files: `snake_case.py`
- Classes: `PascalCase` 
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

## Package Management (CRITICAL)
- ✅ ALWAYS: `uv add <package>`
- ❌ NEVER: manually edit pyproject.toml
- ❌ NEVER: `pip install` or `uv pip install`

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
