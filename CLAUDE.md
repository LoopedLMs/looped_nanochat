# Research Code - Looped LLMs based on nanochat

Forked from [karpathy/nanochat](https://github.com/karpathy/nanochat). Develops a **looped (depth-recurrent) transformer** that scales test-time compute by iterating a recurrent block in latent space. Note: Some comments/variable names may reference the original non-looped architecture.

## Philosophy
Research code optimized for rapid iteration and debugging:
- Simple, hackable implementations > frameworks
- Missing error handling is GOOD (faster bug discovery)
- Understand every component > black-box abstractions
- Compute is scarce (2×A100-SXM4-80GB) — consider memory, FLOPs, and throughput
- Write pytest tests for non-trivial functions; cover behavior/edge cases, not implementation details

## Code Standards
- Check existing modules before writing new functions
- Type hints on all signatures (modern syntax: `str | None`, `list[int]`)
- Use pathlib

## Package Management (CRITICAL)
- ALWAYS: `uv add <package>`
- NEVER: manual edit, `pip install` or `uv pip install`

## Running Code
`uv run python script.py` or activate `.venv` first

## Debugging
Check `.venv` source code directly for library implementation details

## Project Paths
`NANOCHAT_BASE_DIR` (results, checkpoints, plots) is configured in `shells/machine_config.sh`.

## External Knowledge (Notion)
Paper summaries are synced to a Notion database. The project page ID is in `./notion.txt`. When the user references a paper or asks about related work, check Notion first using `notion-search` before re-reading local files or re-downloading.