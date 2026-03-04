# SLURM Job Submission

This directory contains SLURM job scripts for running experiments on the cluster.

## Quick Start

1. **First-time setup** - Create your machine-specific config:
   ```bash
   cp shells/_machine_config.sh.template shells/_machine_config.sh
   # Edit _machine_config.sh with your cluster settings
   ```

2. **Submit a job**:
   ```bash
   ./shells/submit.sh shells/chat_sft.sh
   ```

3. **Override SLURM parameters** (optional):
   ```bash
   ./shells/submit.sh shells/scaling_laws.sh --time=24:00:00
   ./shells/submit.sh shells/chat_eval.sh --mem=128G
   ```

## Architecture

### Configuration (`_machine_config.sh`)
- **Gitignored** - each user/machine has their own config
- Sets environment variables (HF_HOME, NANOCHAT_BASE_DIR, etc.)
- Defines SLURM defaults (partition, QoS, memory, GPUs, email)
- See [_machine_config.sh.template](_machine_config.sh.template) for all options

### Job Scripts (`*.sh`)
- **Tracked in git** - contains the job logic only
- No SLURM headers - pure bash scripts
- Can be run locally or via SLURM
- Sources `_machine_config.sh` for environment setup
- **Naming convention**: Scripts ending with `_cpu.sh` are submitted as CPU jobs; all others as GPU jobs

### Submission Wrapper (`submit.sh`)
- Reads your `_machine_config.sh`
- Auto-detects job type (CPU vs GPU)
- Generates appropriate SLURM headers
- Submits job with `sbatch`
