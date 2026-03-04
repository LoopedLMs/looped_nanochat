#!/bin/bash
# Universal SLURM job submission wrapper
# Usage: ./shells/_submit.sh <script.sh> [script_args...] [-- sbatch_overrides...]
#
# Arguments before "--" are passed to the script itself.
# Arguments after "--" are sbatch overrides (e.g. --time=24:00:00).
#
# Examples:
#   ./shells/_submit.sh shells/chat_sft.sh
#   ./shells/_submit.sh shells/scaling_laws.sh --smoke-test --no-wandb
#   ./shells/_submit.sh shells/scaling_laws.sh --smoke-test -- --time=24:00:00
#
# This script reads SLURM configuration from _machine_config.sh and submits
# the specified script with appropriate resource allocation.
#
# Convention: Scripts ending with _cpu.sh are submitted as CPU jobs.
# All other scripts are submitted as GPU jobs.

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <script.sh> [script_args...] [-- sbatch_overrides...]"
    echo ""
    echo "Examples:"
    echo "  $0 shells/chat_sft.sh"
    echo "  $0 shells/scaling_laws.sh --smoke-test --no-wandb"
    echo "  $0 shells/scaling_laws.sh --smoke-test -- --time=24:00:00"
    echo ""
    echo "Available scripts:"
    for script in shells/*.sh; do
        if [ "$script" != "shells/_submit.sh" ] && [ "$script" != "shells/_machine_config.sh" ]; then
            echo "  - $(basename $script)"
        fi
    done
    echo ""

    # Show configured paths if _machine_config.sh exists
    if [ -f "shells/_machine_config.sh" ]; then
        source "shells/_machine_config.sh"
        echo "Configured paths (from _machine_config.sh):"
        echo "  HF_HOME:            $HF_HOME"
        echo "  HF_DATASETS_CACHE:  $HF_DATASETS_CACHE"
        echo "  NANOCHAT_BASE_DIR:  $NANOCHAT_BASE_DIR"
        echo ""
        echo "SLURM configuration:"
        echo "  GPU Partition:      $SLURM_PARTITION_GPU"
        echo "  GPU QoS:            $SLURM_QOS_GPU"
        echo "  GPUs per job:       $NUM_GPUS"
        if [ -n "$SLURM_GPU_MEM" ]; then
            echo "  GPU Memory (each):  $SLURM_GPU_MEM"
        fi
        echo "  Total Memory:       $SLURM_MEM_GPU"
        echo "  CPU Partition:      $SLURM_PARTITION_CPU"
        echo "  CPU QoS:            $SLURM_QOS_CPU"
        echo "  CPU Memory:         $SLURM_MEM_CPU"
        if [ -n "$SLURM_MAIL_USER" ]; then
            echo "  Mail notifications: $SLURM_MAIL_USER"
        fi
    else
        echo "Note: shells/_machine_config.sh not found. Create it from template:"
        echo "  cp shells/_machine_config.sh.template shells/_machine_config.sh"
    fi

    exit 1
fi

SCRIPT_PATH="$1"
shift  # Remove script path

# Split remaining args: script args go before "--", sbatch overrides after "--".
SCRIPT_ARGS=()
SBATCH_EXTRA_ARGS=()
FOUND_SEPARATOR=false
for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        FOUND_SEPARATOR=true
        continue
    fi
    if [ "$FOUND_SEPARATOR" = true ]; then
        SBATCH_EXTRA_ARGS+=("$arg")
    else
        SCRIPT_ARGS+=("$arg")
    fi
done

# Validate script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script not found: $SCRIPT_PATH"
    exit 1
fi

# Source machine configuration
CONFIG_FILE="shells/_machine_config.sh"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: $CONFIG_FILE not found!"
    echo ""
    echo "Please create it from the template:"
    echo "  cp shells/_machine_config.sh.template shells/_machine_config.sh"
    echo "  # Then edit _machine_config.sh with your cluster settings"
    exit 1
fi

source "$CONFIG_FILE"

# Extract job name from script filename
JOB_NAME="looped_nanochat_$(basename $SCRIPT_PATH .sh)"

# Determine job type based on script name convention
# CPU jobs: scripts ending with _cpu.sh
# GPU jobs: all other scripts
if [[ "$SCRIPT_PATH" =~ _cpu\.sh$ ]]; then
    JOB_TYPE="cpu"
    PARTITION="$SLURM_PARTITION_CPU"
    QOS="$SLURM_QOS_CPU"
    MEM="$SLURM_MEM_CPU"
    GRES=""
    CPUS_PER_TASK="${SLURM_CPUS_CPU:-1}"
else
    JOB_TYPE="gpu"
    PARTITION="$SLURM_PARTITION_GPU"
    QOS="$SLURM_QOS_GPU"
    MEM="$SLURM_MEM_GPU"
    # Build gres string: gpu[:<type>]:<count>[,gpumem:<mem>]
    if [ -n "$SLURM_GPU_TYPE" ]; then
        GRES="--gres=gpu:$SLURM_GPU_TYPE:$NUM_GPUS"
    else
        GRES="--gres=gpu:$NUM_GPUS"
    fi
    if [ -n "$SLURM_GPU_MEM" ]; then
        GRES="$GRES,gpumem:$SLURM_GPU_MEM"
    fi
    CPUS_PER_TASK="${SLURM_CPUS_GPU:-}"
fi

# Create log directory based on script name
LOG_DIR="logs/$(basename $SCRIPT_PATH .sh)"
mkdir -p "$LOG_DIR"

# Build sbatch command
SBATCH_CMD="sbatch"
SBATCH_CMD="$SBATCH_CMD --job-name=$JOB_NAME"
SBATCH_CMD="$SBATCH_CMD --output=$LOG_DIR/%A.out"
SBATCH_CMD="$SBATCH_CMD --error=$LOG_DIR/%A.err"
SBATCH_CMD="$SBATCH_CMD --nodes=1"
SBATCH_CMD="$SBATCH_CMD --ntasks=1"
SBATCH_CMD="$SBATCH_CMD --mem=$MEM"

# Add partition and QoS
if [ -n "$PARTITION" ]; then
    SBATCH_CMD="$SBATCH_CMD --partition=$PARTITION"
fi
if [ -n "$QOS" ]; then
    SBATCH_CMD="$SBATCH_CMD --qos=$QOS"
fi
if [ -n "$SLURM_ACCOUNT" ]; then
    SBATCH_CMD="$SBATCH_CMD --account=$SLURM_ACCOUNT"
fi

# Add GPU resources for GPU jobs
if [ -n "$GRES" ]; then
    SBATCH_CMD="$SBATCH_CMD $GRES"
fi

# Add CPU count
if [ -n "$CPUS_PER_TASK" ]; then
    SBATCH_CMD="$SBATCH_CMD --cpus-per-task=$CPUS_PER_TASK"
fi

# Add mail notifications for GPU jobs (usually longer running)
if [ "$JOB_TYPE" = "gpu" ] && [ -n "$SLURM_MAIL_USER" ]; then
    SBATCH_CMD="$SBATCH_CMD --mail-type=FAIL,END"
    SBATCH_CMD="$SBATCH_CMD --mail-user=$SLURM_MAIL_USER"
fi

# Add default time limits if not overridden
TIME_SET=false
for arg in "${SBATCH_EXTRA_ARGS[@]}"; do
    if [[ "$arg" =~ ^--time= ]] || [[ "$arg" =~ ^-t ]]; then
        TIME_SET=true
        break
    fi
done

if [ "$TIME_SET" = false ]; then
    if [ "$JOB_TYPE" = "cpu" ]; then
        SBATCH_CMD="$SBATCH_CMD --time=01:00:00"
    else
        # Default to 10 hours for GPU jobs
        SBATCH_CMD="$SBATCH_CMD --time=10:00:00"
    fi
fi

# Add any additional user-provided sbatch overrides
for arg in "${SBATCH_EXTRA_ARGS[@]}"; do
    SBATCH_CMD="$SBATCH_CMD $arg"
done

# Add the script path, then script arguments
SBATCH_CMD="$SBATCH_CMD $SCRIPT_PATH"
for arg in "${SCRIPT_ARGS[@]}"; do
    SBATCH_CMD="$SBATCH_CMD $arg"
done

# Show what we're submitting
echo "=============================================="
echo "Submitting SLURM job:"
echo "  Type:      $JOB_TYPE"
echo "  Script:    $SCRIPT_PATH"
echo "  Partition: $PARTITION"
echo "  Memory:    $MEM"
if [ -n "$GRES" ]; then
    if [ -n "$SLURM_GPU_TYPE" ]; then
        echo "  GPUs:      $NUM_GPUS x $SLURM_GPU_TYPE"
    else
        echo "  GPUs:      $NUM_GPUS"
    fi
    if [ -n "$SLURM_GPU_MEM" ]; then
        echo "  GPU Mem:   $SLURM_GPU_MEM (each)"
    fi
fi
if [ -n "$CPUS_PER_TASK" ]; then
    echo "  CPUs:      $CPUS_PER_TASK"
fi
if [ ${#SCRIPT_ARGS[@]} -gt 0 ]; then
    echo "  Script args: ${SCRIPT_ARGS[*]}"
fi
echo "  Logs:      $LOG_DIR/"
echo "=============================================="
echo ""

# Submit the job
echo "Running: $SBATCH_CMD"
echo ""
eval "$SBATCH_CMD"
