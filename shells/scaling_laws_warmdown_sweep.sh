#!/bin/bash
set -eo pipefail

# =============================================================================
# Warmdown Ratio Sweep
# =============================================================================
# Runs a single representative model (s12 at 2.15e18 FLOPs) with
# --save-before-warmdown, then resumes from the pre-warmdown checkpoint
# with different warmdown ratios to find the minimum acceptable ratio.
#
# Usage:
#   bash shells/scaling_laws_warmdown_sweep.sh
#
# Phase 1: Train from scratch with the largest warmdown ratio (saves pre-warmdown checkpoint)
# Phase 2: Resume from pre-warmdown checkpoint with each smaller warmdown ratio
# =============================================================================

LABEL="warmdown_sweep"
SIZE=12
FLOPS="2.15e18"
BATCH_SIZE=32

# Warmdown ratios to sweep (largest first = the initial training run)
WARMDOWN_RATIOS=(0.4 0.3 0.2 0.15 0.1 0.05)
INITIAL_RATIO="${WARMDOWN_RATIOS[0]}"

N_PRELUDE=2
N_RECUR_BLOCK=4
N_CODA=2
N_RECUR=4

NPROC_PER_NODE=${NUM_GPUS:-1}
EVAL_TOKENS=$((100 * 524288))

export OMP_NUM_THREADS=1

#--- One GPU broken, but need to queue two ---
NPROC_PER_NODE=1
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES="${GPUS[1]}"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
#---

cd ~/looped_nanochat

source shells/_machine_config.sh
validate_config || exit 1

uv sync
source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/warmdown_sweep_results_${LABEL}"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "warmdown_ratio,num_iterations,val_bpb,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Common training flags (shared by initial run and resume runs)
COMMON_FLAGS=(
    --size=$SIZE
    --device-batch-size=$BATCH_SIZE
    --target-flops=$FLOPS
    --target-param-data-ratio=-1
    --eval-tokens=$EVAL_TOKENS
    --core-metric-every=-1
    --sample-every=-1
    --window-pattern="L"
    --train-recur-mean=$N_RECUR
    --n-prelude=$N_PRELUDE
    --n-recur-block=$N_RECUR_BLOCK
    --n-coda=$N_CODA
    --input-injection=inject_init_prelude
    --recur-samples-per-step=0
    --bptt-k=$N_RECUR
    --recur-warmup-ratio=0
    --embedding-lr=0.3
    --unembedding-lr=0.004
    --matrix-lr=0.02
    --warmup-ratio=0.0
    --final-lr-frac=0.0
    --prepacked
)

# Model tag used for checkpoints (all runs share the same checkpoint dir)
MODEL_TAG="warmdown_sweep_s${SIZE}"

# =============================================================================
# Phase 1: Initial training with largest warmdown ratio + save-before-warmdown
# =============================================================================
log "Phase 1: Training s=$SIZE at $FLOPS FLOPs with warmdown=$INITIAL_RATIO (saving pre-warmdown checkpoint)"

TAG="${LABEL}_wd${INITIAL_RATIO}"
LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
    "${COMMON_FLAGS[@]}" \
    --warmdown-ratio=$INITIAL_RATIO \
    --save-before-warmdown \
    --save-every=-1 \
    --model-tag="$MODEL_TAG" \
    --run="${TAG}" \
    2>&1 | tee "$LOG_FILE"

# Extract results for the initial ratio
NUM_ITERS=$(grep "Calculated number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')
TRAIN_TIME=$(grep "Total training time:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+(?=m)')
echo "$INITIAL_RATIO,$NUM_ITERS,$VAL_BPB,$TRAIN_TIME" >> "$RESULTS_FILE"
log "  warmdown=$INITIAL_RATIO: val_bpb=$VAL_BPB"

# =============================================================================
# Phase 2: Resume from pre-warmdown checkpoint with different warmdown ratios
# =============================================================================
for ratio in "${WARMDOWN_RATIOS[@]:1}"; do
    log "Phase 2: Resuming with warmdown=$ratio from pre_warmdown checkpoint"

    TAG="${LABEL}_wd${ratio}"
    LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        "${COMMON_FLAGS[@]}" \
        --warmdown-ratio=$ratio \
        --resume-from-step=pre_warmdown \
        --save-every=-1 \
        --model-tag="$MODEL_TAG" \
        --run="${TAG}" \
        2>&1 | tee "$LOG_FILE"

    VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')
    TRAIN_TIME=$(grep "Total training time:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+(?=m)')
    echo "$ratio,$NUM_ITERS,$VAL_BPB,$TRAIN_TIME" >> "$RESULTS_FILE"
    log "  warmdown=$ratio: val_bpb=$VAL_BPB"
done

# =============================================================================
# Summary
# =============================================================================
log "=============================================="
log "Warmdown Sweep Complete"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
