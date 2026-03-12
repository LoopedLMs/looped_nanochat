#!/bin/bash
set -eo pipefail

# =============================================================================
# Scaling Laws Sweep with Cross-Budget Checkpoint Reuse
# =============================================================================
# For each model size, trains budgets in ascending order. Each budget resumes
# from the previous budget's pre-warmdown checkpoint (where LR is still 1.0),
# saving compute vs training each (size, budget) from scratch.
# Lower WARMDOWN_RATIO = more reuse (10% warmdown → ~39% savings, 40% → ~26%).
#
# Usage:
#   bash shells/scaling_laws.sh
# =============================================================================

LABEL="mar12"

# Size -> device batch size mapping (size * 64 = model_dim)
declare -A SIZE_TO_BATCH=(
    [6]=64 [8]=32 [10]=32 [12]=32 [14]=32 [16]=32
    [18]=16 [20]=16 [24]=8  [28]=8
)

# Same 6 sizes across all budgets (enables cross-budget chaining)
# Fixed architecture: 2 prelude + 4×4 recur + 2 coda = 20 effective layers
SIZES=(8 10 12 14 16 18)
FLOPS_BUDGETS=(1e18 2.15e18 4.64e18 1e19)

N_PRELUDE=2
N_RECUR_BLOCK=4
N_CODA=2
N_RECUR=4
WARMDOWN_RATIO=0.2

NPROC_PER_NODE=${NUM_GPUS:-1}
WANDB_RUN="${WANDB_RUN:-scaling_${LABEL}}"
EVAL_TOKENS=$((100 * 524288))  # ~100M tokens for final eval (default is ~10M)
PREPACKED_DIR="${PREPACKED_DIR:-}"  # optional: path to pre-packed data from pretokenize.py

export OMP_NUM_THREADS=1

cd ~/looped_nanochat

source shells/_machine_config.sh
validate_config || exit 1

uv sync
source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_laws_results_${LABEL}"
CHECKPOINT_BASE="$RESULTS_DIR/checkpoints"
mkdir -p "$RESULTS_DIR" "$CHECKPOINT_BASE"
RESULTS_FILE="$RESULTS_DIR/results.csv"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "flops_budget,size,model_dim,params_wte,params_value_embeds,params_lm_head,params_prelude,params_recur_block,params_coda,params_inject,params_scalars,params_total,params_effective,num_iterations,tokens_trained,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_exists() {
    local flops=$1
    local size=$2
    grep -q "^${flops},${size}," "$RESULTS_FILE" 2>/dev/null
}

# Find the pre-warmdown checkpoint from the largest smaller budget for a given size.
# Scans FLOPS_BUDGETS in reverse up to (but excluding) the current budget index.
find_prev_checkpoint() {
    local size=$1
    local current_budget_idx=$2
    for (( i = current_budget_idx - 1; i >= 0; i-- )); do
        local prev_flops="${FLOPS_BUDGETS[$i]}"
        local prev_dir="$CHECKPOINT_BASE/scaling_${prev_flops}_s${size}"
        if [ -f "$prev_dir/model_pre_warmdown.pt" ]; then
            echo "$prev_dir"
            return 0
        fi
    done
    return 1
}

# Common training flags
COMMON_FLAGS=(
    --target-param-data-ratio=-1
    --eval-tokens=$EVAL_TOKENS
    --core-metric-every=999999
    --core-metric-max-per-task=-1
    --sample-every=-1
    --save-every=-1
    --save-before-warmdown
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
    --warmdown-ratio=$WARMDOWN_RATIO
    --final-lr-frac=0.0
    --prepacked
)

# =============================================================================
# Main Loop: budget-first (cheapest first), with cross-budget chaining per size
# =============================================================================

for budget_idx in "${!FLOPS_BUDGETS[@]}"; do
    flops="${FLOPS_BUDGETS[$budget_idx]}"

    log "=============================================="
    log "Compute budget: $flops FLOPs (sizes: ${SIZES[*]})"
    log "=============================================="

    for s in "${SIZES[@]}"; do
        batch_size="${SIZE_TO_BATCH[$s]}"
        if [ -z "$batch_size" ]; then
            log "ERROR: No batch size defined for size=$s in SIZE_TO_BATCH"
            exit 1
        fi

        TAG="scaling_${flops}_s${s}"

        if run_exists "$flops" "$s"; then
            log "Skipping s=$s at $flops FLOPs (already in results)"
            continue
        fi

        log "Training s=$s (batch=$batch_size) at $flops FLOPs..."

        # Auto-detect pre-warmdown checkpoint from a previous (smaller) budget
        RESUME_FLAGS=()
        PREV_CKPT_DIR=$(find_prev_checkpoint "$s" "$budget_idx") && {
            RESUME_FLAGS=(
                --resume-from-step=pre_warmdown
                --resume-checkpoint-dir="$PREV_CKPT_DIR"
            )
            log "  Chaining from ${PREV_CKPT_DIR##*/} pre-warmdown checkpoint"
        }

        # Build optional flags
        OPTIONAL_FLAGS=""
        if [ -n "$PREPACKED_DIR" ]; then
            OPTIONAL_FLAGS="$OPTIONAL_FLAGS --prepacked-dir=$PREPACKED_DIR"
        fi

        START_TIME=$(date +%s)

        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
            --size=$s \
            --device-batch-size=$batch_size \
            --target-flops=$flops \
            --run="${WANDB_RUN}_${TAG}" \
            --model-tag="${TAG}" \
            --checkpoint-base-dir="$CHECKPOINT_BASE" \
            "${COMMON_FLAGS[@]}" \
            "${RESUME_FLAGS[@]}" \
            $OPTIONAL_FLAGS \
            2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"

        END_TIME=$(date +%s)
        TRAIN_TIME=$((END_TIME - START_TIME))

        # Extract training stats from the log
        LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

        # Parameter counts
        PARAMS_WTE=$(grep "wte\s*:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_VE=$(grep "value_embeds\s*:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_LM=$(grep "lm_head\s*:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_PRELUDE=$(grep "prelude\s*:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_RECUR=$(grep "recur_block\s*:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_CODA=$(grep "coda\s*:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_INJECT=$(grep "inject\s*:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_SCALARS=$(grep "scalars\s*:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_TOTAL=$(grep "total\s*:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_EFFECTIVE=$(grep "Effective params\s*:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')

        NUM_ITERS=$(grep "Calculated number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
        TOTAL_BATCH=$(grep "Total batch size" "$LOG_FILE" | tail -1 | grep -oP 'Total batch size \K[\d,]+' | tr -d ',')
        if [ -z "$NUM_ITERS" ] || [ -z "$TOTAL_BATCH" ]; then
            log "ERROR: Could not extract NUM_ITERS or TOTAL_BATCH from $LOG_FILE"
            exit 1
        fi
        TOKENS_TRAINED=$((NUM_ITERS * TOTAL_BATCH))
        MODEL_DIM=$((s * 64))
        VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')

        CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        if [ -z "$CORE_SCORE" ]; then
            log "WARNING: Could not extract CORE score for s=$s"
            CORE_SCORE="0.0"
        fi

        log "  Params: $PARAMS_TOTAL (effective: $PARAMS_EFFECTIVE, recur: $PARAMS_RECUR), Iters: $NUM_ITERS, Val BPB: $VAL_BPB, CORE: $CORE_SCORE"

        echo "$flops,$s,$MODEL_DIM,$PARAMS_WTE,$PARAMS_VE,$PARAMS_LM,$PARAMS_PRELUDE,$PARAMS_RECUR,$PARAMS_CODA,$PARAMS_INJECT,$PARAMS_SCALARS,$PARAMS_TOTAL,$PARAMS_EFFECTIVE,$NUM_ITERS,$TOKENS_TRAINED,$VAL_BPB,$CORE_SCORE,$TRAIN_TIME" >> "$RESULTS_FILE"
    done
done

log "=============================================="
log "Scaling Laws Sweep Complete"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
