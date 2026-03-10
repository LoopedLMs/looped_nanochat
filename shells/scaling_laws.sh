#!/bin/bash
set -eo pipefail

LABEL="feb1"

# Reference: size -> device batch size mapping (size * 64 = model_dim)
declare -A SIZE_TO_BATCH=(
    [6]=64 [8]=32 [10]=32 [12]=32 [14]=32 [16]=32
    [18]=16 [20]=16 [24]=8  [28]=8
)

# Per-budget size sweeps: SIZES_FOR_<budget_index>
# Fixed architecture: 2 prelude + 4×4 recur + 2 coda = 20 effective layers
FLOPS_BUDGETS=(   1e18    2.15e18  4.64e18    1e19)
SIZES_FOR_0=(     6 8 10 12 14)
SIZES_FOR_1=(     8 10 12 14 16)
SIZES_FOR_2=(     12 14 16 18 20)
SIZES_FOR_3=(     14 16 18 20 22)
N_PRELUDE=2
N_RECUR_BLOCK=4
N_CODA=2
N_RECUR=4

NPROC_PER_NODE=${NUM_GPUS:-1} # Number of processes/GPUs to use (from _machine_config.sh, defaults to 1)
WANDB_RUN="${WANDB_RUN:-scaling_${LABEL}}"
EVAL_TOKENS=$((100 * 524288))  # ~100M tokens for final eval (default is ~10M)
PREPACKED_DIR="${PREPACKED_DIR:-}"  # optional: path to pre-packed data from pretokenize.py

export OMP_NUM_THREADS=1 # disable CPU multi-threading for libraries that use OpenMP (NumPy, PyTorch CPU ops, etc.)

cd ~/looped_nanochat

source shells/_machine_config.sh
validate_config || exit 1

uv sync
source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_laws_results_${LABEL}"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Write CSV header only if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "flops_budget,size,model_dim,params_wte,params_value_embeds,params_lm_head,params_prelude,params_recur_block,params_coda,params_inject,params_scalars,params_total,params_effective,num_iterations,tokens_trained,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if a run already exists in results CSV
run_exists() {
    local flops=$1
    local size=$2
    grep -q "^${flops},${size}," "$RESULTS_FILE" 2>/dev/null
}

# =============================================================================
# Main Loop
# =============================================================================

for budget_idx in "${!FLOPS_BUDGETS[@]}"; do
    flops="${FLOPS_BUDGETS[$budget_idx]}"
    sizes_var="SIZES_FOR_${budget_idx}[@]"
    sizes=("${!sizes_var}")

    log "=============================================="
    log "Compute budget: $flops FLOPs (sizes: ${sizes[*]})"
    log "=============================================="

    for s in "${sizes[@]}"; do
        batch_size="${SIZE_TO_BATCH[$s]}"
        if [ -z "$batch_size" ]; then
            log "ERROR: No batch size defined for size=$s in SIZE_TO_BATCH"
            exit 1
        fi

        # Skip if already completed
        if run_exists "$flops" "$s"; then
            log "Skipping s=$s at $flops FLOPs (already in results)"
            continue
        fi

        log "Training s=$s (batch=$batch_size) at $flops FLOPs..."

        # Unique tag for this run
        TAG="scaling_${flops}_s${s}"

        # Record start time
        START_TIME=$(date +%s)

        # Train the model with fixed flops budget
        # The script will auto-calculate num_iterations to hit target_flops
        # CORE eval happens once at the end (999999 ensures only final step)
        # Build optional flags
        OPTIONAL_FLAGS=""
        if [ -n "$PREPACKED_DIR" ]; then
            OPTIONAL_FLAGS="$OPTIONAL_FLAGS --prepacked-dir=$PREPACKED_DIR"
        fi

        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
            --size=$s \
            --device-batch-size=$batch_size \
            --target-flops=$flops \
            --target-param-data-ratio=-1 \
            --run="${WANDB_RUN}_${TAG}" \
            --model-tag="${TAG}" \
            --eval-tokens=$EVAL_TOKENS \
            --core-metric-every=999999 \
            --core-metric-max-per-task=-1 \
            --sample-every=-1 \
            --save-every=-1 \
            --save-before-warmdown \
            --window-pattern="L" \
            --train-recur-mean=$N_RECUR \
            --n-prelude=$N_PRELUDE \
            --n-recur-block=$N_RECUR_BLOCK \
            --n-coda=$N_CODA \
            --input-injection=inject_init_prelude \
            --recur-samples-per-step=0 \
            --bptt-k=$N_RECUR \
            --recur-warmup-ratio=0 \
            --embedding-lr=0.3 \
            --unembedding-lr=0.004 \
            --matrix-lr=0.02 \
            --warmup-ratio=0.0 \
            --warmdown-ratio=0.4 \
            --final-lr-frac=0.0 \
            --prepacked \
            $OPTIONAL_FLAGS \
            2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"

        END_TIME=$(date +%s)
        TRAIN_TIME=$((END_TIME - START_TIME))

        # Extract training stats from the log
        LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

        # Extract detailed parameter counts (for scaling law analysis with different conventions)
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
        # Extract actual batch size from log (auto-computed by Power Lines, varies per model)
        TOTAL_BATCH=$(grep "Total batch size" "$LOG_FILE" | tail -1 | grep -oP 'Total batch size \K[\d,]+' | tr -d ',')
        if [ -z "$NUM_ITERS" ] || [ -z "$TOTAL_BATCH" ]; then
            log "ERROR: Could not extract NUM_ITERS or TOTAL_BATCH from $LOG_FILE"
            exit 1
        fi
        TOKENS_TRAINED=$((NUM_ITERS * TOTAL_BATCH))
        # Model dim
        MODEL_DIM=$((s * 64))
        # Val BPB from final eval
        VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')

        # Extract CORE score from training log (evaluated on final step)
        CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        if [ -z "$CORE_SCORE" ]; then
            log "WARNING: Could not extract CORE score for s=$s"
            CORE_SCORE="0.0"
        fi

        log "  Params: $PARAMS_TOTAL (effective: $PARAMS_EFFECTIVE, recur: $PARAMS_RECUR), Iters: $NUM_ITERS, Val BPB: $VAL_BPB, CORE: $CORE_SCORE"

        # Append to CSV
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
