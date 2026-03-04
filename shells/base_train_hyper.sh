#!/bin/bash
# =============================================================================
# Hyperparameter Tuning — S12 Looped Nanochat
# =============================================================================
#
# 18 runs across 6 stages. Split by commenting out stages you've already run.
# After each stage, inspect results.csv, pick the best value, and update the
# BEST_* variable below before running the next stage.
#
# Stage 1: Muon LR         — 5 runs
# Stage 2: Weight decay    — 5 runs
# Stage 2b: Recur WD scale — 2 runs  (needs code change: --recur-wd-scale)
# Stage 3: Warmdown ratio  — 5 runs
# Stage 4: Embedding LR    — 3 runs
# Stage 5: Unembedding LR  — 3 runs
# =============================================================================

set -euo pipefail

cd ~/looped_nanochat
uv sync
source .venv/bin/activate

source shells/_machine_config.sh
validate_config || exit 1

# Number of processes/GPUs to use (from _machine_config.sh, defaults to 1)
NPROC_PER_NODE=${NUM_GPUS:-1}
EVAL_TOKENS=$((100 * 524288))  # ~100M tokens for final eval (default is ~10M)

# =============================================================================
# UPDATE THESE after each stage
# =============================================================================
BEST_MATRIX_LR=0.02
BEST_WEIGHT_DECAY=0.2
BEST_RECUR_WD_SCALE=1.0
BEST_WARMDOWN_RATIO=0.4
BEST_EMBEDDING_LR=0.3
BEST_UNEMBEDDING_LR=0.004

# =============================================================================
# Fixed config
# =============================================================================
TAG="hp_s12"
RESULTS_DIR="${BASE_DIR}/hp_tune_results_${TAG}"
RESULTS_FILE="${RESULTS_DIR}/results.csv"
mkdir -p "$RESULTS_DIR"

[ ! -f "$RESULTS_FILE" ] && \
    echo "stage,param,value,matrix_lr,weight_decay,warmdown_ratio,embedding_lr,unembedding_lr,val_bpb,time_s" > "$RESULTS_FILE"

run() {
    local STAGE=$1 PNAME=$2 PVAL=$3 MLR=$4 WD=$5 WDR=$6 ELR=$7 ULR=$8
    local EXTRA="${9:-}"
    local RUN_TAG="${TAG}_${STAGE}_${PVAL}"

    echo "[$(date +%H:%M:%S)] ${STAGE} ${PNAME}=${PVAL}"

    local T0=$(date +%s)

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        --size=12 \
        --device-batch-size=32 \
        --target-flops 2.15e18 \
        --target-param-data-ratio=-1 \
        --num-iterations=-1 \
        --run="${RUN_TAG}" \
        --model-tag="${RUN_TAG}" \
        --eval-tokens=$EVAL_TOKENS \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        --window-pattern="LLSSSLLL" \
        --train-recur-mean=4 \
        --n-prelude=2 \
        --n-recur-block=4 \
        --n-coda=2 \
        --input-injection=inject_init_prelude \
        --no-sample-recur \
        --matrix-lr=$MLR \
        --weight-decay=$WD \
        --warmdown-ratio=$WDR \
        --embedding-lr=$ELR \
        --unembedding-lr=$ULR \
        --warmup-ratio=0.0 \
        $EXTRA \
        2>&1 | tee "$RESULTS_DIR/${RUN_TAG}.log"

    local DT=$(( $(date +%s) - T0 ))
    local BPB=$(grep "Validation bpb:" "$RESULTS_DIR/${RUN_TAG}.log" | tail -1 | grep -oP '[\d.]+$')
    echo "  → val_bpb=${BPB:-NaN} (${DT}s)"
    echo "${STAGE},${PNAME},${PVAL},${MLR},${WD},${WDR},${ELR},${ULR},${BPB:-NaN},${DT}" >> "$RESULTS_FILE"
}

# =============================================================================
# STAGE 1: Muon LR                                              [~10-15 hours]
# =============================================================================
# NOTE: Removed 0.02 (default) already tested in scaling laws
for v in 0.01 0.04 0.005 0.08; do
    run s1 matrix_lr $v  $v $BEST_WEIGHT_DECAY $BEST_WARMDOWN_RATIO $BEST_EMBEDDING_LR $BEST_UNEMBEDDING_LR
done

# =============================================================================
# STAGE 2: Weight decay                                         [~10-15 hours]
# → Update BEST_MATRIX_LR above, then uncomment
# =============================================================================
# NOTE: Removed 0.2 (default) already tested in stage 1
#for v in 0.1 0.4 0.05 0.8; do
#     run s2 weight_decay $v  $BEST_MATRIX_LR $v $BEST_WARMDOWN_RATIO $BEST_EMBEDDING_LR $BEST_UNEMBEDDING_LR
#done

# =============================================================================
# STAGE 2b: Recur WD scale                                      [~4-6 hours]
# → Needs --recur-wd-scale in base_train.py
# → Update BEST_WEIGHT_DECAY above, then uncomment
# =============================================================================
#for wd in 0.2 0.4; do
#    for rws in 0.5; do
#        run s2b "wd${wd}_rws" $rws  $BEST_MATRIX_LR $wd $BEST_WARMDOWN_RATIO $BEST_EMBEDDING_LR $BEST_UNEMBEDDING_LR \
#            "--recur-wd-scale=$rws"
#    done
#done

# =============================================================================
# STAGE 3: Warmdown ratio                                       [~8-12 hours]
# → Update BEST_WEIGHT_DECAY above, then uncomment
# =============================================================================
# NOTE: Removed 0.4 (default) already tested in stage 1
#for v in 0.3 0.2 0.5 0.6; do
#    run s3 warmdown_ratio $v  $BEST_MATRIX_LR $BEST_WEIGHT_DECAY $v $BEST_EMBEDDING_LR $BEST_UNEMBEDDING_LR
#done

# =============================================================================
# STAGE 4: Embedding LR                                         [~6-9 hours]
# → Update BEST_WARMDOWN_RATIO above, then uncomment
# =============================================================================
# NOTE: Removed 0.3 (default) already tested in prior stages
# for v in 0.15 0.6; do
#     run s4 embedding_lr $v  $BEST_MATRIX_LR $BEST_WEIGHT_DECAY $BEST_WARMDOWN_RATIO $v $BEST_UNEMBEDDING_LR
# done

# =============================================================================
# STAGE 5: Unembedding LR                                       [~6-9 hours]
# → Update BEST_EMBEDDING_LR above, then uncomment
# =============================================================================
# NOTE: Remove 0.004 (default) already tested in prior stages
# for v in 0.002 0.008; do
#     run s5 unembedding_lr $v  $BEST_MATRIX_LR $BEST_WEIGHT_DECAY $BEST_WARMDOWN_RATIO $BEST_EMBEDDING_LR $v
# done