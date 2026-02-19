#!/bin/bash

#------
# Param info examples:
# S12, R6, BPTT_K=4, SAMPLE_R=false -> DEVICE_BATCH_SIZE=32, 2x A100 80GB Train time = 2:12h
# S12, R6, BPTT_K=6, SAMPLE_R=false -> DEVICE_BATCH_SIZE=16, 2x A100 80GB Train time = 2:14h
#------

cd ~/looped_nanochat
uv sync
source .venv/bin/activate

source slurm/machine_config.sh
validate_config || exit 1

# Number of processes/GPUs to use (from machine_config.sh, defaults to 1)
NPROC_PER_NODE=${SLURM_GPUS:-1}

EVAL_TOKENS=$((100 * 524288))  # ~100M tokens for final eval (default is ~10M)

# --- Experiment config ---
TARGET_FLOPS=2.15e18
SIZE=12
NUM_RECUR=4
SAMPLE_R=false # set to true for recursion sampling
BPTT_K=4
DEVICE_BATCH_SIZE=32
TAG_SUFFIX=""
# --- Less common ---
INPUT_INJECTION=inject_init_prelude  # inject_init_prelude | inject_init_random | passthrough
EMBEDDING_LR=0.3
UNEMBEDDING_LR=0.004
MATRIX_LR=0.02

# --- Derived config ---
if [ "$SAMPLE_R" = true ]; then
    RECUR_SAMPLES_PER_STEP=8
    TAG="r${NUM_RECUR}_sample_${TARGET_FLOPS}_s${SIZE}${TAG_SUFFIX}"
else
    RECUR_SAMPLES_PER_STEP=0
    TAG="r${NUM_RECUR}_${TARGET_FLOPS}_s${SIZE}${TAG_SUFFIX}"
fi
RUN=$TAG

# Run base training, capture output for CSV logging
TRAIN_LOG=$(mktemp)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train \
    --eval-tokens=$EVAL_TOKENS \
    --target-flops $TARGET_FLOPS \
    --model-tag $TAG \
    --run $RUN \
    --device-batch-size $DEVICE_BATCH_SIZE \
    --core-metric-every=-1 \
    --core-metric-max-per-task=-1 \
    --recur-samples-per-step=$RECUR_SAMPLES_PER_STEP \
    --train-recur-mean=$NUM_RECUR \
    --train-recur-max=16 \
    --bptt-k=$BPTT_K \
    --input-injection=$INPUT_INJECTION \
    --embedding-lr=$EMBEDDING_LR \
    --unembedding-lr=$UNEMBEDDING_LR \
    --matrix-lr=$MATRIX_LR \
    --save-every=-1 \
    --size $SIZE 2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}

# Log results to CSV if training succeeded
CSV_FILE=dev/base_train.csv
if [ $TRAIN_EXIT -eq 0 ] && grep -q "Minimum validation bpb" "$TRAIN_LOG"; then
    FLOPS_PER_TOK=$(grep -oP 'Estimated FLOPs per token: \K[\d.e+]+' "$TRAIN_LOG" | head -1)
    TOKENS=$(grep -oP 'Total number of training tokens: \K[\d,]+' "$TRAIN_LOG" | head -1 | tr -d ',')
    BATCH_SIZE=$(grep -oP 'Total batch size \K[\d,]+' "$TRAIN_LOG" | head -1 | tr -d ',')
    STEPS=$(grep -oP 'Calculated number of iterations from target FLOPs: \K[\d,]+' "$TRAIN_LOG" | head -1 | tr -d ',')
    VAL_BPB=$(grep -oP 'Minimum validation bpb: \K[\d.]+' "$TRAIN_LOG")
    TRAIN_TIME_M=$(grep -oP 'Total training time: \K[\d.]+' "$TRAIN_LOG")
    TRAIN_TIME_H=$(python3 -c "print(f'{${TRAIN_TIME_M}/60:.2f}')")
    GPU_NAME=$(grep -oP 'GPU: \K[^|]+' "$TRAIN_LOG" | head -1 | xargs)
    GPUS="${NPROC_PER_NODE}x${GPU_NAME}"
    BRANCH=$(git branch --show-current)

    echo "$(date +%Y-%m-%d),${TAG},${BRANCH},S${SIZE},${NUM_RECUR},${SAMPLE_R},${BPTT_K},${INPUT_INJECTION},${FLOPS_PER_TOK},${TARGET_FLOPS},${TOKENS},${BATCH_SIZE},${DEVICE_BATCH_SIZE},${STEPS},${EMBEDDING_LR},${UNEMBEDDING_LR},${MATRIX_LR},${VAL_BPB},${TRAIN_TIME_H},${GPUS}," >> "$CSV_FILE"
    echo "Results appended to ${CSV_FILE}"
fi
rm -f "$TRAIN_LOG"
