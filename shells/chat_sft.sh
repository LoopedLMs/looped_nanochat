#!/bin/bash

cd ~/looped_nanochat
uv sync
source .venv/bin/activate

source shells/_machine_config.sh
validate_config || exit 1

# Number of processes/GPUs to use (from _machine_config.sh, defaults to 1)
NPROC_PER_NODE=${NUM_GPUS:-1}

# --- Experiment config ---
BASE_MODEL=r4_1.35e19_s20        # base model tag to finetune from
OUTPUT_TAG=r4_1.35e19_s20        # output tag (defaults to BASE_MODEL if empty)
DEVICE_BATCH_SIZE=16
RECUR_SAMPLES_PER_STEP=0         # 0 = fixed recurrence, >0 = sample per step
# --- Less common ---
EMBEDDING_LR=0.2
UNEMBEDDING_LR=0.004
MATRIX_LR=0.02

# --- Eval config ---
EVAL_TASKS=""                    # empty = all tasks (ARC-Easy|ARC-Challenge|MMLU|GSM8K|HumanEval)
EVAL_NUM_RECUR=""                # empty = model default. Comma-separated for sweep, e.g. "4,6,8"
EVAL_BATCH_SIZE=64               # categorical eval batch size
EVAL_GEN_BATCH_SIZE=16           # generative eval batch size
SKIP_EVAL=0                      # set to 1 to skip chat eval after SFT

# --- Derived config ---
OUTPUT_TAG=${OUTPUT_TAG:-$BASE_MODEL}
RUN=$OUTPUT_TAG

# Run chat SFT, capture output for CSV logging
SFT_LOG=$(mktemp)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft \
    -- --model-tag $BASE_MODEL --output-tag $OUTPUT_TAG \
    --device-batch-size $DEVICE_BATCH_SIZE \
    --run $RUN \
    --recur-samples-per-step $RECUR_SAMPLES_PER_STEP \
    --embedding-lr $EMBEDDING_LR \
    --unembedding-lr $UNEMBEDDING_LR \
    --matrix-lr $MATRIX_LR 2>&1 | tee "$SFT_LOG"
SFT_EXIT=${PIPESTATUS[0]}

# Parse SFT results
if [ $SFT_EXIT -eq 0 ] && grep -q "Minimum validation bpb" "$SFT_LOG"; then
    SFT_VAL_BPB=$(grep -oP 'Minimum validation bpb: \K[\d.]+' "$SFT_LOG")
    SFT_TIME_M=$(grep -oP 'Total training time: \K[\d.]+' "$SFT_LOG")
    SFT_TIME_H=$(python3 -c "print(f'{${SFT_TIME_M}/60:.2f}')")
    # Get final step number from last step line
    SFT_STEPS=$(grep -oP 'step \K\d+' "$SFT_LOG" | tail -1)
    GPU_NAME=$(grep -oP 'GPU: \K[^|]+' "$SFT_LOG" | head -1 | xargs)
    GPUS="${NPROC_PER_NODE}x${GPU_NAME}"
    BRANCH=$(git branch --show-current)
    echo "SFT complete: val_bpb=${SFT_VAL_BPB}, steps=${SFT_STEPS}, time=${SFT_TIME_H}h"
else
    echo "SFT training failed or did not complete"
    rm -f "$SFT_LOG"
    exit 1
fi
rm -f "$SFT_LOG"

if [ "$SKIP_EVAL" -eq 1 ]; then
    echo "Skipping chat eval (SKIP_EVAL=1)"
    CSV_FILE=dev/chat_sft.csv
    echo "$(date +%Y-%m-%d),${OUTPUT_TAG},${BASE_MODEL},${BRANCH},${SFT_STEPS},${DEVICE_BATCH_SIZE},${RECUR_SAMPLES_PER_STEP},${EMBEDDING_LR},${UNEMBEDDING_LR},${MATRIX_LR},${SFT_VAL_BPB},,,,,,,${SFT_TIME_H},${GPUS},eval skipped" >> "$CSV_FILE"
    echo "SFT-only results appended to ${CSV_FILE}"
else
    # Run chat eval
    EVAL_LOG=$(mktemp)
    EVAL_ARGS="-i sft -g $OUTPUT_TAG --batch-size $EVAL_BATCH_SIZE --gen-batch-size $EVAL_GEN_BATCH_SIZE"
    if [ -n "$EVAL_TASKS" ]; then
        EVAL_ARGS="$EVAL_ARGS --task-name $EVAL_TASKS"
    fi
    if [ -n "$EVAL_NUM_RECUR" ]; then
        EVAL_ARGS="$EVAL_ARGS --num-recur $EVAL_NUM_RECUR"
    fi
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval \
        -- $EVAL_ARGS 2>&1 | tee "$EVAL_LOG"
    EVAL_EXIT=${PIPESTATUS[0]}

    # Parse eval results and log to CSV
    CSV_FILE=dev/chat_sft.csv
    if [ $EVAL_EXIT -eq 0 ] && grep -q "ChatCORE metric" "$EVAL_LOG"; then
        ARC_E=$(grep -oP 'ARC-Easy accuracy: \K[\d.]+' "$EVAL_LOG" | tail -1)
        ARC_C=$(grep -oP 'ARC-Challenge accuracy: \K[\d.]+' "$EVAL_LOG" | tail -1)
        MMLU=$(grep -oP 'MMLU accuracy: \K[\d.]+' "$EVAL_LOG" | tail -1)
        GSM8K=$(grep -oP 'GSM8K accuracy: \K[\d.]+' "$EVAL_LOG" | tail -1)
        HUMANEVAL=$(grep -oP 'HumanEval accuracy: \K[\d.]+' "$EVAL_LOG" | tail -1)
        CHATCORE=$(grep -oP 'ChatCORE metric: \K[\d.]+' "$EVAL_LOG" | tail -1)

        echo "$(date +%Y-%m-%d),${OUTPUT_TAG},${BASE_MODEL},${BRANCH},${SFT_STEPS},${DEVICE_BATCH_SIZE},${RECUR_SAMPLES_PER_STEP},${EMBEDDING_LR},${UNEMBEDDING_LR},${MATRIX_LR},${SFT_VAL_BPB},${ARC_E},${ARC_C},${MMLU},${GSM8K},${HUMANEVAL},${CHATCORE},${SFT_TIME_H},${GPUS}," >> "$CSV_FILE"
        echo "Results appended to ${CSV_FILE}"
    else
        echo "WARNING: Eval failed or incomplete. Logging SFT results only (no eval scores)."
        echo "$(date +%Y-%m-%d),${OUTPUT_TAG},${BASE_MODEL},${BRANCH},${SFT_STEPS},${DEVICE_BATCH_SIZE},${RECUR_SAMPLES_PER_STEP},${EMBEDDING_LR},${UNEMBEDDING_LR},${MATRIX_LR},${SFT_VAL_BPB},,,,,,,${SFT_TIME_H},${GPUS},eval failed" >> "$CSV_FILE"
        echo "Partial results appended to ${CSV_FILE}"
    fi
    rm -f "$EVAL_LOG"
fi
