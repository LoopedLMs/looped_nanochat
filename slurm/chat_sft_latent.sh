#!/bin/bash

cd ~/looped_nanochat
uv sync
source .venv/bin/activate

source slurm/machine_config.sh
validate_config || exit 1

# Number of processes/GPUs to use (from machine_config.sh, defaults to 1)
NPROC_PER_NODE=${SLURM_GPUS:-1}

# --- Experiment config ---
SFT_MODEL=r4_1.35e19_s20           # SFT checkpoint tag to continue from
OUTPUT_TAG=r4_1.35e19_s20           # output tag (defaults to SFT_MODEL if empty)
DEVICE_BATCH_SIZE=8
RECUR_SAMPLES_PER_STEP=0            # 0 = fixed recurrence, >0 = sample per step
NO_DETACH_WARMUP=0                  # 0 = detach warm_start states (default), 1 = keep gradients
# --- Less common ---
EMBEDDING_LR=0.2
UNEMBEDDING_LR=0.004
MATRIX_LR=0.02
INIT_LR_FRAC=0.5

# --- Eval config ---
EVAL_TASKS=""                       # empty = all tasks
EVAL_NUM_RECUR=""                   # empty = model default. Comma-separated for sweep, e.g. "4,6,8"
EVAL_BATCH_SIZE=64                  # categorical eval batch size
EVAL_GEN_BATCH_SIZE=16              # generative eval batch size
SKIP_EVAL=0                         # set to 1 to skip chat eval after training

# --- Derived config ---
OUTPUT_TAG=${OUTPUT_TAG:-$SFT_MODEL}
RUN=$OUTPUT_TAG

# Build training args
TRAIN_ARGS="--model-tag $SFT_MODEL --output-tag $OUTPUT_TAG \
    --device-batch-size $DEVICE_BATCH_SIZE \
    --run $RUN \
    --recur-samples-per-step $RECUR_SAMPLES_PER_STEP \
    --embedding-lr $EMBEDDING_LR \
    --unembedding-lr $UNEMBEDDING_LR \
    --matrix-lr $MATRIX_LR \
    --init-lr-frac $INIT_LR_FRAC"

if [ "$NO_DETACH_WARMUP" -eq 1 ]; then
    TRAIN_ARGS="$TRAIN_ARGS --no-detach-warmup"
fi

# Run latent channel SFT
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft_latent \
    -- $TRAIN_ARGS

if [ $? -ne 0 ]; then
    echo "Latent SFT training failed"
    exit 1
fi

if [ "$SKIP_EVAL" -eq 1 ]; then
    echo "Skipping chat eval (SKIP_EVAL=1)"
    exit 0
fi

# Run chat eval
EVAL_ARGS="-i sft_latent -g $OUTPUT_TAG --batch-size $EVAL_BATCH_SIZE --gen-batch-size $EVAL_GEN_BATCH_SIZE"
if [ -n "$EVAL_TASKS" ]; then
    EVAL_ARGS="$EVAL_ARGS --task-name $EVAL_TASKS"
fi
if [ -n "$EVAL_NUM_RECUR" ]; then
    EVAL_ARGS="$EVAL_ARGS --num-recur $EVAL_NUM_RECUR"
fi
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval \
    -- $EVAL_ARGS
