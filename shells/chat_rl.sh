#!/bin/bash

cd ~/looped_nanochat
uv sync
source .venv/bin/activate

source shells/machine_config.sh
validate_config || exit 1

# Number of processes/GPUs to use (from machine_config.sh, defaults to 1)
NPROC_PER_NODE=${NUM_GPUS:-1}

# --- Model config ---
MODEL_TAG=r4_1.35e19_s20           # SFT model tag to load from
OUTPUT_TAG=r4_1.35e19_s20_LT           # output tag (defaults to MODEL_TAG if empty)

# --- Training config ---
DEVICE_BATCH_SIZE=16

# --- Optimization ---
EMBEDDING_LR=0.2
UNEMBEDDING_LR=0.004
MATRIX_LR=0.02
WEIGHT_DECAY=0.0
INIT_LR_FRAC=0.05

# --- Recurrence options ---
USE_REC_WARM_START=false            # carry recurrent state when decoding tokens
KV_BUDGET=-1                         # -1 = full cache (budget=num_recur), 1 = cache final only
RECUR_SAMPLES_PER_STEP=0             # 0 = fixed recurrence, >0 = sample per step
LATENT_THOUGHTS=true               # cache recurrent states from rollout and replay during training

# --- Eval / checkpointing ---
EVAL_EVERY=60                       # evaluate pass@k every N steps
EVAL_EXAMPLES=400                   # number of examples for pass@k evaluation
SAVE_EVERY=60                       # save checkpoint every N steps

# --- Derived config ---
OUTPUT_TAG=${OUTPUT_TAG:-$MODEL_TAG}
RUN=$OUTPUT_TAG

# Build optional arguments
EXTRA_ARGS=""
if [ "$USE_REC_WARM_START" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-rec-warm-start"
fi
if [ "$LATENT_THOUGHTS" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --latent-thoughts"
fi

# Run chat RL
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl \
    -- --model-tag $MODEL_TAG --output-tag $OUTPUT_TAG \
    --run $RUN \
    --device-batch-size $DEVICE_BATCH_SIZE \
    --embedding-lr $EMBEDDING_LR \
    --unembedding-lr $UNEMBEDDING_LR \
    --matrix-lr $MATRIX_LR \
    --weight-decay $WEIGHT_DECAY \
    --init-lr-frac $INIT_LR_FRAC \
    --recur-samples-per-step $RECUR_SAMPLES_PER_STEP \
    --kv-budget $KV_BUDGET \
    --eval-every $EVAL_EVERY \
    --eval-examples $EVAL_EXAMPLES \
    --save-every $SAVE_EVERY \
    $EXTRA_ARGS
