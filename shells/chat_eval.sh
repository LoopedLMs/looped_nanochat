#!/bin/bash

cd ~/looped_nanochat
uv sync
source .venv/bin/activate

source shells/_machine_config.sh
validate_config || exit 1

# Number of processes/GPUs to use (from _machine_config.sh, defaults to 1)
NPROC_PER_NODE=${NUM_GPUS:-1}

# Run chat evaluation
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval \
    -- -i sft -g s20 \
    --batch-size 32 \
    --kv-budget 1 \
    --use-rec-warm-start \
    --num-recur "2,4,8,16,32"