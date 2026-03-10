#!/bin/bash
set -eo pipefail

# Pre-tokenize and pre-pack the dataset for efficient scaling law runs.
# Produces ready-to-train Parquet shards where each row is a packed sequence.
#
# Usage:
#   ./shells/_submit.sh shells/pretokenize_cpu.sh
#   ./shells/_submit.sh shells/pretokenize_cpu.sh -- --time=24:00:00
#
# Optional env vars:
#   SEQ_LEN=2048        Sequence length T (default: 2048)
#   MAX_SHARDS=-1       Limit input shards for quick testing (default: all)
#   PUSH_TO_HUB=        HuggingFace repo ID to upload to (default: skip)

cd ~/looped_nanochat

source shells/_machine_config.sh
validate_config || exit 1

uv sync
source .venv/bin/activate

SEQ_LEN="${SEQ_LEN:-2048}"
MAX_SHARDS="${MAX_SHARDS:--1}"
OUTPUT_DIR="$NANOCHAT_BASE_DIR/prepacked_T${SEQ_LEN}"

echo "=============================================="
echo "Pre-tokenizing dataset"
echo "  Sequence length: $SEQ_LEN"
echo "  Max shards:      $MAX_SHARDS"
echo "  Output:          $OUTPUT_DIR"
echo "=============================================="

OPTIONAL_FLAGS=""
if [ "$MAX_SHARDS" != "-1" ]; then
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --max-shards=$MAX_SHARDS"
fi
if [ -n "$PUSH_TO_HUB" ]; then
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --push-to-hub=$PUSH_TO_HUB"
fi

python -m scripts.pretokenize \
    --seq-len="$SEQ_LEN" \
    --output-dir="$OUTPUT_DIR" \
    $OPTIONAL_FLAGS

echo ""
echo "Done! Output: $OUTPUT_DIR"
echo "To use in training: PREPACKED_DIR=$OUTPUT_DIR ./shells/_submit.sh shells/scaling_laws.sh"
