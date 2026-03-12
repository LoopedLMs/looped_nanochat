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
#   DOWNLOAD_HUB=       HuggingFace repo ID to download from
#                        (default: KristianS7/nanochat-prepacked-fineweb-edu)
#   ONLY=               Download only "tokenizer" or "data" (default: both)
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
DOWNLOAD_HUB="${DOWNLOAD_HUB:-KristianS7/nanochat-prepacked-fineweb-edu}"
OUTPUT_DIR="$NANOCHAT_BASE_DIR/prepacked_T${SEQ_LEN}"

# --- Download mode (default) ---
echo "=============================================="
echo "Downloading pre-packed dataset from HuggingFace"
echo "  Repo:   $DOWNLOAD_HUB"
echo "  Output: $OUTPUT_DIR"
if [ -n "$ONLY" ]; then
    echo "  Only:   $ONLY"
fi
echo "=============================================="

DOWNLOAD_FLAGS="--download=$DOWNLOAD_HUB --output-dir=$OUTPUT_DIR"
if [ -n "$ONLY" ]; then
    DOWNLOAD_FLAGS="$DOWNLOAD_FLAGS --only=$ONLY"
fi

python -u -m scripts.pretokenize $DOWNLOAD_FLAGS

echo ""
echo "Done! Output: $OUTPUT_DIR"
echo "To use in training: PREPACKED_DIR=$OUTPUT_DIR ./shells/_submit.sh shells/scaling_laws.sh"
