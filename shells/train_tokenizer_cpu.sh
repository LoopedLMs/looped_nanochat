#!/bin/bash

cd ~/looped_nanochat
uv sync
source .venv/bin/activate

source shells/machine_config.sh
validate_config || exit 1

# train the tokenizer with vocab size 2**16 = 65536 on ~10B characters of data
# With value embeddings, we could half the vocab size (see nanochat)
python -m scripts.tok_train --vocab-size 65536 --max-chars 10_000_000_000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval
