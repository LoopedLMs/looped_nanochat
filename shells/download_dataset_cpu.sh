#!/bin/bash

cd ~/looped_nanochat
uv sync
source .venv/bin/activate

source shells/machine_config.sh
validate_config || exit 1

python -m nanochat.dataset -n 700
