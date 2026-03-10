"""Inspect a pre-packed dataset: validate metadata, print stats, and show detokenized rows.

Usage:
    uv run python dev/debugging/inspect_prepacked.py /path/to/prepacked_T2048
    uv run python dev/debugging/inspect_prepacked.py /path/to/prepacked_T2048 --num-rows 5 --shards 0 2 10
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from nanochat.tokenizer import get_tokenizer


def inspect(prepacked_dir: Path, num_rows: int = 3, shard_indices: list[int] | None = None):
    tokenizer = get_tokenizer()
    bos_id = tokenizer.get_bos_token_id()

    # --- Metadata ---
    meta_path = prepacked_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print("=== meta.json ===")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        row_capacity = meta["row_capacity"]
    else:
        print("WARNING: no meta.json found")
        row_capacity = None

    # --- Shards ---
    shards = sorted(prepacked_dir.glob("train-*.parquet"))
    print(f"\n=== {len(shards)} shards ===")
    if not shards:
        print("No shards found!")
        return

    # Pick which shards to inspect
    if shard_indices is None:
        shard_indices = [0, len(shards) // 2, len(shards) - 1]
    shard_indices = [i for i in shard_indices if 0 <= i < len(shards)]

    for si in shard_indices:
        shard_path = shards[si]
        pf = pq.ParquetFile(shard_path)
        table = pf.read()
        tokens_col = table.column("tokens")
        total_rows_in_shard = len(tokens_col)

        print(f"\n--- shard {si}: {shard_path.name} ({total_rows_in_shard} rows, {pf.num_row_groups} row groups) ---")

        for row_idx in range(min(num_rows, total_rows_in_shard)):
            tokens = tokens_col[row_idx].as_py()

            # Validate length
            if row_capacity and len(tokens) != row_capacity:
                print(f"  [!] row {row_idx}: length {len(tokens)} != expected {row_capacity}")

            # Count documents (BOS-delimited)
            bos_positions = [i for i, t in enumerate(tokens) if t == bos_id]
            num_docs = len(bos_positions)

            print(f"\n  row {row_idx}: {len(tokens)} tokens, {num_docs} docs (BOS at {bos_positions})")

            # Detokenize each document segment
            for doc_idx in range(num_docs):
                start = bos_positions[doc_idx]
                end = bos_positions[doc_idx + 1] if doc_idx + 1 < num_docs else len(tokens)
                doc_tokens = tokens[start:end]
                text = tokenizer.decode(doc_tokens)
                # Truncate for display
                preview = text[:200].replace("\n", "\\n")
                if len(text) > 200:
                    preview += "..."
                print(f"    doc {doc_idx} ({end - start} tokens): {preview}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect pre-packed dataset")
    parser.add_argument("prepacked_dir", type=Path, help="path to prepacked directory")
    parser.add_argument("--num-rows", type=int, default=3, help="rows to show per shard (default: 3)")
    parser.add_argument("--shards", type=int, nargs="*", default=None, help="shard indices to inspect (default: first, middle, last)")
    args = parser.parse_args()
    inspect(args.prepacked_dir, num_rows=args.num_rows, shard_indices=args.shards)
