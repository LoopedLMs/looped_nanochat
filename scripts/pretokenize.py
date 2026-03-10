"""
Pre-tokenize and pre-pack the FineWeb-Edu dataset for efficient scaling law runs.

Produces ready-to-train Parquet files where each row is a BOS-aligned best-fit
packed sequence of T+1 tokens. During training, the dataloader just reads rows
directly — no tokenization or packing overhead.

Usage:
    uv run python -m scripts.pretokenize
    uv run python -m scripts.pretokenize --max-shards 10  # quick test with 10 shards
    uv run python -m scripts.pretokenize --push-to-hub ORG/REPO_NAME
"""

import argparse
import json
import random
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer


def pack_batch(doc_buffer: list[list[int]], row_capacity: int, rows_per_batch: int) -> list[list[int]]:
    """
    BOS-aligned best-fit pack documents from doc_buffer into fixed-length rows.

    Same algorithm as nanochat/dataloader.py: for each row, find the largest doc
    that fits entirely, repeat until nothing fits, then crop the shortest doc to
    fill remaining space. Every row starts with BOS.

    Args:
        doc_buffer: Mutable list of tokenized documents (consumed in-place).
        row_capacity: Number of tokens per row (T+1).
        rows_per_batch: Number of rows to produce.

    Returns:
        List of packed token lists, each of length row_capacity.
    """
    rows = []
    for _ in range(rows_per_batch):
        if not doc_buffer:
            break
        row = []
        pos = 0
        while pos < row_capacity and doc_buffer:
            remaining = row_capacity - pos

            # Find largest doc that fits entirely
            best_idx = -1
            best_len = 0
            for i, doc in enumerate(doc_buffer):
                doc_len = len(doc)
                if doc_len <= remaining and doc_len > best_len:
                    best_idx = i
                    best_len = doc_len

            if best_idx >= 0:
                doc = doc_buffer.pop(best_idx)
                row.extend(doc)
                pos += len(doc)
            else:
                # No doc fits — crop shortest to fill remaining space
                shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                doc = doc_buffer.pop(shortest_idx)
                row.extend(doc[:remaining])
                pos += remaining

        assert len(row) == row_capacity, f"Row length {len(row)} != {row_capacity}"
        rows.append(row)
    return rows


def pretokenize(
    output_dir: Path,
    seq_len: int = 2048,
    buffer_size: int = 1000,
    rows_per_batch: int = 256,
    rows_per_shard: int = 10_000,
    max_shards: int = -1,
    tokenizer_threads: int = 8,
    tokenizer_batch_size: int = 128,
    split: str = "train",
):
    """
    Pre-tokenize and pre-pack the dataset into Parquet shards.

    Args:
        output_dir: Where to write output Parquet files.
        seq_len: Sequence length T (rows will have T+1 tokens for input/target offset).
        buffer_size: Document buffer size for best-fit packing.
        rows_per_batch: Documents packed independently per batch (no cross-batch state).
        rows_per_shard: Rows per output Parquet shard file.
        max_shards: Stop after this many input shards (-1 = all).
        tokenizer_threads: Threads for tiktoken batch encoding.
        tokenizer_batch_size: Documents per tokenizer batch call.
        split: "train" or "val".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    row_capacity = seq_len + 1  # T+1 for input/target offset

    parquet_paths = list_parquet_files()
    if split == "train":
        parquet_paths = parquet_paths[:-1]
    else:
        parquet_paths = parquet_paths[-1:]

    if max_shards > 0:
        parquet_paths = parquet_paths[:max_shards]

    total_rows = 0
    total_tokens_packed = 0
    total_docs_tokenized = 0
    shard_idx = 0
    shard_rows: list[list[int]] = []
    doc_buffer: list[list[int]] = []

    t_start = time.time()

    for pq_idx, filepath in enumerate(parquet_paths):
        pf = pq.ParquetFile(filepath)
        shard_name = Path(filepath).name
        t_shard = time.time()

        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()

            # Tokenize in batches
            for i in range(0, len(texts), tokenizer_batch_size):
                batch = texts[i : i + tokenizer_batch_size]
                token_lists = tokenizer.encode(batch, prepend=bos_token, num_threads=tokenizer_threads)
                doc_buffer.extend(token_lists)
                total_docs_tokenized += len(token_lists)

                # Pack whenever buffer is full enough
                while len(doc_buffer) >= buffer_size:
                    packed_rows = pack_batch(doc_buffer, row_capacity, rows_per_batch)
                    if not packed_rows:
                        break
                    shard_rows.extend(packed_rows)
                    total_rows += len(packed_rows)
                    total_tokens_packed += len(packed_rows) * row_capacity

                    # Write shard when full
                    while len(shard_rows) >= rows_per_shard:
                        _write_shard(output_dir, shard_idx, shard_rows[:rows_per_shard], split)
                        shard_rows = shard_rows[rows_per_shard:]
                        shard_idx += 1

        elapsed = time.time() - t_shard
        total_elapsed = time.time() - t_start
        print(
            f"[{pq_idx + 1}/{len(parquet_paths)}] {shard_name}: "
            f"{elapsed:.1f}s | "
            f"docs: {total_docs_tokenized:,} | "
            f"rows: {total_rows:,} | "
            f"shards written: {shard_idx} | "
            f"total: {total_elapsed:.0f}s"
        )

    # Drain remaining docs in buffer
    while doc_buffer:
        packed_rows = pack_batch(doc_buffer, row_capacity, min(rows_per_batch, len(doc_buffer)))
        if not packed_rows:
            break
        shard_rows.extend(packed_rows)
        total_rows += len(packed_rows)
        total_tokens_packed += len(packed_rows) * row_capacity

    # Write final shard
    if shard_rows:
        _write_shard(output_dir, shard_idx, shard_rows, split)
        shard_idx += 1

    # Write metadata
    meta = {
        "tokenizer": "RustBPETokenizer",
        "vocab_size": tokenizer.get_vocab_size(),
        "bos_token_id": bos_token,
        "seq_len": seq_len,
        "row_capacity": row_capacity,
        "total_rows": total_rows,
        "total_tokens": total_tokens_packed,
        "total_docs": total_docs_tokenized,
        "num_shards": shard_idx,
        "rows_per_batch": rows_per_batch,
        "buffer_size": buffer_size,
        "split": split,
        "input_shards": len(parquet_paths),
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total_elapsed = time.time() - t_start
    print(f"\nDone! {total_rows:,} rows in {shard_idx} shards ({total_elapsed:.0f}s)")
    print(f"Total tokens: {total_tokens_packed:,} ({total_tokens_packed / 1e9:.2f}B)")
    print(f"Output: {output_dir}")
    return meta


def _write_shard(output_dir: Path, shard_idx: int, rows: list[list[int]], split: str):
    """Write a single Parquet shard."""
    assert all(
        max(row) <= 65535 for row in rows
    ), "Token values exceed uint16 range (65535). Use a tokenizer with vocab_size <= 65536."
    # Shuffle rows so batches get a random mix of packing patterns
    # (without this, best-fit produces a systematic ordering: rows with
    # few long docs first, rows with many short docs last)
    random.shuffle(rows)
    # Store tokens as fixed-length lists of uint16
    table = pa.table({"tokens": pa.array(rows, type=pa.list_(pa.uint16()))})
    filename = f"{split}-{shard_idx:05d}.parquet"
    pq.write_table(table, output_dir / filename)


def push_to_hub(output_dir: Path, repo_id: str):
    """Upload the pre-packed dataset and tokenizer to HuggingFace Hub.

    Repo layout:
        tokenizer/tokenizer.pkl  — tokenizer (small, downloadable separately)
        train-*.parquet          — pre-packed training shards
        meta.json                — dataset metadata
    """
    from huggingface_hub import HfApi

    # Save tokenizer into output_dir so it gets uploaded together
    tokenizer = get_tokenizer()
    tokenizer.save(str(output_dir / "tokenizer"))

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


def download_from_hub(
    repo_id: str,
    local_dir: Path,
    only: str | None = None,
):
    """Download a pre-packed dataset from HuggingFace Hub.

    Args:
        repo_id: HuggingFace dataset repo ID.
        local_dir: Local directory to download into.
        only: Optional filter — "tokenizer" for just the tokenizer,
              "data" for just shards + meta, None for everything.
    """
    from huggingface_hub import snapshot_download

    allow_patterns = None
    if only == "tokenizer":
        allow_patterns = ["tokenizer/*"]
    elif only == "data":
        allow_patterns = ["*.parquet", "meta.json"]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    print(f"Downloaded to {local_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-tokenize and pre-pack FineWeb-Edu for training")
    parser.add_argument("--seq-len", type=int, default=2048, help="sequence length T (default: 2048)")
    parser.add_argument("--buffer-size", type=int, default=1000, help="document buffer size for best-fit packing")
    parser.add_argument("--rows-per-batch", type=int, default=256, help="rows packed per independent batch")
    parser.add_argument("--rows-per-shard", type=int, default=10000, help="rows per output Parquet shard")
    parser.add_argument("--max-shards", type=int, default=-1, help="max input shards to process (-1 = all)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="dataset split")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory (default: NANOCHAT_BASE_DIR/prepacked_T<seq_len>)")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HuggingFace repo ID to upload to (e.g. ORG/dataset-name)")
    parser.add_argument("--download", type=str, default=None, help="HuggingFace repo ID to download from (e.g. ORG/dataset-name)")
    parser.add_argument("--only", type=str, default=None, choices=["tokenizer", "data"], help="download only tokenizer or data (used with --download)")
    args = parser.parse_args()

    if args.output_dir is None:
        base_dir = get_base_dir()
        output_dir = Path(base_dir) / f"prepacked_T{args.seq_len}"
    else:
        output_dir = Path(args.output_dir)

    if args.download:
        download_from_hub(args.download, output_dir, only=args.only)
    else:
        meta = pretokenize(
            output_dir=output_dir,
            seq_len=args.seq_len,
            buffer_size=args.buffer_size,
            rows_per_batch=args.rows_per_batch,
            rows_per_shard=args.rows_per_shard,
            max_shards=args.max_shards,
            split=args.split,
        )

        if args.push_to_hub:
            push_to_hub(output_dir, args.push_to_hub)
