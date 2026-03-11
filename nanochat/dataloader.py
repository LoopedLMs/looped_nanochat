"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import json
from pathlib import Path

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files()
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

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
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets


# =============================================================================
# Pre-packed dataloader: reads pre-tokenized + pre-packed Parquet files
# =============================================================================

def _list_prepacked_shards(prepacked_dir: str, split: str = "train") -> list[Path]:
    """List pre-packed Parquet shard files in sorted order."""
    d = Path(prepacked_dir)
    files = sorted(d.glob(f"{split}-*.parquet"))
    assert files, f"No {split}-*.parquet files found in {prepacked_dir}"
    return files


def prepacked_data_loader(
    prepacked_dir: str,
    B: int,
    T: int,
    device: str = "cuda",
    resume_state: dict | None = None,
):
    """
    Dataloader that reads pre-packed Parquet files directly.

    Each row in the Parquet files is a packed sequence of T+1 tokens, already
    BOS-aligned and best-fit packed. This loader just reads rows and yields
    (inputs, targets, state_dict) with zero data processing overhead.

    DDP sharding: rank k reads every Nth row (N = world_size).

    Args:
        prepacked_dir: Directory containing pre-packed Parquet shards.
        B: Batch size (rows per yield).
        T: Sequence length (each row has T+1 tokens).
        device: Target device for tensors.
        resume_state: State dict from a previous yield to resume from.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    shard_paths = _list_prepacked_shards(prepacked_dir)

    # Validate that pre-packed data matches expected sequence length
    meta_path = Path(prepacked_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["row_capacity"] == T + 1, (
            f"Pre-packed data has row_capacity={meta['row_capacity']} but T+1={T + 1}"
        )

    use_cuda = device == "cuda"
    row_capacity = T + 1

    # Pre-allocate buffers (same pattern as the tokenizing loader)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)

    # Resume state: skip directly to the right shard/row instead of scanning
    resume_shard_idx = 0
    resume_rg_idx = 0
    resume_row_in_rg = 0
    epoch = 1
    if resume_state and "shard_idx" in resume_state:
        resume_shard_idx = resume_state["shard_idx"]
        resume_rg_idx = resume_state["rg_idx"]
        resume_row_in_rg = resume_state["row_in_rg"]
        epoch = resume_state.get("epoch", 1)

    # Start global_row_idx aligned to batch boundary so batch_pos starts at 0
    resume_row_idx = (resume_state or {}).get("row_idx", 0)
    global_row_idx = (resume_row_idx // (B * ddp_world_size)) * (B * ddp_world_size)

    while True:  # infinite iteration (multi-epoch)
        for shard_idx, shard_path in enumerate(shard_paths):
            if shard_idx < resume_shard_idx:
                continue

            pf = pq.ParquetFile(shard_path)
            for rg_idx in range(pf.num_row_groups):
                if shard_idx == resume_shard_idx and rg_idx < resume_rg_idx:
                    continue

                table = pf.read_row_group(rg_idx)
                all_rows = table.column("tokens").to_pylist()

                assert len(all_rows) >= ddp_world_size, (
                    f"Row group {rg_idx} in {shard_path} has {len(all_rows)} rows "
                    f"but ddp_world_size={ddp_world_size}. All row groups must have "
                    f"at least as many rows as GPUs to prevent DDP hangs."
                )

                start_row = resume_row_in_rg if (shard_idx == resume_shard_idx and rg_idx == resume_rg_idx) else ddp_rank
                for row_in_rg in range(start_row, len(all_rows), ddp_world_size):
                    tokens = all_rows[row_in_rg]
                    batch_pos = (global_row_idx // ddp_world_size) % B

                    row_buffer[batch_pos, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

                    if batch_pos == B - 1:
                        # Full batch — copy to GPU
                        cpu_inputs.copy_(row_buffer[:, :-1])
                        cpu_targets.copy_(row_buffer[:, 1:])
                        state_dict = {
                            "row_idx": global_row_idx,
                            "shard_idx": shard_idx,
                            "rg_idx": rg_idx,
                            "row_in_rg": row_in_rg + ddp_world_size,
                            "epoch": epoch,
                        }
                        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
                        yield inputs, targets, state_dict

                    global_row_idx += ddp_world_size

        # Reset skip state for next epoch
        resume_shard_idx = 0
        resume_rg_idx = 0
        resume_row_in_rg = 0
        epoch += 1
