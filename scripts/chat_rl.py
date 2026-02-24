"""
Reinforcement learning on GSM8K via "GRPO".

I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use DAPO style normalization that is token-level, not sequence-level.
4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.

1 GPU:
python -m scripts.chat_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import argparse
import itertools
import os
import time
from collections.abc import Generator
from contextlib import nullcontext

import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_gradient_stats,
    compute_init,
    get_base_dir,
    get_num_recur_for_microstep,
    print0,
    sample_num_recurs_for_step,
)
from nanochat.engine import Engine
from nanochat.report import get_report
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--output-tag", type=str, default=None, help="model tag to save to (defaults to model-tag)")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument(
    "--examples-per-step",
    type=int,
    default=16,
    help="total examples per optimization step across all ranks",
)
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument(
    "--unembedding-lr",
    type=float,
    default=0.004,
    help="learning rate for unembedding parameters (Adam)",
)
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.0,
    help="weight decay for embedding/unembedding parameters (Adam)",
)
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
# Recurrence options
parser.add_argument('-rws', '--use-rec-warm-start', action='store_true', help='Use recurrent warm-start (carry recurrent state when decoding tokens)')
parser.add_argument('-kb', '--kv-budget', type=int, default=1, help='KV-cache budget for recurrences. 1=cache final only, -1=full cache (budget=num_recur)')
parser.add_argument(
    "--recur-samples-per-step",
    type=int,
    default=0,
    help="Number of different num_recur values per step (0 = fixed, use model default)",
)
parser.add_argument('--latent-thoughts', action='store_true', help='Cache recurrent states from rollout generation and replay (detached) during training forward pass')
# Gradient tracking
parser.add_argument(
    "--track-gradients",
    type=str,
    choices=["none", "basic", "detailed"],
    default="basic",
    help="Gradient tracking level: none (disabled), basic (global norm), detailed (per-component norms)",
)
args = parser.parse_args()
user_config = vars(args).copy()
# latent_thoughts implies warm_start during generation (need states to cache)
generation_use_warm_start = args.use_rec_warm_start or args.latent_thoughts
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

# Init model and tokenizer
model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer) # for sampling rollouts

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")

# Shared num_recur for the current training step (maintains on-policy consistency)
current_step_num_recur = None


def resolve_kv_budget(kv_budget: int, num_recur: int | None) -> int:
    """Resolve kv_budget: -1 means full cache (budget = num_recur)."""
    if kv_budget == -1:
        return num_recur if num_recur is not None else int(model.config.train_recur_mean)
    return kv_budget


@torch.no_grad()
def get_batch() -> Generator[tuple[list[list[int]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None], None, None]:
    assistant_end = tokenizer.encode_special("<|assistant_end|>")  # ok to use this token, it's only for padding and isn't used in the loss.
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)  # each rank is responsible for different examples in the training data
    for example_idx in itertools.cycle(rank_indices):
        # First get the full conversation of both user and assistant messages
        conversation = train_task[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        model.eval()  # ensure the model is in eval mode
        generated_token_sequences = []
        masks = []
        latent_caches = []
        num_sampling_steps = args.num_samples // args.device_batch_size  # go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF  # positive half of int32
            with autocast_ctx:
                generated_token_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=args.device_batch_size,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=seed,  # must make sure to change the seed for each sampling step
                    num_recur=current_step_num_recur,  # use the step's num_recur for on-policy consistency
                    use_warm_start=generation_use_warm_start,
                    kv_budget=resolve_kv_budget(args.kv_budget, current_step_num_recur),
                )
            if args.latent_thoughts and engine._latent_cache is not None:
                latent_caches.append(torch.cat(engine._latent_cache, dim=1))  # (device_bs, num_steps+1, n_embd)
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate the rewards for each sample
        rewards = []
        for sample_tokens in generated_token_sequences:
            # Get just the generated tokens (after the prompt)
            generated_tokens = sample_tokens[prefix_length:]
            # Decode the generated response
            generated_text = tokenizer.decode(generated_tokens)
            # Calculate the reward
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad the sequences so that their lengths (in time) match
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # Stack up the sequences and masks into PyTorch tensors
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Generate autoregressive inputs and targets to the Transformer
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()  # clone to avoid in-place modification:
        targets[mask_ids[:, 1:] == 0] = -1  # <-- inplace modification right here. -1 is the ignore index
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # Calculate the advantages by simply subtracting the mean (instead of z-score (x-mu)/sigma)
        mu = rewards.mean()
        advantages = rewards - mu
        # Build warm_start tensor and mask for latent thought replay during training
        if args.latent_thoughts and latent_caches:
            T = inputs.size(1)  # max_length - 1
            n_embd = latent_caches[0].size(-1)
            warm_start_states = torch.zeros(inputs.size(0), T, n_embd, device=device, dtype=ptdtype)
            warm_start_mask = torch.zeros(inputs.size(0), T, device=device, dtype=torch.bool)
            offset = 0
            for batch_cache in latent_caches:
                bs = batch_cache.size(0)
                num_to_place = min(batch_cache.size(1), T - prefix_length)
                if num_to_place > 0:
                    warm_start_states[offset:offset + bs, prefix_length:prefix_length + num_to_place] = batch_cache[:, :num_to_place]
                    warm_start_mask[offset:offset + bs, prefix_length:prefix_length + num_to_place] = True
                offset += bs
        else:
            warm_start_states = None
            warm_start_mask = None
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        yield generated_token_sequences, inputs, targets, rewards, advantages, warm_start_states, warm_start_mask, prefix_length


# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k
def run_gsm8k_eval(
    task: GSM8K,
    tokenizer,
    engine: Engine,
    max_examples: int | None = None,
    num_samples: int = 1,
    max_completion_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 50,
    use_warm_start: bool = False,
    kv_budget: int = 1,
) -> Generator[dict, None, None]:
    """
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        assert num_samples <= args.device_batch_size  # usually this is true. we can add a loop if not...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
            num_recur=None,
            use_warm_start=use_warm_start,
            kv_budget=resolve_kv_budget(kv_budget, None),
        )
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record


# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]  # save the initial learning so we can decay easily later


# Learning rate scheduler: simple rampdown to zero over num_steps
def get_lr_multiplier(it: int) -> float:
    lrm: float = 1.0 - it / num_steps
    return lrm


# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}")  # total batch size in sequences/step
assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size  # per GPU
print0(f"Calculated examples per rank: {examples_per_rank}")

# Validate recur-samples-per-step (maps examples across ranks to num_recur values)
if args.recur_samples_per_step:
    if args.recur_samples_per_step > args.examples_per_step:
        raise ValueError(
            f"--recur-samples-per-step ({args.recur_samples_per_step}) cannot exceed total examples per step ({args.examples_per_step}). "
            f"Decrease --recur-samples-per-step or increase --examples-per-step."
        )
    if args.examples_per_step % args.recur_samples_per_step != 0:
        raise ValueError(
            f"Total examples per step ({args.examples_per_step} = {ddp_world_size} ranks * {examples_per_rank} examples) must be evenly divisible by --recur-samples-per-step ({args.recur_samples_per_step}). "
            f"Adjust --examples-per-step or --recur-samples-per-step."
        )
    examples_per_sample = args.examples_per_step // args.recur_samples_per_step
    print0(f"Recurrence sampling: {args.recur_samples_per_step} global samples per step ({examples_per_sample} examples each across {ddp_world_size} ranks)")
else:
    print0(f"Recurrence sampling: fixed (using model default)")

# Kick off the training loop
batch_iterator = get_batch()
for step in range(num_steps):
    # Pre-sample all num_recur values for this step (global across all ranks)
    sampled_num_recurs = sample_num_recurs_for_step(
        recur_samples_per_step=args.recur_samples_per_step,
        mean_recur=model.config.train_recur_mean,
        sigma=0.5,
        max_recur=model.config.train_recur_max,
        ddp=ddp,
        master_process=master_process,
        device=device,
    )
    # Set default for generation (eval uses model default, first example will override)
    current_step_num_recur = None

    # Evaluate the model once in a while and log to wandb
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device)  # pass@k for k=1..device_batch_size
        with autocast_ctx:
            records_iter = run_gsm8k_eval(
                val_task,
                tokenizer,
                engine,
                num_samples=args.device_batch_size,
                max_examples=args.eval_examples,
                temperature=1.0,
                use_warm_start=generation_use_warm_start,
                kv_budget=args.kv_budget,
            )
            records = list(records_iter)  # collect all records
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item()  # normalize by the total number of records
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
        wandb_run.log(
            {
                "step": step,
                **log_passk,
            }
        )

    # Forward/Backward on rollouts over multiple examples in the dataset
    rewards_list = []
    sequence_lengths = []
    had_backward = False  # track whether any backward pass was performed this step
    num_zero_advantage = 0  # count questions where all advantages are zero (no learning signal)
    num_truncated = 0  # count sequences that hit max_new_tokens without EOS
    num_total_sequences = 0
    surrogate_loss_accum = 0.0
    t_generation = 0.0
    t_training = 0.0
    for example_step in range(examples_per_rank):
        # Get num_recur for this example (on-policy: same for generation and training)
        current_step_num_recur = get_num_recur_for_microstep(
            sampled_num_recurs=sampled_num_recurs,
            micro_step=example_step,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            grad_accum_steps=examples_per_rank,
            recur_samples_per_step=args.recur_samples_per_step,
        )
        # Get one batch corresponding to one example in the training dataset
        t0 = time.perf_counter()
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all, ws_all, ws_mask_all, prefix_len = next(batch_iterator)
        t_generation += time.perf_counter() - t0
        # Track truncated completions (generated part hit max_new_tokens without EOS)
        for seq in sequences_all:
            num_total_sequences += 1
            if len(seq) - prefix_len >= args.max_new_tokens:
                num_truncated += 1
        # Skip training if all advantages are zero (no learning signal, e.g. all rewards identical)
        if (advantages_all == 0).all():
            num_zero_advantage += 1
            rewards_list.append(rewards_all.mean().item())
            sequence_lengths.extend(len(seq) for seq in sequences_all)
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Skipped (zero advantage)")
            continue
        # Evaluate the loss and gradients
        t0 = time.perf_counter()
        model.train()  # ensure the model is in train mode
        # We need one more loop because we can never exceed the device_batch_size
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # Latent thought replay: pass cached (detached) warm_start states from generation
            ws = ws_all[b0:b1].detach() if ws_all is not None else None
            ws_mask = ws_mask_all[b0:b1] if ws_mask_all is not None else None
            # Use the same num_recur that was used for rollout generation (on-policy consistency)
            # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction="none", num_recur=current_step_num_recur,
                              warm_start_state=ws, warm_start_mask=ws_mask).view_as(inputs)  # (B, T)
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            loss = -pg_obj
            loss.backward()
            had_backward = True
            surrogate_loss_accum += loss.item()
            print0(
                f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}"
            )
        t_training += time.perf_counter() - t0
        # For logging
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # A bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp:  # aggregate across ranks
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    # For logging: report num_recur info
    if sampled_num_recurs is None:
        logged_num_recur = int(model.config.train_recur_mean)
        logged_num_recur_str = f"{logged_num_recur}"
    else:
        logged_num_recur = sum(sampled_num_recurs) / len(sampled_num_recurs)
        logged_num_recur_str = f"{sampled_num_recurs} (mean={logged_num_recur:.1f})"
    print0(
        f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f} | num_recur: {logged_num_recur_str}"
    )
    wandb_run.log(
        {
            "step": step,
            "reward/mean": mean_reward,
            "reward/frac_zero_advantage": num_zero_advantage / examples_per_rank,
            "train/surrogate_loss": surrogate_loss_accum,
            "completions/clipped_ratio": num_truncated / max(num_total_sequences, 1),
            "completions/mean_length": mean_sequence_length,
            "num_recur": logged_num_recur,
            "time/generation": t_generation,
            "time/training": t_training,
        }
    )

    # Update the model parameters (skip only if NO rank had a backward pass)
    lrm = get_lr_multiplier(step)
    if ddp:
        # All ranks must agree: if any rank had a backward, all must participate in the optimizer step
        had_backward_tensor = torch.tensor(float(had_backward), device=device)
        dist.all_reduce(had_backward_tensor, op=dist.ReduceOp.MAX)
        any_rank_had_backward = had_backward_tensor.item() > 0
        # Ranks that didn't run backward still need valid (zero) grads for the collective ops
        if any_rank_had_backward and not had_backward:
            for p in model.parameters():
                if p.requires_grad and p.grad is None:
                    p.grad = torch.zeros_like(p)
    else:
        any_rank_had_backward = had_backward
    if any_rank_had_backward:
        model_health_stats = compute_gradient_stats(model, args.track_gradients)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        optimizer.step()
        model.zero_grad(set_to_none=True)
    else:
        model_health_stats = {}
        print0(f"Step {step}/{num_steps} | Skipping optimizer step (no gradients on any rank)")
    wandb_run.log(
        {
            "step": step,
            "lrm": lrm,
            **{f"model_health/{k}": v for k, v in model_health_stats.items()},
        }
    )

    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        size = model.config.size
        output_dirname = args.output_tag or args.model_tag or f"s{size}"  # e.g. s12
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        model_config_kwargs = model.config.__dict__  # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,  # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            },
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
section_name = "Chat RL"
if args.model_tag is not None:
    section_name += f" {args.model_tag}"
get_report().log(
    section=section_name,
    data=[
        user_config,  # CLI args
    ],
)

wandb_run.finish()  # wandb run finish
compute_cleanup()
