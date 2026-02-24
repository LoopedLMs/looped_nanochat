"""
Evaluate the Chat model.
All the generic code lives here, and all the evaluation-specific
code lives in nanochat directory and is imported from here.

Example runs:
python -m scripts.chat_eval -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -a ARC-Easy
"""

import argparse
import os
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist

from nanochat.checkpoint_manager import load_model
from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_dist_info,
    print0,
)
from nanochat.engine import Engine
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.spellingbee import SpellingBee

# -----------------------------------------------------------------------------
# Generative evaluation loop (we go one problem at a time, sample, evaluate)


def run_generative_eval(
    task_object,
    tokenizer,
    model,
    engine,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    max_problems=None,
    num_recur=None,
    use_warm_start=False,
    kv_budget=1,
    batch_size=1,
):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Indices assigned to this rank
    rank_indices = list(range(ddp_rank, num_problems, ddp_world_size))

    # Use batched evaluation when num_samples == 1 and batch_size > 1
    use_batching = num_samples == 1 and batch_size > 1

    num_passed, total = 0, 0

    if use_batching:
        # Process problems in batches
        for batch_start in range(0, len(rank_indices), batch_size):
            batch_indices = rank_indices[batch_start : batch_start + batch_size]
            conversations = [task_object[i] for i in batch_indices]
            encoded_prompts = [tokenizer.render_for_completion(conv) for conv in conversations]

            # Batched generation: one completion per prompt
            results, _ = engine.generate_multi(
                encoded_prompts,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                num_recur=num_recur,
                use_warm_start=use_warm_start,
                kv_budget=kv_budget,
            )

            # Decode and evaluate each result
            for idx, (conv, encoded_prompt, result_tokens) in enumerate(
                zip(conversations, encoded_prompts, results)
            ):
                prefix_length = len(encoded_prompt)
                completion = tokenizer.decode(result_tokens[prefix_length:])
                passed = task_object.evaluate(conv, completion)
                total += 1
                num_passed += int(passed)

            print(
                f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100 * num_passed / total:.2f}%)",
                end="",
                flush=True,
            )
    else:
        # Original sequential path (num_samples > 1 or batch_size == 1)
        for i in rank_indices:
            conversation = task_object[i]
            encoded_prompt = tokenizer.render_for_completion(conversation)
            results, _ = engine.generate_batch(
                encoded_prompt,
                num_samples=num_samples,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                num_recur=num_recur,
                use_warm_start=use_warm_start,
                kv_budget=kv_budget,
            )
            prefix_length = len(encoded_prompt)
            completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
            outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
            passed = any(outcomes)
            total += 1
            num_passed += int(passed)

            print(
                f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100 * num_passed / total:.2f}%)",
                end="",
                flush=True,
            )

    # Finish the in-place progress line with a newline before final summary
    print()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100 * num_passed / total:.2f}%)")

    # Return the accuracy
    return num_passed / total


# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.


def run_categorical_eval(
    task_object,
    tokenizer,
    model,
    batch_size,
    max_problems=None,
    num_recur=None,
):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()  # use BOS as pad token is ok, these positions are ignored

    # We'll process batches of independent problems at a time because there is no sampling needed
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    def ceil_div(x, y):
        return -(-x // y)

    num_batches = ceil_div(num_problems, batch_size)

    # Run the evaluation
    letter_to_id_cache = {}  # many letters will repeat often, let's save the tokenizer some work
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # Prepare the batch of problems. They might all be of different length, so we pad/collate them.
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations]  # TODO: remake the way this works
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids]  # where the last token is (and the predicted answer)
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        # Get the logits for the whole batch of conversations in parallel (efficiency win here)
        with torch.no_grad():
            logits, _ = model(prompt_ids, num_recur=num_recur)  # (B, T, V)

        # Focus on the available answer on just the letters corresponding to choices
        # Note that this helps the evaluation a lot because it specifically narrows the focus to only the available letters
        # The much harder alternative would be to just generate from the Assistant and check if it responded with the correct
        # letter (e.g. A, B, C, D), but evaluations typically make the task easier in this way.
        for idx, conversation in enumerate(conversations):
            # get the token ids of all the available letters of this problem
            letters = conversation["letters"]
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            # focus logits just down to the answer position and the available letters of the answer
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            # get the argmax letter (the predicted answer)
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            # evaluate the outcome
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed / total
    print0(f"Final: {num_passed}/{total} ({100 * average:.2f}%)")
    return average


# -----------------------------------------------------------------------------


def run_chat_eval(
    task_name,
    model,
    tokenizer,
    engine,
    cat_batch_size=8,
    gen_batch_size=4,
    num_samples=1,
    max_new_tokens=512,
    temperature=0.0,
    top_k=50,
    max_problems=None,
    num_recur=None,
    use_warm_start=False,
    kv_budget=1,
):
    # Create the evaluation object
    task_module = {
        "HumanEval": HumanEval,
        "MMLU": partial(MMLU, subset="all", split="test"),
        "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
        "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
        "GSM8K": partial(GSM8K, subset="main", split="test"),
        "SpellingBee": partial(SpellingBee, size=256, split="test"),
    }[task_name]
    task_object = task_module()
    # Run the evaluation
    if task_object.eval_type == "generative":
        acc = run_generative_eval(
            task_object,
            tokenizer,
            model,
            engine,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            max_problems=max_problems,
            num_recur=num_recur,
            use_warm_start=use_warm_start,
            kv_budget=kv_budget,
            batch_size=gen_batch_size,
        )
    elif task_object.eval_type == "categorical":
        acc = run_categorical_eval(
            task_object,
            tokenizer,
            model,
            cat_batch_size,
            max_problems=max_problems,
            num_recur=num_recur,
        )
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    return acc


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="Source of the model: sft|sft_latent|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Task name. Default = all tasks. Use | to split multiple tasks.")
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size for categorical evaluation')
    parser.add_argument('-bg', '--gen-batch-size', type=int, default=4, help='Batch size for generative evaluation (when num_samples=1). Smaller than -b because decode KV cache is much larger.')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='Max problems to evaluate')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    parser.add_argument('-r', '--num-recur', type=str, default=None, help='Number of recurrences for recursive transformer (optional, uses model default if not specified)')
    parser.add_argument('-rws', '--use-rec-warm-start', action='store_true', help='Use recurrent warm-start (carry recurrent state when decoding tokens)')
    parser.add_argument('-kb', '--kv-budget', type=str, default='1', help='KV-cache budget for recurrences. Single value broadcasts to all num-recur. Comma-separated list must match num-recur length. -1 means "same as num-recur" (cache all iterations). Default=1')
    args = parser.parse_args()

    # Parse num_recur argument - can be single value or comma-separated list
    # None is resolved to the model's default after loading
    recur_values = [None]
    if args.num_recur is not None:
        recur_values = [int(x.strip()) for x in args.num_recur.split(",")]

    # Parse kv_budget argument - same format as num_recur, with -1 meaning "match num_recur"
    raw_kv_budgets = [int(x.strip()) for x in args.kv_budget.split(",")]
    if len(raw_kv_budgets) == 1:
        # Broadcast single value to all recur_values
        raw_kv_budgets = raw_kv_budgets * len(recur_values)
    elif len(raw_kv_budgets) != len(recur_values):
        parser.error(f"--kv-budget list length ({len(raw_kv_budgets)}) must match --num-recur length ({len(recur_values)})")

    # Resolve -1 to match the corresponding num_recur value
    kv_budgets = []
    for kv_b, nr in zip(raw_kv_budgets, recur_values):
        if kv_b == -1:
            if nr is None:
                parser.error("--kv-budget -1 requires explicit --num-recur (cannot resolve against model default)")
            kv_budgets.append(nr)
        else:
            assert kv_b >= 1, f"kv_budget must be >= 1 or -1, got {kv_b}"
            kv_budgets.append(kv_b)

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # Resolve None num_recur to the model's default
    recur_values = [int(model.config.train_recur_mean) if v is None else v for v in recur_values]

    # Get the tasks to evaluate on
    all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
    baseline_accuracies = {
        "ARC-Easy": 0.25,  # multiple choice 1 of 4 => 25%
        "ARC-Challenge": 0.25,  # multiple choice 1 of 4 => 25%
        "MMLU": 0.25,  # multiple choice 1 of 4 => 25%
        "GSM8K": 0.0,  # open-ended => 0%
        "HumanEval": 0.0,  # open-ended => 0%
        "SpellingBee": 0.0,  # open-ended => 0%
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split("|")

    # Summary CSV: one row per (num_recur, kv_budget) with per-task accuracies + ChatCORE
    base_dir = get_base_dir()
    eval_dir = os.path.join(base_dir, "chat_eval")
    os.makedirs(eval_dir, exist_ok=True)
    base_slug = f"{args.source}_{args.model_tag}_step{meta['step']:06d}"
    summary_csv_path = os.path.join(eval_dir, f"{base_slug}.csv")

    def read_completed_rows(csv_path: str) -> set[str]:
        """Read CSV and return set of 'num_recur,kv_budget,warm_start' keys already present."""
        if not os.path.exists(csv_path):
            return set()
        completed = set()
        with open(csv_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or line.startswith("num_recur"):
                    continue
                parts = line.split(",", 3)
                if len(parts) >= 3:
                    completed.add(f"{parts[0].strip()},{parts[1].strip()},{parts[2].strip()}")
        return completed

    completed_rows = read_completed_rows(summary_csv_path)

    # Write comment header with hyperparameters (once, when creating the file)
    if ddp_rank == 0 and not os.path.exists(summary_csv_path):
        with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(f"# source={args.source}, model_tag={args.model_tag}, step={meta['step']}"
                    f", temperature={args.temperature}, top_k={args.top_k}"
                    f", num_samples={args.num_samples}, max_new_tokens={args.max_new_tokens}"
                    f", warm_start={args.use_rec_warm_start}, dtype={args.dtype}"
                    f", tasks={';'.join(task_names)}\n")

    warm_start = args.use_rec_warm_start
    for num_recur, kv_budget in zip(recur_values, kv_budgets):
        row_key = f"{num_recur},{kv_budget},{warm_start}"
        if row_key in completed_rows:
            print0(f"\nSkipping num_recur={num_recur}, kv_budget={kv_budget}, warm_start={warm_start} (already in CSV)")
            continue

        print0(f"\n{'=' * 80}")
        print0(f"Evaluating with num_recur={num_recur if num_recur is not None else 'default'}, kv_budget={kv_budget}")
        print0(f"{'=' * 80}")

        results = {}
        for task_name in task_names:
            with autocast_ctx:
                acc = run_chat_eval(
                    task_name,
                    model,
                    tokenizer,
                    engine,
                    cat_batch_size=args.batch_size,
                    gen_batch_size=args.gen_batch_size,
                    num_samples=args.num_samples,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    max_problems=args.max_problems,
                    num_recur=num_recur,
                    use_warm_start=args.use_rec_warm_start,
                    kv_budget=kv_budget,
                )
                results[task_name] = acc
                print0(f"{task_name} accuracy: {100 * acc:.2f}%")

        # Calculate ChatCORE metric (mean centered accuracy across all tasks)
        missing_tasks = [t for t in all_tasks if t not in results]
        full_results = {t: results.get(t, 0.0) for t in all_tasks}
        centered_mean = 0
        for task_name, acc in full_results.items():
            baseline_acc = baseline_accuracies[task_name]
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(all_tasks)
        if missing_tasks:
            print0(f"  Note: {', '.join(missing_tasks)} not tested, assumed 0 for ChatCORE")
        print0(f"ChatCORE metric: {chatcore_metric:.4f}")

        # Append summary row to CSV
        if ddp_rank == 0:
            columns = ["num_recur", "kv_budget", "warm_start"]
            values = [str(num_recur), str(kv_budget), str(warm_start)]
            for task in task_names:
                columns.append(task)
                values.append(f"{results.get(task, 0.0):.6f}")
            columns.append("ChatCORE")
            values.append(f"{chatcore_metric:.6f}")

            # Write header row if this is the first data row
            file_size = os.path.getsize(summary_csv_path) if os.path.exists(summary_csv_path) else 0
            write_header = file_size == 0 or all(
                line.startswith("#") for line in open(summary_csv_path, encoding="utf-8") if line.strip()
            )
            with open(summary_csv_path, "a", encoding="utf-8", newline="") as f:
                if write_header:
                    f.write(",".join(columns) + "\n")
                f.write(",".join(values) + "\n")
            print0(f"Summary appended to: {summary_csv_path}")

    compute_cleanup()
