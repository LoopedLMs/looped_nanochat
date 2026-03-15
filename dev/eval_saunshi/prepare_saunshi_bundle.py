"""
Prepare the Saunshi math word problem eval bundle.

Downloads SVAMP, ASDiv, and MAWPS from HuggingFace and writes JSONL files
to ~/.cache/nanochat/saunshi_bundle/ along with a saunshi.yaml config.

Usage:
    uv run python dev/eval_saunshi/prepare_saunshi_bundle.py

Extend this script to add more task groups (closed-book QA, open-book QA,
reasoning primitives) later.
"""

import json
import random
import re
import sys
from pathlib import Path

import yaml

from nanochat.common import get_base_dir


def get_bundle_dir() -> Path:
    bundle_dir = Path(get_base_dir()) / "saunshi_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "eval_data").mkdir(exist_ok=True)
    return bundle_dir


# -----------------------------------------------------------------------------
# SVAMP  (ChilleD/SVAMP, test split)
# Fields: ID, Body, Question, Equation, Answer, Type
# -----------------------------------------------------------------------------

def prepare_svamp(out_dir: Path) -> int:
    from datasets import load_dataset

    ds = load_dataset("ChilleD/SVAMP", split="test")
    records = []
    for item in ds:
        question = item["Body"].strip().rstrip(".") + ". " + item["Question"].strip()
        fval = float(item["Answer"])
        answer = str(int(fval)) if fval == int(fval) else str(fval)
        records.append({"context": f"Question: {question}", "continuation": f" {answer}"})

    out_path = out_dir / "svamp.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"SVAMP: wrote {len(records)} records → {out_path}")
    return len(records)


# -----------------------------------------------------------------------------
# ASDiv  (EleutherAI/asdiv, validation split)
# Fields: body, question, answer, solution_type, formula, numbers
# Answer may include units like "9 (apples)" — strip parenthetical.
# -----------------------------------------------------------------------------

def prepare_asdiv(out_dir: Path) -> int:
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/asdiv", split="validation")
    records = []
    for item in ds:
        question = item["body"].strip().rstrip(".") + " " + item["question"].strip()
        raw_answer = item["answer"].strip()
        # Strip units in parentheses e.g. "9 (apples)" → "9"
        answer = re.sub(r"\s*\(.*?\)", "", raw_answer).strip()
        records.append({"context": f"Question: {question}", "continuation": f" {answer}"})

    out_path = out_dir / "asdiv.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"ASDiv: wrote {len(records)} records → {out_path}")
    return len(records)


# -----------------------------------------------------------------------------
# MAWPS  (mwpt5/MAWPS, train split)
# Fields: Question, Numbers, Answer, Equation
# Questions use N_00, N_01, ... placeholders; Numbers is comma-sep values.
# -----------------------------------------------------------------------------

def prepare_mawps(out_dir: Path) -> int:
    from datasets import load_dataset

    ds = load_dataset("mwpt5/MAWPS", split="train")
    records = []
    skipped = 0
    for item in ds:
        question_raw = item["Question"].strip()
        numbers_str = item.get("Numbers", "") or ""
        answer_raw = str(item["Answer"]).strip()

        # Substitute N_00, N_01, ... placeholders with actual numbers (space-separated).
        # Convert floats to ints where possible (e.g. "8.0" → "8").
        def fmt_num(s: str) -> str:
            try:
                f = float(s)
                return str(int(f)) if f == int(f) else s
            except ValueError:
                return s

        if numbers_str.strip():
            numbers = [fmt_num(n) for n in numbers_str.split()]
            question = question_raw
            for i, num in enumerate(numbers):
                placeholder = f"N_{i:02d}"
                question = question.replace(placeholder, num)
        else:
            question = question_raw

        # Skip if any unresolved placeholder remains
        if re.search(r"N_\d{2}", question):
            skipped += 1
            continue

        # Convert float answers to int string where possible (6.0 → "6", 6.5 → "6.5")
        try:
            fval = float(answer_raw)
            answer = str(int(fval)) if fval == int(fval) else answer_raw
        except ValueError:
            answer = answer_raw
        records.append({"context": f"Question: {question}", "continuation": f" {answer}"})

    out_path = out_dir / "mawps.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"MAWPS: wrote {len(records)} records → {out_path} (skipped {skipped} with unresolved placeholders)")
    return len(records)


# ==============================================================================
# Closed book QA
# context = "Question: {q}", continuation = " {answer}", delimiter = "\nAnswer:"
# ==============================================================================

def prepare_triviaqa(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    records = []
    for item in ds:
        answer = item["answer"]["value"].strip()
        if not answer:
            continue
        records.append({"context": f"Question: {item['question'].strip()}", "continuation": f" {answer}"})
    out_path = out_dir / "triviaqa.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"TriviaQA: wrote {len(records)} records → {out_path}")
    return len(records)


def prepare_nq(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("nq_open", split="validation")
    records = []
    for item in ds:
        answers = item["answer"]
        if not answers:
            continue
        records.append({"context": f"Question: {item['question'].strip()}", "continuation": f" {answers[0].strip()}"})
    out_path = out_dir / "nq.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"NQ: wrote {len(records)} records → {out_path}")
    return len(records)


def prepare_webq(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("web_questions", split="test")
    records = []
    for item in ds:
        answers = item["answers"]
        if not answers:
            continue
        records.append({"context": f"Question: {item['question'].strip()}", "continuation": f" {answers[0].strip()}"})
    out_path = out_dir / "webq.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"WebQ: wrote {len(records)} records → {out_path}")
    return len(records)


def prepare_tydiqa_nocontext(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="validation")
    records = []
    for item in ds:
        if item["id"].split("-")[0] != "english":
            continue
        answers = item["answers"]["text"]
        if not answers:
            continue
        records.append({"context": f"Question: {item['question'].strip()}", "continuation": f" {answers[0].strip()}"})
    out_path = out_dir / "tydiqa_nocontext.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"TydiQA-NoContext: wrote {len(records)} records → {out_path}")
    return len(records)


# -----------------------------------------------------------------------------
# YAML config
# -----------------------------------------------------------------------------

# ==============================================================================
# Open book QA
# context = "Passage: {passage}\nQuestion: {q}", continuation = " {answer}", delimiter = "\nAnswer:"
# ==============================================================================

def prepare_tydiqa_goldp(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="validation")
    records = []
    for item in ds:
        if item["id"].split("-")[0] != "english":
            continue
        answers = item["answers"]["text"]
        if not answers:
            continue
        records.append({
            "context": f"Passage: {item['context'].strip()}\nQuestion: {item['question'].strip()}",
            "continuation": f" {answers[0].strip()}",
        })
    out_path = out_dir / "tydiqa_goldp.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"TydiQA-GoldP: wrote {len(records)} records → {out_path}")
    return len(records)


def prepare_squadv2(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    records = []
    for item in ds:
        answers = item["answers"]["text"]
        if not answers:  # unanswerable
            continue
        records.append({
            "context": f"Passage: {item['context'].strip()}\nQuestion: {item['question'].strip()}",
            "continuation": f" {answers[0].strip()}",
        })
    out_path = out_dir / "squadv2.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"SquadV2: wrote {len(records)} records → {out_path}")
    return len(records)


def prepare_drop(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("ucinlp/drop", split="validation")
    records = []
    for item in ds:
        spans = item["answers_spans"]["spans"]
        if not spans:
            continue
        records.append({
            "context": f"Passage: {item['passage'].strip()}\nQuestion: {item['question'].strip()}",
            "continuation": f" {spans[0].strip()}",
        })
    out_path = out_dir / "drop.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Drop: wrote {len(records)} records → {out_path}")
    return len(records)


def prepare_coqa(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/coqa", split="validation")
    records = []
    for item in ds:
        passage = item["story"].strip()
        for q, a in zip(item["questions"], item["answers"]["input_text"]):
            if not a.strip():
                continue
            records.append({
                "context": f"Passage: {passage}\nQuestion: {q.strip()}",
                "continuation": f" {a.strip()}",
            })
    out_path = out_dir / "coqa.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"CoQA: wrote {len(records)} records → {out_path}")
    return len(records)


def prepare_quac(out_dir: Path) -> int:
    # allenai/quac has a broken legacy loader in datasets>=4.0; load via parquet
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="allenai/quac",
        filename="data/validation-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        passage = str(row.get("background", "") or "").strip()
        questions = row.get("questions") or []
        orig_answers = row.get("orig_answers") or {}
        texts = orig_answers.get("texts") if isinstance(orig_answers, dict) else []
        if not isinstance(questions, list) or not isinstance(texts, list):
            continue
        for q, a in zip(questions, texts):
            a = str(a).strip()
            if not a or a.lower() == "cannotanswer":
                continue
            records.append({
                "context": f"Passage: {passage}\nQuestion: {str(q).strip()}",
                "continuation": f" {a}",
            })
    out_path = out_dir / "quac.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"QuAC: wrote {len(records)} records → {out_path}")
    return len(records)


# ==============================================================================
# Reasoning primitives — synthetic variable assignment
#
# Depth-0: each var assigned a direct integer. "a=1, b=2, c=6, b" + "=" + "2"
# Depth-1: base vars have integer values; derived vars alias a base var.
# Two surface formats: math (comma-sep) and code (newline Python-style).
# ==============================================================================

def _make_var_assign_d0(rng: random.Random, n_vars: int = 5, n_vals: int = 10) -> dict:
    letters = list("abcdefghijklmnopqrstuvwxyz")
    vars_ = rng.sample(letters, n_vars)
    assignments = [(v, str(rng.randint(0, n_vals - 1))) for v in vars_]
    query_var = rng.choice(vars_)
    return {"assignments": assignments, "query": query_var, "answer": dict(assignments)[query_var]}


def _make_var_assign_d1(rng: random.Random, n_base: int = 4, n_derived: int = 3, n_vals: int = 10) -> dict:
    letters = list("abcdefghijklmnopqrstuvwxyz")
    base_vars = rng.sample(letters, n_base)
    derived_vars = rng.sample([v for v in letters if v not in base_vars], n_derived)
    base_vals = {v: str(rng.randint(0, n_vals - 1)) for v in base_vars}
    derived_aliases = {v: rng.choice(base_vars) for v in derived_vars}
    query_var = rng.choice(derived_vars)
    answer = base_vals[derived_aliases[query_var]]
    all_assignments = [(v, base_vals[v]) for v in base_vars] + [(v, derived_aliases[v]) for v in derived_vars]
    rng.shuffle(all_assignments)
    return {"assignments": all_assignments, "query": query_var, "answer": answer}


def _to_math(ex: dict) -> dict:
    # context ends with query var; delimiter "=" completes the expression
    parts = [f"{v}={val}" for v, val in ex["assignments"]]
    return {"context": ", ".join(parts) + f", {ex['query']}", "continuation": ex["answer"]}


def _to_code(ex: dict) -> dict:
    # context ends with query var; delimiter " = " completes the assignment
    lines = [f"{v} = {val}" for v, val in ex["assignments"]]
    return {"context": "\n".join(lines) + f"\n{ex['query']}", "continuation": ex["answer"]}


def prepare_reasoning_primitives(out_dir: Path, n_examples: int = 1000, seed: int = 42) -> None:
    rng = random.Random(seed)
    tasks = [
        ("var_assign_d0_math", _make_var_assign_d0, _to_math),
        ("var_assign_d0_code", _make_var_assign_d0, _to_code),
        ("var_assign_d1_math", _make_var_assign_d1, _to_math),
        ("var_assign_d1_code", _make_var_assign_d1, _to_code),
    ]
    for label, maker, formatter in tasks:
        records = [formatter(maker(rng)) for _ in range(n_examples)]
        out_path = out_dir / f"{label}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"{label}: wrote {len(records)} records → {out_path}")


# -----------------------------------------------------------------------------
# YAML config helpers
# -----------------------------------------------------------------------------

def _task(label, uri, group, delimiter, num_fewshot=5):
    return {"label": label, "dataset_uri": uri, "task_type": "language_modeling",
            "num_fewshot": num_fewshot, "continuation_delimiter": delimiter, "group": group}


ALL_TASKS = [
    # math word problems
    _task("svamp",            "svamp.jsonl",            "math_word_problems", " Answer:"),
    _task("asdiv",            "asdiv.jsonl",            "math_word_problems", " Answer:"),
    _task("mawps",            "mawps.jsonl",            "math_word_problems", " Answer:"),
    # closed book QA
    _task("triviaqa",         "triviaqa.jsonl",         "closed_book_qa", "\nAnswer:"),
    _task("nq",               "nq.jsonl",               "closed_book_qa", "\nAnswer:"),
    _task("webq",             "webq.jsonl",             "closed_book_qa", "\nAnswer:"),
    _task("tydiqa_nocontext", "tydiqa_nocontext.jsonl", "closed_book_qa", "\nAnswer:"),
    # open book QA (3-shot: passages make prompts long)
    _task("tydiqa_goldp", "tydiqa_goldp.jsonl", "open_book_qa", "\nAnswer:", num_fewshot=3),
    _task("squadv2",      "squadv2.jsonl",      "open_book_qa", "\nAnswer:", num_fewshot=3),
    _task("drop",         "drop.jsonl",         "open_book_qa", "\nAnswer:", num_fewshot=3),
    _task("coqa",         "coqa.jsonl",         "open_book_qa", "\nAnswer:", num_fewshot=3),
    # reasoning primitives (synthetic)
    _task("var_assign_d0_math", "var_assign_d0_math.jsonl", "reasoning_primitives", "="),
    _task("var_assign_d0_code", "var_assign_d0_code.jsonl", "reasoning_primitives", " = "),
    _task("var_assign_d1_math", "var_assign_d1_math.jsonl", "reasoning_primitives", "="),
    _task("var_assign_d1_code", "var_assign_d1_code.jsonl", "reasoning_primitives", " = "),
]


def write_yaml(bundle_dir: Path, tasks: list[dict]) -> None:
    groups = list(dict.fromkeys(t["group"] for t in tasks))
    config = {"groups": groups, "tasks": tasks}
    out_path = bundle_dir / "saunshi.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config: wrote {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    bundle_dir = get_bundle_dir()
    data_dir = bundle_dir / "eval_data"

    print(f"Bundle dir: {bundle_dir}\n")

    errors = []

    datasets = [
        ("SVAMP",            prepare_svamp),
        ("ASDiv",            prepare_asdiv),
        ("MAWPS",            prepare_mawps),
        ("TriviaQA",         prepare_triviaqa),
        ("NQ",               prepare_nq),
        ("WebQ",             prepare_webq),
        ("TydiQA-NoContext", prepare_tydiqa_nocontext),
        ("TydiQA-GoldP",     prepare_tydiqa_goldp),
        ("SquadV2",          prepare_squadv2),
        ("Drop",             prepare_drop),
        ("CoQA",             prepare_coqa),
    ]
    for name, fn in datasets:
        try:
            fn(data_dir)
        except Exception as e:
            print(f"ERROR {name}: {e}", file=sys.stderr)
            errors.append(name)

    try:
        prepare_reasoning_primitives(data_dir)
    except Exception as e:
        print(f"ERROR ReasoningPrimitives: {e}", file=sys.stderr)
        errors.append("ReasoningPrimitives")

    if errors:
        print(f"\nFailed: {errors}. Fix errors above and re-run.", file=sys.stderr)
        sys.exit(1)

    write_yaml(bundle_dir, ALL_TASKS)


if __name__ == "__main__":
    main()
