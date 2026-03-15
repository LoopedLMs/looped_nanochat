# Saunshi Downstream Eval — Implementation Overview

Based on the 16-task benchmark from [Saunshi et al. 2502.17416](https://arxiv.org/abs/2502.17416).
Implementation: `dev/eval_saunshi/prepare_saunshi_bundle.py` (data prep) + `scripts/base_eval.py` (eval).

---

## How Evaluation Works

All tasks use **teacher-forcing exact-match** via the `language_modeling` task type in `nanochat/core_eval.py`:

1. Build a prompt: `[fewshot examples] + [context] + [delimiter]` (no continuation).
2. Build the full sequence: same prompt + continuation tokens.
3. Run the model autoregressively on the full sequence.
4. At each position `i` in the continuation span, check if `argmax(logits[i-1]) == target_token[i]`.
5. A sample is **correct** only if **all** continuation tokens are predicted exactly right.

**Metric**: accuracy per task → averaged within each group → averaged across 4 groups = `saunshi_metric`.

**Few-shot**: examples are sampled from the same dataset (excluding the current item), seeded by `1234 + idx` for reproducibility. Formatted identically to the test item.

---

## Group 1: Math Word Problems

**Context format**: `Question: {problem text}`
**Continuation delimiter**: ` Answer:`
**Few-shot**: 5

Full prompt looks like:
```
Question: Mary has 3 apples and buys 4 more. How many does she have? Answer: 7

Question: Tom had 10 balls and lost 3. How many left? Answer: 7

Question: {test question} Answer: {answer}
```

### SVAMP
**Source**: `ChilleD/SVAMP`, test split (300 examples)
**Fields used**: `Body` + `Question` → concatenated as the problem; `Answer` → numeric answer
**Processing**: float answers cast to int where possible (`6.0 → "6"`)

**Example**:
```
context:    "Question: Winter is almost here and most animals are migrating to warmer countries.
             There are 41 bird families living near the mountain. If 35 bird families flew away
             to asia and 62 bird families flew ..."
continuation: " 27"
```

### ASDiv
**Source**: `EleutherAI/asdiv`, validation split (2305 examples)
**Fields used**: `body` + `question`; `answer`
**Processing**: strip parenthetical units from answer — `"9 (apples)"` → `"9"` via `re.sub(r'\s*\(.*?\)', '', answer)`

**Example**:
```
context:    "Question: Seven red apples and two green apples are in the basket.
             How many apples are in the basket?"
continuation: " 9"
```

### MAWPS
**Source**: `mwpt5/MAWPS`, train split (1772 examples, 0 skipped)
**Fields used**: `Question` (contains `N_00`, `N_01`, ... placeholders), `Numbers` (space-separated floats), `Answer`
**Processing**:
1. Split `Numbers` on whitespace → substitute `N_00`, `N_01`, ... into question text
2. Format floats as ints where exact (`"8.0"` → `"8"`, `"28.3"` stays `"28.3"`)
3. Skip examples where any placeholder remains unresolved
4. Same int-cast on answer

**Example**:
```
context:    "Question: Mary is baking a cake. The recipe wants 8 cups of flour.
             She already put in 2 cups. How many cups does she need to add?"
continuation: " 6"
```

---

## Group 2: Closed Book QA

**Context format**: `Question: {question}`
**Continuation delimiter**: `\nAnswer:`
**Few-shot**: 5

No passage is given — the model must answer from parametric memory alone.

### TriviaQA
**Source**: `mandarjoshi/trivia_qa`, config `rc.nocontext`, validation split (17944 examples)
**Fields used**: `question`, `answer["value"]`

**Example**:
```
context:    "Question: Who was the man behind The Chipmunks?"
continuation: " David Seville"
```

### NQ (Natural Questions)
**Source**: `nq_open`, validation split (3610 examples with short answers)
**Fields used**: `question`, first element of `answer` list
**Processing**: skip examples with empty answer list

**Example**:
```
context:    "Question: when was the last time anyone was on the moon"
continuation: " 14 December 1972 UTC"
```

### WebQ (Web Questions)
**Source**: `web_questions`, test split (2032 examples)
**Fields used**: `question`, first element of `answers` list

**Example**:
```
context:    "Question: what does jamaican people speak?"
continuation: " Jamaican Creole English Language"
```

### TydiQA-NoContext
**Source**: `google-research-datasets/tydiqa`, config `secondary_task`, validation split
**Fields used**: `question`, `answers["text"][0]`
**Processing**: filter to English only via `item["id"].split("-")[0] == "english"` (440 examples)

**Example**:
```
context:    "Question: What is a way to increase your wound healing speed?"
continuation: " Wound care"
```

---

## Group 3: Open Book QA

**Context format**: `Passage: {passage}\nQuestion: {question}`
**Continuation delimiter**: `\nAnswer:`
**Few-shot**: 3 (reduced from 5 because passages make prompts long)

A relevant passage is provided — the model must extract or infer the answer from it.

### TydiQA-GoldP
**Source**: Same as TydiQA-NoContext (`secondary_task` validation, English only, 440 examples)
**Difference from NoContext**: `context` field (the passage) is prepended

**Example**:
```
context:    "Passage: Wound care encourages and speeds wound healing via cleaning and
             protection from reinjury or infection. Depending on each patient's needs,
             it can range from the simplest first aid to entire nursing specialties...
             Question: What is a way to increase your wound healing speed?"
continuation: " Wound care"
```

### SQuAD v2
**Source**: `rajpurkar/squad_v2`, validation split (5928 answerable examples)
**Processing**: skip unanswerable questions (`answers["text"]` is empty)

**Example**:
```
context:    "Passage: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni)
             were the people who in the 10th and 11th centuries gave their name to
             Normandy, a region in France...
             Question: In what country is Normandy located?"
continuation: " France"
```

### DROP
**Source**: `ucinlp/drop`, validation split (9535 examples)
**Fields used**: `passage`, `question`, `answers_spans["spans"][0]`

**Example**:
```
context:    "Passage: Hoping to rebound from their loss to the Patriots, the Raiders
             stayed at home for a Week 16 duel with the Houston Texans...
             Question: Who caught the first touchdown of the game?"
continuation: " Chaz Schilens"
```

### CoQA
**Source**: `stanfordnlp/coqa`, validation split (7983 QA pairs)
**Structure**: each dataset item is a passage with multiple questions → flattened to one record per QA pair
**Fields used**: `story` (passage), `questions` list, `answers["input_text"]` list

**Example**:
```
context:    "Passage: Once upon a time, in a barn near a farm house, there lived a little
             white kitten named Cotton...
             Question: What color was Cotton?"
continuation: " white"
```

---

## Group 4: Reasoning Primitives (Synthetic)

Synthetic tasks testing lookup and one-hop aliasing. No external dataset — generated with a fixed seed.
**1000 examples per task**, seed 42.

Two **depths** × two **surface formats** = 4 tasks.

### Depth 0 — Direct Lookup

All variables are assigned a direct integer value. The query is one of those variables.
**Answer**: the integer directly assigned to the query variable.

#### var_assign_d0_math
**Format**: comma-separated assignments, query variable at end
**Delimiter**: `=` (no space — math style)

**Example**:
```
context:    "u=3, d=3, a=2, x=1, i=8, u"
continuation: "3"
```
Prompt ends with `u=`, model must predict `3`.

#### var_assign_d0_code
**Format**: one assignment per line (Python-style), query variable on final line
**Delimiter**: ` = ` (with spaces — code style)

**Example**:
```
context:    "y = 8\nl = 9\nz = 9\nv = 3\ne = 1\nv"
continuation: "3"
```
Prompt ends with `v = `, model must predict `3`.

### Depth 1 — One-Hop Alias

Base variables have integer values; derived variables alias a base variable (one level of indirection).
The query is always a derived variable — the model must follow the alias chain to find the integer.
**Answer**: the integer value of the aliased base variable.

#### var_assign_d1_math
**Format**: same comma-separated style, assignments shuffled
**Delimiter**: `=`

**Example**:
```
context:    "e=g, n=8, k=9, r=k, y=k, g=1, h=4, y"
continuation: "9"
```
`y` aliases `k`; `k=9` → answer is `9`.

#### var_assign_d1_code
**Format**: same newline Python style, assignments shuffled
**Delimiter**: ` = `

**Example**:
```
context:    "t = 4\ng = o\nj = 1\no = 0\nm = t\nc = 3\na = o\na"
continuation: "0"
```
`a` aliases `o`; `o=0` → answer is `0`.

---

## Running

**Prepare bundle** (one-time, or to add new tasks):
```bash
uv run python dev/eval_saunshi/prepare_saunshi_bundle.py
```
Output goes to `$NANOCHAT_BASE_DIR/saunshi_bundle/`.

**Run eval** (on cluster):
```bash
NUM_GPUS=1 SLURM_GPU_TYPE=rtx_6000 ./shells/_submit.sh shells/base_eval.sh
```

**CSV output** columns: `num_recur`, `saunshi_metric`, `saunshi_{group}` (×4), `saunshi_{task}` (×15).
