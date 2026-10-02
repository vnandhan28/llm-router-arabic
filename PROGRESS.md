# Progress log: Arabic LLM Router

What has been done so far, step by step. All numbers are from actual runs.

---

## Phase 0: Project setup ✅

Commit `43b1e35`

**Built**
- Folder layout:
  - `src/`: `config.py`, `data.py`, `query_models.py`, `parse.py`, `router.py`, `evaluate.py`
  - `data/`: ignored by git
  - `results/`: committed
  - `tests/`, `notebooks/`
- `.gitignore`: excludes `.env`, `data/`, `.venv/`, model caches and `__pycache__`.
- `.env.example` with an empty `HF_TOKEN=`. The real token lives only in `.env`, which is never committed.
- `README.md` skeleton and an MIT `LICENSE`. The author name is still `<YOUR NAME>`.
- `src/config.py`: every setting in one place (models, sample size, seed, paths), plus loading `HF_TOKEN` with python-dotenv.
- A virtual environment in `.venv/` (Python 3.12.10), with pinned versions in `requirements.txt`:
  - torch 2.14.0 (CPU build), transformers 5.17.0, accelerate 1.15.0
  - huggingface_hub 1.33.0, datasets 5.0.1, python-dotenv 1.2.3
  - pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, matplotlib 3.11.2
  - tqdm 4.70.1, pytest 9.1.1, jupyter 1.1.1

**Token safety**
- The token was first pasted into `.env.example`. It was moved to `.env` and `.env.example` was reset.
- It reappeared once, when VS Code saved the file again. It was reset again before any commit.
- A search of every commit finds no token.

---

## Phase 1: Data check ✅

Commit `b31cdf6`

**Dataset:** `MBZUAI/ArabicMMLU`, config `All`
- `test` split: 14,455 questions. This is the one we use.
- `dev` split: 120 few-shot examples. Not used.

**Raw columns:** `ID, Source, Country, Group, Subject, Level, Question, Context, Answer Key, Option 1 … Option 5, is_few_shot`

**Column mapping (agreed)**

| Our column | Source | Notes |
|---|---|---|
| `id` | `ID` | unique |
| `subject` | `Subject` + `Level` | 40 groups, e.g. "History (High)". Subject alone has only 21 values. |
| `question` | `Question` | |
| `context` | `Context` | 707 questions have a passage (e.g. a poem). **Kept** and placed above the question. |
| `options` | `Option 1..5` | only non-empty ones: 2 options (1,874 questions), 3 (2,121), 4 (10,120), 5 (340) |
| `answer_letter` | `Answer Key` | already A–E |

**Data facts found**
- Correct-answer letters in the full dataset: A 4,509, B 3,846, D 3,037, C 2,924, E 139. "A" is the most common correct answer (31%).
- Subject sizes range from 1,409 (Biology, High) down to 27 (Arabic Language, Middle and Computer Science, Middle).
- 81 questions have duplicate option texts, because diacritics were lost in the dataset.
  - In **40** of them, the correct answer's text appears more than once, so no model can pick the "right" letter.
  - **Decision: those 40 were dropped.**
  - **Clean dataset: 14,415 questions, 40 subjects.**

**Sampling (agreed): proportional stratified by subject**, fixed seed 42
- Each subject gets questions in proportion to its size, with a minimum of 1 when N ≥ 40.
- For N=1500, every subject gets at least 3 questions (the largest gets 146).
- Samples are saved as `data/sample_<N>.jsonl`. Files created so far: `sample_10`, `sample_20`, `sample_1500`.

**Tests:** `tests/test_data.py` has 4 tests (allocation sums correctly, is proportional, never exceeds a group's size, is reproducible).

---

## Phase 2: Querying the models 🔄 (in progress)

Commit `4054597`, plus the uncommitted `scripts/run_model.bat`

**Prompt:** the same for both models:

```
[context, if any]
<question>
A. <option 1>
B. <option 2>
...
Answer with one letter only.
```

Greedy decoding / temperature 0, `max_new_tokens = 8`.

**Parser** (`src/parse.py`)
- Understands:
  - `B`, `B.`, `(D)`, `**D**`
  - `Answer: B`, `The answer is (C)`
  - `الإجابة: B`, `الجواب هو ب`
  - Arabic option letters `أ ب ج د هـ`
- Returns nothing (counted as wrong) when:
  - the reply is empty
  - there is no letter
  - it names two letters ("A or B")
  - the letter is beyond the number of options
- `tests/test_parse.py` has 22 tests. With the Phase 1 tests, **26 tests pass** in total.

**Resumable runs** (`src/query_models.py`)
- Each answer is appended to `data/responses.jsonl` as soon as it's done.
- Each line records: id, role, model, raw reply, parsed letter, correct, parse_failed, seconds.
- A rerun skips `(id, model)` pairs that are already answered.
- `--role weak` and `--role strong` run the two models separately, so they never share the 8 GB of RAM at the same time.
- `--table` rebuilds `data/routing_table.csv` (`id, subject, prompt, weak_correct, strong_correct`).

### Step 2a: API test call

- `Qwen/Qwen2.5-72B-Instruct` through Hugging Face Inference Providers answered a test question correctly ("B").
- 1.3 s, 45 tokens, estimated cost $0.000016.

### Step 2b: Pilot, N=10 (first model pair)

Weak = `Qwen2.5-1.5B-Instruct` (local CPU), strong = `Qwen2.5-72B-Instruct` (API).

| | Weak 1.5B | Strong 72B |
|---|---|---|
| Accuracy | 40.0% (4/10) | 66.7% (6/9) |
| Parse failures | 0 | 0 |
| Time per question | 8.6 s (the first took 202 s to warm up) | 0.65 s |

- On 3 questions the weak model was wrong and the strong one right.
- The average prompt was 284 tokens (range 42 to 2,092).
- **Problem:** on the 10th strong call the API returned **"402 Payment Required: monthly included credits depleted"**.
- Pilot files were kept under new names:
  - `data/responses_pilot_qwen1.5b_vs_72b_api.jsonl`
  - `data/routing_table_pilot_qwen1.5b_vs_72b_api.csv`
  - `data/pilot_log_qwen1.5b_vs_72b_api.txt`

### Step 2c: Decision: switch to two local models ($0)

**Option 3 chosen.**
- Weak = `Qwen/Qwen2.5-0.5B-Instruct`, strong = `Qwen/Qwen2.5-3B-Instruct`, both on CPU in bfloat16.
- `STRONG_BACKEND = "local"` in `config.py`. It can be switched back to `"api"` later.
- N raised to **1500**.

Trade-offs to note in the README:
- The quality gap between 0.5B and 3B is smaller than between 1.5B and 72B, so the router has less room to help.
- The 3B model needs about 6.5 GB of RAM. When checked, only 1.2 GB of the 7.7 GB was free, so close other apps before the strong run.

### Step 2d: Launching the long runs

- tmux isn't available on Windows. Instead, each run gets **its own console window**, which keeps going if VS Code closes.
- First attempt: a PowerShell window. Python never started, and the cause was not found.
- Second attempt: `scripts\run_model.bat weak` in a cmd window. **This works.** It logs to `data\run_weak.log`.

### Step 2e: Weak run finished ✅

`Qwen2.5-0.5B-Instruct` on all 1,500 questions, with no errors.

| | Value |
|---|---|
| Accuracy | **37.0%** |
| Random-guess accuracy on the same questions | 29.6% |
| Parse failures | 15 (1.0%). All are real non-answers: 11 replies were `"ABCD"`, and 4 were `"C"` on 2-option questions. |
| Speed | median 3.1 s per question |

- **Letter bias:** the model answers "A" 48.6% of the time, but "A" is correct only 32.7% of the time.
- **Lost time:** one question took about 3 hours because the laptop went to sleep.

### Step 2f: Strong run ⏳ (pending)

- Postponed by choice: only 0.4–0.5 GB of RAM was free, and the 3B model needs about 6.5 GB.
- **Decision:** first build and test all the remaining code (Phases 3–7) on a **practice table**, then run the strong model once at the end.

---

## Practice mode (for testing before the strong run)

- `src/practice.py` builds `data/practice/routing_table.csv` from:
  - the **real** weak answers
  - **simulated** strong answers: right 85% of the time when weak is right, 45% when weak is wrong
- With `PRACTICE=1`, every step reads and writes `data/practice/`, never `results/`.
- **No practice number is a result.** Practice runs only show that the code works.

---

## Phase 3: Labels and splits ✅ (code)

`src/labels.py`, tests in `tests/test_labels.py`

- **Label:** `need_strong = weak wrong AND strong right`.
- **Statistics:**
  - weak and strong accuracy, % need_strong, both right / both wrong
  - the oracle ceiling and random-guess accuracy
  - a per-subject table
- **Automatic warnings** if:
  - weak accuracy is above 90%
  - weak accuracy is within 5 points of guessing
  - strong is not better than weak
- **Splits**, saved to `splits.json` so every router uses the same ones:
  - (a) random 70/30 with seed 42, stratified on `need_strong`: 1,050 train / 450 test;
  - (b) GroupKFold by subject, 5 folds of about 300 questions and 8 unseen subjects each.

---

## Phase 4: Routers ✅ (code)

`src/router.py`, tests in `tests/test_router.py`

Every router has the same two steps: `fit(train)`, then `score(df)`, which returns a value in [0, 1].

- **Random** and **Length**: baselines.
- **TF-IDF:** character 2–5-grams plus class-balanced logistic regression.
- **Embedding:** `intfloat/multilingual-e5-small` with the `"query: "` prefix and mean pooling, plus logistic regression.
  - The embeddings of all 1,500 real prompts are cached in `data/embeddings_e5_small.npz` (384 dimensions), so the final run reuses them.
- **Oracle:** uses the true labels; an upper bound only.

**Bug fixed:** `np.savez` adds `.npz` to the filename, so the cache path was corrected.

Practice check: the AUROC of the real routers was 0.46–0.53. That is expected, because the simulated strong answers contain no learnable pattern. The oracle scored 1.0, which shows the scoring logic is right.

---

## Phase 5: Evaluation ✅ (code)

`src/evaluate.py`, tests in `tests/test_evaluate.py`

- **Curve:** accuracy when the top p% (by score) go to strong, for p = 0, 5, …, 100.
- **AUC:** trapezoid area under that curve, with the x-axis running from 0 to 1.
- **APGR** = (AUC − weak acc) / (strong acc − weak acc).
- **% strong calls needed** to reach 90% / 95% of strong accuracy. This uses every possible cut-off, not just 5% steps.
- **Bootstrap 95% CI for APGR:** 1,000 resamples of the test questions.
- **Outputs:**
  - `metrics_random_split.csv`, `metrics_unseen_subjects.csv`
  - `curve_random_split.png`, `curve_unseen_subjects.png`
  - `scores_*.csv`, which Phase 6 uses
- **Tests on a hand-worked 4-question example.** One test failed at first because of an arithmetic slip in the test itself. The code was right: AUC = 0.78125.

Practice check: the random router's APGR was about 0.46–0.50, close to the theoretical 0.5. The oracle's was about 0.95–0.99.

---

## Phase 6: Failure analysis ✅ (code)

`src/failures.py`, tests in `tests/test_failures.py`

- **Operating point:** send the top X% to strong, where X = the % needed to reach 90% of strong accuracy. If that is never reached, use 50%.
- **15 worst decisions** of the embedding router on the random-split test set → `failure_examples.csv`:
  - **missed** (lowest scores): sent to weak, weak was wrong, strong was right;
  - **wasted** (highest scores): sent to strong, but weak was already right.
- **Per-subject AUROC, miss rate and waste rate** in the unseen-subject setup → `failure_by_subject.csv`.
  - Only subjects with at least 20 questions and at least 3 need_strong questions are ranked.
- **Pattern check** on the worst decisions vs. the whole test set: number of options, context passage, prompt length.

---

## Phase 7: Demo and README ✅

- **Demo** (`src/demo.py`, tests in `tests/test_demo.py`):
  - `python -m src.demo "<question with options>"` or `--file question.txt`
  - prints the router score, the chosen model and its answer
  - loads **only the chosen model**, so it fits in 8 GB of RAM
  - `--route-only` and `--threshold` options
  - The router is trained on all questions and saved to `router_embedding.joblib`. Its cut-off comes from the 90% operating point.
  - Tested: on "capital of the UAE", the weak model answered **B. أبوظبي** (correct).
- **`scripts/run_analysis.bat`:** one command that runs Phases 3–6 on the real data and retrains the demo router.
  - Tested: before the strong run, it stops cleanly with "need answers from both models".
- **`README.md`:** problem, data and licenses, models, method, results (real weak-model numbers only; the rest is marked pending), limitations, how to reproduce, project layout, Built on, and What I added.

**Tests:** 48 passing.

---

## Final run ✅

### Strong run

- `Qwen2.5-3B-Instruct`, median 25.5 s per question. RAM was almost full (0.1 GB free).
- **Stopped by choice after 1,011 questions**, to save about 4.5 more hours.
  - The questions were processed in shuffled order, so these 1,011 are a random subset of the 1,500.
  - They cover all 40 subjects.
- 0 errors, 6 parse failures (0.6%). The answers file was checked: all 2,511 lines are valid.

### Analysis

`scripts\run_analysis.bat`, with the log in `data/run_analysis.log`.

**The two models**
- Weak accuracy is **38.2%**, strong **53.1%**, and random guessing 29.6%.
- `need_strong` = 29.4%. In 14.4% of questions the weak model was right and the strong one wrong.
- **Oracle ceiling: 67.6%.**

**Routers**
- No learned router clearly beats random routing. AUROC was 0.51–0.53.
- Unseen-subject APGR [95% CI]:

  | Router | APGR [95% CI] |
  |---|---|
  | random | 0.485 [0.403, 0.562] |
  | length | 0.595 [0.516, 0.694] |
  | tfidf | 0.545 [0.467, 0.630] |
  | embedding | 0.547 [0.468, 0.630] |
  | oracle | 1.318 |

**Why:** the weak model answers "A" 49.6% of the time, while "A" is correct 32.6% of the time.

| True answer | Weak accuracy | `need_strong` rate |
|---|---|---|
| A | 66.1% | 14.2% |
| B–E | 24.7% | 36.7% |

The label therefore depends mostly on the hidden answer key, which a router cannot see.

**Outputs**
- README results, limitations and the opening finding are filled in with these real numbers.
- `results/` holds the metrics CSVs, both curve plots, the per-subject stats and the failure analysis.
- The demo router was retrained on the 1,011 real questions. Its cut-off is 0.493 (about 46.1% of questions to strong).

## Possible next steps (only if wanted)

- **Finish the remaining 489 strong answers** (about 4.5 h). This narrows the confidence intervals.
- **Shuffle the option order**, or use a less biased weak model (e.g. Qwen2.5-1.5B), to give routers a learnable signal.
- **Optional Phase 8:** English→Arabic transfer with RouteLLM's MMLU data, and a RouteLLM `Router` subclass.

## Commits so far

| Commit | Message |
|---|---|
| `43b1e35` | Phase 0: project skeleton, pinned requirements, config with dotenv |
| `b31cdf6` | Phase 1: load and clean ArabicMMLU, proportional stratified sampling by subject |
| `4054597` | Phase 2: prompt builder, letter parser with tests, resumable model querying |
