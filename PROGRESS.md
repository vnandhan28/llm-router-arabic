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
- Second attempt: `scripts\run_model.bat weak` in a cmd window. **This works:**
  - It logs to `data\run_weak.log`.
  - At the last check it was downloading the 0.5B model (447 MB so far). No questions had been answered yet.
  - Work stopped here for the day.

---

## Next steps

1. Confirm the weak run is answering questions:
   ```powershell
   Get-Content data\run_weak.log -Wait -Encoding utf8
   ```
   If the window was closed, restart it with `scripts\run_model.bat weak`. Finished questions are skipped.
2. After the weak run: close heavy apps, then run `scripts\run_model.bat strong`.
3. Rebuild the routing table with `python -m src.query_models --table` and review accuracy, parse-failure rate and speed.
4. Commit `scripts/run_model.bat` (and this file).
5. Phase 3: labels (`need_strong`), statistics, random 70/30 split and GroupKFold-by-subject split.

## Commits so far

| Commit | Message |
|---|---|
| `43b1e35` | Phase 0: project skeleton, pinned requirements, config with dotenv |
| `b31cdf6` | Phase 1: load and clean ArabicMMLU, proportional stratified sampling by subject |
| `4054597` | Phase 2: prompt builder, letter parser with tests, resumable model querying |
