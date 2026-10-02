# Arabic LLM Router

A RouteLLM-style router for Arabic: for each multiple-choice question it decides whether a
small, cheap model is enough or whether the question should go to a larger, more expensive
model. Evaluated on ArabicMMLU.

> **Status:** all code is written and tested. The small-model run is complete; the
> large-model run is still pending, so the results section below is not filled in yet.
> No number in this README comes from simulated data.

## Problem: cost vs. quality

Large language models answer better but cost more per call (money, latency, hardware).
Many questions are easy enough for a small model. A **router** looks at a question
*before* answering it and predicts whether the large model is actually needed. A good
router gets close to large-model accuracy while sending only part of the traffic to the
large model. RouteLLM ([Ong et al., 2024](https://arxiv.org/abs/2406.18665)) studied this
for English. This project repeats the evaluation for **Arabic**.

## Data and licenses

- **ArabicMMLU** ([MBZUAI/ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU),
  Koto et al., 2024), licensed **CC BY-NC 4.0** (non-commercial).
  - Modern Standard Arabic school and professional exam questions, `test` split:
    14,455 questions.
  - **Subject** = `Subject` + `Level`, which gives 40 groups (e.g. "History (High)").
  - Questions with a `Context` passage (707) keep the passage above the question.
  - **40 questions removed:** the text of their correct option also appears in another
    option (the diacritics seem to have been lost), so they cannot be answered.
    That leaves **14,415 questions**.
  - **Working sample:** 1,500 questions, stratified by subject in proportion to subject
    size (every subject has at least 3), seed 42.
- **Code in this repo:** MIT (see `LICENSE`). The dataset is not redistributed here; it
  is downloaded from Hugging Face.

## Models

| Role | Model | Where it runs |
|---|---|---|
| Weak (cheap) | `Qwen/Qwen2.5-0.5B-Instruct` | local CPU, bfloat16 |
| Strong (expensive) | `Qwen/Qwen2.5-3B-Instruct` | local CPU, bfloat16 |
| Router embeddings | `intfloat/multilingual-e5-small` | local CPU |

Both models get exactly the same prompt: the question, the options labelled A–E, and
"Answer with one letter only.". Decoding is greedy, with at most 8 new tokens. A parser
(`src/parse.py`, unit-tested) extracts the letter. A reply with no single valid letter
counts as wrong.

**Why two local models?** The original plan used Qwen2.5-72B through Hugging Face
Inference Providers as the strong model. The free monthly credits ran out during a
10-question pilot, so to keep the budget at $0 both models now run locally. A 0.5B-vs-3B
gap is much smaller than a small-vs-frontier gap, so this is a harder and less typical
setting for routing (see Limitations).

## Method

1. **Collect answers.** Ask both models every question in the sample and record whether
   each was correct (`data/routing_table.csv`). The runs can resume after a crash.
2. **Label.** `need_strong = weak wrong AND strong right`. These are the only questions
   where paying for the strong model changes the outcome.
3. **Routers.** Each one outputs a score in [0, 1] for "needs the strong model", the same
   role as RouteLLM's `calculate_strong_win_rate`:
   - **Random:** a baseline.
   - **Length:** a baseline; longer prompts get higher scores.
   - **TF-IDF:** character 2–5-grams plus logistic regression (class-balanced).
   - **Embedding:** multilingual-e5-small (`"query: "` prefix, mean pooling) plus
     logistic regression.
   - **Oracle:** sees the true labels; an upper bound only.
4. **Two evaluation setups:**
   - (a) a random 70/30 split, stratified on the label;
   - (b) **unseen subjects:** GroupKFold by subject with 5 folds, so the router is always
     tested on subjects it never saw during training.
5. **Metrics (RouteLLM-style).**
   - Send the top *p*% of questions by score to the strong model, for *p* = 0, 5, …, 100,
     and record overall accuracy at each *p*.
   - **AUC** is the area under accuracy vs. fraction routed.
   - **APGR** = (AUC − weak acc) / (strong acc − weak acc). It is about 0.5 for random
     routing, and 1 or more means strong-model accuracy at little cost.
   - Also reported: the % of strong calls needed to reach 90% and 95% of strong-model
     accuracy.
   - **95% bootstrap CI** for APGR from 1,000 resamples of the test questions.
6. **Failure analysis.**
   - The embedding router's 15 most confident wrong decisions:
     - *missed:* sent to weak, which was wrong, while strong was right;
     - *wasted:* sent to strong, while weak was already right.
   - Per-subject AUROC in the unseen-subject setup.

## Results

**Pending the strong-model run.** What is measured so far:

| | Value |
|---|---|
| Weak model (Qwen2.5-0.5B) accuracy, 1,500 questions | **37.0%** |
| Accuracy of random guessing (accounts for 2–5 options per question) | 29.6% |
| Weak-model parse failures | 15 / 1,500 (1.0%): 11 replies were `"ABCD"`, 4 named a letter beyond the options |
| Share of weak answers that are "A" / share of questions whose correct answer is "A" | 48.6% / 32.7% |

The weak model has a strong bias toward answering "A".

_After the strong run: metrics tables from `results/metrics_*.csv` and the curves
`results/curve_*.png` go here._

## Limitations

- **The results depend on this specific model pair.** A 0.5B vs. 3B pair (both Qwen 2.5)
  is a narrow quality gap. Routing between a small model and a frontier model, as in
  RouteLLM, can behave very differently.
- **The data is ArabicMMLU:** Modern Standard Arabic school-exam questions. It contains
  no dialects and is not real user traffic.
- **Possible benchmark contamination:** pretrained models may have seen these questions.
- **Sample size:** 1,500 questions (450 in the random test split). This limits precision;
  confidence intervals are reported. Per-subject numbers rest on 20–146 questions and
  only show large differences.
- **Letter bias:** the weak model's preference for "A" means some of its correct answers
  are lucky guesses.

## How to reproduce

Windows (PowerShell). On macOS/Linux, use `source .venv/bin/activate` and run the
`python -m` lines directly.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # add a Hugging Face token (only needed for API models)

python -m src.data              # load + clean ArabicMMLU, write the stratified sample
python -m pytest -q             # 48 unit tests

scripts\run_model.bat weak      # weak model on all 1,500 questions (own window, resumable)
scripts\run_model.bat strong    # strong model (needs ~6.5 GB free RAM)
scripts\run_analysis.bat        # labels, routers, evaluation, failure analysis -> results\

python -m src.demo "ما عاصمة الإمارات العربية المتحدة؟
A. دبي
B. أبوظبي
C. الشارقة"                       # or: python -m src.demo --file question.txt
```

**Practice mode.** `src/practice.py` builds a routing table with *simulated* strong
answers, so the analysis code can be tested before the strong run. With the environment
variable `PRACTICE=1`, every step reads and writes `data/practice/` instead of
`results/`. Practice numbers are never reported.

## Project layout

```
src/config.py        all settings (models, N, seed, paths)
src/data.py          load, clean and sample ArabicMMLU
src/query_models.py  prompts, local/API models, resumable answering, routing table
src/parse.py         answer-letter parser
src/labels.py        need_strong label, stats, random + unseen-subject splits
src/router.py        random / length / TF-IDF / embedding / oracle routers
src/evaluate.py      accuracy-vs-cost curves, AUC, APGR, bootstrap CIs, plots
src/failures.py      worst routing decisions, per-subject difficulty
src/demo.py          command-line demo
src/practice.py      simulated table for testing the pipeline
scripts/             Windows launchers (run_model.bat, run_analysis.bat)
tests/               unit tests
```

## Built on

- **RouteLLM:** I. Ong, A. Almahairi, V. Wu, W.-L. Chiang, T. Wu, J. E. Gonzalez,
  M. W. Kadous, I. Stoica. *RouteLLM: Learning to Route LLMs with Preference Data.*
  arXiv:2406.18665, 2024. Code: https://github.com/lm-sys/RouteLLM. The evaluation
  idea (accuracy vs. % strong calls, APGR) and the router scoring interface come
  from this work.
- **ArabicMMLU:** F. Koto et al. *ArabicMMLU: Assessing Massive Multitask Language
  Understanding in Arabic.* Findings of ACL, 2024. MBZUAI.
- Models: Qwen2.5 (Alibaba Cloud), multilingual-E5 (Microsoft).

## What I added

- A RouteLLM-style evaluation for **Arabic**, on ArabicMMLU, with a small open model pair
  that runs on a laptop CPU at $0.
- An **unseen-subject** evaluation (GroupKFold by subject) that tests whether routers
  transfer to topics they were not trained on.
- **Bootstrap confidence intervals** for APGR, and "% strong calls to reach 90% / 95% of
  strong accuracy".
- A data-quality fix: removing 40 ArabicMMLU questions whose correct option is
  duplicated.
- An answer parser that handles both Latin and Arabic option letters.
- Resumable model runs.
- Failure analysis of the router's most confident mistakes and of the subjects that are
  hardest to route.
- A command-line demo that loads only the model the router picks.
