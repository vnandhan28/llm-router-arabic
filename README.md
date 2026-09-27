# Arabic LLM Router

A RouteLLM-style router that decides, for each Arabic multiple-choice question,
whether a cheap small model is enough or whether to call a strong large model.

> Status: work in progress. Sections marked TODO are filled in as phases complete.

## Problem
TODO: cost vs. quality trade-off.

## Data and licenses
- [ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) (Koto et al., 2024, MBZUAI), CC BY-NC 4.0.
- Code in this repo: MIT (see `LICENSE`).

## Models
- Weak: `Qwen/Qwen2.5-1.5B-Instruct` (local, CPU)
- Strong: TODO (Hugging Face Inference Providers)

## Method
TODO

## Results
TODO (only real numbers from `results/`)

## Limitations
TODO

## How to reproduce
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env        # then put your Hugging Face token in .env
```

## Built on
- RouteLLM: Ong et al., 2024, arXiv:2406.18665 — https://github.com/lm-sys/RouteLLM
- ArabicMMLU: Koto et al., 2024 — MBZUAI

## What I added
TODO
