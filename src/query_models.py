"""Phase 2: ask the weak and strong models each question; resumable.

Usage:
  python -m src.query_models --role weak     # only the weak model (run each role separately
  python -m src.query_models --role strong   #   so two models never share the 8 GB RAM)
  python -m src.query_models --table         # just rebuild data/routing_table.csv
  python -m src.query_models --test-strong   # exactly ONE call to check the strong model

Every finished answer is appended to data/responses.jsonl immediately, so after a
crash the same command skips (id, model) pairs that are already done.
"""
import argparse
import json
import sys
import time

import pandas as pd

from src import config
from src.data import load_clean, sample_path, stratified_sample
from src.parse import LETTERS, parse_letter

INSTRUCTION = "Answer with one letter only."


def build_prompt(question: str, options: list[str], context: str | None = None) -> str:
    """Same prompt for both models: [context] + question + options A-E + instruction."""
    # Missing context can arrive as None or NaN (after a JSON round-trip), so check for text.
    parts = [context.strip()] if isinstance(context, str) and context.strip() else []
    parts.append(question.strip())
    parts += [f"{LETTERS[i]}. {opt.strip()}" for i, opt in enumerate(options)]
    parts.append(INSTRUCTION)
    return "\n".join(parts)


# ---------- models ----------

class LocalModel:
    """Local Hugging Face model on CPU, greedy decoding."""

    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.torch = torch
        print(f"Loading {name} on CPU (first time downloads it)...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(name)
        # bfloat16 halves memory (2 bytes/param: ~1 GB for 0.5B, ~6 GB for 3B).
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
        self.model.eval()

    def ask(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(new_tokens, skip_special_tokens=True)


class ApiModel:
    """Large model through Hugging Face Inference Providers (chat completion)."""

    def __init__(self, name: str):
        from huggingface_hub import InferenceClient

        if not config.HF_TOKEN:
            sys.exit("HF_TOKEN missing: put it in .env")
        self.name = name
        self.client = InferenceClient(token=config.HF_TOKEN)
        self.last_usage = None

    def ask(self, prompt: str, retries: int = 3) -> str:
        for attempt in range(1, retries + 1):
            try:
                resp = self.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.name,
                    max_tokens=config.MAX_NEW_TOKENS,
                    temperature=config.TEMPERATURE,
                )
                self.last_usage = getattr(resp, "usage", None)
                return resp.choices[0].message.content or ""
            except Exception as e:  # network / rate limit / provider error
                status = getattr(getattr(e, "response", None), "status_code", None)
                if attempt == retries or status in (401, 402, 403):  # auth/billing: retry won't help
                    raise
                wait = 5 * attempt
                print(f"  strong call failed ({e.__class__.__name__}: {e}); retry in {wait}s",
                      flush=True)
                time.sleep(wait)


# ---------- resumable run ----------

ROLE_MODELS = {"weak": config.WEAK_MODEL, "strong": config.STRONG_MODEL}


def make_model(role: str):
    name = ROLE_MODELS[role]
    if role == "strong" and config.STRONG_BACKEND == "api":
        return ApiModel(name)
    return LocalModel(name)


def load_done() -> set[tuple[int, str]]:
    """(id, model) pairs already answered. Keyed on model name, so answers from an
    older model pair are never mistaken for the current one."""
    if not config.RESPONSES_PATH.exists():
        return set()
    with open(config.RESPONSES_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {(r["id"], r["model"]) for r in records}


def get_sample(n: int) -> pd.DataFrame:
    path = sample_path(n)
    if path.exists():
        return pd.read_json(path, lines=True)
    sample = stratified_sample(load_clean(), n)
    sample.to_json(path, orient="records", lines=True, force_ascii=False)
    return sample


def with_prompts(sample: pd.DataFrame) -> pd.DataFrame:
    sample = sample.copy()
    sample["prompt"] = [build_prompt(r.question, r.options, r.context) for r in sample.itertuples()]
    return sample


def run(n: int, role: str):
    sample = with_prompts(get_sample(n))
    name = ROLE_MODELS[role]
    done = load_done()
    rows = [r for r in sample.itertuples() if (r.id, name) not in done]
    print(f"{role} = {name} | N={len(sample)} | to do: {len(rows)} "
          f"(already done: {len(sample) - len(rows)})", flush=True)
    if not rows:
        build_routing_table(sample)
        return

    model = make_model(role)
    started = time.perf_counter()
    with open(config.RESPONSES_PATH, "a", encoding="utf-8") as out:
        for i, r in enumerate(rows, 1):
            t0 = time.perf_counter()
            try:
                raw = model.ask(r.prompt)
            except Exception as e:
                # Not saved, so a rerun will try this question again.
                print(f"[{role} {i}/{len(rows)}] id={r.id} FAILED: {e}", flush=True)
                if config.STRONG_BACKEND == "api" and "402" in str(e):
                    sys.exit("Out of API credits: stopping. Rerun later to resume.")
                continue
            secs = time.perf_counter() - t0
            parsed = parse_letter(raw, n_options=len(r.options))
            rec = {
                "id": int(r.id), "role": role, "model": name,
                "raw": raw, "parsed": parsed, "answer": r.answer_letter,
                "correct": parsed == r.answer_letter, "parse_failed": parsed is None,
                "seconds": round(secs, 2),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            eta_min = (time.perf_counter() - started) / i * (len(rows) - i) / 60
            flag = " PARSE-FAIL" if parsed is None else ""
            print(f"[{role} {i}/{len(rows)}] id={r.id} raw={raw!r} -> {parsed} "
                  f"(answer {r.answer_letter}, {'OK' if rec['correct'] else 'wrong'}) "
                  f"{secs:.1f}s | ETA {eta_min:.0f} min{flag}", flush=True)

    build_routing_table(sample)


def build_routing_table(sample: pd.DataFrame):
    """Join weak and strong answers of the CURRENT model pair into one row per question."""
    if "prompt" not in sample:
        sample = with_prompts(sample)
    resp = pd.read_json(config.RESPONSES_PATH, lines=True)
    resp = resp[resp["id"].isin(sample["id"]) & resp["model"].isin(ROLE_MODELS.values())]
    resp = resp.assign(role=resp["model"].map({v: k for k, v in ROLE_MODELS.items()}))

    print("\n=== Summary (current model pair) ===")
    for role, g in resp.groupby("role"):
        print(f"{role:6s} {g['model'].iloc[0]:28s} answered {len(g)} | accuracy "
              f"{g['correct'].mean():.1%} | parse failures {g['parse_failed'].sum()} "
              f"({g['parse_failed'].mean():.1%}) | median {g['seconds'].median():.2f}s/question")

    wide = resp.pivot_table(index="id", columns="role", values="correct", aggfunc="last")
    if not {"weak", "strong"} <= set(wide.columns):
        print("routing table not written yet: need answers from both models")
        return False
    table = sample[["id", "subject", "prompt"]].merge(
        wide.rename(columns={"weak": "weak_correct", "strong": "strong_correct"}),
        left_on="id", right_index=True, how="inner",
    ).dropna(subset=["weak_correct", "strong_correct"])
    table[["weak_correct", "strong_correct"]] = table[["weak_correct", "strong_correct"]].astype(bool)
    table.to_csv(config.ROUTING_TABLE_PATH, index=False, encoding="utf-8")
    print(f"routing table: {len(table)} complete rows -> {config.ROUTING_TABLE_PATH.name}")
    if len(table) < len(sample):
        print(f"WARNING: {len(sample) - len(table)} questions still miss an answer from one model")
    return True


def test_strong():
    """Exactly one call (API or local, per config) on a toy Arabic question."""
    prompt = build_prompt("ما عاصمة الإمارات العربية المتحدة؟", ["دبي", "أبوظبي", "الشارقة", "العين"])
    print("Prompt sent:\n" + prompt + "\n")
    m = make_model("strong")
    t0 = time.perf_counter()
    raw = m.ask(prompt)
    print(f"Model: {m.name}")
    print(f"Raw reply: {raw!r} -> parsed: {parse_letter(raw, 4)} (correct: B)")
    print(f"Time: {time.perf_counter() - t0:.1f}s | usage: {getattr(m, 'last_usage', None)}")


class Tee:
    """Write to the console and a log file at the same time (Windows has no `tee`/tmux)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="also write all output to this file")
    ap.add_argument("--role", choices=["weak", "strong"])
    ap.add_argument("--table", action="store_true", help="only rebuild the routing table")
    ap.add_argument("--test-strong", action="store_true")
    ap.add_argument("-n", type=int, default=config.N_SAMPLES)
    args = ap.parse_args()
    if args.log:
        log_file = open(args.log, "a", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)
    if args.test_strong:
        test_strong()
    elif args.table:
        if not build_routing_table(get_sample(args.n)):
            sys.exit(1)
    elif args.role:
        run(args.n, args.role)
    else:
        ap.error("choose --role weak, --role strong, --table or --test-strong")
