"""Phase 7: route one Arabic multiple-choice question and answer it.

  python -m src.demo "ما عاصمة الإمارات؟
  A. دبي
  B. أبوظبي
  C. الشارقة"
  python -m src.demo --file question.txt      # easier on Windows terminals with Arabic
  python -m src.demo --route-only "..."       # only show the routing decision
  python -m src.demo --retrain                # retrain the saved router

Prints the router score, the chosen model and that model's answer. Only the chosen
model is loaded, so the 8 GB laptop never holds both at once.

The router is the embedding router trained on ALL questions in the routing table.
Cut-off: a question goes to the strong model if its score is in the top X% of
training scores, where X = % strong calls needed to reach 90% of strong accuracy
(Phase 5, random split). Use --threshold to set the cut-off yourself.
"""
import argparse
import re
import sys

import joblib
import numpy as np
import pandas as pd

from src import config
from src.parse import ARABIC_TO_LATIN, parse_letter
from src.query_models import LocalModel, build_prompt

ROUTER_PATH = config.WORK_DIR / "router_embedding.joblib"
_OPTION_LINE = re.compile(r"^\s*[\(\[]?([A-Ea-eأابجده]ـ?)\s*[\)\].:\-]\s*(.+)$")
_INLINE_SPLIT = re.compile(r"\s+(?=[\(\[]?[A-E]\s*[\)\].:]\s)")


def parse_question(text: str) -> tuple[str, list[str]]:
    """Split free text into (question, options). Options are lines like 'A. ...',
    'B) ...', '(C) ...' or Arabic 'أ) ...'; they may also sit on one line."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) == 1:
        lines = _INLINE_SPLIT.split(lines[0])
    question, options = [], []
    for ln in lines:
        m = _OPTION_LINE.match(ln)
        if m and (options or question):  # first line is always (part of) the question
            options.append(m.group(2).strip())
        else:
            question.append(ln.strip())
    return " ".join(question), options[:5]


def train_and_save():
    from src.labels import load_table
    from src.router import EmbeddingRouter, embed

    df = load_table()
    router = EmbeddingRouter().fit(df)
    metrics = pd.read_csv(config.RESULTS_DIR / "metrics_random_split.csv").set_index("router")
    pct = metrics.loc["embedding", "pct_strong_for_90"]
    pct = 50.0 if np.isnan(pct) else float(pct)
    train_scores = router.clf.predict_proba(embed(df))[:, 1]
    threshold = float(np.quantile(train_scores, 1 - pct / 100))
    bundle = {"router": router, "threshold": threshold, "pct_strong": pct,
              "n_train": len(df), "practice": config.PRACTICE}
    joblib.dump(bundle, ROUTER_PATH)
    print(f"Trained embedding router on {len(df)} questions; cut-off {threshold:.3f} "
          f"(sends ~{pct:.1f}% to strong) -> {ROUTER_PATH.relative_to(config.ROOT)}")
    return bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?", help="question text with options A-E")
    ap.add_argument("--file", help="read the question from a UTF-8 text file")
    ap.add_argument("--threshold", type=float, help="score cut-off for the strong model")
    ap.add_argument("--route-only", action="store_true", help="do not load a model to answer")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if config.PRACTICE:
        print("PRACTICE MODE: router trained on SIMULATED strong answers - for testing only.\n")

    bundle = train_and_save() if args.retrain or not ROUTER_PATH.exists() else joblib.load(ROUTER_PATH)
    if not (args.question or args.file):
        return
    text = open(args.file, encoding="utf-8").read() if args.file else args.question

    question, options = parse_question(text)
    if len(options) < 2:
        sys.exit("Could not find at least 2 options. Put each option on its own line, "
                 "e.g. 'A. ...', 'B. ...'.")
    prompt = build_prompt(question, options)
    print("Prompt:\n" + prompt + "\n")

    threshold = args.threshold if args.threshold is not None else bundle["threshold"]
    score = float(bundle["router"].score_texts([prompt])[0])
    role = "strong" if score >= threshold else "weak"
    model_name = config.STRONG_MODEL if role == "strong" else config.WEAK_MODEL
    print(f"Router score:  {score:.3f}  (cut-off {threshold:.3f})")
    print(f"Chosen model:  {role} -> {model_name}")
    if args.route_only:
        return

    raw = LocalModel(model_name).ask(prompt)
    letter = parse_letter(raw, n_options=len(options))
    answer = f"{letter}. {options['ABCDE'.index(letter)]}" if letter else "(no valid letter)"
    print(f"Model reply:   {raw!r}")
    print(f"Answer:        {answer}")


if __name__ == "__main__":
    main()
