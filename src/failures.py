"""Phase 6: where does the embedding router make its worst decisions?

Operating point: route the top X% of questions (by score) to the strong model, where
X = % strong calls needed to reach 90% of strong accuracy (from Phase 5); 50% if never.
Two kinds of bad decisions:
  missed : sent to weak, weak was wrong, strong would have been right (costs accuracy)
  wasted : sent to strong, but weak was already right               (costs money)
"Worst" = the most confident ones: missed with the lowest scores, wasted with the highest.

  python -m src.failures                    # real data (run src.evaluate first)
  $env:PRACTICE=1; python -m src.failures   # practice data
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.evaluate import routing_order
from src.query_models import get_sample, with_prompts

ROUTER = "embedding"
N_EXAMPLES = 15
MIN_SUBJECT_N = 20     # subjects smaller than this are too noisy to rank
MIN_SUBJECT_POS = 3    # need a few need_strong questions to compute AUROC


def operating_point(setup_tag: str) -> float:
    m = pd.read_csv(config.RESULTS_DIR / f"metrics_{setup_tag}.csv").set_index("router")
    pct = m.loc[ROUTER, "pct_strong_for_90"]
    return 50.0 if np.isnan(pct) else float(pct)


def mark_decisions(scored: pd.DataFrame, pct: float) -> pd.DataFrame:
    """Add sent_to_strong + missed/wasted flags for routing the top pct% to strong."""
    scored = scored.reset_index(drop=True)
    k = int(round(pct / 100 * len(scored)))
    sent = np.zeros(len(scored), dtype=bool)
    sent[routing_order(scored["score"].to_numpy())[:k]] = True
    return scored.assign(
        sent_to_strong=sent,
        missed=~sent & scored["need_strong"],
        wasted=sent & scored["weak_correct"],
    )


def worst_examples(d: pd.DataFrame, n: int = N_EXAMPLES) -> pd.DataFrame:
    n_missed = min(int(d["missed"].sum()), (n + 1) // 2)
    n_wasted = min(int(d["wasted"].sum()), n - n_missed)
    missed = d[d["missed"]].nsmallest(n_missed, "score").assign(error="missed")
    wasted = d[d["wasted"]].nlargest(n_wasted, "score").assign(error="wasted")
    return pd.concat([missed, wasted], ignore_index=True)


def subject_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subj, g in d.groupby("subject"):
        pos = int(g["need_strong"].sum())
        ok = len(g) >= MIN_SUBJECT_N and MIN_SUBJECT_POS <= pos < len(g)
        rows.append({
            "subject": subj, "n": len(g), "need_strong": pos,
            "auroc": roc_auc_score(g["need_strong"], g["score"]) if ok else np.nan,
            "missed_rate": g["missed"].mean(), "wasted_rate": g["wasted"].mean(),
        })
    return pd.DataFrame(rows).sort_values("auroc", na_position="last").reset_index(drop=True)


def main():
    if config.PRACTICE:
        print("=" * 70 + "\nPRACTICE MODE: strong answers are SIMULATED. Numbers are not real.\n"
              + "=" * 70)
    sample = with_prompts(get_sample(config.N_SAMPLES)).set_index("id")
    sample["n_options"] = sample["options"].str.len()
    sample["has_context"] = sample["context"].apply(lambda c: isinstance(c, str))
    sample["prompt_chars"] = sample["prompt"].str.len()
    features = ["n_options", "has_context", "prompt_chars"]

    # 1) Worst individual decisions: random-split test set.
    pct = operating_point("random_split")
    sc = pd.read_csv(config.WORK_DIR / "scores_random_split.csv")
    d = mark_decisions(sc[sc["router"] == ROUTER], pct)
    print(f"Embedding router, random-split test set (n={len(d)}), "
          f"top {pct:.1f}% sent to strong:")
    print(f"  missed (needed strong, sent to weak): {d['missed'].sum()} "
          f"of {d['need_strong'].sum()} need_strong questions")
    print(f"  wasted (weak was right, sent to strong): {d['wasted'].sum()} "
          f"of {d['sent_to_strong'].sum()} strong calls")

    ex = worst_examples(d)
    d = d.join(sample[features], on="id")
    ex = ex.join(sample[features + ["question"]], on="id")
    cols = ["error", "id", "subject", "score", "weak_correct", "strong_correct",
            "n_options", "has_context", "prompt_chars", "question"]
    out = config.RESULTS_DIR / "failure_examples.csv"
    ex[cols].round(3).to_csv(out, index=False, encoding="utf-8-sig")  # -sig: Excel shows Arabic
    print(f"\n{len(ex)} worst decisions -> {out.relative_to(config.ROOT)}")
    print(ex[cols[:-1]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 2) Which subjects are hardest to route: unseen-subject setup (every question scored
    #    by a router that never saw its subject).
    pct_u = operating_point("unseen_subjects")
    su = pd.read_csv(config.WORK_DIR / "scores_unseen_subjects.csv")
    du = mark_decisions(su[su["router"] == ROUTER], pct_u)
    subj = subject_table(du)
    out = config.RESULTS_DIR / "failure_by_subject.csv"
    subj.round(3).to_csv(out, index=False, encoding="utf-8-sig")

    ranked = subj.dropna(subset=["auroc"])
    print(f"\nPer subject, unseen-subject setup (top {pct_u:.1f}% to strong). "
          f"{len(ranked)} of {len(subj)} subjects have enough data "
          f"(n >= {MIN_SUBJECT_N}, >= {MIN_SUBJECT_POS} need_strong) for AUROC:")
    print("Hardest to route (lowest AUROC):")
    print(ranked.head(5).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("Easiest to route (highest AUROC):")
    print(ranked.tail(5).iloc[::-1].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"Full table -> {out.relative_to(config.ROOT)}")

    # 3) Short factual pattern check on the worst examples.
    print("\nPatterns among the worst decisions vs. the whole test set:")
    labels = {"n_options": "mean number of options", "has_context": "share with context passage",
              "prompt_chars": "mean prompt length (chars)"}
    for col, label in labels.items():
        parts = [f"{kind} {ex.loc[ex['error'] == kind, col].mean():.2f}"
                 for kind in ("missed", "wasted") if (ex["error"] == kind).any()]
        print(f"  {label:28s} {' | '.join(parts)} | all test {d[col].mean():.2f}")


if __name__ == "__main__":
    main()
