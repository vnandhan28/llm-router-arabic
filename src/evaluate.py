"""Phase 5: RouteLLM-style cost/accuracy curves, AUC, APGR, bootstrap CIs.

For a router's scores: send the top p% of questions (highest score) to the strong
model and the rest to the weak one, for p = 0, 5, ..., 100. Overall accuracy at each
p gives the curve. Then:
  AUC  = area under accuracy vs. fraction-to-strong (x from 0 to 1) = mean accuracy
  APGR = (AUC - weak_acc) / (strong_acc - weak_acc)
         0 ~ no better than random mixing, 1 ~ strong accuracy at no extra cost
  % strong calls needed to reach 90% / 95% of the strong model's accuracy
  bootstrap 95% CI for APGR (resample the test questions with replacement)

  python -m src.evaluate                    # real data -> results/
  $env:PRACTICE=1; python -m src.evaluate   # practice data -> data/practice/results/
"""
import numpy as np
import pandas as pd

from src import config
from src.router import ROUTERS, load_splits, scores_for_splits

TARGETS = (0.90, 0.95)


# ---------- metric functions (pure numpy, unit-tested) ----------

def routing_order(scores: np.ndarray) -> np.ndarray:
    """Indices from highest to lowest score; ties keep their original order."""
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")


def accuracy_curve(weak: np.ndarray, strong: np.ndarray, scores: np.ndarray,
                   pcts=config.STRONG_PCTS) -> np.ndarray:
    """Overall accuracy when the top p% by score go to the strong model."""
    weak, strong = np.asarray(weak, float), np.asarray(strong, float)
    n = len(weak)
    gain = (strong - weak)[routing_order(scores)]  # change in correct answers per routed question
    cum_gain = np.concatenate([[0.0], np.cumsum(gain)])
    k = np.round(np.asarray(pcts) / 100 * n).astype(int)
    return (weak.sum() + cum_gain[k]) / n


def auc(curve: np.ndarray, pcts=config.STRONG_PCTS) -> float:
    """Trapezoid area under accuracy vs. fraction routed (x in [0, 1])."""
    x = np.asarray(pcts) / 100
    y = np.asarray(curve)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def apgr(auc_value: float, weak_acc: float, strong_acc: float) -> float:
    gap = strong_acc - weak_acc
    return float("nan") if gap <= 0 else (auc_value - weak_acc) / gap


def pct_strong_to_reach(weak, strong, scores, target_fraction: float) -> float:
    """Smallest % of questions sent to strong so accuracy >= target_fraction * strong_acc.

    Uses every possible cut-off (not just 5% steps) for precision; NaN if never reached.
    """
    weak, strong = np.asarray(weak, float), np.asarray(strong, float)
    n = len(weak)
    acc = (weak.sum() + np.concatenate([[0.0], np.cumsum((strong - weak)[routing_order(scores)])])) / n
    goal = target_fraction * strong.mean()
    hits = np.nonzero(acc >= goal - 1e-12)[0]
    return float("nan") if len(hits) == 0 else 100 * hits[0] / n


def evaluate_scores(weak, strong, scores) -> dict:
    weak_acc, strong_acc = float(np.mean(weak)), float(np.mean(strong))
    curve = accuracy_curve(weak, strong, scores)
    a = auc(curve)
    out = {"weak_acc": weak_acc, "strong_acc": strong_acc, "auc": a,
           "apgr": apgr(a, weak_acc, strong_acc)}
    for t in TARGETS:
        out[f"pct_strong_for_{int(t * 100)}"] = pct_strong_to_reach(weak, strong, scores, t)
    return out


def bootstrap_apgr(weak, strong, scores, n_boot=config.N_BOOTSTRAP, seed=config.SEED):
    """95% percentile CI for APGR, resampling questions with replacement."""
    weak, strong, scores = map(np.asarray, (weak, strong, scores))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(weak), len(weak))
        w, s = weak[idx], strong[idx]
        v = apgr(auc(accuracy_curve(w, s, scores[idx])), w.mean(), s.mean())
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


# ---------- plotting ----------

# Reference categorical palette (dataviz skill), fixed per router so colors never shift.
STYLE = {
    "tfidf":     dict(color="#2a78d6", lw=2, ls="-", label="TF-IDF + LR"),
    "embedding": dict(color="#eb6834", lw=2, ls="-", label="Embedding (e5-small) + LR"),
    "length":    dict(color="#1baf7a", lw=2, ls="-", label="Length baseline"),
    "random":    dict(color="#898781", lw=2, ls="--", label="Random baseline"),
    "oracle":    dict(color="#0b0b0b", lw=1.5, ls=":", label="Oracle (upper bound)"),
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def plot_curves(curves: dict[str, np.ndarray], weak_acc, strong_acc, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.asarray(config.STRONG_PCTS)
    for name, st in STYLE.items():
        if name in curves:
            ax.plot(x, 100 * curves[name], marker="o", ms=3, **st)
    for y, label in [(weak_acc, "weak model only"), (strong_acc, "strong model only")]:
        ax.axhline(100 * y, color=GRID, lw=1, zorder=0)
        ax.text(101, 100 * y, f"{label}\n{100 * y:.1f}%", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of questions sent to the strong model (cost)", color=INK)
    ax.set_ylabel("Overall accuracy (%)", color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------- main ----------

def main():
    from src.labels import load_table

    tag = "PRACTICE (simulated strong answers, not real)" if config.PRACTICE else "ArabicMMLU"
    if config.PRACTICE:
        print("=" * 70 + "\nPRACTICE MODE: strong answers are SIMULATED. Numbers are not real.\n"
              + "=" * 70)
    df = load_table()
    splits = load_splits()

    setups = {"random": "random_split", "unseen_subjects": "unseen_subjects"}
    rows = {s: [] for s in setups}
    curves = {s: {} for s in setups}
    all_scores = {s: [] for s in setups}

    for name in ROUTERS:
        res = scores_for_splits(df, name, splits)
        for setup, scored in res.items():
            w, s, sc = scored["weak_correct"], scored["strong_correct"], scored["score"]
            m = evaluate_scores(w, s, sc)
            lo, hi = bootstrap_apgr(w.to_numpy(), s.to_numpy(), sc.to_numpy())
            rows[setup].append({"router": name, **m, "apgr_ci_low": lo, "apgr_ci_high": hi,
                                "n_test": len(scored)})
            curves[setup][name] = accuracy_curve(w, s, sc)
            all_scores[setup].append(scored[["id", "subject", "weak_correct", "strong_correct",
                                             "need_strong", "score"]].assign(router=name))
        print(f"  evaluated {name}")

    for setup, file_tag in setups.items():
        table = pd.DataFrame(rows[setup])
        cols = ["router", "n_test", "weak_acc", "strong_acc", "auc", "apgr", "apgr_ci_low",
                "apgr_ci_high", "pct_strong_for_90", "pct_strong_for_95"]
        table = table[cols]
        table.round(4).to_csv(config.RESULTS_DIR / f"metrics_{file_tag}.csv", index=False)
        pd.concat(all_scores[setup]).to_csv(config.WORK_DIR / f"scores_{file_tag}.csv",
                                            index=False, encoding="utf-8")

        label = "Random 70/30 split" if setup == "random" else "Unseen subjects (5-fold GroupKFold)"
        plot_curves(curves[setup], table["weak_acc"].iloc[0], table["strong_acc"].iloc[0],
                    f"Accuracy vs. cost: {label}\n{tag}",
                    config.RESULTS_DIR / f"curve_{file_tag}.png")

        print(f"\n=== {label} (n_test={table['n_test'].iloc[0]}) ===")
        show = table.drop(columns=["n_test", "weak_acc", "strong_acc"]).copy()
        show["APGR [95% CI]"] = [f"{a:.3f} [{l:.3f}, {h:.3f}]" for a, l, h in
                                 zip(show.pop("apgr"), show.pop("apgr_ci_low"), show.pop("apgr_ci_high"))]
        print(f"weak acc {table['weak_acc'].iloc[0]:.1%} | strong acc {table['strong_acc'].iloc[0]:.1%}")
        print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nSaved metrics + curve plots -> {config.RESULTS_DIR.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
