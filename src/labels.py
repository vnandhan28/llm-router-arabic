"""Phase 3: routing labels, basic statistics, and the two evaluation splits.

  python -m src.labels                 # real data (needs the strong-model run)
  $env:PRACTICE=1; python -m src.labels   # practice data (simulated strong answers)

Label: need_strong = weak wrong AND strong right
       (the only questions where paying for the strong model changes the outcome).
Splits (saved to splits.json so every router uses exactly the same ones):
  (a) random 70/30, fixed seed, stratified on need_strong so both parts have a
      similar share of positives;
  (b) unseen subjects: GroupKFold by subject, 5 folds, so test subjects are never
      seen in training.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from src import config
from src.query_models import get_sample


def load_table(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or config.ROUTING_TABLE_PATH, encoding="utf-8")
    df["weak_correct"] = df["weak_correct"].astype(bool)
    df["strong_correct"] = df["strong_correct"].astype(bool)
    df["need_strong"] = add_label(df["weak_correct"], df["strong_correct"])
    return df


def add_label(weak_correct: pd.Series, strong_correct: pd.Series) -> pd.Series:
    return (~weak_correct) & strong_correct


def chance_accuracy(n_options: pd.Series) -> float:
    """Expected accuracy of uniform random guessing, given each question's option count."""
    return float((1 / n_options).mean())


def summarize(df: pd.DataFrame, n_options: pd.Series | None = None) -> pd.DataFrame:
    """Print overall stats + warnings; return the per-subject table."""
    weak, strong = df["weak_correct"].mean(), df["strong_correct"].mean()
    oracle = (df["weak_correct"] | df["strong_correct"]).mean()
    print(f"Questions:            {len(df)}")
    print(f"Weak accuracy:        {weak:.1%}")
    print(f"Strong accuracy:      {strong:.1%}")
    print(f"need_strong:          {df['need_strong'].mean():.1%}  "
          f"(weak wrong, strong right: {df['need_strong'].sum()})")
    print(f"Both right:           {(df['weak_correct'] & df['strong_correct']).mean():.1%}")
    print(f"Both wrong:           {(~df['weak_correct'] & ~df['strong_correct']).mean():.1%}")
    print(f"Weak right, strong wrong: {(df['weak_correct'] & ~df['strong_correct']).mean():.1%}")
    print(f"Oracle router ceiling: {oracle:.1%}  (best possible: strong only when needed)")

    if n_options is not None:
        chance = chance_accuracy(n_options)
        print(f"Random-guess accuracy: {chance:.1%}")
        if weak - chance < config.NEAR_CHANCE_MARGIN:
            print("WARNING: weak model is near chance level; it is barely answering, "
                  "so 'route everything to strong' may be hard to beat.")
    if weak > config.WEAK_TOO_GOOD:
        print("WARNING: weak accuracy > 90%; routing has little room to help.")
    if strong <= weak:
        print("WARNING: strong model is not better than weak; routing makes no sense.")

    per_subject = df.groupby("subject").agg(
        n=("id", "size"),
        weak_acc=("weak_correct", "mean"),
        strong_acc=("strong_correct", "mean"),
        need_strong_rate=("need_strong", "mean"),
    ).sort_values("need_strong_rate", ascending=False)
    return per_subject


def make_splits(df: pd.DataFrame, seed: int = config.SEED) -> dict:
    """Return both split setups as lists of question ids."""
    train_ids, test_ids = train_test_split(
        df["id"], test_size=config.TEST_SIZE, random_state=seed, stratify=df["need_strong"]
    )
    folds = []
    for tr, te in GroupKFold(n_splits=config.N_GROUP_FOLDS).split(df, groups=df["subject"]):
        folds.append({
            "train": df["id"].iloc[tr].tolist(),
            "test": df["id"].iloc[te].tolist(),
            "test_subjects": sorted(df["subject"].iloc[te].unique()),
        })
    return {
        "random": {"train": sorted(train_ids.tolist()), "test": sorted(test_ids.tolist())},
        "unseen_subjects": folds,
    }


def main():
    if config.PRACTICE:
        print("=" * 70 + "\nPRACTICE MODE: strong answers are SIMULATED. Numbers are not real.\n"
              + "=" * 70)
    df = load_table()
    sample = get_sample(config.N_SAMPLES).set_index("id")
    n_options = df["id"].map(sample["options"].str.len())

    per_subject = summarize(df, n_options)
    out = config.RESULTS_DIR / "label_stats_by_subject.csv"
    per_subject.round(3).to_csv(out, encoding="utf-8")
    print(f"\nPer subject (top 10 by need_strong rate; full table -> "
          f"{out.relative_to(config.ROOT)}):")
    print(per_subject.head(10).to_string(float_format=lambda x: f"{x:.2f}"))

    splits = make_splits(df)
    config.SPLITS_PATH.write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding="utf-8")
    r = splits["random"]
    is_pos = df.set_index("id")["need_strong"]
    print(f"\nRandom split: train {len(r['train'])} / test {len(r['test'])} | need_strong "
          f"train {is_pos[r['train']].mean():.1%}, test {is_pos[r['test']].mean():.1%}")
    print("Unseen-subject folds:")
    for k, f in enumerate(splits["unseen_subjects"], 1):
        print(f"  fold {k}: train {len(f['train'])} / test {len(f['test'])} questions, "
              f"{len(f['test_subjects'])} test subjects")
    print(f"Splits saved -> {config.SPLITS_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
