"""Phase 1: load ArabicMMLU, inspect structure, build a clean stratified sample.

Column mapping (confirmed against the real data):
  ID -> id, Subject + Level -> subject (40 groups), Question -> question,
  Context -> context (707 questions, kept), Option 1..5 -> options (non-empty only),
  Answer Key (already A-E) -> answer_letter. Only the 'test' split is used;
  'dev' holds 120 few-shot examples. 40 questions whose correct option text is
  duplicated in another option are dropped as unanswerable.
"""
import numpy as np
import pandas as pd
from datasets import get_dataset_config_names, load_dataset

from src import config

LETTERS = "ABCDE"


def inspect():
    """Print config names, splits, columns and 3 example rows. No assumptions."""
    configs = get_dataset_config_names(config.DATASET_NAME)
    print(f"{len(configs)} configs:", configs)

    ds = load_dataset(config.DATASET_NAME, "All")
    print("\nSplits:", {name: len(split) for name, split in ds.items()})
    for name, split in ds.items():
        print(f"\n--- split '{name}' columns ---")
        for col, feat in split.features.items():
            print(f"  {col}: {feat}")
        print(f"\n--- 3 example rows from '{name}' ---")
        for i in range(3):
            print(split[i])


def load_clean(verbose: bool = False) -> pd.DataFrame:
    """Return one row per question: id, subject, question, context, options, answer_letter."""
    raw = load_dataset(config.DATASET_NAME, "All", split="test").to_pandas()

    level = raw["Level"].map(lambda lv: f" ({lv})" if isinstance(lv, str) else "")
    option_cols = [f"Option {i}" for i in range(1, 6)]
    options = raw[option_cols].apply(lambda row: [o for o in row if isinstance(o, str)], axis=1)

    df = pd.DataFrame({
        "id": raw["ID"],
        "subject": raw["Subject"] + level,
        "question": raw["Question"].str.strip(),
        "context": raw["Context"].where(raw["Context"].notna(), None),
        "options": options,
        "answer_letter": raw["Answer Key"].str.strip(),
    })

    # Sanity checks: fail loudly instead of silently producing bad data.
    assert df["id"].is_unique
    assert df["answer_letter"].isin(list(LETTERS)).all()
    n_opts = df["options"].str.len()
    assert (n_opts >= 2).all()
    assert (df["answer_letter"].map(LETTERS.index) < n_opts).all(), "answer points to missing option"

    # Drop questions where the correct answer's text appears in more than one option
    # (diacritics lost in the dataset), so no model could pick the "right" letter.
    ambiguous = df.apply(
        lambda r: r["options"].count(r["options"][LETTERS.index(r["answer_letter"])]) > 1, axis=1
    )
    if verbose:
        print(f"Dropped {ambiguous.sum()} ambiguous questions (correct option text duplicated)")
    return df[~ambiguous].reset_index(drop=True)


def allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Split n across groups proportionally to their sizes (largest-remainder method).

    Every group gets at least 1 when n >= number of groups (and never more than its size).
    """
    n = min(n, int(sizes.sum()))
    raw = sizes / sizes.sum() * n
    alloc = np.floor(raw).astype(int)
    if n >= len(sizes):
        alloc = alloc.clip(lower=1)
    alloc = alloc.clip(upper=sizes)

    remainder = (raw - alloc).sort_values(ascending=False)
    while alloc.sum() < n:  # hand out leftovers to groups with the largest remainder
        for g in remainder.index:
            if alloc.sum() == n:
                break
            if alloc[g] < sizes[g]:
                alloc[g] += 1
    while alloc.sum() > n:  # the min-1 rule can overshoot: take back from the biggest groups
        g = alloc.idxmax()
        alloc[g] -= 1
    return alloc


def stratified_sample(df: pd.DataFrame, n: int, seed: int = config.SEED) -> pd.DataFrame:
    """Proportional stratified sample by subject, reproducible with a fixed seed."""
    alloc = allocate(df["subject"].value_counts(), n)
    parts = [
        df[df["subject"] == subj].sample(k, random_state=seed)
        for subj, k in alloc.items() if k > 0
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def sample_path(n: int):
    return config.DATA_DIR / f"sample_{n}.jsonl"


def main():
    df = load_clean(verbose=True)
    print(f"Clean dataframe: {len(df)} questions, {df['subject'].nunique()} subjects")
    print("Columns:", list(df.columns))

    print("\nQuestions per subject:")
    print(df["subject"].value_counts().to_string())

    sample = stratified_sample(df, config.N_SAMPLES)
    out = sample_path(config.N_SAMPLES)
    sample.to_json(out, orient="records", lines=True, force_ascii=False)
    print(f"\nSample of N={len(sample)} saved to {out.relative_to(config.ROOT)}")
    print(f"Subjects in sample: {sample['subject'].nunique()}")
    print(sample["subject"].value_counts().to_string())
    print("\nAnswer letters in sample:", sample["answer_letter"].value_counts().sort_index().to_dict())
    print("With context:", sample["context"].notna().sum())

    print("\nFirst 3 sampled rows:")
    for row in sample.head(3).itertuples():
        print(f"  id={row.id} | {row.subject} | answer={row.answer_letter}")
        print(f"    Q: {row.question}")
        print(f"    options: {row.options}")


if __name__ == "__main__":
    main()
