"""Phase 4: routers that score how much a question needs the strong model (0-1).

Every router has the same two methods:
  fit(train_df)  -> learn from questions with known need_strong labels
  score(df)      -> array of scores in [0, 1]; higher = send to the strong model
This is the role of RouteLLM's calculate_strong_win_rate.

  python -m src.router                    # real data
  $env:PRACTICE=1; python -m src.router   # practice data
"""
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from src import config


class RandomRouter:
    """Baseline: a coin flip. Any useful router must beat this."""

    def __init__(self, seed: int = config.SEED):
        self.seed = seed

    def fit(self, train_df):
        return self

    def score(self, df):
        return np.random.default_rng(self.seed).random(len(df))


class LengthRouter:
    """Baseline: longer prompts are assumed harder. Score = share of training prompts
    that are shorter than this one (so it is always between 0 and 1)."""

    def fit(self, train_df):
        self.train_lengths = np.sort(train_df["prompt"].str.len().to_numpy())
        return self

    def score(self, df):
        lengths = df["prompt"].str.len().to_numpy()
        return np.searchsorted(self.train_lengths, lengths, side="right") / len(self.train_lengths)


class TfidfRouter:
    """Character n-grams (works for Arabic without a tokenizer) + logistic regression."""

    def fit(self, train_df):
        self.model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                            sublinear_tf=True, max_features=50_000),
            # balanced: need_strong is the minority class, so weigh it up
            LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0),
        )
        self.model.fit(train_df["prompt"], train_df["need_strong"])
        return self

    def score(self, df):
        return self.model.predict_proba(df["prompt"])[:, 1]


class EmbeddingRouter:
    """multilingual-e5-small sentence embeddings (cached to disk) + logistic regression."""

    def fit(self, train_df):
        self.clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
        self.clf.fit(embed(train_df), train_df["need_strong"])
        return self

    def score(self, df):
        return self.clf.predict_proba(embed(df))[:, 1]

    def score_texts(self, prompts: list[str]):
        """Score new prompts that are not in the dataset (used by the demo)."""
        return self.clf.predict_proba(_compute_embeddings(prompts, verbose=False))[:, 1]


class OracleRouter:
    """Upper bound: cheats by looking at the true label. Not a real router."""

    def fit(self, train_df):
        return self

    def score(self, df):
        return df["need_strong"].astype(float).to_numpy()


ROUTERS = {
    "random": RandomRouter,
    "length": LengthRouter,
    "tfidf": TfidfRouter,
    "embedding": EmbeddingRouter,
    "oracle": OracleRouter,
}


# ---------- embeddings (computed once, cached by question id) ----------

_EMBED_MEMO: dict[int, np.ndarray] = {}


def _load_cache() -> dict[int, np.ndarray]:
    path = config.EMBED_CACHE_PATH
    if not _EMBED_MEMO and path.exists():
        data = np.load(path)
        _EMBED_MEMO.update(zip(data["ids"].tolist(), data["vectors"]))
    return _EMBED_MEMO


def _compute_embeddings(texts: list[str], batch_size: int = 16, verbose: bool = True) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.EMBED_MODEL)
    model = AutoModel.from_pretrained(config.EMBED_MODEL).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = ["query: " + t for t in texts[i:i + batch_size]]  # e5 expects this prefix
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)  # mean over real tokens (e5 recipe)
        out.append(torch.nn.functional.normalize(pooled, dim=-1).numpy())
        if verbose:
            print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}", end="\r", flush=True)
    if verbose:
        print()
    return np.vstack(out)


def embed(df: pd.DataFrame) -> np.ndarray:
    cache = _load_cache()
    missing = df.loc[~df["id"].isin(cache.keys()), ["id", "prompt"]].drop_duplicates("id")
    if len(missing):
        print(f"Computing {len(missing)} embeddings with {config.EMBED_MODEL} (cached afterwards)")
        vecs = _compute_embeddings(missing["prompt"].tolist())
        cache.update(zip(missing["id"].tolist(), vecs))
        ids = np.array(list(cache.keys()))
        np.savez(config.EMBED_CACHE_PATH, ids=ids, vectors=np.vstack([cache[i] for i in ids]))
    return np.vstack([cache[i] for i in df["id"]])


# ---------- scoring on the saved splits ----------

def load_splits() -> dict:
    return json.loads(config.SPLITS_PATH.read_text(encoding="utf-8"))


def scores_for_splits(df: pd.DataFrame, name: str, splits: dict) -> dict[str, pd.DataFrame]:
    """Fit on train, score on test, for both evaluation setups.

    Returns {"random": test rows + score, "unseen_subjects": all rows scored out-of-fold}.
    """
    by_id = df.set_index("id", drop=False)
    out = {}

    r = splits["random"]
    router = ROUTERS[name]().fit(by_id.loc[r["train"]])
    test = by_id.loc[r["test"]].reset_index(drop=True)
    out["random"] = test.assign(score=router.score(test))

    parts = []
    for k, f in enumerate(splits["unseen_subjects"]):
        router = ROUTERS[name]().fit(by_id.loc[f["train"]])
        test = by_id.loc[f["test"]].reset_index(drop=True)
        parts.append(test.assign(score=router.score(test), fold=k))
    out["unseen_subjects"] = pd.concat(parts, ignore_index=True)
    return out


def main():
    from src.labels import load_table

    if config.PRACTICE:
        print("=" * 70 + "\nPRACTICE MODE: strong answers are SIMULATED. Numbers are not real.\n"
              + "=" * 70)
    df = load_table()
    splits = load_splits()
    embed(df)  # compute / load all embeddings once up front

    print("\nSanity check: AUROC of each router at spotting need_strong questions")
    print("(0.5 = no better than a coin flip, 1.0 = perfect)\n")
    print(f"{'router':10s} {'random split':>13s} {'unseen subjects':>16s}")
    for name in ROUTERS:
        res = scores_for_splits(df, name, splits)
        aucs = [roc_auc_score(res[s]["need_strong"], res[s]["score"]) for s in res]
        print(f"{name:10s} {aucs[0]:13.3f} {aucs[1]:16.3f}")


if __name__ == "__main__":
    main()
