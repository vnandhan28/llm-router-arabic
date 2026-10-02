import numpy as np
import pandas as pd

from src.router import LengthRouter, OracleRouter, RandomRouter, TfidfRouter


def _toy(n=80):
    # "hard" questions contain a marker word and are labelled need_strong
    hard = np.arange(n) % 2 == 0
    return pd.DataFrame({
        "id": np.arange(n),
        "prompt": [("صعب جدا " if h else "سهل ") + f"سؤال {i}" for i, h in enumerate(hard)],
        "need_strong": hard,
    })


def test_scores_are_in_unit_interval():
    df = _toy()
    for router in [RandomRouter(), LengthRouter(), TfidfRouter(), OracleRouter()]:
        s = router.fit(df).score(df)
        assert len(s) == len(df)
        assert (s >= 0).all() and (s <= 1).all()


def test_random_router_is_reproducible():
    df = _toy()
    assert np.array_equal(RandomRouter(seed=1).score(df), RandomRouter(seed=1).score(df))


def test_length_router_ranks_longer_prompts_higher():
    train = pd.DataFrame({"prompt": ["a", "aa", "aaa", "aaaa"]})
    s = LengthRouter().fit(train).score(pd.DataFrame({"prompt": ["a", "aaaa"]}))
    assert s[0] < s[1]


def test_tfidf_router_learns_an_obvious_pattern():
    df = _toy()
    s = TfidfRouter().fit(df).score(df)
    assert s[df["need_strong"]].mean() > s[~df["need_strong"]].mean() + 0.3


def test_oracle_router_returns_the_labels():
    df = _toy()
    assert OracleRouter().score(df).tolist() == df["need_strong"].astype(float).tolist()
