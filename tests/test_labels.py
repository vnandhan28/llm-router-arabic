import numpy as np
import pandas as pd

from src.labels import add_label, chance_accuracy, make_splits


def test_need_strong_only_when_weak_wrong_and_strong_right():
    weak = pd.Series([True, True, False, False])
    strong = pd.Series([True, False, True, False])
    assert add_label(weak, strong).tolist() == [False, False, True, False]


def test_chance_accuracy():
    # one 2-option question (50%) and one 4-option question (25%) -> 37.5%
    assert chance_accuracy(pd.Series([2, 4])) == 0.375


def _toy_table(n=200, n_subjects=10):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "id": np.arange(n),
        "subject": [f"s{i % n_subjects}" for i in range(n)],
        "need_strong": rng.random(n) < 0.3,
    })


def test_random_split_is_disjoint_and_70_30():
    df = _toy_table()
    r = make_splits(df)["random"]
    assert set(r["train"]).isdisjoint(r["test"])
    assert len(r["train"]) + len(r["test"]) == len(df)
    assert len(r["test"]) == 60


def test_unseen_subject_folds_never_share_a_subject():
    df = _toy_table()
    subj = df.set_index("id")["subject"]
    folds = make_splits(df)["unseen_subjects"]
    assert len(folds) == 5
    for f in folds:
        assert set(subj[f["train"]]).isdisjoint(subj[f["test"]])
    # every question is tested exactly once across the folds
    all_test = sorted(i for f in folds for i in f["test"])
    assert all_test == df["id"].tolist()
