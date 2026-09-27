import pandas as pd

from src.data import allocate, stratified_sample


def test_allocate_sums_to_n_and_is_proportional():
    sizes = pd.Series({"big": 700, "mid": 200, "small": 100})
    alloc = allocate(sizes, 10)
    assert alloc.sum() == 10
    assert alloc.to_dict() == {"big": 7, "mid": 2, "small": 1}


def test_allocate_gives_every_group_one_when_n_is_large_enough():
    sizes = pd.Series({"a": 1000, "b": 5, "c": 5})
    alloc = allocate(sizes, 10)
    assert alloc.sum() == 10
    assert (alloc >= 1).all()


def test_allocate_never_exceeds_group_size():
    sizes = pd.Series({"a": 3, "b": 2})
    assert allocate(sizes, 100).to_dict() == {"a": 3, "b": 2}


def test_stratified_sample_is_reproducible():
    df = pd.DataFrame({"id": range(100), "subject": ["x"] * 60 + ["y"] * 40})
    s1 = stratified_sample(df, 10, seed=1)
    s2 = stratified_sample(df, 10, seed=1)
    assert s1["id"].tolist() == s2["id"].tolist()
    assert s1["subject"].value_counts().to_dict() == {"x": 6, "y": 4}
