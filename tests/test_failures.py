import pandas as pd

from src.failures import mark_decisions, worst_examples


def _scored():
    #            q0     q1     q2     q3
    return pd.DataFrame({
        "id": [0, 1, 2, 3],
        "score": [0.9, 0.1, 0.6, 0.3],
        "weak_correct": [True, False, False, True],
        "need_strong": [False, True, True, False],
    })


def test_mark_decisions_top_half_to_strong():
    d = mark_decisions(_scored(), 50)  # top 2 by score: q0, q2
    assert d["sent_to_strong"].tolist() == [True, False, True, False]
    assert d["missed"].tolist() == [False, True, False, False]   # q1 needed strong, sent to weak
    assert d["wasted"].tolist() == [True, False, False, False]   # q0 weak was right, sent to strong


def test_worst_examples_picks_most_confident_mistakes():
    ex = worst_examples(mark_decisions(_scored(), 50), n=2)
    assert ex.set_index("error")["id"].to_dict() == {"missed": 1, "wasted": 0}
