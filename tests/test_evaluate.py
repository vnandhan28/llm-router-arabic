"""Metric checks on a tiny hand-made example, worked out on paper.

4 questions:          q0     q1     q2     q3
  weak correct         0      1      0      1      -> weak acc 50%
  strong correct       1      1      1      0      -> strong acc 75%
  router score        0.9    0.1    0.6    0.3
Routing order by score: q0, q2, q3, q1
  0 sent to strong: 2/4 = 0.50
  1 (q0):           3/4 = 0.75   (q0 flips wrong -> right)
  2 (q0,q2):        4/4 = 1.00   (q2 flips wrong -> right)
  3 (+q3):          3/4 = 0.75   (q3 flips right -> wrong)
  4 (+q1):          3/4 = 0.75   (q1 right either way)
"""
import numpy as np
import pytest

from src.evaluate import (accuracy_curve, apgr, auc, bootstrap_apgr, pct_strong_to_reach,
                          routing_order)

WEAK = np.array([0, 1, 0, 1])
STRONG = np.array([1, 1, 1, 0])
SCORES = np.array([0.9, 0.1, 0.6, 0.3])
PCTS = [0, 25, 50, 75, 100]


def test_routing_order_highest_first_and_stable_ties():
    assert routing_order(SCORES).tolist() == [0, 2, 3, 1]
    assert routing_order([0.5, 0.5, 1.0]).tolist() == [2, 0, 1]


def test_accuracy_curve_matches_hand_calculation():
    curve = accuracy_curve(WEAK, STRONG, SCORES, pcts=PCTS)
    assert curve.tolist() == [0.50, 0.75, 1.00, 0.75, 0.75]


def test_curve_endpoints_are_weak_and_strong_accuracy():
    curve = accuracy_curve(WEAK, STRONG, SCORES, pcts=PCTS)
    assert curve[0] == WEAK.mean() and curve[-1] == STRONG.mean()


def test_auc_trapezoid():
    # 0.25 * [(0.50+0.75) + (0.75+1.00) + (1.00+0.75) + (0.75+0.75)] / 2
    # = 0.25 * (0.625 + 0.875 + 0.875 + 0.75) = 0.25 * 3.125 = 0.78125
    curve = np.array([0.50, 0.75, 1.00, 0.75, 0.75])
    assert auc(curve, pcts=PCTS) == pytest.approx(0.78125)


def test_auc_of_flat_line_is_its_height():
    assert auc(np.full(21, 0.6)) == pytest.approx(0.6)


def test_apgr():
    # (0.78125 - 0.5) / (0.75 - 0.5) = 1.125  (> 1 is possible: routing can beat
    # strong-only when the weak model is right where the strong one is wrong)
    assert apgr(0.78125, 0.5, 0.75) == pytest.approx(1.125)
    # random mixing traces the straight line weak->strong: area = midpoint -> APGR 0.5
    assert apgr(0.625, 0.5, 0.75) == pytest.approx(0.5)
    assert np.isnan(apgr(0.7, 0.6, 0.6))  # no gap -> undefined


def test_pct_strong_to_reach():
    # 90% of strong acc = 0.675 -> first reached with 1 of 4 questions = 25%
    assert pct_strong_to_reach(WEAK, STRONG, SCORES, 0.90) == 25.0
    # 100% of strong acc = 0.75 -> also first reached at 25%
    assert pct_strong_to_reach(WEAK, STRONG, SCORES, 1.00) == 25.0
    # unreachable target -> NaN
    assert np.isnan(pct_strong_to_reach(WEAK, STRONG, SCORES, 1.5))


def test_bootstrap_ci_is_ordered_and_reproducible():
    rng = np.random.default_rng(0)
    w = rng.random(200) < 0.4
    s = rng.random(200) < 0.7
    sc = rng.random(200)
    lo1, hi1 = bootstrap_apgr(w, s, sc, n_boot=200, seed=1)
    lo2, hi2 = bootstrap_apgr(w, s, sc, n_boot=200, seed=1)
    assert lo1 <= hi1 and (lo1, hi1) == (lo2, hi2)
