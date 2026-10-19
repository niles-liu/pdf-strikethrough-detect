"""Golden tests for `bench.stats`: hand-worked examples, and the seeded draws PREREG.md fixes."""

import math

import numpy as np
import pytest

from bench.common import seed
from bench.stats import (
    bootstrap_ci,
    cohen_kappa,
    f1_sign_flip_p,
    holm,
    kappa_ci,
    largest_remainder,
    relabel_p,
    sign_flip_p,
    wilson,
)


def ratio(s: np.ndarray) -> np.ndarray:
    return s[..., 0] / s[..., 1]


def test_seeds_match_prereg():
    assert (seed("sign_flip"), seed("bootstrap"), seed("kappa")) == (20261012, 20261017, 20261022)


# sign_flip_p, exact path


def test_sign_flip_only_identity_reaches_observed():
    # T_obs = 6; of the 8 patterns only (+,+,+) sums to 6.
    assert sign_flip_p(np.array([1, 2, 3])) == 1 / 8


def test_sign_flip_ties_count():
    # T_obs = 0; (+,+) = 0, (+,-) = 2, (-,+) = -2, (-,-) = 0, so three of four reach 0.
    assert sign_flip_p(np.array([1, -1])) == 3 / 4


def test_sign_flip_zero_cluster_doubles_ties():
    # T_obs = 2; flipping the 0 changes nothing: (+,±) = 2, (-,±) = -2.
    assert sign_flip_p(np.array([2, 0])) == 1 / 2
    assert sign_flip_p(np.zeros(3)) == 1.0


def test_sign_flip_float_ties():
    # 0.1 + 0.2 - 0.3 is 0 only up to float error; T >= 0 for (+,+,+), (+,+,-), (+,-,-), (-,+,-)
    # and (-,-,-), which ties the observed 0 exactly in reals.
    assert sign_flip_p(np.array([0.1, 0.2, -0.3])) == 5 / 8


def test_sign_flip_no_clusters():
    assert sign_flip_p(np.array([])) == 1.0


def test_sign_flip_exact_up_to_16():
    assert sign_flip_p(np.ones(16)) == 1 / 2**16


# sign_flip_p, random path


def test_sign_flip_random_reproducible():
    d = np.linspace(-1, 2, 20)
    p = sign_flip_p(d)
    assert p == sign_flip_p(d)
    assert 0 < p <= 1


def test_sign_flip_random_matches_documented_draw():
    d = np.random.default_rng(0).normal(0.2, 1, 25)
    rng = np.random.Generator(np.random.PCG64(20261012))
    signs = rng.integers(0, 2, size=(10_000, 25)) * 2 - 1
    count = np.count_nonzero(signs @ d >= d.sum() - 1e-9 * np.abs(d).sum())
    assert sign_flip_p(d) == (1 + count) / 10_001


def test_sign_flip_random_formula():
    # 20 equal positive values: only the all-plus pattern (chance 2^-20 per draw) reaches T_obs.
    assert sign_flip_p(np.ones(20)) == 1 / 10_001
    # All zeros: every draw ties, so p = (1 + 10,000) / 10,001.
    assert sign_flip_p(np.zeros(20)) == 1.0


# f1_sign_flip_p


def test_f1_sign_flip_hand_example():
    # (hits, misses, false alarms): cluster 1 A (3,1,0), B (1,3,0); cluster 2 A (0,2,1), B (2,0,0).
    # (+,+): A (3,3,1) 6/10 = 0.600, B (3,3,0) 6/9 = 0.667, diff -0.067 (observed)
    # (+,-): A (5,1,0) 10/11 = 0.909, B (1,5,1) 2/8 = 0.250, diff +0.659
    # (-,+): A (1,5,1) 0.250, B (5,1,0) 0.909, diff -0.659
    # (-,-): A (3,3,0) 0.667, B (3,3,1) 0.600, diff +0.067
    # Three of four reach -0.067.
    a = np.array([[3, 1, 0], [0, 2, 1]])
    b = np.array([[1, 3, 0], [2, 0, 0]])
    assert f1_sign_flip_p(a, b) == 3 / 4
    # Swapping the sides: only (+,+) and (-,+) reach the observed +0.067.
    assert f1_sign_flip_p(b, a) == 2 / 4


def test_f1_sign_flip_ties_and_empty_pools():
    # Every pattern gives F1 0 against 0 (the 0/0 rule), so all tie.
    assert f1_sign_flip_p(np.zeros((3, 3)), np.zeros((3, 3))) == 1.0
    assert f1_sign_flip_p(np.zeros((0, 3)), np.zeros((0, 3))) == 1.0


def test_f1_sign_flip_random_reproducible():
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 20, size=(20, 3)), rng.integers(0, 20, size=(20, 3))
    p = f1_sign_flip_p(a, b)
    assert p == f1_sign_flip_p(a, b)
    assert 1 / 10_001 <= p <= 1


def test_f1_sign_flip_rejects_bad_shapes():
    with pytest.raises(ValueError):
        f1_sign_flip_p(np.zeros((2, 3)), np.zeros((3, 3)))


# bootstrap_ci


def test_bootstrap_constant_ratio_collapses():
    assert bootstrap_ci(np.array([[1, 2], [2, 4], [3, 6]]), ratio) == (0.5, 0.5)


def test_bootstrap_matches_documented_draw():
    counts = np.array([[3, 10], [0, 4], [7, 9], [1, 12], [5, 5]])
    idx = np.random.Generator(np.random.PCG64(20261017)).integers(0, 5, size=(10_000, 5))
    values = ratio(counts[idx].sum(axis=1))
    lo, hi = bootstrap_ci(counts, ratio)
    assert (lo, hi) == pytest.approx(tuple(np.percentile(values, [2.5, 97.5])), abs=1e-12)
    assert (lo, hi) == bootstrap_ci(counts, ratio)


def test_bootstrap_drops_undefined_replicates():
    # A draw of only the empty cluster is 0/0; every other draw pools to 1/2.
    assert bootstrap_ci(np.array([[1, 2], [0, 0]]), ratio) == (0.5, 0.5)
    assert all(math.isnan(x) for x in bootstrap_ci(np.zeros((2, 2)), ratio))
    assert all(math.isnan(x) for x in bootstrap_ci(np.zeros((0, 2)), ratio))


# holm


def test_holm():
    # Sorted 0.01, 0.03, 0.04 times 3, 2, 1 gives 0.03, 0.06, 0.04; the running maximum lifts 0.04.
    adjusted = holm([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    order = np.argsort([0.01, 0.04, 0.03])
    assert np.all(np.diff(np.array(adjusted)[order]) >= 0)
    assert holm([0.5]) == [0.5]
    assert holm([0.6, 0.7]) == [1.0, 1.0]
    assert holm([]) == []


# wilson


def test_wilson():
    assert wilson(0, 10)[0] == 0.0
    assert wilson(0, 10)[1] == pytest.approx(0.2775, abs=1e-4)
    assert wilson(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert wilson(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)
    assert wilson(0, 0) == (0.0, 1.0)


# cohen_kappa and kappa_ci


def textbook_raters() -> tuple[list[str], list[str]]:
    # 50 items: both yes 20, A yes B no 5, A no B yes 10, both no 15.
    # p_o = 35/50 = 0.7, p_e = 0.5*0.6 + 0.5*0.4 = 0.5, so kappa = 0.4.
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    return a, b


def test_cohen_kappa_textbook():
    a, b = textbook_raters()
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa(b, a) == pytest.approx(0.4)


def test_cohen_kappa_three_categories():
    # Diagonal 2, 1, 1 of 6; marginals A (3, 2, 1)/6, B (2, 2, 2)/6.
    # p_o = 4/6, p_e = (6 + 4 + 2)/36 = 1/3, so kappa = (2/3 - 1/3) / (2/3) = 0.5.
    a = ["struck", "struck", "struck", "not struck", "not struck", "can't tell"]
    b = ["struck", "struck", "not struck", "not struck", "can't tell", "can't tell"]
    assert cohen_kappa(a, b) == pytest.approx(0.5)


def test_cohen_kappa_perfect_and_degenerate():
    labels = ["struck", "not struck", "can't tell", "struck"]
    assert cohen_kappa(labels, labels) == 1.0
    assert cohen_kappa(["struck"] * 4, ["struck"] * 4) == 1.0  # p_e = 1


def test_cohen_kappa_rejects_mismatched_items():
    with pytest.raises(ValueError):
        cohen_kappa(["struck"], ["struck", "not struck"])


def test_kappa_ci():
    a, b = textbook_raters()
    lo, hi = kappa_ci(a, b)
    assert (lo, hi) == kappa_ci(a, b)
    assert -1 <= lo < 0.4 < hi <= 1
    labels = ["struck", "not struck", "can't tell", "struck"]
    assert kappa_ci(labels, labels) == (1.0, 1.0)


# largest_remainder


def test_largest_remainder():
    assert largest_remainder([1, 1, 1], 10) == [4, 3, 3]  # equal remainders: lower index first
    assert largest_remainder([1, 2, 3], 12) == [2, 4, 6]
    assert largest_remainder([5, 3, 2], 7) == [4, 2, 1]  # quotas 3.5, 2.1, 1.4
    assert largest_remainder([0, 1, 1], 3) == [0, 2, 1]
    assert largest_remainder([1, 1], 0) == [0, 0]
    for weights, total in (([0.2, 0.3, 0.5], 7), ([3, 1, 4, 1, 5], 23), ([1] * 7, 160)):
        assert sum(largest_remainder(weights, total)) == total
    with pytest.raises(ValueError):
        largest_remainder([0, 0], 3)


def test_relabel_p_matches_brute_force_enumeration():
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.integers(0, 6, size=(int(rng.integers(1, 5)), 4))
        totals = [sum(x[i, j] for i, j in enumerate(js))
                  for js in np.ndindex(*(4,) * len(x))]  # fmt: skip
        t = x[:, -1].sum()
        upper = sum(v >= t for v in totals) / len(totals)
        lower = sum(v <= t for v in totals) / len(totals)
        assert relabel_p(x) == pytest.approx(min(1.0, 2 * min(upper, lower)))


def test_relabel_p_doesnt_read_a_lumpy_start_as_drift():
    # each start run misses a 3-token block on a third of the pages; the end run is clean. A
    # sign-flip test on end minus the starts' mean reads p = 2/512 here, which is not exact.
    x = np.zeros((9, 4), dtype=int)
    for page in range(9):
        x[page, page // 3] = 3
    assert relabel_p(x) == pytest.approx(2 * 0.75**9)  # 0.150
    assert relabel_p(np.zeros((0, 4), dtype=int)) == 1.0
    with pytest.raises(ValueError):
        relabel_p(np.zeros(4))
