"""The statistics' primitives (PREREG.md, Statistics and The audit), in NumPy only.

Which arms, sides and clusters enter a test is `report.py`'s business: these functions take
per-cluster numbers and return p-values, intervals and apportionments. Every random draw comes from
NumPy's PCG64 seeded from `config/seeds.toml`, so each result reproduces exactly.
"""

import math
from collections.abc import Callable, Sequence
from fractions import Fraction

import numpy as np

from bench.common import seed

EXACT_MAX = 16  # up to this many clusters, a sign-flip test enumerates all 2^n patterns
REPLICATES = 10_000  # random sign patterns beyond EXACT_MAX, and bootstrap replicates
Z95 = 1.959963984540054  # the standard normal's 97.5% quantile


def _rng(name: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed(name)))


def _sign_patterns(n: int) -> tuple[np.ndarray, bool]:
    """Sign patterns as rows (+1 keeps a cluster, -1 flips it), and whether they are all 2^n."""
    if n <= EXACT_MAX:
        return 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1), True
    return _rng("sign_flip").integers(0, 2, size=(REPLICATES, n)) * 2 - 1, False


def _p_value(t: np.ndarray, t_obs: float, tol: float, exact: bool) -> float:
    """Share of patterns whose statistic reaches the observed one, ties within `tol` included."""
    count = int(np.count_nonzero(t >= t_obs - tol))
    return count / t.size if exact else (1 + count) / (REPLICATES + 1)


def sign_flip_p(d: np.ndarray) -> float:
    """One-sided sign-flip p-value for "sum(d) > 0", with d_c = E_A − E_B − b·S_c per cluster.

    The caller has dropped clusters with S = 0. The statistic is T = Σ s_c·d_c, and p is the share
    of sign patterns s with T ≥ T_obs, ties included (to within 1e-9·Σ|d_c|, for float error). For
    n ≤ 16 all 2^n patterns are enumerated, so n = 0 gives 1.0. Otherwise p = (1 + count) / 10,001
    over 10,000 patterns drawn as `Generator(PCG64(seed("sign_flip"))).integers(0, 2,
    size=(10_000, n)) * 2 - 1`: row r is pattern r, column c the sign of cluster c in the order
    given, so the caller must pass clusters in a fixed order.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 1:
        raise ValueError("d must hold one value per cluster")
    signs, exact = _sign_patterns(d.size)
    return _p_value(signs @ d, float(d.sum()), 1e-9 * float(np.abs(d).sum()), exact)


def _f1(c: np.ndarray) -> np.ndarray:
    """F1 = 2h / (2h + m + f) over the last axis's (hits, misses, false alarms); 0 when 0/0."""
    num = 2 * c[..., 0]
    den = num + c[..., 1] + c[..., 2]
    return np.divide(num, den, out=np.zeros(den.shape), where=den > 0)


def f1_sign_flip_p(a: np.ndarray, b: np.ndarray) -> float:
    """One-sided sign-flip p-value for "pooled F1 of A > pooled F1 of B".

    `a` and `b` hold each cluster's (hits, misses, false alarms), shape (n_clusters, 3), and every
    cluster is kept. A pattern swaps A's and B's rows in its flipped clusters and recomputes
    F1_A − F1_B over the pooled sums; p is the share of patterns whose difference reaches the
    observed one, ties included (to within 1e-9). The patterns are `sign_flip_p`'s: all 2^n for
    n ≤ 16, otherwise the same 10,000 seeded draws, with p = (1 + count) / 10,001.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("a and b must both have shape (n_clusters, 3)")
    signs, exact = _sign_patterns(len(a))
    moved = (signs < 0).astype(float) @ (b - a)  # what each pattern's swaps add to A's pool
    a_sum, b_sum = a.sum(axis=0), b.sum(axis=0)
    t = _f1(a_sum + moved) - _f1(b_sum - moved)
    return _p_value(t, float(_f1(a_sum) - _f1(b_sum)), 1e-9, exact)


def relabel_p(x: np.ndarray) -> float:
    """Two-sided p-value that one run's counts differ from its page's other runs', exactly, by
    relabeling (the canary; the effort ablation).

    `x` holds integer counts, a row per page and a column per run, the run tested last. With no
    difference a page's runs are exchangeable, so each of the runs^n ways of taking one of each
    page's runs as the tested one is equally likely. The statistic, the tested run's total over the
    pages, orders them as the sum of its count minus the others' mean does, and its distribution
    is the convolution of the pages' (each run's count with probability 1/runs). p is twice the
    smaller share at least, or at most, the observed total, ties included, capped at 1; no pages
    give 1.0.
    """
    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("x must hold one row per page and a column per run, at least two")
    pmf = np.ones(1)
    for row in x:
        pmf = np.convolve(pmf, np.bincount(row - row.min()) / row.size)
    t = int(x[:, -1].sum() - x.min(axis=1).sum())
    return min(1.0, 2 * min(float(pmf[t:].sum()), float(pmf[: t + 1].sum())))


def _draw_counts(name: str, n: int, n_rep: int) -> np.ndarray:
    """How often each of n items is drawn in each of n_rep resamples with replacement, (n_rep, n).

    Draws `Generator(PCG64(seed(name))).integers(0, n, size=(n_rep, n))`, row r resample r.
    """
    idx = _rng(name).integers(0, n, size=(n_rep, n))
    flat = (np.arange(n_rep)[:, None] * n + idx).ravel()
    return np.bincount(flat, minlength=n_rep * n).reshape(n_rep, n)


def bootstrap_ci(
    counts: np.ndarray,
    stat: Callable[[np.ndarray], np.ndarray | float],
    n_rep: int = REPLICATES,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Paired cluster bootstrap percentile interval (PREREG.md, Statistics).

    `counts` holds each cluster's summed counts, shape (n_clusters, k); both sides' counts as
    columns of one array is what pairs them. Each replicate draws n_clusters rows with replacement,
    `Generator(PCG64(seed("bootstrap"))).integers(0, n, size=(n_rep, n))`, and pools them by
    summing. `stat` must be vectorized, mapping pooled sums (n_rep, k) to (n_rep,), e.g.
    `lambda s: s[..., 0] / s[..., 1]`. Returns the alpha/2 and 1 − alpha/2 percentiles (NumPy's
    default linear method) over the replicates where `stat` is finite: one giving nan (0/0) or an
    infinity is dropped. No clusters, or no defined replicate, gives (nan, nan).
    """
    counts = np.asarray(counts, dtype=float)
    if len(counts) == 0:
        return math.nan, math.nan
    pooled = _draw_counts("bootstrap", len(counts), n_rep) @ counts  # = summing the drawn rows
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.asarray(stat(pooled), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order (PREREG.md, the decision rule).

    The i-th smallest of m p-values (i from 1) is multiplied by m − i + 1, then a running maximum
    makes the sequence monotone, capped at 1.
    """
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    out = np.empty(p.size)
    out[order] = np.minimum(1.0, np.maximum.accumulate((p.size - np.arange(p.size)) * p[order]))
    return out.tolist()


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials; (0.0, 1.0) when n is 0."""
    if n == 0:
        return 0.0, 1.0
    centre = (k + z * z / 2) / (n + z * z)
    half = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return max(0.0, centre - half), min(1.0, centre + half)


def _codes(a: Sequence[str], b: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """The two raters' labels as integer codes over the union of their categories."""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("the raters must label the same, non-empty list of items")
    _, codes = np.unique(np.asarray([*a, *b]), return_inverse=True)
    return codes[: len(a)], codes[len(a) :]


def _kappa(w: np.ndarray, ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """κ for each row of item weights w (n_rows, n_items): a resample's draw counts, or ones."""
    onehot = np.eye(max(ca.max(), cb.max()) + 1)
    total = w.sum(axis=1)
    p_o = w @ (ca == cb).astype(float) / total
    p_e = ((w @ onehot[ca]) * (w @ onehot[cb])).sum(axis=1) / total**2
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (p_o - p_e) / (1 - p_e)
    return np.where(p_e == 1, (p_o == 1).astype(float), kappa)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's κ for two raters' categorical labels of the same items (PREREG.md, The audit).

    κ = (p_o − p_e) / (1 − p_e). When p_e = 1 (both raters gave every item one and the same label)
    κ is 0/0 and is taken as 1.0 if observed agreement is perfect, else 0.0.
    """
    ca, cb = _codes(a, b)
    return float(_kappa(np.ones((1, len(ca))), ca, cb)[0])


def kappa_ci(a: Sequence[str], b: Sequence[str], n_rep: int = REPLICATES) -> tuple[float, float]:
    """95% percentile interval for Cohen's κ, from an item-level bootstrap.

    Items are resampled with replacement, `Generator(PCG64(seed("kappa"))).integers(0, n,
    size=(n_rep, n))`, and κ recomputed per resample under `cohen_kappa`'s rules.
    """
    ca, cb = _codes(a, b)
    lo, hi = np.percentile(_kappa(_draw_counts("kappa", len(ca), n_rep), ca, cb), [2.5, 97.5])
    return float(lo), float(hi)


def largest_remainder(weights: Sequence[float], total: int) -> list[int]:
    """Apportion `total` in proportion to `weights` by largest remainder, ties to the lower index.

    Quotas are computed exactly (as fractions), so equal remainders tie exactly.
    """
    w = [Fraction(x) for x in weights]
    s = sum(w)
    if total < 0 or any(x < 0 for x in w) or (total > 0 and s == 0):
        raise ValueError("need total >= 0 and non-negative weights with a positive sum")
    if total == 0:
        return [0] * len(w)
    quotas = [total * x / s for x in w]
    seats = [math.floor(q) for q in quotas]
    by_remainder = sorted(range(len(w)), key=lambda i: (seats[i] - quotas[i], i))
    for i in by_remainder[: total - sum(seats)]:
        seats[i] += 1
    return seats
