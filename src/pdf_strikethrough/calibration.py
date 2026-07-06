"""Threshold calibration for the CNN operating point (R-cal).

StrikeNet emits a strike probability in [0, 1]; a deployment turns that into a struck/clean
decision at a threshold. These helpers pick that threshold *from labeled data* instead of the
shipped default, so an operating point can be stated as a guarantee ("catch >= 98% of deletions")
rather than a magic number:

  - :func:`threshold_for_recall` / :func:`threshold_for_precision` — the most extreme threshold on
    a calibration set that still meets a target recall / precision.
  - :func:`conformal_threshold` — a distribution-free (split-conformal) threshold: with the
    finite-sample correction, a genuinely struck word scores at or above it with probability
    >= 1 - alpha, so 1 - alpha is a *guaranteed* recall floor on exchangeable future data.
  - :func:`pr_curve` — precision/recall across all thresholds, for plotting the tradeoff.

Feed the labeled crops that :func:`pdf_strikethrough.active.dump_crops` produces (probabilities
from each row's ``cnn_prob``, labels you filled into ``label``). Wire the chosen threshold in with
``ScanConfig.recall_first(cnn_p_hi=...)`` / ``ScanConfig.precision_first(cnn_p_hi=...)``.
"""
from __future__ import annotations

import numpy as np


def _as_arrays(probs, labels):
    p = np.asarray(probs, dtype=float).reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(bool)
    if p.shape != y.shape:
        raise ValueError(f"probs and labels differ in length: {p.size} vs {y.size}")
    if p.size == 0:
        raise ValueError("need at least one (prob, label) pair to calibrate")
    return p, y


def threshold_for_recall(probs, labels, target_recall):
    """Highest threshold ``t`` such that classifying ``prob >= t`` as struck still recalls at
    least `target_recall` of the truly-struck words in the calibration set. Higher thresholds
    trade recall for precision, so this returns the most precise threshold meeting the recall
    floor. Raises ValueError if there are no struck (positive) examples."""
    p, y = _as_arrays(probs, labels)
    if not y.any():
        raise ValueError("no struck (positive) examples to measure recall against")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    pos = np.sort(p[y])                          # struck-word probabilities, ascending
    n = pos.size
    # recall at threshold t = fraction of positives with prob >= t. At most floor(n*(1-target))
    # positives may fall below t; the highest such t is that positive's probability.
    k = min(int(np.floor(n * (1.0 - target_recall))), n - 1)
    return float(pos[k])


def threshold_for_precision(probs, labels, target_precision):
    """Lowest threshold ``t`` such that classifying ``prob >= t`` as struck yields at least
    `target_precision` precision on the calibration set (maximizing recall subject to the
    precision floor). Raises ValueError if no threshold reaches the target."""
    p, y = _as_arrays(probs, labels)
    if not 0.0 < target_precision <= 1.0:
        raise ValueError("target_precision must be in (0, 1]")
    order = np.argsort(-p)                        # consider words in descending-probability order
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    precision = tp / np.maximum(tp + fp, 1)
    ok = (precision >= target_precision) & (tp > 0)
    if not ok.any():
        raise ValueError(f"no threshold reaches precision {target_precision} on this set")
    return float(ps[np.max(np.flatnonzero(ok))])  # deepest inclusion still ok -> lowest threshold


def conformal_threshold(pos_probs, alpha=0.05):
    """Split-conformal threshold on the struck class. Given the CNN probabilities of a calibration
    set of KNOWN-struck words, return the threshold ``t`` such that a future struck word scores
    ``>= t`` with probability at least ``1 - alpha`` (exchangeability assumed). Uses the
    finite-sample-corrected empirical quantile, so the recall floor holds distribution-free.
    `alpha` is the tolerated miss rate (e.g. 0.05 -> a 95% recall guarantee). Returns 0.0 when the
    calibration set is too small to exclude anything at this alpha (accept everything as struck)."""
    p = np.asarray(pos_probs, dtype=float).reshape(-1)
    if p.size == 0:
        raise ValueError("need at least one struck-word probability")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    k = int(np.floor(alpha * (p.size + 1)))       # order statistic of the conformal quantile
    if k < 1:
        return 0.0
    return float(np.sort(p)[k - 1])


def pr_curve(probs, labels):
    """Return ``(thresholds, precision, recall)`` arrays over the distinct probability thresholds
    (descending), for plotting the CNN's precision/recall tradeoff on a labeled set. Entry ``i`` is
    the metric when classifying ``prob >= thresholds[i]`` as struck. Raises ValueError if there are
    no struck examples (recall undefined)."""
    p, y = _as_arrays(probs, labels)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("no struck (positive) examples; recall is undefined")
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    keep = np.ones(ps.size, dtype=bool)           # one point per distinct threshold (run's last)
    keep[:-1] = ps[1:] != ps[:-1]
    return ps[keep], precision[keep], recall[keep]
