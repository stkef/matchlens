"""Evaluation metrics, exactly as defined in docs/PRD.md.

All functions take self-free candidates (see retrievers.base.drop_self) plus the label_group of every
pool row. The queries are the first n pool rows (n = number of candidate rows); any rows after them are
distractors that can be retrieved but are never queried. "True matches" of a query are the other rows
in its group; self always counts as predicted and as truth for F1, matching the Shopee competition metric.
"""

from __future__ import annotations

import numpy as np

from matchlens.retrievers.base import Candidates


def match_matrix(cands: Candidates, labels: np.ndarray) -> np.ndarray:
    """(n, k) bool: candidate j of query i is a true match."""
    valid = cands.indices >= 0
    cand_labels = labels[np.where(valid, cands.indices, 0)]
    return valid & (cand_labels == labels[:len(cands.indices), None])


def group_sizes(labels: np.ndarray, n_queries: int | None = None) -> np.ndarray:
    """Size of each query row's group within the whole pool (including itself)."""
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return counts[inverse][:n_queries]


def recall_at_k(cands: Candidates, labels: np.ndarray, k: int) -> float:
    hits = match_matrix(cands, labels)[:, :k].sum(axis=1)
    others = group_sizes(labels, len(hits)) - 1
    has_match = others > 0
    if not has_match.any():
        return float("nan")
    return float(np.mean(hits[has_match] / np.minimum(k, others[has_match])))


def mrr(cands: Candidates, labels: np.ndarray) -> float:
    is_match = match_matrix(cands, labels)
    has_match = group_sizes(labels, len(is_match)) > 1
    first = np.argmax(is_match, axis=1)
    rr = np.where(is_match.any(axis=1), 1.0 / (first + 1), 0.0)
    return float(rr[has_match].mean()) if has_match.any() else float("nan")


def f1_per_listing(cands: Candidates, labels: np.ndarray, threshold: float, min_matches: int = 0) -> np.ndarray:
    """Competition F1 for each query: predicted = self + candidates scoring >= threshold."""
    return f1_curve(cands, labels, np.array([threshold]), min_matches)[0]


def f1_curve(cands: Candidates, labels: np.ndarray, thresholds: np.ndarray, min_matches: int = 0) -> np.ndarray:
    """(n_thresholds, n_queries) per-listing F1 for every threshold, vectorised.

    min_matches: always predict at least this many top candidates, even below the threshold. Every
    Shopee product has at least two listings, so every listing has at least one other match.
    """
    is_match = match_matrix(cands, labels)
    truth = group_sizes(labels, len(is_match))
    out = np.empty((len(thresholds), len(is_match)))
    for t_i, t in enumerate(thresholds):
        selected = (cands.indices >= 0) & (cands.scores >= t)
        if min_matches:
            selected[:, :min_matches] |= cands.indices[:, :min_matches] >= 0
        tp = 1 + (selected & is_match).sum(axis=1)
        pred = 1 + selected.sum(axis=1)
        out[t_i] = 2 * tp / (pred + truth)
    return out


def precision_at_recall(cands: Candidates, labels: np.ndarray, target_recall: float = 0.9) -> float:
    """Pairwise precision at the loosest threshold that reaches target pairwise recall.

    Total positives counts every true pair, retrieved or not, so a retriever that never surfaces 90% of
    pairs gets NaN instead of a flattering number.
    """
    is_match = match_matrix(cands, labels)
    valid = cands.indices >= 0
    scores, hits = cands.scores[valid], is_match[valid]
    total_pos = int((group_sizes(labels, len(is_match)) - 1).sum())
    if total_pos == 0 or len(scores) == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(hits[order])
    recall = tp / total_pos
    precision = tp / np.arange(1, len(tp) + 1)
    reached = recall >= target_recall
    return float(precision[reached].max()) if reached.any() else float("nan")
