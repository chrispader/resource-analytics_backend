"""Structural quality metrics for resource×role matrices."""

import numpy as np


def density(values: np.ndarray) -> float:
    """Share of non-empty cells; invariant under row/column reordering."""
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(values) / values.size)


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(a_bool, b_bool).sum()
    return float(intersection / union)


def neighbor_similarity_coherence(values: np.ndarray) -> float:
    """Mean Jaccard similarity between adjacent rows or columns."""
    if values.shape[0] < 2:
        return 1.0
    similarities = [
        jaccard_similarity(values[index], values[index + 1])
        for index in range(values.shape[0] - 1)
    ]
    return float(np.mean(similarities))


def count_runs(row: np.ndarray) -> int:
    runs = 0
    inside_run = False
    for value in row:
        is_filled = value != 0
        if is_filled and not inside_run:
            runs += 1
            inside_run = True
            continue
        if not is_filled:
            inside_run = False
    return runs


def average_fragmentation(values: np.ndarray) -> float:
    """Mean number of separate filled-cell runs per row (or column when transposed)."""
    if values.shape[0] == 0:
        return 0.0
    return float(np.mean([count_runs(row) for row in values]))
