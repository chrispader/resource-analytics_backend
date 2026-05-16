"""Row/column reordering variants for matrix comparison (future use)."""

import numpy as np

from evaluation.matrix_model import ResourceRoleMatrix
from evaluation.metrics import jaccard_similarity


def reorder_matrix(
    matrix: ResourceRoleMatrix,
    row_order: list[int],
    column_order: list[int],
) -> ResourceRoleMatrix:
    return ResourceRoleMatrix(
        resources=[matrix.resources[index] for index in row_order],
        roles=[matrix.roles[index] for index in column_order],
        values=matrix.values[np.ix_(row_order, column_order)],
    )


def degree_based_ordering(matrix: ResourceRoleMatrix) -> ResourceRoleMatrix:
    row_degrees = matrix.values.sum(axis=1)
    column_degrees = matrix.values.sum(axis=0)
    row_order = list(np.argsort(-row_degrees))
    column_order = list(np.argsort(-column_degrees))
    return reorder_matrix(matrix, row_order, column_order)


def greedy_similarity_order(values: np.ndarray) -> list[int]:
    item_count = values.shape[0]
    if item_count == 0:
        return []
    remaining = set(range(item_count))
    order = [0]
    remaining.remove(0)
    while remaining:
        current = order[-1]
        next_index = max(
            remaining,
            key=lambda candidate: jaccard_similarity(values[current], values[candidate]),
        )
        order.append(next_index)
        remaining.remove(next_index)
    return order


def similarity_based_ordering(matrix: ResourceRoleMatrix) -> ResourceRoleMatrix:
    row_order = greedy_similarity_order(matrix.values)
    column_order = greedy_similarity_order(matrix.values.T)
    return reorder_matrix(matrix, row_order, column_order)


def random_ordering(matrix: ResourceRoleMatrix, seed: int) -> ResourceRoleMatrix:
    rng = np.random.default_rng(seed)
    row_order = list(rng.permutation(len(matrix.resources)))
    column_order = list(rng.permutation(len(matrix.roles)))
    return reorder_matrix(matrix, row_order, column_order)
