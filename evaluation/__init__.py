"""Structural, perceptual, and Plotly heuristic evaluation."""

from evaluation.evaluate import (
    MatrixEvaluationResult,
    MatrixEvaluationResultsBundle,
    evaluate_current_ordering,
    evaluate_ordering_variants,
    evaluate_resource_role_matrix,
)
from evaluation.matrix_model import ResourceRoleMatrix, resource_role_matrix_from_mapping
from evaluation.plotly_heuristics import (
    HeuristicReport,
    PUBLISHED_HEURISTIC_RULES,
    evaluate_plotly_figure,
    extract_plotly_features,
)

__all__ = [
    "MatrixEvaluationResult",
    "MatrixEvaluationResultsBundle",
    "HeuristicReport",
    "PUBLISHED_HEURISTIC_RULES",
    "ResourceRoleMatrix",
    "evaluate_current_ordering",
    "evaluate_ordering_variants",
    "evaluate_resource_role_matrix",
    "evaluate_plotly_figure",
    "extract_plotly_features",
    "resource_role_matrix_from_mapping",
]
