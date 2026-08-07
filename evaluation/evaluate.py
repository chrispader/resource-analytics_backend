"""Combined evaluation for resource×role matrix orderings."""

from dataclasses import asdict, dataclass, field

import numpy as np

from evaluation.color_metrics import (
    ColorDiscriminabilityResult,
    evaluate_color_discriminability,
)
from evaluation.matrix_model import ResourceRoleMatrix
from evaluation.metrics import (
    average_fragmentation,
    degree_order_agreement,
    density,
    neighbor_similarity_coherence,
)

DEFAULT_BACKGROUND_COLOR = "#ffffff"
DEFAULT_EMPTY_CELL_COLOR = "#ededed"

ORDERING_VARIANT_ROW_DEGREE = "row_degree"
ORDERING_VARIANT_ALPHABETICAL = "alphabetical"
ORDERING_VARIANT_DEGREE = "degree_based"
ORDERING_VARIANT_SIMILARITY = "similarity_based"
ORDERING_VARIANT_RANDOM = "random"

ENABLED_ORDERING_VARIANTS = (
    ORDERING_VARIANT_ROW_DEGREE,
    ORDERING_VARIANT_ALPHABETICAL,
    ORDERING_VARIANT_DEGREE,
    ORDERING_VARIANT_SIMILARITY,
)


def random_ordering_variant_key(seed: int) -> str:
    return f"{ORDERING_VARIANT_RANDOM}_{seed}"


@dataclass(frozen=True)
class MatrixEvaluationResult:
    variant: str
    seed: int | None
    resource_count: int
    role_count: int
    filled_cells: int
    density: float
    row_coherence: float
    column_coherence: float
    row_fragmentation: float
    column_fragmentation: float
    degree_order_agreement: float

    def to_dict(self) -> dict:
        return asdict(self)

    def ordering_metrics_dict(self) -> dict:
        """Return only fields that can change under row/column permutation."""
        return {
            "variant": self.variant,
            "seed": self.seed,
            "row_coherence": self.row_coherence,
            "column_coherence": self.column_coherence,
            "row_fragmentation": self.row_fragmentation,
            "column_fragmentation": self.column_fragmentation,
            "degree_order_agreement": self.degree_order_agreement,
        }


@dataclass(frozen=True)
class MatrixEvaluationResultsBundle:
    """
    Evaluation results for one dataset across matrix orderings.

    orderings: fixed variants (row_degree, degree_based, similarity_based, …).
    random_baselines: repeated random orderings (one result per seed); empty until enabled.
    """

    color_discriminability: ColorDiscriminabilityResult
    orderings: dict[str, MatrixEvaluationResult] = field(default_factory=dict)
    random_baselines: tuple[MatrixEvaluationResult, ...] = ()

    def to_dict(self) -> dict:
        reference = self.orderings[ORDERING_VARIANT_ROW_DEGREE]
        return {
            "resource_count": reference.resource_count,
            "role_count": reference.role_count,
            "filled_cells": reference.filled_cells,
            "density": reference.density,
            "color_discriminability": self.color_discriminability.to_dict(),
            "orderings": {
                variant: result.ordering_metrics_dict()
                for variant, result in self.orderings.items()
            },
            "random_baselines": [
                result.ordering_metrics_dict() for result in self.random_baselines
            ],
        }


def role_palette(palette: list[str], role_count: int) -> list[str]:
    """Return role colors without cycling a qualitative palette.

    Once there are more roles than distinct palette entries, column position and
    labels identify roles and one shared assignment color represents filled cells.
    """
    if role_count <= 0 or not palette:
        return []
    if role_count <= len(palette):
        return palette[:role_count]
    return [palette[0]] * role_count


def build_ordering_variants(base_matrix: ResourceRoleMatrix) -> dict[str, ResourceRoleMatrix]:
    """Matrices per fixed ordering variant."""
    from evaluation.ordering import (
        alphabetical_ordering,
        degree_based_ordering,
        similarity_based_ordering,
    )

    return {
        ORDERING_VARIANT_ROW_DEGREE: base_matrix,
        ORDERING_VARIANT_ALPHABETICAL: alphabetical_ordering(base_matrix),
        ORDERING_VARIANT_DEGREE: degree_based_ordering(base_matrix),
        ORDERING_VARIANT_SIMILARITY: similarity_based_ordering(base_matrix),
    }


def evaluate_ordering_variants(
    base_matrix: ResourceRoleMatrix,
    palette: list[str],
    *,
    background_color: str = DEFAULT_BACKGROUND_COLOR,
    empty_cell_color: str = DEFAULT_EMPTY_CELL_COLOR,
    include_random_baselines: bool = False,
    random_seed_count: int = 100,
) -> MatrixEvaluationResultsBundle:
    """
    Evaluate all enabled ordering variants. Random baselines are optional (off by default).
    """
    role_colors = role_palette(palette, len(base_matrix.roles))
    distinct_role_colors = list(dict.fromkeys(role_colors))
    color_evaluation = evaluate_color_discriminability(
        distinct_role_colors,
        empty_cell_color,
        background_color,
    )
    ordering_results: dict[str, MatrixEvaluationResult] = {}
    for variant_name, variant_matrix in build_ordering_variants(base_matrix).items():
        if variant_name not in ENABLED_ORDERING_VARIANTS:
            continue
        ordering_results[variant_name] = evaluate_resource_role_matrix(
            variant_matrix,
            variant=variant_name,
        )

    random_results: list[MatrixEvaluationResult] = []
    if include_random_baselines:
        from evaluation.ordering import random_ordering

        for seed in range(random_seed_count):
            random_matrix = random_ordering(base_matrix, seed=seed)
            random_results.append(
                evaluate_resource_role_matrix(
                    random_matrix,
                    variant=random_ordering_variant_key(seed),
                    seed=seed,
                )
            )

    return MatrixEvaluationResultsBundle(
        color_discriminability=color_evaluation,
        orderings=ordering_results,
        random_baselines=tuple(random_results),
    )


def evaluate_resource_role_matrix(
    matrix: ResourceRoleMatrix,
    *,
    variant: str = ORDERING_VARIANT_ROW_DEGREE,
    seed: int | None = None,
) -> MatrixEvaluationResult:
    """
    Evaluate structural metrics for one matrix ordering.

    Higher row/column coherence suggests similar entities are neighbors.
    Lower fragmentation suggests more compact filled-cell runs.
    """
    return MatrixEvaluationResult(
        variant=variant,
        seed=seed,
        resource_count=len(matrix.resources),
        role_count=len(matrix.roles),
        filled_cells=int(np.count_nonzero(matrix.values)),
        density=density(matrix.values),
        row_coherence=neighbor_similarity_coherence(matrix.values),
        column_coherence=neighbor_similarity_coherence(matrix.values.T),
        row_fragmentation=average_fragmentation(matrix.values),
        column_fragmentation=average_fragmentation(matrix.values.T),
        degree_order_agreement=degree_order_agreement(matrix.values),
    )


def evaluate_row_degree_ordering(
    matrix: ResourceRoleMatrix,
) -> MatrixEvaluationResult:
    """Evaluate the row-degree order used by the matrix visualization."""
    return evaluate_resource_role_matrix(
        matrix,
        variant=ORDERING_VARIANT_ROW_DEGREE,
        seed=None,
    )
