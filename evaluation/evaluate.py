"""Combined evaluation for resource×role matrix orderings."""

from dataclasses import asdict, dataclass, field

import numpy as np

from evaluation.color_metrics import (
    contrast_ratio,
    delta_e_2000,
    palette_discriminability,
)
from evaluation.matrix_model import ResourceRoleMatrix
from evaluation.metrics import (
    average_fragmentation,
    density,
    neighbor_similarity_coherence,
)

DEFAULT_BACKGROUND_COLOR = "#ffffff"
DEFAULT_EMPTY_CELL_COLOR = "#ededed"

ORDERING_VARIANT_CURRENT = "current"
ORDERING_VARIANT_ALPHABETICAL = "alphabetical"
ORDERING_VARIANT_DEGREE = "degree_based"
ORDERING_VARIANT_SIMILARITY = "similarity_based"
ORDERING_VARIANT_RANDOM = "random"

ENABLED_ORDERING_VARIANTS = (
    ORDERING_VARIANT_CURRENT,
    ORDERING_VARIANT_ALPHABETICAL,
    ORDERING_VARIANT_DEGREE,
    ORDERING_VARIANT_SIMILARITY,
)


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
    min_delta_e: float
    mean_delta_e: float
    max_delta_e: float
    min_contrast_ratio: float
    empty_cell_contrast_ratio: float
    empty_cell_delta_e: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MatrixEvaluationResultsBundle:
    """
    Evaluation results for one dataset across matrix orderings.

    orderings: fixed variants (current, degree_based, similarity_based, …).
    random_baselines: repeated random orderings (one result per seed); empty until enabled.
    """

    orderings: dict[str, MatrixEvaluationResult] = field(default_factory=dict)
    random_baselines: tuple[MatrixEvaluationResult, ...] = ()

    def to_dict(self) -> dict:
        return {
            "orderings": {
                variant: result.to_dict()
                for variant, result in self.orderings.items()
            },
            "random_baselines": [
                result.to_dict() for result in self.random_baselines
            ],
        }


def role_palette(palette: list[str], role_count: int) -> list[str]:
    if role_count <= 0:
        return []
    return palette[:role_count]


def build_ordering_variants(base_matrix: ResourceRoleMatrix) -> dict[str, ResourceRoleMatrix]:
    """Matrices per fixed ordering variant."""
    from evaluation.ordering import (
        alphabetical_ordering,
        degree_based_ordering,
        similarity_based_ordering,
    )

    return {
        ORDERING_VARIANT_CURRENT: base_matrix,
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
    ordering_results: dict[str, MatrixEvaluationResult] = {}
    for variant_name, variant_matrix in build_ordering_variants(base_matrix).items():
        if variant_name not in ENABLED_ORDERING_VARIANTS:
            continue
        ordering_results[variant_name] = evaluate_resource_role_matrix(
            variant_matrix,
            palette,
            variant=variant_name,
            background_color=background_color,
            empty_cell_color=empty_cell_color,
        )

    random_results: list[MatrixEvaluationResult] = []
    if include_random_baselines:
        from evaluation.ordering import random_ordering

        for seed in range(random_seed_count):
            random_matrix = random_ordering(base_matrix, seed=seed)
            random_results.append(
                evaluate_resource_role_matrix(
                    random_matrix,
                    palette,
                    variant=ORDERING_VARIANT_RANDOM,
                    seed=seed,
                    background_color=background_color,
                    empty_cell_color=empty_cell_color,
                )
            )

    return MatrixEvaluationResultsBundle(
        orderings=ordering_results,
        random_baselines=tuple(random_results),
    )


def evaluate_resource_role_matrix(
    matrix: ResourceRoleMatrix,
    palette: list[str],
    *,
    variant: str = ORDERING_VARIANT_CURRENT,
    seed: int | None = None,
    background_color: str = DEFAULT_BACKGROUND_COLOR,
    empty_cell_color: str = DEFAULT_EMPTY_CELL_COLOR,
) -> MatrixEvaluationResult:
    """
    Evaluate structural and color metrics for one matrix ordering.

    Higher row/column coherence suggests similar entities are neighbors.
    Lower fragmentation suggests more compact filled-cell runs.
    Color metrics are proxies for perceptual separability, not user performance.
    """
    role_colors = role_palette(palette, len(matrix.roles))
    color_distances = palette_discriminability(role_colors)

    contrast_values = [
        contrast_ratio(color, background_color) for color in role_colors
    ]
    contrast_values.append(contrast_ratio(empty_cell_color, background_color))

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
        min_delta_e=color_distances["min_delta_e"],
        mean_delta_e=color_distances["mean_delta_e"],
        max_delta_e=color_distances["max_delta_e"],
        min_contrast_ratio=float(np.min(contrast_values)) if contrast_values else 0.0,
        empty_cell_contrast_ratio=contrast_ratio(empty_cell_color, background_color),
        empty_cell_delta_e=delta_e_2000(empty_cell_color, background_color),
    )


def evaluate_current_ordering(
    matrix: ResourceRoleMatrix,
    palette: list[str],
    *,
    background_color: str = DEFAULT_BACKGROUND_COLOR,
    empty_cell_color: str = DEFAULT_EMPTY_CELL_COLOR,
) -> MatrixEvaluationResult:
    """Evaluate the visualization's current resource/role order."""
    return evaluate_resource_role_matrix(
        matrix,
        palette,
        variant=ORDERING_VARIANT_CURRENT,
        seed=None,
        background_color=background_color,
        empty_cell_color=empty_cell_color,
    )
