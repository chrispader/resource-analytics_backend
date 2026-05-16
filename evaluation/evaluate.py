"""Combined evaluation for resource×role matrix orderings."""

from dataclasses import asdict, dataclass

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
CURRENT_ORDERING_VARIANT = "current"


@dataclass(frozen=True)
class MatrixEvaluationResult:
    variant: str
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


def role_palette(palette: list[str], role_count: int) -> list[str]:
    if role_count <= 0:
        return []
    return palette[:role_count]


def evaluate_resource_role_matrix(
    matrix: ResourceRoleMatrix,
    palette: list[str],
    *,
    variant: str = CURRENT_ORDERING_VARIANT,
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
        variant=CURRENT_ORDERING_VARIANT,
        background_color=background_color,
        empty_cell_color=empty_cell_color,
    )
