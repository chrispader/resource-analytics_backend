import numpy as np
import pytest

from evaluation.color_metrics import contrast_ratio, delta_e_2000, palette_discriminability
from evaluation.evaluate import evaluate_current_ordering
from evaluation.matrix_model import ResourceRoleMatrix, resource_role_matrix_from_mapping
from evaluation.metrics import (
    average_fragmentation,
    count_runs,
    density,
    jaccard_similarity,
    neighbor_similarity_coherence,
)
from evaluation.ordering import degree_based_ordering, random_ordering, reorder_matrix


def test_density_empty_matrix():
    values = np.zeros((0, 0), dtype=int)
    assert density(values) == 0.0


def test_density_all_empty():
    values = np.zeros((2, 3), dtype=int)
    assert density(values) == 0.0


def test_density_all_filled():
    values = np.ones((2, 2), dtype=int)
    assert density(values) == 1.0


def test_density_half_filled():
    values = np.array([[1, 0], [0, 1]], dtype=int)
    assert density(values) == 0.5


def test_jaccard_identical_rows():
    row = np.array([1, 1, 0])
    assert jaccard_similarity(row, row) == 1.0


def test_jaccard_disjoint_rows():
    assert jaccard_similarity(np.array([1, 0]), np.array([0, 1])) == 0.0


def test_jaccard_partial_overlap():
    assert jaccard_similarity(np.array([1, 1, 0]), np.array([1, 0, 0])) == pytest.approx(1 / 2)


def test_jaccard_two_empty_rows():
    assert jaccard_similarity(np.array([0, 0]), np.array([0, 0])) == 1.0


def test_fragmentation_cases():
    assert count_runs(np.array([0, 0, 0])) == 0
    assert count_runs(np.array([1, 1, 1])) == 1
    assert count_runs(np.array([1, 0, 1])) == 2
    assert count_runs(np.array([0, 1, 1, 0, 1])) == 2


def test_average_fragmentation_empty():
    assert average_fragmentation(np.zeros((0, 3), dtype=int)) == 0.0


def test_reorder_preserves_shape_and_fill_count():
    matrix = resource_role_matrix_from_mapping(
        resources=["r1", "r2"],
        roles=["a", "b"],
        mapping=[[True, False], [False, True]],
    )
    reordered = reorder_matrix(matrix, row_order=[1, 0], column_order=[1, 0])
    assert reordered.values.shape == matrix.values.shape
    assert reordered.values.sum() == matrix.values.sum()


def test_random_ordering_deterministic_for_seed():
    matrix = resource_role_matrix_from_mapping(
        resources=["r1", "r2", "r3"],
        roles=["a", "b"],
        mapping=[
            [True, False],
            [False, True],
            [True, True],
        ],
    )
    first = random_ordering(matrix, seed=42)
    second = random_ordering(matrix, seed=42)
    assert first.resources == second.resources
    assert first.roles == second.roles
    np.testing.assert_array_equal(first.values, second.values)


def test_matrix_model_validation():
    with pytest.raises(ValueError):
        ResourceRoleMatrix(
            resources=["r1"],
            roles=["a", "b"],
            values=np.zeros((1, 1), dtype=int),
        )


def test_contrast_ratio_same_color():
    assert contrast_ratio("#ffffff", "#ffffff") == pytest.approx(1.0)


def test_contrast_ratio_black_white():
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, rel=0.01)


def test_delta_e_same_color():
    assert delta_e_2000("#ffffff", "#ffffff") == pytest.approx(0.0, abs=1e-6)


def test_palette_discriminability_single_color():
    result = palette_discriminability(["#ff0000"])
    assert result["min_delta_e"] == 0.0
    assert result["mean_delta_e"] == 0.0


def test_evaluate_current_ordering_smoke():
    matrix = resource_role_matrix_from_mapping(
        resources=["Alice", "Bob"],
        roles=["Buyer", "Approver"],
        mapping=[[True, False], [False, True]],
    )
    result = evaluate_current_ordering(
        matrix,
        palette=["#E68A75", "#75E68A"],
    )
    assert result.variant == "current"
    assert result.resource_count == 2
    assert result.role_count == 2
    assert result.filled_cells == 2
    assert result.density == 0.5
    assert 0.0 <= result.row_coherence <= 1.0
