import json

import numpy as np
import pandas as pd
import pytest

from evaluation.color_metrics import (
    color_discriminability_score,
    contrast_ratio,
    delta_e_2000,
    evaluate_color_discriminability,
    palette_discriminability,
)
from evaluation.evaluate import (
    ORDERING_VARIANT_ALPHABETICAL,
    ORDERING_VARIANT_DEGREE,
    ORDERING_VARIANT_ROW_DEGREE,
    ORDERING_VARIANT_SIMILARITY,
    evaluate_ordering_variants,
    evaluate_row_degree_ordering,
    random_ordering_variant_key,
    role_palette,
)
from evaluation.matrix_model import ResourceRoleMatrix, resource_role_matrix_from_mapping
from evaluation.metrics import (
    average_fragmentation,
    blockiness,
    count_runs,
    density,
    jaccard_similarity,
    neighbor_similarity_coherence,
)
from evaluation.ordering import alphabetical_ordering, random_ordering, reorder_matrix
from pm import (
    PlotlyHeuristicFindingModel,
    PlotlyHeuristicReportModel,
    PlotlyHeuristicRuleLabels,
    ResourceRoleMatrixColorDiscriminability,
    ResourceRoleMatrixData,
    ResourceRoleMatrixEvaluationModel,
    ResourceRoleMatrixEvaluations,
    ResourceRoleMatrixMetricBound,
    ResourceRoleMatrixQualityEvaluation,
    resource_role_matrix,
    resource_role_matrix_evaluation,
)


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


def test_blockiness_rewards_coherent_blocks():
    coherent_blocks = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ]
    )
    checkerboard = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ]
    )
    assert blockiness(coherent_blocks) > blockiness(checkerboard)
    assert blockiness(np.ones((2, 2), dtype=int)) == 1.0


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


def test_alphabetical_ordering_sorts_resources_and_roles():
    matrix = resource_role_matrix_from_mapping(
        resources=["bob", "Alice"],
        roles=["Reviewer", "approver"],
        mapping=[
            [True, False],
            [False, True],
        ],
    )
    ordered = alphabetical_ordering(matrix)
    assert ordered.resources == ["Alice", "bob"]
    assert ordered.roles == ["approver", "Reviewer"]
    np.testing.assert_array_equal(
        ordered.values,
        np.array(
            [
                [True, False],
                [False, True],
            ],
            dtype=np.int8,
        ),
    )


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


def test_color_discriminability_score_is_normalized():
    score = color_discriminability_score(
        ["#ff0000", "#00ff00"],
        empty_cell_color="#000000",
        background_color="#ffffff",
    )
    assert 0.0 <= score <= 1.0


def test_color_discriminability_score_rejects_duplicate_role_colors():
    assert color_discriminability_score(
        ["#ff0000", "#ff0000"],
        empty_cell_color="#000000",
        background_color="#ffffff",
    ) == pytest.approx(0.0)


def test_color_discriminability_score_rewards_separation_from_background():
    well_separated = color_discriminability_score(
        ["#ff0000"],
        empty_cell_color="#000000",
        background_color="#ffffff",
    )
    barely_separated = color_discriminability_score(
        ["#ff0000"],
        empty_cell_color="#fefefe",
        background_color="#ffffff",
    )
    assert well_separated > barely_separated


def test_evaluate_color_discriminability_reports_color_distance_and_contrast():
    role_colors = ["#000000", "#ff0000"]
    empty_cell_color = "#777777"
    background_color = "#ffffff"

    result = evaluate_color_discriminability(
        role_colors,
        empty_cell_color,
        background_color,
    )

    expected_distances = palette_discriminability(
        [*role_colors, empty_cell_color, background_color]
    )
    expected_contrasts = [
        contrast_ratio(color, background_color)
        for color in [*role_colors, empty_cell_color]
    ]
    assert result.score == pytest.approx(
        color_discriminability_score(
            role_colors,
            empty_cell_color,
            background_color,
        )
    )
    assert result.min_delta_e == pytest.approx(
        expected_distances["min_delta_e"]
    )
    assert result.mean_delta_e == pytest.approx(
        expected_distances["mean_delta_e"]
    )
    assert result.max_delta_e == pytest.approx(
        expected_distances["max_delta_e"]
    )
    assert result.min_contrast_ratio == pytest.approx(min(expected_contrasts))
    assert result.mean_contrast_ratio == pytest.approx(np.mean(expected_contrasts))
    assert result.max_contrast_ratio == pytest.approx(max(expected_contrasts))


def test_role_palette_matches_visualization_color_cycling():
    assert role_palette(["#111111", "#222222"], 5) == [
        "#111111",
        "#222222",
        "#111111",
        "#222222",
        "#111111",
    ]


def test_evaluate_row_degree_ordering_smoke():
    matrix = resource_role_matrix_from_mapping(
        resources=["Alice", "Bob"],
        roles=["Buyer", "Approver"],
        mapping=[[True, False], [False, True]],
    )
    result = evaluate_row_degree_ordering(
        matrix,
    )
    assert result.variant == ORDERING_VARIANT_ROW_DEGREE
    assert result.resource_count == 2
    assert result.role_count == 2
    assert result.filled_cells == 2
    assert result.density == 0.5
    assert 0.0 <= result.blockiness <= 1.0
    assert 0.0 <= result.row_coherence <= 1.0


def test_evaluate_ordering_variants_bundle_shape(monkeypatch):
    matrix = resource_role_matrix_from_mapping(
        resources=["Alice", "Bob"],
        roles=["Buyer", "Approver"],
        mapping=[[True, False], [False, True]],
    )
    color_evaluation_calls = 0

    def track_color_evaluation(
        role_colors: list[str],
        empty_cell_color: str,
        background_color: str,
    ):
        nonlocal color_evaluation_calls
        color_evaluation_calls += 1
        return evaluate_color_discriminability(
            role_colors,
            empty_cell_color,
            background_color,
        )

    monkeypatch.setattr(
        "evaluation.evaluate.evaluate_color_discriminability",
        track_color_evaluation,
    )
    bundle = evaluate_ordering_variants(matrix, palette=["#E68A75", "#75E68A"])
    assert color_evaluation_calls == 1
    assert set(bundle.orderings) == {
        ORDERING_VARIANT_ROW_DEGREE,
        ORDERING_VARIANT_ALPHABETICAL,
        ORDERING_VARIANT_DEGREE,
        ORDERING_VARIANT_SIMILARITY,
    }
    assert (
        bundle.orderings[ORDERING_VARIANT_ROW_DEGREE].variant
        == ORDERING_VARIANT_ROW_DEGREE
    )
    assert bundle.random_baselines == ()
    payload = bundle.to_dict()
    assert payload["resource_count"] == 2
    assert payload["role_count"] == 2
    assert payload["filled_cells"] == 2
    assert payload["density"] == 0.5
    assert 0.0 <= payload["color_discriminability"]["score"] <= 1.0
    assert payload["color_discriminability"]["min_delta_e"] >= 0.0
    assert payload["color_discriminability"]["mean_delta_e"] >= 0.0
    assert payload["color_discriminability"]["max_delta_e"] >= 0.0
    assert payload["color_discriminability"]["min_contrast_ratio"] >= 1.0
    assert payload["color_discriminability"]["mean_contrast_ratio"] >= 1.0
    assert payload["color_discriminability"]["max_contrast_ratio"] >= 1.0
    assert "orderings" in payload
    assert "random_baselines" in payload
    assert ORDERING_VARIANT_ROW_DEGREE in payload["orderings"]
    assert "resource_count" not in payload["orderings"][ORDERING_VARIANT_ROW_DEGREE]
    assert "density" not in payload["orderings"][ORDERING_VARIANT_ROW_DEGREE]
    assert (
        "color_discriminability"
        not in payload["orderings"][ORDERING_VARIANT_ROW_DEGREE]
    )
    assert "min_delta_e" not in payload["orderings"][ORDERING_VARIANT_ROW_DEGREE]


def test_evaluate_ordering_variants_random_baseline_count_and_seeds():
    matrix = resource_role_matrix_from_mapping(
        resources=["Alice", "Bob", "Charlie"],
        roles=["Buyer", "Approver"],
        mapping=[
            [True, False],
            [False, True],
            [True, True],
        ],
    )
    bundle = evaluate_ordering_variants(
        matrix,
        palette=["#E68A75", "#75E68A"],
        include_random_baselines=True,
        random_seed_count=100,
    )
    assert len(bundle.random_baselines) == 100
    assert [result.seed for result in bundle.random_baselines] == list(range(100))
    assert [result.variant for result in bundle.random_baselines] == [
        random_ordering_variant_key(seed) for seed in range(100)
    ]


def test_resource_role_matrix_evaluation_response_contract():
    fixed_orderings = {
        variant: ResourceRoleMatrixQualityEvaluation(
            variant=variant,
            seed=None,
            row_coherence=1.0,
            column_coherence=1.0,
            row_fragmentation=1.0,
            column_fragmentation=1.0,
            blockiness=1.0,
        )
        for variant in (
            ORDERING_VARIANT_ROW_DEGREE,
            ORDERING_VARIANT_ALPHABETICAL,
            ORDERING_VARIANT_DEGREE,
            ORDERING_VARIANT_SIMILARITY,
        )
    }
    payload = ResourceRoleMatrixEvaluationModel(
        evaluations=ResourceRoleMatrixEvaluations(
            color_discriminability=ResourceRoleMatrixColorDiscriminability(
                score=0.75,
                min_delta_e=12.0,
                mean_delta_e=30.0,
                max_delta_e=50.0,
                min_contrast_ratio=2.0,
                mean_contrast_ratio=4.0,
                max_contrast_ratio=7.0,
            ),
            orderings=fixed_orderings,
            random_baselines=[
                ResourceRoleMatrixQualityEvaluation(
                    variant=random_ordering_variant_key(seed),
                    seed=seed,
                    row_coherence=0.0,
                    column_coherence=0.0,
                    row_fragmentation=1.0,
                    column_fragmentation=1.0,
                    blockiness=0.0,
                )
                for seed in range(100)
            ],
        ),
        metric_bounds={
            "row_coherence": ResourceRoleMatrixMetricBound(
                lower=0.0, upper=1.0, higher_is_better=True
            ),
            "column_coherence": ResourceRoleMatrixMetricBound(
                lower=0.0, upper=1.0, higher_is_better=True
            ),
            "blockiness": ResourceRoleMatrixMetricBound(
                lower=0.0, upper=1.0, higher_is_better=True
            ),
            "row_fragmentation": ResourceRoleMatrixMetricBound(
                lower=0.0, upper=1.0, higher_is_better=False
            ),
            "column_fragmentation": ResourceRoleMatrixMetricBound(
                lower=0.0, upper=1.0, higher_is_better=False
            ),
        },
        plots={
            variant: {"data": [], "layout": {}}
            for variant in (
                ORDERING_VARIANT_ROW_DEGREE,
                ORDERING_VARIANT_ALPHABETICAL,
                ORDERING_VARIANT_DEGREE,
                ORDERING_VARIANT_SIMILARITY,
                "random_0",
            )
        },
        heuristic_report=PlotlyHeuristicReportModel(
            summary={"warning": 0, "advice": 1, "pass": 1},
            feature_summary={"trace_types": ["heatmap"]},
            findings=[
                PlotlyHeuristicFindingModel(
                    rule_id="plot-title",
                    heuristic="A plot should have a title",
                    assistance="automatic-check",
                    status="pass",
                    message="The plot has a title.",
                    labels=PlotlyHeuristicRuleLabels(labeling=["title"]),
                    evidence={"title": "Resource × Role matrix"},
                    source="https://doi.org/10.1007/978-3-030-90436-4_33",
                )
            ],
        ),
    ).model_dump()

    assert set(payload) == {
        "evaluations",
        "metric_bounds",
        "plots",
        "heuristic_report",
    }
    assert set(payload["plots"]) == {
        ORDERING_VARIANT_ROW_DEGREE,
        ORDERING_VARIANT_ALPHABETICAL,
        ORDERING_VARIANT_DEGREE,
        ORDERING_VARIANT_SIMILARITY,
        "random_0",
    }
    assert payload["metric_bounds"]["row_coherence"] == {
        "lower": 0.0,
        "upper": 1.0,
        "higher_is_better": True,
    }
    assert payload["metric_bounds"]["row_fragmentation"]["higher_is_better"] is False
    assert payload["evaluations"]["color_discriminability"] == {
        "score": 0.75,
        "min_delta_e": 12.0,
        "mean_delta_e": 30.0,
        "max_delta_e": 50.0,
        "min_contrast_ratio": 2.0,
        "mean_contrast_ratio": 4.0,
        "max_contrast_ratio": 7.0,
    }
    assert len(payload["evaluations"]["random_baselines"]) == 100
    assert payload["evaluations"]["random_baselines"][99]["variant"] == "random_99"
    assert payload["evaluations"]["random_baselines"][99]["seed"] == 99

    row_degree_evaluation = payload["evaluations"]["orderings"][
        ORDERING_VARIANT_ROW_DEGREE
    ]
    assert row_degree_evaluation == {
        "variant": ORDERING_VARIANT_ROW_DEGREE,
        "seed": None,
        "row_coherence": 1.0,
        "column_coherence": 1.0,
        "row_fragmentation": 1.0,
        "column_fragmentation": 1.0,
        "blockiness": 1.0,
    }
    assert "resource_count" not in row_degree_evaluation
    assert "density" not in row_degree_evaluation
    assert "min_delta_e" not in row_degree_evaluation
    assert payload["heuristic_report"]["findings"][0]["rule_id"] == "plot-title"


def test_resource_role_matrix_response_includes_all_ordering_plots():
    dataframe = pd.DataFrame(
        {
            "Resource": ["Alice", "Bob"],
            "Role": ["Buyer", "Approver"],
            "Activity": ["Create", "Approve"],
        }
    )

    payload = resource_role_matrix(dataframe).model_dump()

    assert set(payload["plots"]) == {
        ORDERING_VARIANT_ROW_DEGREE,
        ORDERING_VARIANT_ALPHABETICAL,
        ORDERING_VARIANT_DEGREE,
        ORDERING_VARIANT_SIMILARITY,
        "random_0",
    }
    assert json.loads(payload["plot"]) == payload["plots"][ORDERING_VARIANT_ROW_DEGREE]


def test_resource_role_matrix_evaluation_includes_plotly_heuristic_report():
    dataframe = pd.DataFrame(
        {
            "Resource": ["Alice", "Bob"],
            "Role": ["Buyer", "Approver"],
            "Activity": ["Create", "Approve"],
        }
    )

    payload = resource_role_matrix_evaluation(dataframe).model_dump()

    assert set(payload["heuristic_report"]["summary"]) == {
        "warning",
        "advice",
        "pass",
    }
    findings = {
        finding["rule_id"]: finding
        for finding in payload["heuristic_report"]["findings"]
    }
    assert findings["plot-title"]["status"] == "pass"
    assert findings["data-ink-ratio"]["status"] == "advice"
