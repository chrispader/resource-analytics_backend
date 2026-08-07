import csv
import json

import pytest

from evaluation.evaluate import ENABLED_ORDERING_VARIANTS
from evaluation.color_metrics import evaluate_color_discriminability
from evaluation.evaluate import role_palette
from evaluation.plotly_heuristics import PUBLISHED_HEURISTIC_RULES
from experiments.role_matrix.generate import generate_factorial_experiment
from experiments.role_matrix.workflow import evaluate_experiment
from pm import RESOURCE_ROLE_MATRIX_PALETTE


def _rows(path):
    with path.open(encoding="utf-8", newline="") as artifact:
        return list(csv.DictReader(artifact))


def _tiny_experiment(tmp_path):
    config = {
        "resource_sizes": {"small": 12},
        "role_counts": [6],
        "target_densities": [0.5],
        "profile_diversities": ["low", "medium", "high"],
        "seeds": [0],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    dataset_root = tmp_path / "datasets"
    generate_factorial_experiment(dataset_root, config_path)
    return dataset_root / "manifest.csv", config_path


def test_workflow_exports_expected_rows_and_coverage(tmp_path):
    manifest, config = _tiny_experiment(tmp_path)
    artifacts = evaluate_experiment(
        manifest,
        tmp_path / "results",
        random_seed_count=3,
        config_path=config,
    )

    fixed = _rows(artifacts["fixed_ordering_metrics"])
    random = _rows(artifacts["random_baseline_metrics"])
    heuristics = _rows(artifacts["heuristic_findings"])
    comparisons = _rows(artifacts["random_comparisons"])
    condition_aggregates = _rows(artifacts["condition_aggregates"])

    assert len(fixed) == 3 * len(ENABLED_ORDERING_VARIANTS)
    assert len(random) == 3 * 3
    assert len(heuristics) == (
        3 * len(ENABLED_ORDERING_VARIANTS) * len(PUBLISHED_HEURISTIC_RULES)
    )
    assert len(comparisons) == len(fixed) * 6
    assert len(condition_aggregates) == 3 * len(ENABLED_ORDERING_VARIANTS) * 6
    assert {row["variant"] for row in fixed} == set(ENABLED_ORDERING_VARIANTS)
    assert all(0 <= float(row["group_contiguity"]) <= 1 for row in fixed + random)
    assert {row["status"] for row in heuristics} <= {
        "pass",
        "warning",
        "advice",
        "not_applicable",
    }


def test_more_than_five_roles_uses_application_color_fallback(tmp_path):
    manifest, config = _tiny_experiment(tmp_path)
    artifacts = evaluate_experiment(
        manifest,
        tmp_path / "results",
        random_seed_count=1,
        config_path=config,
    )
    summary = _rows(artifacts["dataset_summaries"])[0]
    role_colors = role_palette(RESOURCE_ROLE_MATRIX_PALETTE, 6)
    expected = evaluate_color_discriminability(
        list(dict.fromkeys(role_colors)),
        "#ededed",
        "#ffffff",
    )

    assert float(summary["score"]) == expected.score
    assert float(summary["min_delta_e"]) == expected.min_delta_e


def test_workflow_outputs_are_byte_reproducible(tmp_path):
    manifest, config = _tiny_experiment(tmp_path)
    first = evaluate_experiment(
        manifest,
        tmp_path / "first",
        random_seed_count=2,
        config_path=config,
    )
    second = evaluate_experiment(
        manifest,
        tmp_path / "second",
        random_seed_count=2,
        config_path=config,
    )

    assert first.keys() == second.keys()
    for artifact_name in first:
        assert first[artifact_name].read_bytes() == second[artifact_name].read_bytes()


def test_random_comparison_uses_metric_direction(tmp_path):
    manifest, config = _tiny_experiment(tmp_path)
    artifacts = evaluate_experiment(
        manifest,
        tmp_path / "results",
        random_seed_count=3,
        config_path=config,
    )
    comparisons = _rows(artifacts["random_comparisons"])

    directions = {
        row["metric"]: row["higher_is_better"] for row in comparisons
    }
    assert directions["row_fragmentation"] == "False"
    assert directions["column_fragmentation"] == "False"
    assert directions["group_contiguity"] == "True"
    assert all(0 <= float(row["directional_percentile"]) <= 1 for row in comparisons)


def test_workflow_rejects_tampered_event_log(tmp_path):
    manifest, config = _tiny_experiment(tmp_path)
    manifest_rows = _rows(manifest)
    event_log = manifest.parent / manifest_rows[0]["path"]
    event_log.write_text(event_log.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate_experiment(
            manifest,
            tmp_path / "results",
            random_seed_count=1,
            config_path=config,
        )
