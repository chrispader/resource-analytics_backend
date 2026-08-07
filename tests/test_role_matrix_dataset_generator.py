import csv
import json
from dataclasses import replace

import pytest

from experiments.role_matrix.generate import (
    DatasetCondition,
    generate_dataset,
    generate_factorial_experiment,
)


BASE_CONDITION = DatasetCondition(
    size="small",
    resource_count=40,
    role_count=10,
    target_density=0.15,
    profile_diversity="low",
    seed=0,
)


def _read_rows(path):
    with path.open(encoding="utf-8", newline="") as event_log:
        return list(csv.DictReader(event_log))


def test_generated_log_has_expected_matrix_properties(tmp_path):
    result = generate_dataset(BASE_CONDITION, tmp_path)
    rows = _read_rows(result.path)

    resources = {row["Resource"] for row in rows}
    roles = {row["Role"] for row in rows}
    pairs = {(row["Resource"], row["Role"]) for row in rows}

    assert len(resources) == BASE_CONDITION.resource_count
    assert len(roles) == BASE_CONDITION.role_count
    assert len(rows) == len(pairs) == result.assignment_count
    assert result.realized_density == pytest.approx(BASE_CONDITION.target_density)
    assert {row["Profile Group"] for row in rows} == {
        "PG-01",
        "PG-02",
        "PG-03",
        "PG-04",
        "PG-05",
    }
    assert len({row["Case ID"] for row in rows}) == BASE_CONDITION.resource_count


def test_diversity_changes_profiles_but_preserves_row_degrees(tmp_path):
    results = {
        level: generate_dataset(
            replace(BASE_CONDITION, profile_diversity=level),
            tmp_path / level,
        )
        for level in ("low", "medium", "high")
    }

    def row_degrees(result):
        counts = {}
        for row in _read_rows(result.path):
            counts[row["Resource"]] = counts.get(row["Resource"], 0) + 1
        return counts

    def group_membership(result):
        return {
            row["Resource"]: row["Profile Group"]
            for row in _read_rows(result.path)
        }

    assert row_degrees(results["low"]) == row_degrees(results["medium"])
    assert row_degrees(results["medium"]) == row_degrees(results["high"])
    assert group_membership(results["low"]) == group_membership(results["medium"])
    assert group_membership(results["medium"]) == group_membership(results["high"])
    assert all(result.profile_group_count == 5 for result in results.values())
    assert (
        results["low"].unique_profile_count
        < results["medium"].unique_profile_count
        < results["high"].unique_profile_count
    )
    assert (
        results["low"].normalized_profile_entropy
        < results["medium"].normalized_profile_entropy
        < results["high"].normalized_profile_entropy
    )


def test_generation_is_byte_reproducible(tmp_path):
    first = generate_dataset(BASE_CONDITION, tmp_path / "first")
    second = generate_dataset(BASE_CONDITION, tmp_path / "second")

    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()


def test_factorial_workflow_writes_manifest(tmp_path):
    config = {
        "resource_sizes": {"small": 40},
        "role_counts": [10],
        "target_densities": [0.15],
        "profile_diversities": ["low", "medium", "high"],
        "seeds": [0],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    results = generate_factorial_experiment(tmp_path / "output", config_path)

    assert len(results) == 3
    manifest_rows = _read_rows(tmp_path / "output" / "manifest.csv")
    assert len(manifest_rows) == 3
    assert {row["profile_diversity"] for row in manifest_rows} == {
        "low",
        "medium",
        "high",
    }
