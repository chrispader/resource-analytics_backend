import csv
import json
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from evaluation.metrics import degree_order_agreement, neighbor_similarity_coherence
from experiments.role_matrix.generate import (
    DatasetCondition,
    MIN_GROUP_SEPARATION,
    PROFILE_DIVERSITY_TARGETS,
    PROFILE_DIVERSITY_TOLERANCE,
    _conditions,
    _degree_group_cramers_v,
    _group_jaccard_separation,
    _profile_groups,
    _row_degrees,
    _assign_profiles,
    generate_dataset,
    generate_factorial_experiment,
    load_config,
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
    assert all(result.profile_group_count == 3 for result in results.values())
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


def test_all_factorial_conditions_meet_design_invariants():
    matched = {}
    alphabetical_degree_scores = []
    degree_scores_by_factor_cell = defaultdict(list)
    alphabetical_profile_coherence = []
    shuffled_profile_coherence = []
    alphabetical_group_contiguity = []
    group_contiguity_by_factor_cell = defaultdict(list)
    degree_group_associations = []
    degree_group_associations_by_factor_cell = defaultdict(list)
    for condition in _conditions(load_config()):
        groups = _profile_groups(condition)
        degrees = _row_degrees(condition, groups)
        profiles, _, _, _, realized_diversity, minimum_entropy, maximum_entropy = _assign_profiles(
            condition,
            degrees,
            groups,
        )
        _, _, separation = _group_jaccard_separation(profiles, groups)

        assert len(set(degrees)) >= 2
        assert max(degrees) > min(degrees)
        assert all(
            len(
                {
                    degree
                    for degree, assigned_group in zip(degrees, groups)
                    if assigned_group == group
                }
            )
            >= 2
            for group in set(groups)
        )
        for degree in set(degrees):
            counts_by_group = [
                sum(
                    observed_degree == degree and observed_group == group
                    for observed_degree, observed_group in zip(degrees, groups)
                )
                for group in sorted(set(groups))
            ]
            assert max(counts_by_group) - min(counts_by_group) <= 1
        assert sum(degrees) == round(
            condition.resource_count
            * condition.role_count
            * condition.target_density
        )
        assert set().union(*map(set, profiles)) == set(range(condition.role_count))
        assert all(
            group in profile
            and not ({0, 1, 2} - {group}).intersection(profile)
            for profile, group in zip(profiles, groups)
        )
        assert separation >= MIN_GROUP_SEPARATION
        assert realized_diversity == pytest.approx(
            PROFILE_DIVERSITY_TARGETS[condition.profile_diversity],
            abs=PROFILE_DIVERSITY_TOLERANCE,
        )
        assert 0 <= minimum_entropy <= maximum_entropy <= 1

        matrix = np.zeros(
            (condition.resource_count, condition.role_count),
            dtype=np.int8,
        )
        for resource_index, profile in enumerate(profiles):
            matrix[resource_index, list(profile)] = 1
        degree_score = degree_order_agreement(matrix)
        alphabetical_degree_scores.append(degree_score)
        factor_cell = (
            condition.size,
            condition.role_count,
            condition.target_density,
            condition.profile_diversity,
        )
        degree_scores_by_factor_cell[factor_cell].append(degree_score)
        alphabetical_profile_coherence.append(
            neighbor_similarity_coherence(matrix)
        )
        shuffled_order = np.random.default_rng(condition.seed + 991).permutation(
            condition.resource_count
        )
        shuffled_profile_coherence.append(
            neighbor_similarity_coherence(matrix[shuffled_order])
        )
        positions = defaultdict(list)
        for position, group in enumerate(groups):
            positions[group].append(position)
        group_contiguity = sum(
            (len(group_positions) / len(groups))
            * (
                len(group_positions)
                / (group_positions[-1] - group_positions[0] + 1)
            )
            for group_positions in positions.values()
        )
        alphabetical_group_contiguity.append(group_contiguity)
        group_contiguity_by_factor_cell[factor_cell].append(group_contiguity)
        degree_group_association = _degree_group_cramers_v(degrees, groups)
        degree_group_associations.append(degree_group_association)
        degree_group_associations_by_factor_cell[factor_cell].append(
            degree_group_association
        )

        key = (
            condition.size,
            condition.resource_count,
            condition.role_count,
            condition.target_density,
            condition.seed,
        )
        previous = matched.setdefault(key, (groups, degrees))
        assert groups == previous[0]
        assert degrees == previous[1]

    overall_degree_agreement = sum(alphabetical_degree_scores) / len(
        alphabetical_degree_scores
    )
    assert 0.45 <= overall_degree_agreement <= 0.55
    assert all(
        0.40 <= sum(scores) / len(scores) <= 0.60
        for scores in degree_scores_by_factor_cell.values()
    )
    alphabetical_mean = sum(alphabetical_profile_coherence) / len(
        alphabetical_profile_coherence
    )
    shuffled_mean = sum(shuffled_profile_coherence) / len(
        shuffled_profile_coherence
    )
    assert abs(alphabetical_mean - shuffled_mean) <= 0.02
    assert sum(alphabetical_group_contiguity) / len(
        alphabetical_group_contiguity
    ) <= 0.40
    assert all(
        sum(scores) / len(scores) <= 0.40
        for scores in group_contiguity_by_factor_cell.values()
    )
    assert sum(degree_group_associations) / len(degree_group_associations) <= 0.05
    assert all(
        sum(scores) / len(scores) <= 0.08
        for scores in degree_group_associations_by_factor_cell.values()
    )
