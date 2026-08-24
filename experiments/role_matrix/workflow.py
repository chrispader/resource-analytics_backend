"""Offline generation and evaluation workflow for the matrix experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

import pandas as pd

from evaluation.evaluate import (
    DEFAULT_BACKGROUND_COLOR,
    DEFAULT_EMPTY_CELL_COLOR,
    ENABLED_ORDERING_VARIANTS,
    build_ordering_variants,
    evaluate_resource_role_matrix,
    role_palette,
)
from evaluation.color_metrics import evaluate_color_discriminability
from evaluation.matrix_model import ResourceRoleMatrix, resource_role_matrix_from_mapping
from evaluation.ordering import random_ordering
from evaluation.plotly_heuristics import (
    PUBLISHED_HEURISTIC_RULES,
    evaluate_plotly_figure,
)
from experiments.role_matrix.generate import (
    CONFIG_PATH,
    MIN_GROUP_SEPARATION,
    PROFILE_DIVERSITY_TARGETS,
    PROFILE_DIVERSITY_TOLERANCE,
    _degree_group_cramers_v,
    _group_jaccard_separation,
    _normalized_entropy,
    _normalized_profile_entropy,
    generate_factorial_experiment,
    load_config,
)
from pm import (
    RESOURCE_ROLE_MATRIX_PALETTE,
    compute_resource_role_matrix_context,
    resource_role_figure_for_matrix,
    resource_role_matrix_data,
)


STRUCTURAL_METRICS = (
    "row_coherence",
    "column_coherence",
    "row_fragmentation",
    "column_fragmentation",
    "degree_order_agreement",
    "group_contiguity",
)
HIGHER_IS_BETTER = {
    "row_coherence": True,
    "column_coherence": True,
    "row_fragmentation": False,
    "column_fragmentation": False,
    "degree_order_agreement": True,
    "group_contiguity": True,
}
FACTOR_COLUMNS = (
    "size",
    "resource_count",
    "role_count",
    "target_density",
    "profile_diversity",
)
ANALYSIS_FACTORS = (
    "resource_count",
    "role_count",
    "target_density",
    "profile_diversity",
)
BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_FILES = (
    "experiments/role_matrix/generate.py",
    "experiments/role_matrix/workflow.py",
    "experiments/role_matrix/config.json",
    "evaluation/evaluate.py",
    "evaluation/metrics.py",
    "evaluation/ordering.py",
    "evaluation/color_metrics.py",
    "evaluation/matrix_model.py",
    "evaluation/plotly_heuristics.py",
    "requirements.txt",
    "pm.py",
)


def planted_group_contiguity(
    ordered_resources: Sequence[str],
    resource_groups: Mapping[str, str],
) -> float:
    """Weighted compactness of planted groups in an ordered resource list.

    A group's score is its resource count divided by its occupied row span.
    The overall value is weighted by group size. One means that every planted
    group forms one uninterrupted row block.
    """
    if not ordered_resources:
        return 0.0

    positions: dict[str, list[int]] = defaultdict(list)
    for position, resource in enumerate(ordered_resources):
        group = resource_groups.get(resource)
        if group is None:
            raise ValueError(f"Missing planted group for resource {resource}")
        positions[group].append(position)

    total = len(ordered_resources)
    return sum(
        (len(group_positions) / total)
        * (
            len(group_positions)
            / (group_positions[-1] - group_positions[0] + 1)
        )
        for group_positions in positions.values()
    )


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if not rows:
        raise ValueError("manifest.csv contains no datasets")
    return sorted(rows, key=lambda row: row["path"])


def _validate_manifest_against_config(
    manifest_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, object],
) -> None:
    sizes = config.get("resource_sizes")
    if not isinstance(sizes, dict) or not sizes:
        raise ValueError("Config resource_sizes must be a non-empty object")
    required_lists = (
        "role_counts",
        "target_densities",
        "profile_diversities",
        "seeds",
    )
    if any(not isinstance(config.get(key), list) or not config[key] for key in required_lists):
        raise ValueError("Config factorial levels must be non-empty lists")

    expected = {
        (
            str(size),
            int(resource_count),
            int(role_count),
            float(density),
            str(diversity),
            int(seed),
        )
        for (size, resource_count), role_count, density, diversity, seed in itertools.product(
            sizes.items(),
            config["role_counts"],
            config["target_densities"],
            config["profile_diversities"],
            config["seeds"],
        )
    }
    observed = [
        (
            row["size"],
            int(row["resource_count"]),
            int(row["role_count"]),
            float(row["target_density"]),
            row["profile_diversity"],
            int(row["seed"]),
        )
        for row in manifest_rows
    ]
    if len(observed) != len(set(observed)):
        raise ValueError("Manifest contains duplicate factorial conditions")
    observed_set = set(observed)
    if observed_set != expected:
        missing = sorted(expected - observed_set, key=str)
        unexpected = sorted(observed_set - expected, key=str)
        raise ValueError(
            "Manifest/config factorial mismatch: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty artifact: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _factor_values(manifest_row: Mapping[str, str]) -> dict[str, object]:
    return {
        "size": manifest_row["size"],
        "resource_count": int(manifest_row["resource_count"]),
        "role_count": int(manifest_row["role_count"]),
        "target_density": float(manifest_row["target_density"]),
        "profile_diversity": manifest_row["profile_diversity"],
        "seed": int(manifest_row["seed"]),
    }


def _matrix_from_log(
    log_path: Path,
) -> tuple[pd.DataFrame, Any, ResourceRoleMatrix, dict[str, str]]:
    frame = pd.read_csv(log_path)
    required = {"Resource", "Role", "Activity", "Profile Group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{log_path} is missing columns: {sorted(missing)}")

    memberships = frame[["Resource", "Profile Group"]].drop_duplicates()
    if memberships["Resource"].duplicated().any():
        raise ValueError(f"{log_path} assigns a resource to multiple planted groups")
    resource_groups = dict(
        zip(memberships["Resource"].astype(str), memberships["Profile Group"].astype(str))
    )

    context = compute_resource_role_matrix_context(frame)
    matrix_data = resource_role_matrix_data(context)
    matrix = resource_role_matrix_from_mapping(
        resources=matrix_data.resources,
        roles=matrix_data.roles,
        mapping=matrix_data.mapping,
    )
    return frame, context, matrix, resource_groups


def _validate_log_against_manifest(
    log_path: Path,
    manifest_row: Mapping[str, str],
    matrix: ResourceRoleMatrix,
    resource_groups: Mapping[str, str],
) -> None:
    expected_hash = manifest_row.get("sha256")
    if not expected_hash:
        raise ValueError(f"Manifest has no SHA-256 value for {log_path.name}")
    observed_hash = _sha256(log_path)
    if observed_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch for {log_path.name}: "
            f"expected {expected_hash}, observed {observed_hash}"
        )

    expected_resources = int(manifest_row["resource_count"])
    expected_roles = int(manifest_row["role_count"])
    if len(matrix.resources) != expected_resources:
        raise ValueError(
            f"Resource count mismatch for {log_path.name}: "
            f"expected {expected_resources}, observed {len(matrix.resources)}"
        )
    if len(matrix.roles) != expected_roles:
        raise ValueError(
            f"Role count mismatch for {log_path.name}: "
            f"expected {expected_roles}, observed {len(matrix.roles)}"
        )

    observed_density = float(matrix.values.sum() / matrix.values.size)
    realized_density = float(manifest_row["realized_density"])
    if not math.isclose(observed_density, realized_density, abs_tol=1e-8):
        raise ValueError(
            f"Realized density mismatch for {log_path.name}: "
            f"manifest {realized_density}, observed {observed_density}"
        )
    target_density = float(manifest_row["target_density"])
    rounding_tolerance = (0.5 / (expected_resources * expected_roles)) + 1e-12
    if not math.isclose(observed_density, target_density, abs_tol=rounding_tolerance):
        raise ValueError(
            f"Target density mismatch for {log_path.name}: "
            f"target {target_density}, observed {observed_density}"
        )

    profiles = [
        tuple(int(index) for index, value in enumerate(row) if value)
        for row in matrix.values
    ]
    degrees = [len(profile) for profile in profiles]
    groups = [resource_groups[resource] for resource in matrix.resources]
    degree_mean = sum(degrees) / len(degrees)
    degree_standard_deviation = math.sqrt(
        sum((degree - degree_mean) ** 2 for degree in degrees) / len(degrees)
    )
    within_jaccard, between_jaccard, separation = _group_jaccard_separation(
        profiles,
        groups,
    )
    derived_statistics = {
        "unique_profile_count": len(set(profiles)),
        "normalized_profile_entropy": _normalized_profile_entropy(profiles),
        "profile_group_count": len(set(groups)),
        "degree_value_count": len(set(degrees)),
        "degree_standard_deviation": degree_standard_deviation,
        "degree_coefficient_of_variation": degree_standard_deviation / degree_mean,
        "normalized_degree_entropy": _normalized_entropy(degrees),
        "degree_group_cramers_v": _degree_group_cramers_v(degrees, groups),
        "within_group_jaccard": within_jaccard,
        "between_group_jaccard": between_jaccard,
        "group_jaccard_separation": separation,
    }
    integer_statistics = {
        "unique_profile_count",
        "profile_group_count",
        "degree_value_count",
    }
    for statistic, observed_value in derived_statistics.items():
        manifest_value = manifest_row.get(statistic)
        if manifest_value is None:
            raise ValueError(f"Manifest has no {statistic} value for {log_path.name}")
        matches = (
            int(manifest_value) == observed_value
            if statistic in integer_statistics
            else math.isclose(float(manifest_value), observed_value, abs_tol=1e-8)
        )
        if not matches:
            raise ValueError(
                f"{statistic} mismatch for {log_path.name}: "
                f"manifest {manifest_value}, observed {observed_value}"
            )

    diversity_level = manifest_row["profile_diversity"]
    configured_target = PROFILE_DIVERSITY_TARGETS[diversity_level]
    manifest_target = float(manifest_row["target_relative_profile_diversity"])
    realized_diversity = float(manifest_row["realized_relative_profile_diversity"])
    if not math.isclose(manifest_target, configured_target, abs_tol=1e-12):
        raise ValueError(
            f"Profile-diversity target mismatch for {log_path.name}: "
            f"configured {configured_target}, manifest {manifest_target}"
        )
    if abs(realized_diversity - configured_target) > PROFILE_DIVERSITY_TOLERANCE:
        raise ValueError(
            f"Realized profile diversity is outside configured tolerance for "
            f"{log_path.name}: target {configured_target}, realized {realized_diversity}"
        )
    if separation < MIN_GROUP_SEPARATION:
        raise ValueError(
            f"Planted-group separation is below the configured minimum for "
            f"{log_path.name}: minimum {MIN_GROUP_SEPARATION}, observed {separation}"
        )


def _metric_row(
    dataset_id: str,
    factors: Mapping[str, object],
    matrix: ResourceRoleMatrix,
    resource_groups: Mapping[str, str],
    variant: str,
    random_seed: int | None,
) -> dict[str, object]:
    result = evaluate_resource_role_matrix(
        matrix,
        variant=variant,
        seed=random_seed,
    )
    return {
        "dataset_id": dataset_id,
        **factors,
        "variant": variant,
        "random_seed": "" if random_seed is None else random_seed,
        "resource_count_observed": result.resource_count,
        "role_count_observed": result.role_count,
        "filled_cells": result.filled_cells,
        "density": result.density,
        "row_coherence": result.row_coherence,
        "column_coherence": result.column_coherence,
        "row_fragmentation": result.row_fragmentation,
        "column_fragmentation": result.column_fragmentation,
        "degree_order_agreement": result.degree_order_agreement,
        "group_contiguity": planted_group_contiguity(
            matrix.resources,
            resource_groups,
        ),
    }


def _heuristic_rows(
    dataset_id: str,
    factors: Mapping[str, object],
    variant: str,
    figure: Any,
) -> list[dict[str, object]]:
    report = evaluate_plotly_figure(figure)
    findings = {finding.rule_id: finding for finding in report.findings}
    rows = []
    for rule in PUBLISHED_HEURISTIC_RULES:
        finding = findings.get(rule.rule_id)
        rows.append(
            {
                "dataset_id": dataset_id,
                **factors,
                "variant": variant,
                "rule_id": rule.rule_id,
                "heuristic": rule.heuristic,
                "assistance": rule.assistance,
                "status": finding.status if finding else "not_applicable",
                "message": finding.message if finding else "",
                "recommendation": (
                    finding.recommendation if finding and finding.recommendation else ""
                ),
                "labels_json": json.dumps(
                    finding.labels.__dict__ if finding else rule.labels.__dict__,
                    sort_keys=True,
                ),
                "evidence_json": json.dumps(
                    finding.evidence if finding else {},
                    sort_keys=True,
                ),
            }
        )
    return rows


def _comparison_rows(
    fixed_rows: Sequence[Mapping[str, object]],
    random_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    random_by_dataset: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in random_rows:
        random_by_dataset[str(row["dataset_id"])].append(row)

    comparisons = []
    for fixed in fixed_rows:
        baselines = random_by_dataset[str(fixed["dataset_id"])]
        for metric in STRUCTURAL_METRICS:
            values = [float(row[metric]) for row in baselines]
            fixed_value = float(fixed[metric])
            mean = fmean(values)
            standard_deviation = pstdev(values)
            higher_is_better = HIGHER_IS_BETTER[metric]
            percentile = (
                sum(value <= fixed_value for value in values) / len(values)
                if higher_is_better
                else sum(value >= fixed_value for value in values) / len(values)
            )
            signed_difference = (
                fixed_value - mean if higher_is_better else mean - fixed_value
            )
            comparisons.append(
                {
                    "dataset_id": fixed["dataset_id"],
                    **{key: fixed[key] for key in (*FACTOR_COLUMNS, "seed")},
                    "variant": fixed["variant"],
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "fixed_value": fixed_value,
                    "random_mean": mean,
                    "random_std": standard_deviation,
                    "directional_difference_from_random_mean": signed_difference,
                    "directional_percentile": percentile,
                }
            )
    return comparisons


def _factor_aggregate_rows(
    fixed_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in fixed_rows:
        for factor in FACTOR_COLUMNS:
            for metric in STRUCTURAL_METRICS:
                key = (factor, str(row[factor]), str(row["variant"]), metric)
                grouped[key].append(float(row[metric]))

    rows = []
    for (factor, value, variant, metric), values in sorted(grouped.items()):
        rows.append(
            {
                "factor": factor,
                "factor_value": value,
                "variant": variant,
                "metric": metric,
                "higher_is_better": HIGHER_IS_BETTER[metric],
                "dataset_count": len(values),
                "mean": fmean(values),
                "std": pstdev(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def _condition_aggregate_rows(
    fixed_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    group_columns = (*FACTOR_COLUMNS, "variant")
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in fixed_rows:
        grouped[tuple(row[column] for column in group_columns)].append(row)

    rows = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        group_values = dict(zip(group_columns, key))
        for metric in STRUCTURAL_METRICS:
            values = [float(row[metric]) for row in group_rows]
            rows.append(
                {
                    **group_values,
                    "metric": metric,
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                    "seed_count": len(values),
                    "mean": fmean(values),
                    "std": pstdev(values),
                    "min": min(values),
                    "max": max(values),
                }
            )
    return rows


def _factor_interaction_rows(
    fixed_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for factor_a, factor_b in itertools.combinations(ANALYSIS_FACTORS, 2):
        for row in fixed_rows:
            for metric in STRUCTURAL_METRICS:
                key = (
                    factor_a,
                    row[factor_a],
                    factor_b,
                    row[factor_b],
                    row["variant"],
                    metric,
                )
                grouped[key].append(float(row[metric]))

    rows = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        factor_a, value_a, factor_b, value_b, variant, metric = key
        rows.append(
            {
                "factor_a": factor_a,
                "factor_a_value": value_a,
                "factor_b": factor_b,
                "factor_b_value": value_b,
                "variant": variant,
                "metric": metric,
                "higher_is_better": HIGHER_IS_BETTER[metric],
                "observation_count": len(values),
                "mean": fmean(values),
                "std": pstdev(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def _git_provenance() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BACKEND_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=BACKEND_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _dependency_versions() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "plotly", "scikit-image", "pytest")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def evaluate_experiment(
    manifest_path: Path,
    output_root: Path,
    *,
    random_seed_count: int = 100,
    config_path: Path = CONFIG_PATH,
) -> dict[str, Path]:
    if random_seed_count < 1:
        raise ValueError("random_seed_count must be positive")

    manifest_rows = _read_manifest(manifest_path)
    config = load_config(config_path)
    _validate_manifest_against_config(manifest_rows, config)
    output_root.mkdir(parents=True, exist_ok=True)
    fixed_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    heuristic_rows: list[dict[str, object]] = []

    for manifest_row in manifest_rows:
        log_path = manifest_path.parent / manifest_row["path"]
        dataset_id = log_path.stem
        factors = _factor_values(manifest_row)
        _, context, matrix, resource_groups = _matrix_from_log(log_path)
        _validate_log_against_manifest(
            log_path,
            manifest_row,
            matrix,
            resource_groups,
        )
        variants = build_ordering_variants(matrix)

        role_colors = role_palette(RESOURCE_ROLE_MATRIX_PALETTE, len(matrix.roles))
        distinct_role_colors = list(dict.fromkeys(role_colors))
        color = evaluate_color_discriminability(
            distinct_role_colors,
            DEFAULT_EMPTY_CELL_COLOR,
            DEFAULT_BACKGROUND_COLOR,
        )
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                **factors,
                "path": manifest_row["path"],
                "input_sha256": _sha256(log_path),
                "realized_density": float(manifest_row["realized_density"]),
                "unique_profile_count": int(manifest_row["unique_profile_count"]),
                "normalized_profile_entropy": float(
                    manifest_row["normalized_profile_entropy"]
                ),
                "baseline_generated_profile_entropy": float(
                    manifest_row["baseline_generated_profile_entropy"]
                ),
                "maximum_generated_profile_entropy": float(
                    manifest_row["maximum_generated_profile_entropy"]
                ),
                "profile_group_count": int(manifest_row["profile_group_count"]),
                "target_relative_profile_diversity": float(
                    manifest_row["target_relative_profile_diversity"]
                ),
                "realized_relative_profile_diversity": float(
                    manifest_row["realized_relative_profile_diversity"]
                ),
                "degree_value_count": int(manifest_row["degree_value_count"]),
                "degree_standard_deviation": float(
                    manifest_row["degree_standard_deviation"]
                ),
                "degree_coefficient_of_variation": float(
                    manifest_row["degree_coefficient_of_variation"]
                ),
                "normalized_degree_entropy": float(
                    manifest_row["normalized_degree_entropy"]
                ),
                "degree_group_cramers_v": float(
                    manifest_row["degree_group_cramers_v"]
                ),
                "within_group_jaccard": float(manifest_row["within_group_jaccard"]),
                "between_group_jaccard": float(manifest_row["between_group_jaccard"]),
                "group_jaccard_separation": float(
                    manifest_row["group_jaccard_separation"]
                ),
                **color.to_dict(),
            }
        )

        for variant in ENABLED_ORDERING_VARIANTS:
            ordered_matrix = variants[variant]
            fixed_rows.append(
                _metric_row(
                    dataset_id,
                    factors,
                    ordered_matrix,
                    resource_groups,
                    variant,
                    None,
                )
            )
            figure = resource_role_figure_for_matrix(context, ordered_matrix)
            heuristic_rows.extend(
                _heuristic_rows(dataset_id, factors, variant, figure)
            )

        for random_seed in range(random_seed_count):
            random_matrix = random_ordering(matrix, seed=random_seed)
            random_rows.append(
                _metric_row(
                    dataset_id,
                    factors,
                    random_matrix,
                    resource_groups,
                    "random",
                    random_seed,
                )
            )

    comparison_rows = _comparison_rows(fixed_rows, random_rows)
    aggregate_rows = _factor_aggregate_rows(fixed_rows)
    condition_aggregate_rows = _condition_aggregate_rows(fixed_rows)
    interaction_rows = _factor_interaction_rows(fixed_rows)
    artifacts = {
        "fixed_ordering_metrics": output_root / "fixed_ordering_metrics.csv",
        "random_baseline_metrics": output_root / "random_baseline_metrics.csv",
        "dataset_summaries": output_root / "dataset_summaries.csv",
        "heuristic_findings": output_root / "heuristic_findings.csv",
        "random_comparisons": output_root / "random_comparisons.csv",
        "factor_aggregates": output_root / "factor_aggregates.csv",
        "condition_aggregates": output_root / "condition_aggregates.csv",
        "factor_interactions": output_root / "factor_interactions.csv",
        "run_metadata": output_root / "run_metadata.json",
        "experiment_config": output_root / "experiment_config.json",
    }
    known_artifact_paths = {path.resolve() for path in artifacts.values()}
    ignored_preexisting_output_files = sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path.resolve() not in known_artifact_paths
    )
    input_hashes = {
        row["path"]: row["sha256"]
        for row in manifest_rows
    }
    code_hashes = {
        relative_path: _sha256(BACKEND_ROOT / relative_path)
        for relative_path in PROVENANCE_FILES
        if (BACKEND_ROOT / relative_path).exists()
    }

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}-staging-",
            dir=output_root.parent,
        )
    )
    staged = {name: staging_root / path.name for name, path in artifacts.items()}
    try:
        _write_csv(staged["fixed_ordering_metrics"], fixed_rows)
        _write_csv(staged["random_baseline_metrics"], random_rows)
        _write_csv(staged["dataset_summaries"], dataset_rows)
        _write_csv(staged["heuristic_findings"], heuristic_rows)
        _write_csv(staged["random_comparisons"], comparison_rows)
        _write_csv(staged["factor_aggregates"], aggregate_rows)
        _write_csv(staged["condition_aggregates"], condition_aggregate_rows)
        _write_csv(staged["factor_interactions"], interaction_rows)
        shutil.copyfile(config_path, staged["experiment_config"])

        output_hashes = {
            path.name: _sha256(path)
            for name, path in staged.items()
            if name != "run_metadata"
        }
        metadata = {
            "schema_version": 1,
            "dataset_count": len(manifest_rows),
            "deterministic_orderings": list(ENABLED_ORDERING_VARIANTS),
            "random_seed_count": random_seed_count,
            "random_seeds": list(range(random_seed_count)),
            "structural_metrics": [
                {
                    "name": metric,
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                }
                for metric in STRUCTURAL_METRICS
            ],
            "heuristic_rule_count": len(PUBLISHED_HEURISTIC_RULES),
            "artifact_files": sorted(path.name for path in artifacts.values()),
            "manifest_sha256": _sha256(manifest_path),
            "experiment_config_sha256": _sha256(config_path),
            "input_sha256": input_hashes,
            "output_sha256": output_hashes,
            "code_sha256": code_hashes,
            "backend_git": _git_provenance(),
            "python_version": platform.python_version(),
            "dependency_versions": _dependency_versions(),
            "analysis_scope": "descriptive factorial summaries; no inferential claims",
            "provenance_scope": (
                "Hashes cover the directly invoked local experiment, matrix, metric, "
                "ordering, color, heuristic, rendering, and dependency-declaration files."
            ),
            "ignored_preexisting_output_files": ignored_preexisting_output_files,
        }
        _write_json(staged["run_metadata"], metadata)

        for name, destination in artifacts.items():
            staged[name].replace(destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate event logs")
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--config", type=Path, default=CONFIG_PATH)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate generated logs")
    evaluate_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--random-seeds", type=int, default=100)
    evaluate_parser.add_argument("--config", type=Path, default=CONFIG_PATH)

    full_parser = subparsers.add_parser("full", help="Generate and evaluate")
    full_parser.add_argument("--output", type=Path, required=True)
    full_parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    full_parser.add_argument("--random-seeds", type=int, default=100)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "generate":
        results = generate_factorial_experiment(args.output, args.config)
        print(f"Generated {len(results)} event logs in {args.output}")
        return
    if args.command == "evaluate":
        evaluate_experiment(
            args.manifest,
            args.output,
            random_seed_count=args.random_seeds,
            config_path=args.config,
        )
        print(f"Evaluation artifacts written to {args.output}")
        return

    dataset_root = args.output / "datasets"
    result_root = args.output / "results"
    results = generate_factorial_experiment(dataset_root, args.config)
    evaluate_experiment(
        dataset_root / "manifest.csv",
        result_root,
        random_seed_count=args.random_seeds,
        config_path=args.config,
    )
    print(f"Generated and evaluated {len(results)} logs in {args.output}")


if __name__ == "__main__":
    main()
