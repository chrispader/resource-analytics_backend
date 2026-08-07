"""Offline generation and evaluation workflow for the matrix experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
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
from experiments.role_matrix.generate import CONFIG_PATH, generate_factorial_experiment
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
        _validate_log_against_manifest(log_path, manifest_row, matrix)
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
                "profile_group_count": int(manifest_row["profile_group_count"]),
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
    artifacts = {
        "fixed_ordering_metrics": output_root / "fixed_ordering_metrics.csv",
        "random_baseline_metrics": output_root / "random_baseline_metrics.csv",
        "dataset_summaries": output_root / "dataset_summaries.csv",
        "heuristic_findings": output_root / "heuristic_findings.csv",
        "random_comparisons": output_root / "random_comparisons.csv",
        "factor_aggregates": output_root / "factor_aggregates.csv",
        "condition_aggregates": output_root / "condition_aggregates.csv",
        "run_metadata": output_root / "run_metadata.json",
        "experiment_config": output_root / "experiment_config.json",
    }
    _write_csv(artifacts["fixed_ordering_metrics"], fixed_rows)
    _write_csv(artifacts["random_baseline_metrics"], random_rows)
    _write_csv(artifacts["dataset_summaries"], dataset_rows)
    _write_csv(artifacts["heuristic_findings"], heuristic_rows)
    _write_csv(artifacts["random_comparisons"], comparison_rows)
    _write_csv(artifacts["factor_aggregates"], aggregate_rows)
    _write_csv(artifacts["condition_aggregates"], condition_aggregate_rows)

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
    }
    _write_json(artifacts["run_metadata"], metadata)
    shutil.copyfile(config_path, artifacts["experiment_config"])
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
