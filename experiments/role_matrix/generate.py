"""Generate deterministic event logs for the matrix-ordering experiment.

The design varies resource count, role count, density, and profile diversity
independently. Each condition uses the same row-degree sequence for all three
diversity levels, so diversity does not accidentally change density.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence


CONFIG_PATH = Path(__file__).with_name("config.json")
DIVERSITY_LEVELS = ("low", "medium", "high")
PROFILE_DIVERSITY_TARGETS = {"low": 0.25, "medium": 0.60, "high": 0.90}
PROFILE_DIVERSITY_TOLERANCE = 0.12
PROFILE_GROUP_COUNT = 3
MIN_GROUP_SEPARATION = 0.05
EVENT_LOG_COLUMNS = (
    "Case ID",
    "Start Timestamp",
    "Complete Timestamp",
    "Activity",
    "Resource",
    "Role",
    "Profile Group",
)


@dataclass(frozen=True)
class DatasetCondition:
    size: str
    resource_count: int
    role_count: int
    target_density: float
    profile_diversity: str
    seed: int


@dataclass(frozen=True)
class GeneratedDataset:
    condition: DatasetCondition
    path: Path
    assignment_count: int
    realized_density: float
    unique_profile_count: int
    normalized_profile_entropy: float
    baseline_generated_profile_entropy: float
    maximum_generated_profile_entropy: float
    target_relative_profile_diversity: float
    realized_relative_profile_diversity: float
    profile_group_count: int
    degree_value_count: int
    degree_standard_deviation: float
    degree_coefficient_of_variation: float
    normalized_degree_entropy: float
    degree_group_cramers_v: float
    within_group_jaccard: float
    between_group_jaccard: float
    group_jaccard_separation: float
    sha256: str

    def manifest_row(self, root: Path) -> dict[str, object]:
        row = asdict(self.condition)
        row.update(
            {
                "path": self.path.relative_to(root).as_posix(),
                "assignment_count": self.assignment_count,
                "realized_density": round(self.realized_density, 8),
                "unique_profile_count": self.unique_profile_count,
                "normalized_profile_entropy": round(
                    self.normalized_profile_entropy, 8
                ),
                "baseline_generated_profile_entropy": round(
                    self.baseline_generated_profile_entropy, 8
                ),
                "maximum_generated_profile_entropy": round(
                    self.maximum_generated_profile_entropy, 8
                ),
                "target_relative_profile_diversity": (
                    self.target_relative_profile_diversity
                ),
                "realized_relative_profile_diversity": round(
                    self.realized_relative_profile_diversity, 8
                ),
                "profile_group_count": self.profile_group_count,
                "degree_value_count": self.degree_value_count,
                "degree_standard_deviation": round(
                    self.degree_standard_deviation, 8
                ),
                "degree_coefficient_of_variation": round(
                    self.degree_coefficient_of_variation, 8
                ),
                "normalized_degree_entropy": round(
                    self.normalized_degree_entropy, 8
                ),
                "degree_group_cramers_v": round(
                    self.degree_group_cramers_v, 8
                ),
                "within_group_jaccard": round(self.within_group_jaccard, 8),
                "between_group_jaccard": round(self.between_group_jaccard, 8),
                "group_jaccard_separation": round(
                    self.group_jaccard_separation, 8
                ),
                "sha256": self.sha256,
            }
        )
        return row


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    design = config.get("design_parameters")
    if design is not None:
        expected = {
            "profile_group_count": PROFILE_GROUP_COUNT,
            "profile_diversity_targets": PROFILE_DIVERSITY_TARGETS,
            "profile_diversity_tolerance": PROFILE_DIVERSITY_TOLERANCE,
            "minimum_group_jaccard_separation": MIN_GROUP_SEPARATION,
            "degree_band_proportions": [0.25, 0.50, 0.25],
            "degree_group_stratification_max_count_difference": 1,
        }
        if design != expected:
            raise ValueError(
                "Config design_parameters do not match the implemented generator design"
            )
    return config


def _condition_seed(condition: DatasetCondition) -> int:
    return _salted_condition_seed(condition, "condition")


def _salted_condition_seed(condition: DatasetCondition, salt: str) -> int:
    key = (
        f"{condition.resource_count}:{condition.role_count}:"
        f"{condition.target_density:.4f}:{condition.seed}:{salt}"
    )
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def _row_degrees(condition: DatasetCondition, groups: Sequence[int]) -> list[int]:
    target_assignments = round(
        condition.resource_count * condition.role_count * condition.target_density
    )
    target_assignments = min(
        condition.resource_count * condition.role_count,
        max(condition.resource_count, target_assignments),
    )
    mean_degree = target_assignments / condition.resource_count
    spread = max(1, round(condition.role_count * 0.10))
    low_count = condition.resource_count // 4
    high_count = condition.resource_count // 4
    desired_degrees = (
        [mean_degree - spread] * low_count
        + [mean_degree] * (condition.resource_count - low_count - high_count)
        + [mean_degree + spread] * high_count
    )
    maximum_degree = condition.role_count - 2
    desired_degrees = [
        min(maximum_degree, max(1.0, degree))
        for degree in desired_degrees
    ]
    for _ in range(4):
        difference = target_assignments - sum(desired_degrees)
        if abs(difference) < 1e-9:
            break
        adjustable = [
            index
            for index, degree in enumerate(desired_degrees)
            if (difference > 0 and degree < maximum_degree)
            or (difference < 0 and degree > 1)
        ]
        shift = difference / len(adjustable)
        for index in adjustable:
            desired_degrees[index] = min(
                maximum_degree,
                max(1.0, desired_degrees[index] + shift),
            )

    degrees = [math.floor(degree) for degree in desired_degrees]
    remainder = target_assignments - sum(degrees)
    fractional_order = sorted(
        range(len(degrees)),
        key=lambda index: (desired_degrees[index] - degrees[index], -index),
        reverse=True,
    )
    for index in fractional_order[:remainder]:
        degrees[index] += 1

    if len(set(degrees)) < 2 or max(degrees) - min(degrees) < 1:
        raise RuntimeError(f"Could not create a non-degenerate degree distribution: {condition}")

    indices_by_group: dict[int, list[int]] = defaultdict(list)
    for resource_index, group in enumerate(groups):
        indices_by_group[group].append(resource_index)
    for group, indices in indices_by_group.items():
        random.Random(
            _salted_condition_seed(condition, f"degree-placement:{group}")
        ).shuffle(indices)
    assigned = [0] * condition.resource_count
    remaining_capacity = {
        group: len(indices)
        for group, indices in indices_by_group.items()
    }
    degrees_by_group = {group: [] for group in indices_by_group}
    degree_counts = Counter(degrees)
    groups_in_order = sorted(indices_by_group)
    for degree in sorted(degree_counts, reverse=True):
        count = degree_counts[degree]
        base_count, remainder = divmod(count, len(groups_in_order))
        if any(remaining_capacity[group] < base_count for group in groups_in_order):
            raise RuntimeError("Balanced degree stratification exceeded group capacity")
        for group in groups_in_order:
            degrees_by_group[group].extend([degree] * base_count)
            remaining_capacity[group] -= base_count

        tie_order = list(groups_in_order)
        random.Random(
            _salted_condition_seed(condition, f"degree-stratum:{degree}")
        ).shuffle(tie_order)
        remainder_groups = sorted(
            tie_order,
            key=lambda group: remaining_capacity[group],
            reverse=True,
        )[:remainder]
        for group in remainder_groups:
            degrees_by_group[group].append(degree)
            remaining_capacity[group] -= 1

    if any(remaining_capacity.values()):
        raise RuntimeError("Balanced degree stratification left unassigned resources")
    for group, indices in indices_by_group.items():
        group_degrees = degrees_by_group[group]
        if len(group_degrees) != len(indices):
            raise RuntimeError("Degree stratum does not match planted-group size")
        for resource_index, degree in zip(indices, group_degrees):
            assigned[resource_index] = degree
    return assigned


def _profile_groups(condition: DatasetCondition) -> list[int]:
    group_count = min(PROFILE_GROUP_COUNT, condition.resource_count)
    resource_indices = list(range(condition.resource_count))
    random.Random(_condition_seed(condition)).shuffle(resource_indices)
    groups = [0] * condition.resource_count
    for offset, resource_index in enumerate(resource_indices):
        groups[resource_index] = offset % group_count
    return groups


def _common_roles(role_count: int, group_count: int) -> list[int]:
    return list(range(group_count, role_count))


def _coverage_profiles(
    role_count: int,
    degree: int,
    group: int,
    group_count: int,
) -> list[tuple[int, ...]]:
    common = _common_roles(role_count, group_count)
    common_slots = degree - 1
    profile_count = max(1, math.ceil(len(common) / max(1, common_slots * group_count)))
    profiles = []
    for index in range(profile_count):
        selected = [group]
        for offset in range(common_slots):
            common_index = (index * common_slots * group_count) + (group * common_slots) + offset
            selected.append(common[common_index % len(common)])
        profiles.append(tuple(sorted(selected)))
    return profiles


def _random_unique_profile(
    role_count: int,
    degree: int,
    group: int,
    group_count: int,
    rng: random.Random,
) -> tuple[int, ...]:
    common = _common_roles(role_count, group_count)
    selected = []
    available = list(common)
    weights = [
        2 if (role - group_count) % group_count == group else 1
        for role in available
    ]
    for _ in range(degree - 1):
        role = rng.choices(available, weights=weights, k=1)[0]
        selected.append(role)
        selected_index = available.index(role)
        available.pop(selected_index)
        weights.pop(selected_index)
    return tuple(sorted([group, *selected]))


def _profile_pool(
    role_count: int,
    degree: int,
    group: int,
    group_count: int,
    needed: int,
    rng: random.Random,
    globally_seen: set[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    common_count = role_count - group_count
    constrained_capacity = math.comb(common_count, degree - 1)
    maximum = min(needed, constrained_capacity)
    profiles = list(
        dict.fromkeys(
            _coverage_profiles(role_count, degree, group, group_count)
        )
    )
    profiles = [profile for profile in profiles if profile not in globally_seen][:maximum]
    seen = set(profiles) | globally_seen

    while len(profiles) < maximum:
        profile = _random_unique_profile(
            role_count,
            degree,
            group,
            group_count,
            rng,
        )
        if profile in seen:
            continue
        seen.add(profile)
        profiles.append(profile)

    globally_seen.update(profiles)
    return profiles


def _assign_profiles(
    condition: DatasetCondition,
    degrees: Sequence[int],
    groups: Sequence[int],
) -> tuple[list[tuple[int, ...]], int, int, int, float, float, float]:
    indices_by_group_and_degree: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (degree, group) in enumerate(zip(degrees, groups)):
        indices_by_group_and_degree[(group, degree)].append(index)

    base_rng = random.Random(_condition_seed(condition))
    group_count = len(set(groups))
    globally_seen_by_degree: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    pools: dict[tuple[int, int], list[tuple[int, ...]]] = {}
    minimum_counts: dict[tuple[int, int], int] = {}
    for group, degree in sorted(indices_by_group_and_degree):
        indices = indices_by_group_and_degree[(group, degree)]
        key = (group, degree)
        pool = _profile_pool(
            condition.role_count,
            degree,
            group,
            group_count,
            len(indices),
            base_rng,
            globally_seen_by_degree[degree],
        )
        pools[key] = pool
        minimum_counts[key] = min(
            len(pool),
            len(
                _coverage_profiles(
                    condition.role_count,
                    degree,
                    group,
                    group_count,
                )
            ),
        )

    def profiles_for_counts(
        counts: dict[tuple[int, int], int],
    ) -> list[tuple[int, ...]]:
        candidate_assignments: list[tuple[int, ...] | None] = [None] * len(degrees)
        for key in sorted(indices_by_group_and_degree):
            indices = list(indices_by_group_and_degree[key])
            group, degree = key
            random.Random(
                _salted_condition_seed(
                    condition,
                    f"profile-placement:{group}:{degree}",
                )
            ).shuffle(indices)
            templates = pools[key][: counts[key]]
            for offset, resource_index in enumerate(indices):
                candidate_assignments[resource_index] = templates[offset % len(templates)]
        return [profile for profile in candidate_assignments if profile is not None]

    minimum_total = sum(minimum_counts.values())
    maximum_counts = {key: len(pool) for key, pool in pools.items()}
    maximum_total = sum(maximum_counts.values())
    minimum_entropy = _normalized_profile_entropy(profiles_for_counts(minimum_counts))
    maximum_entropy = _normalized_profile_entropy(profiles_for_counts(maximum_counts))
    entropy_range = maximum_entropy - minimum_entropy
    target = PROFILE_DIVERSITY_TARGETS[condition.profile_diversity]
    template_counts = dict(minimum_counts)
    best_counts = dict(template_counts)
    best_distance = float("inf")
    allocation_keys = sorted(template_counts)
    allocation_cursor = 0

    while True:
        candidate_profiles = profiles_for_counts(template_counts)
        candidate_entropy = _normalized_profile_entropy(candidate_profiles)
        normalized_entropy = (
            (candidate_entropy - minimum_entropy) / entropy_range
            if entropy_range > 0
            else 0.0
        )
        distance = abs(normalized_entropy - target)
        if distance < best_distance:
            best_distance = distance
            best_counts = dict(template_counts)
        if template_counts == maximum_counts:
            break
        changed = False
        for _ in allocation_keys:
            key = allocation_keys[allocation_cursor % len(allocation_keys)]
            allocation_cursor += 1
            if template_counts[key] >= maximum_counts[key]:
                continue
            template_counts[key] += 1
            changed = True
            break
        if not changed:
            break

    assignments = profiles_for_counts(best_counts)
    selected_total = sum(best_counts.values())
    selected_entropy = _normalized_profile_entropy(assignments)
    realized_normalized_entropy = (
        (selected_entropy - minimum_entropy) / entropy_range
        if entropy_range > 0
        else 0.0
    )

    return (
        assignments,
        minimum_total,
        maximum_total,
        selected_total,
        realized_normalized_entropy,
        minimum_entropy,
        maximum_entropy,
    )


def _normalized_profile_entropy(profiles: Sequence[tuple[int, ...]]) -> float:
    if len(profiles) <= 1:
        return 0.0
    counts = Counter(profiles)
    entropy = -sum(
        (count / len(profiles)) * math.log(count / len(profiles))
        for count in counts.values()
    )
    return entropy / math.log(len(profiles))


def _normalized_entropy(values: Sequence[int]) -> float:
    if len(values) <= 1:
        return 0.0
    counts = Counter(values)
    entropy = -sum(
        (count / len(values)) * math.log(count / len(values))
        for count in counts.values()
    )
    return entropy / math.log(len(values))


def _jaccard(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    first_set = set(first)
    second_set = set(second)
    return len(first_set & second_set) / len(first_set | second_set)


def _group_jaccard_separation(
    profiles: Sequence[tuple[int, ...]],
    groups: Sequence[int],
) -> tuple[float, float, float]:
    within = []
    between = []
    for first in range(len(profiles)):
        for second in range(first + 1, len(profiles)):
            target = within if groups[first] == groups[second] else between
            target.append(_jaccard(profiles[first], profiles[second]))
    within_mean = sum(within) / len(within)
    between_mean = sum(between) / len(between)
    return within_mean, between_mean, within_mean - between_mean


def _degree_group_cramers_v(
    degrees: Sequence[int],
    groups: Sequence[int],
) -> float:
    degree_values = sorted(set(degrees))
    group_values = sorted(set(groups))
    if len(degree_values) < 2 or len(group_values) < 2:
        return 0.0
    table = {
        (degree, group): sum(
            observed_degree == degree and observed_group == group
            for observed_degree, observed_group in zip(degrees, groups)
        )
        for degree in degree_values
        for group in group_values
    }
    total = len(degrees)
    chi_squared = 0.0
    for degree in degree_values:
        row_total = sum(table[(degree, group)] for group in group_values)
        for group in group_values:
            column_total = sum(table[(other_degree, group)] for other_degree in degree_values)
            expected = row_total * column_total / total
            chi_squared += (table[(degree, group)] - expected) ** 2 / expected
    denominator = total * min(len(degree_values) - 1, len(group_values) - 1)
    return math.sqrt(chi_squared / denominator)


def _dataset_filename(condition: DatasetCondition) -> str:
    density = str(int(round(condition.target_density * 100))).zfill(2)
    return (
        f"role_matrix_{condition.size}_r{condition.resource_count}_"
        f"c{condition.role_count}_d{density}_{condition.profile_diversity}_"
        f"seed{condition.seed}.csv"
    )


def _event_rows(
    profiles: Sequence[tuple[int, ...]],
    groups: Sequence[int],
) -> Iterable[dict[str, str]]:
    origin = datetime(2026, 1, 5, 8, 0, 0)
    event_index = 0
    for resource_index, (profile, group) in enumerate(zip(profiles, groups), start=1):
        resource = f"R-{resource_index:04d}"
        for role_index in profile:
            start = origin + timedelta(minutes=event_index * 5)
            complete = start + timedelta(minutes=4)
            event_index += 1
            yield {
                "Case ID": f"C-{resource_index:06d}",
                "Start Timestamp": start.strftime("%Y/%m/%d %H:%M:%S.%f"),
                "Complete Timestamp": complete.strftime("%Y/%m/%d %H:%M:%S.%f"),
                "Activity": f"Perform Role {role_index + 1:02d} Work",
                "Resource": resource,
                "Role": f"Role {role_index + 1:02d}",
                "Profile Group": f"PG-{group + 1:02d}",
            }


def generate_dataset(condition: DatasetCondition, output_root: Path) -> GeneratedDataset:
    if condition.profile_diversity not in DIVERSITY_LEVELS:
        raise ValueError(f"Unknown profile diversity: {condition.profile_diversity}")
    if not 0 < condition.target_density <= 1:
        raise ValueError("target_density must be in (0, 1]")
    if condition.resource_count < 1 or condition.role_count < 1:
        raise ValueError("resource_count and role_count must be positive")

    groups = _profile_groups(condition)
    degrees = _row_degrees(condition, groups)
    (
        profiles,
        _,
        _,
        _,
        realized_diversity,
        minimum_profile_entropy,
        maximum_profile_entropy,
    ) = _assign_profiles(condition, degrees, groups)
    represented_roles = set(itertools.chain.from_iterable(profiles))
    if represented_roles != set(range(condition.role_count)):
        raise RuntimeError("Generated matrix does not represent every configured role")
    within_jaccard, between_jaccard, group_separation = _group_jaccard_separation(
        profiles, groups
    )
    if group_separation < MIN_GROUP_SEPARATION:
        raise RuntimeError(
            f"Planted-group Jaccard separation is too small for {condition}: "
            f"{group_separation:.4f}"
        )
    target_diversity = PROFILE_DIVERSITY_TARGETS[condition.profile_diversity]
    if abs(realized_diversity - target_diversity) > PROFILE_DIVERSITY_TOLERANCE:
        raise RuntimeError(
            f"Normalized profile entropy misses its target for {condition}: "
            f"target {target_diversity:.2f}, realized {realized_diversity:.4f}"
        )

    output_directory = output_root / condition.size
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / _dataset_filename(condition)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=EVENT_LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(_event_rows(profiles, groups))

    assignments = sum(degrees)
    degree_mean = assignments / len(degrees)
    return GeneratedDataset(
        condition=condition,
        path=output_path,
        assignment_count=assignments,
        realized_density=assignments / (condition.resource_count * condition.role_count),
        unique_profile_count=len(set(profiles)),
        normalized_profile_entropy=_normalized_profile_entropy(profiles),
        baseline_generated_profile_entropy=minimum_profile_entropy,
        maximum_generated_profile_entropy=maximum_profile_entropy,
        target_relative_profile_diversity=target_diversity,
        realized_relative_profile_diversity=realized_diversity,
        profile_group_count=len(set(groups)),
        degree_value_count=len(set(degrees)),
        degree_standard_deviation=math.sqrt(
            sum((degree - degree_mean) ** 2 for degree in degrees) / len(degrees)
        ),
        degree_coefficient_of_variation=(
            math.sqrt(
                sum((degree - degree_mean) ** 2 for degree in degrees)
                / len(degrees)
            )
            / degree_mean
        ),
        normalized_degree_entropy=_normalized_entropy(degrees),
        degree_group_cramers_v=_degree_group_cramers_v(degrees, groups),
        within_group_jaccard=within_jaccard,
        between_group_jaccard=between_jaccard,
        group_jaccard_separation=group_separation,
        sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
    )


def _conditions(config: dict[str, object]) -> Iterable[DatasetCondition]:
    sizes = config["resource_sizes"]
    assert isinstance(sizes, dict)
    for (size, resource_count), role_count, density, diversity, seed in itertools.product(
        sizes.items(),
        config["role_counts"],
        config["target_densities"],
        config["profile_diversities"],
        config["seeds"],
    ):
        yield DatasetCondition(
            size=str(size),
            resource_count=int(resource_count),
            role_count=int(role_count),
            target_density=float(density),
            profile_diversity=str(diversity),
            seed=int(seed),
        )


def _validate_diversity_order(results: Sequence[GeneratedDataset]) -> None:
    grouped: dict[tuple[object, ...], dict[str, GeneratedDataset]] = defaultdict(dict)
    for result in results:
        condition = result.condition
        key = (
            condition.size,
            condition.resource_count,
            condition.role_count,
            condition.target_density,
            condition.seed,
        )
        grouped[key][condition.profile_diversity] = result

    for key, levels in grouped.items():
        if set(levels) != set(DIVERSITY_LEVELS):
            continue
        counts = [levels[level].unique_profile_count for level in DIVERSITY_LEVELS]
        entropies = [levels[level].normalized_profile_entropy for level in DIVERSITY_LEVELS]
        if not counts[0] < counts[1] < counts[2]:
            raise RuntimeError(f"Profile counts are not monotonic for {key}: {counts}")
        if not entropies[0] < entropies[1] < entropies[2]:
            raise RuntimeError(f"Profile entropies are not monotonic for {key}: {entropies}")


def generate_factorial_experiment(
    output_root: Path,
    config_path: Path = CONFIG_PATH,
) -> list[GeneratedDataset]:
    config = load_config(config_path)
    results = [generate_dataset(condition, output_root) for condition in _conditions(config)]
    _validate_diversity_order(results)

    manifest_path = output_root / "manifest.csv"
    rows = [result.manifest_row(output_root) for result in results]
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory that will receive generated event logs and manifest.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Experiment configuration JSON",
    )
    args = parser.parse_args()
    results = generate_factorial_experiment(args.output, args.config)
    print(f"Generated {len(results)} event logs in {args.output}")


if __name__ == "__main__":
    main()
