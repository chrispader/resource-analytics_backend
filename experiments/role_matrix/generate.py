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
    profile_group_count: int
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
                "profile_group_count": self.profile_group_count,
                "sha256": self.sha256,
            }
        )
        return row


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    with path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def _condition_seed(condition: DatasetCondition) -> int:
    key = (
        f"{condition.resource_count}:{condition.role_count}:"
        f"{condition.target_density:.4f}:{condition.seed}"
    )
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def _row_degrees(condition: DatasetCondition) -> list[int]:
    target_assignments = round(
        condition.resource_count * condition.role_count * condition.target_density
    )
    target_assignments = min(
        condition.resource_count * condition.role_count,
        max(condition.resource_count, target_assignments),
    )
    base_degree, remainder = divmod(target_assignments, condition.resource_count)
    degrees = [base_degree + (index < remainder) for index in range(condition.resource_count)]
    random.Random(_condition_seed(condition)).shuffle(degrees)
    return degrees


def _profile_groups(condition: DatasetCondition) -> list[int]:
    group_count = min(5, condition.resource_count)
    resource_indices = list(range(condition.resource_count))
    random.Random(_condition_seed(condition)).shuffle(resource_indices)
    groups = [0] * condition.resource_count
    for offset, resource_index in enumerate(resource_indices):
        groups[resource_index] = offset % group_count
    return groups


def _affinity_roles(role_count: int, group: int, group_count: int) -> list[int]:
    return [role for role in range(role_count) if role % group_count == group]


def _coverage_profiles(
    role_count: int,
    degree: int,
    group: int,
    group_count: int,
) -> list[tuple[int, ...]]:
    affinity = _affinity_roles(role_count, group, group_count)
    other_roles = [role for role in range(role_count) if role not in affinity]
    profile_count = max(1, math.ceil(len(affinity) / degree))
    profiles = []
    for index in range(profile_count):
        selected = affinity[index * degree : (index + 1) * degree]
        fill_candidates = affinity + other_roles
        fill_offset = index * degree
        while len(selected) < degree:
            candidate = fill_candidates[fill_offset % len(fill_candidates)]
            fill_offset += 1
            if candidate not in selected:
                selected.append(candidate)
        profiles.append(tuple(sorted(selected)))
    return profiles


def _random_unique_profile(
    role_count: int,
    degree: int,
    group: int,
    group_count: int,
    rng: random.Random,
) -> tuple[int, ...]:
    affinity = _affinity_roles(role_count, group, group_count)
    weights = [4 if role in affinity else 1 for role in range(role_count)]
    available = list(range(role_count))
    selected = []
    for _ in range(degree):
        role = rng.choices(available, weights=weights, k=1)[0]
        selected.append(role)
        selected_index = available.index(role)
        available.pop(selected_index)
        weights.pop(selected_index)
    return tuple(sorted(selected))


def _profile_pool(
    role_count: int,
    degree: int,
    group: int,
    group_count: int,
    needed: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    maximum = min(needed, math.comb(role_count, degree))
    profiles = list(
        dict.fromkeys(
            _coverage_profiles(role_count, degree, group, group_count)
        )
    )[:maximum]
    seen = set(profiles)

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

    return profiles


def _template_count(
    level: str,
    group_size: int,
    role_count: int,
    degree: int,
    group_count: int,
) -> int:
    maximum = min(group_size, math.comb(role_count, degree))
    affinity_size = math.ceil(role_count / group_count)
    minimum = min(maximum, max(1, math.ceil(affinity_size / degree)))
    if level == "low":
        return minimum
    if level == "medium":
        return min(maximum, max(minimum, round(math.sqrt(minimum * maximum))))
    if level == "high":
        return maximum
    raise ValueError(f"Unknown profile diversity: {level}")


def _assign_profiles(
    condition: DatasetCondition,
    degrees: Sequence[int],
    groups: Sequence[int],
) -> list[tuple[int, ...]]:
    indices_by_group_and_degree: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (degree, group) in enumerate(zip(degrees, groups)):
        indices_by_group_and_degree[(group, degree)].append(index)

    assignments: list[tuple[int, ...] | None] = [None] * len(degrees)
    base_rng = random.Random(_condition_seed(condition))
    group_count = len(set(groups))
    for group, degree in sorted(indices_by_group_and_degree):
        indices = indices_by_group_and_degree[(group, degree)]
        pool = _profile_pool(
            condition.role_count,
            degree,
            group,
            group_count,
            len(indices),
            base_rng,
        )
        template_count = _template_count(
            condition.profile_diversity,
            len(indices),
            condition.role_count,
            degree,
            group_count,
        )
        templates = pool[:template_count]
        for offset, resource_index in enumerate(indices):
            assignments[resource_index] = templates[offset % len(templates)]

    return [profile for profile in assignments if profile is not None]


def _normalized_profile_entropy(profiles: Sequence[tuple[int, ...]]) -> float:
    if len(profiles) <= 1:
        return 0.0
    counts = Counter(profiles)
    entropy = -sum(
        (count / len(profiles)) * math.log(count / len(profiles))
        for count in counts.values()
    )
    return entropy / math.log(len(profiles))


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

    degrees = _row_degrees(condition)
    groups = _profile_groups(condition)
    profiles = _assign_profiles(condition, degrees, groups)
    represented_roles = set(itertools.chain.from_iterable(profiles))
    if represented_roles != set(range(condition.role_count)):
        raise RuntimeError("Generated matrix does not represent every configured role")

    output_directory = output_root / condition.size
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / _dataset_filename(condition)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=EVENT_LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(_event_rows(profiles, groups))

    assignments = sum(degrees)
    return GeneratedDataset(
        condition=condition,
        path=output_path,
        assignment_count=assignments,
        realized_density=assignments / (condition.resource_count * condition.role_count),
        unique_profile_count=len(set(profiles)),
        normalized_profile_entropy=_normalized_profile_entropy(profiles),
        profile_group_count=len(set(groups)),
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
