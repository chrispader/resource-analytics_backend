"""Dataset generation utilities for the Resource x Role matrix experiment."""

from .generate import (
    DatasetCondition,
    GeneratedDataset,
    generate_dataset,
    generate_factorial_experiment,
)

__all__ = [
    "DatasetCondition",
    "GeneratedDataset",
    "generate_dataset",
    "generate_factorial_experiment",
]
