"""Matrix data model independent from Plotly."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResourceRoleMatrix:
    """Binary resource×role matrix with explicit row/column label order."""

    resources: list[str]
    roles: list[str]
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError("values must be a 2-D array")
        if values.shape[0] != len(self.resources):
            raise ValueError("values row count must match len(resources)")
        if values.shape[1] != len(self.roles):
            raise ValueError("values column count must match len(roles)")
        object.__setattr__(self, "values", values)


def resource_role_matrix_from_mapping(
    resources: list[str],
    roles: list[str],
    mapping: list[list[bool]],
) -> ResourceRoleMatrix:
    """Build a matrix model from API-style row/column labels and boolean mapping."""
    values = np.array(mapping, dtype=np.int8)
    return ResourceRoleMatrix(resources=resources, roles=roles, values=values)
