# Plan: Evaluation Suite for Resource × Role Matrix Quality Metrics

## Context

The project contains a **Resource × Role Matrix** visualization created with **Plotly in Python**. The matrix shows relationships between process-mining resources and roles:

- Rows represent resources.
- Columns represent roles.
- Cells indicate whether a resource has a role, usually as a binary value `0` or `1`.

The goal is to add a programmatic evaluation suite that computes selected visualization quality metrics for the matrix. The evaluation should support bachelor-thesis documentation by making the matrix layout and visual encoding measurable and reproducible.

The evaluation should not claim that the visualization is universally “good”. Instead, it should measure selected structural and perceptual properties that are relevant for matrix readability and pattern recognition.

---

## Main Objective

Implement a Python-based evaluation suite for the Resource × Role Matrix that can:

1. Build or receive a clean matrix data model independent from Plotly.
2. Evaluate matrix quality metrics for the current ordering.
3. Optionally evaluate alternative row/column orderings.
4. Evaluate color discriminability of the visual encoding.
5. Export results as CSV and/or JSON for thesis documentation.
6. Provide reproducible results using fixed random seeds where randomness is involved.

---

## Important Design Principle

Evaluation must be based on the **matrix data model**, not on the Plotly figure object.

The expected pipeline should be:

```text
event log / resource-role data
→ ResourceRoleMatrix data model
→ quality metric evaluation
→ Plotly visualization
```

Do not inspect the Plotly figure for basic structural metrics. Plotly rendering should only be relevant if later evaluating pixel-level issues such as label overlap or exported image quality.

---

## Proposed File Structure

Create or adapt a structure similar to this:

```text
evaluation/
  __init__.py
  matrix_model.py
  metrics.py
  color_metrics.py
  ordering.py
  evaluate.py
  export.py
  results/
    matrix_quality_results.csv
    matrix_quality_results.json
```

If the project already has a better structure, integrate these modules accordingly instead of creating a separate top-level folder.

---

## Step 1: Define the Matrix Data Model

Create a small data model that represents the matrix independently from Plotly.

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ResourceRoleMatrix:
    resources: list[str]
    roles: list[str]
    values: np.ndarray  # shape: resources x roles, usually binary 0/1
```

### Requirements

- `values.shape[0]` must match `len(resources)`.
- `values.shape[1]` must match `len(roles)`.
- The model should work for binary matrices first.
- If weighted values already exist, the metric functions should either handle them explicitly or convert them to binary for binary metrics.

### Acceptance Criteria

- Existing matrix generation code can produce a `ResourceRoleMatrix` instance.
- Plotly visualization generation can still use the same data.
- Evaluation code does not depend on Plotly.

---

## Step 2: Implement Structural Matrix Metrics

Implement the following metrics in `metrics.py`.

---

### 2.1 Matrix Density

Density describes how full the matrix is.

Formula:

```text
density = non_empty_cells / total_cells
```

Implementation:

```python
import numpy as np

def density(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0

    return float(np.count_nonzero(values) / values.size)
```

### Interpretation

- Low density: sparse matrix.
- Medium density: often easier for pattern recognition.
- High density: potentially visually overloaded.

### Important Note

Density does **not** change when rows or columns are reordered. It describes dataset/matrix complexity, not ordering quality.

---

### 2.2 Row Similarity Coherence

Row coherence measures whether similar resources are placed close to each other.

Use Jaccard similarity between neighboring rows.

```python
import numpy as np

def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)

    union = np.logical_or(a_bool, b_bool).sum()

    if union == 0:
        return 1.0

    intersection = np.logical_and(a_bool, b_bool).sum()
    return float(intersection / union)


def neighbor_similarity_coherence(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        return 1.0

    similarities = [
        jaccard_similarity(values[index], values[index + 1])
        for index in range(values.shape[0] - 1)
    ]

    return float(np.mean(similarities))
```

For rows:

```python
row_coherence = neighbor_similarity_coherence(matrix.values)
```

### Interpretation

- Higher score means neighboring resources have more similar role profiles.
- Lower score means similar resources are more scattered.

---

### 2.3 Column Similarity Coherence

Column coherence measures whether similar roles are placed close to each other.

Use the same function, but transpose the matrix.

```python
column_coherence = neighbor_similarity_coherence(matrix.values.T)
```

### Interpretation

- Higher score means neighboring roles are assigned to similar resource sets.
- Lower score means similar roles are more scattered.

---

### 2.4 Row Fragmentation

Fragmentation measures how scattered filled cells are.

For each row, count the number of separate filled-cell runs.

Example:

```text
0 1 1 0 1 0 1 1
```

This row has three runs:

```text
[1 1], [1], [1 1]
```

Implementation:

```python
import numpy as np

def count_runs(row: np.ndarray) -> int:
    runs = 0
    inside_run = False

    for value in row:
        is_filled = value != 0

        if is_filled and not inside_run:
            runs += 1
            inside_run = True
            continue

        if not is_filled:
            inside_run = False

    return runs


def average_fragmentation(values: np.ndarray) -> float:
    if values.shape[0] == 0:
        return 0.0

    return float(np.mean([count_runs(row) for row in values]))
```

For rows:

```python
row_fragmentation = average_fragmentation(matrix.values)
```

### Interpretation

- Lower fragmentation means filled cells form more compact visual patterns.
- Higher fragmentation means filled cells are more scattered.

---

### 2.5 Column Fragmentation

Column fragmentation is the same metric applied to the transposed matrix.

```python
column_fragmentation = average_fragmentation(matrix.values.T)
```

### Interpretation

- Lower column fragmentation means roles form more compact resource patterns.
- Higher column fragmentation means role assignments are visually scattered across resources.

---

## Step 3: Implement Color Discriminability Metrics

Implement these in `color_metrics.py`.

Color metrics evaluate the visual encoding, not the matrix structure.

For a binary matrix, the most important checks are:

1. Filled cell color vs. empty cell color.
2. Cell color vs. background color.
3. Label/text color vs. background color, if labels are evaluated.
4. Optional: multiple colors against each other if the matrix uses more than two colors.

---

### 3.1 WCAG Contrast Ratio

Use contrast ratio for approximate accessibility-oriented separation.

```python
def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    normalized = hex_color.lstrip("#")

    return tuple(
        int(normalized[index:index + 2], 16) / 255
        for index in (0, 2, 4)
    )


def linearize_srgb_channel(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92

    return ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    red, green, blue = hex_to_rgb01(hex_color)

    red_linear = linearize_srgb_channel(red)
    green_linear = linearize_srgb_channel(green)
    blue_linear = linearize_srgb_channel(blue)

    return 0.2126 * red_linear + 0.7152 * green_linear + 0.0722 * blue_linear


def contrast_ratio(hex_a: str, hex_b: str) -> float:
    luminance_a = relative_luminance(hex_a)
    luminance_b = relative_luminance(hex_b)

    lighter = max(luminance_a, luminance_b)
    darker = min(luminance_a, luminance_b)

    return (lighter + 0.05) / (darker + 0.05)
```

### Suggested Interpretation

For text labels:

- `>= 4.5:1`: good for normal text.
- `>= 3:1`: often acceptable for large text or non-text UI elements.
- `< 3:1`: weak contrast.

For matrix cells, do not claim strict WCAG compliance unless evaluating text. Instead, describe it as an accessibility-oriented proxy for visual separability.

---

### 3.2 CIEDE2000 Delta E

Use Delta E to measure perceptual distance between colors.

Dependency:

```bash
pip install scikit-image
```

Implementation:

```python
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000


def hex_to_rgb_array(hex_color: str) -> np.ndarray:
    return np.array(hex_to_rgb01(hex_color), dtype=float).reshape(1, 1, 3)


def delta_e_2000(hex_a: str, hex_b: str) -> float:
    lab_a = rgb2lab(hex_to_rgb_array(hex_a))
    lab_b = rgb2lab(hex_to_rgb_array(hex_b))

    return float(deltaE_ciede2000(lab_a, lab_b)[0, 0])
```

For a palette:

```python
from itertools import combinations
import numpy as np


def palette_discriminability(colors: list[str]) -> dict[str, float]:
    distances = [
        delta_e_2000(color_a, color_b)
        for color_a, color_b in combinations(colors, 2)
    ]

    if not distances:
        return {
            "min_delta_e": 0.0,
            "mean_delta_e": 0.0,
            "max_delta_e": 0.0,
        }

    return {
        "min_delta_e": float(np.min(distances)),
        "mean_delta_e": float(np.mean(distances)),
        "max_delta_e": float(np.max(distances)),
    }
```

### Suggested Interpretation

Use Delta E mainly to compare palettes or check whether colors are clearly separated. Avoid hard universal thresholds unless the thesis cites a specific source for them.

---

### 3.3 Optional: Color-Vision Deficiency Simulation

This is optional but useful if the visualization uses multiple colors.

Dependency:

```bash
pip install colorspacious
```

Implementation:

```python
import numpy as np
from colorspacious import cspace_convert


def rgb01_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(
        f"{round(channel * 255):02x}"
        for channel in rgb
    )


def simulate_cvd_rgb01(
    rgb: tuple[float, float, float],
    deficiency: str,
    severity: int = 100,
) -> tuple[float, float, float]:
    cvd_space = {
        "name": "sRGB1+CVD",
        "cvd_type": deficiency,
        "severity": severity,
    }

    simulated = cspace_convert(np.array(rgb), cvd_space, "sRGB1")
    clipped = np.clip(simulated, 0, 1)

    return tuple(float(value) for value in clipped)


def simulate_cvd_hex(
    hex_color: str,
    deficiency: str,
    severity: int = 100,
) -> str:
    simulated_rgb = simulate_cvd_rgb01(
        hex_to_rgb01(hex_color),
        deficiency=deficiency,
        severity=severity,
    )

    return rgb01_to_hex(simulated_rgb)


def palette_discriminability_under_cvd(colors: list[str]) -> dict[str, dict[str, float]]:
    deficiencies = ["deuteranomaly", "protanomaly", "tritanomaly"]

    result = {}

    for deficiency in deficiencies:
        simulated_colors = [
            simulate_cvd_hex(color, deficiency)
            for color in colors
        ]

        result[deficiency] = palette_discriminability(simulated_colors)

    return result
```

---

## Step 4: Implement Ordering Variants

Implement ordering helpers in `ordering.py` so that matrix variants can be compared.

At minimum evaluate:

1. Current ordering.
2. Degree-based ordering.
3. Similarity-based ordering.
4. Random baseline ordering with fixed seeds.

Alphabetical ordering is optional if labels are meaningful and stable.

---

### 4.1 Apply Explicit Row/Column Order

```python
import numpy as np


def reorder_matrix(
    matrix: ResourceRoleMatrix,
    row_order: list[int],
    column_order: list[int],
) -> ResourceRoleMatrix:
    return ResourceRoleMatrix(
        resources=[matrix.resources[index] for index in row_order],
        roles=[matrix.roles[index] for index in column_order],
        values=matrix.values[np.ix_(row_order, column_order)],
    )
```

---

### 4.2 Degree-Based Ordering

Sort resources by number of assigned roles and roles by number of assigned resources.

```python
def degree_based_ordering(matrix: ResourceRoleMatrix) -> ResourceRoleMatrix:
    row_degrees = matrix.values.sum(axis=1)
    column_degrees = matrix.values.sum(axis=0)

    row_order = list(np.argsort(-row_degrees))
    column_order = list(np.argsort(-column_degrees))

    return reorder_matrix(matrix, row_order, column_order)
```

---

### 4.3 Similarity-Based Greedy Ordering

A simple bachelor-thesis-friendly method:

1. Start with one row.
2. Find the most similar remaining row.
3. Append it.
4. Repeat.

```python
def greedy_similarity_order(values: np.ndarray) -> list[int]:
    item_count = values.shape[0]

    if item_count == 0:
        return []

    remaining = set(range(item_count))
    order = [0]
    remaining.remove(0)

    while remaining:
        current = order[-1]
        next_index = max(
            remaining,
            key=lambda candidate: jaccard_similarity(values[current], values[candidate]),
        )
        order.append(next_index)
        remaining.remove(next_index)

    return order


def similarity_based_ordering(matrix: ResourceRoleMatrix) -> ResourceRoleMatrix:
    row_order = greedy_similarity_order(matrix.values)
    column_order = greedy_similarity_order(matrix.values.T)

    return reorder_matrix(matrix, row_order, column_order)
```

### Important Note

This is a heuristic, not an optimal matrix reordering algorithm. Document it as a simple similarity-based baseline.

---

### 4.4 Random Baseline Ordering

Random baselines are important for scientific validity.

```python
def random_ordering(matrix: ResourceRoleMatrix, seed: int) -> ResourceRoleMatrix:
    rng = np.random.default_rng(seed)

    row_order = list(rng.permutation(len(matrix.resources)))
    column_order = list(rng.permutation(len(matrix.roles)))

    return reorder_matrix(matrix, row_order, column_order)
```

Use repeated random runs, for example:

```python
RANDOM_SEEDS = list(range(100))
```

---

## Step 5: Combine Everything in an Evaluation Function

Create `evaluate.py` with a result model and evaluation functions.

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MatrixEvaluationResult:
    dataset: str
    variant: str
    resource_count: int
    role_count: int
    density: float
    row_coherence: float
    column_coherence: float
    row_fragmentation: float
    column_fragmentation: float
    min_delta_e: float
    mean_delta_e: float
    min_contrast_ratio: float
```

Evaluation function:

```python
def evaluate_resource_role_matrix(
    dataset: str,
    variant: str,
    matrix: ResourceRoleMatrix,
    palette: list[str],
    background_color: str = "#ffffff",
) -> MatrixEvaluationResult:
    color_distances = palette_discriminability(palette)

    contrast_values = [
        contrast_ratio(color, background_color)
        for color in palette
    ]

    return MatrixEvaluationResult(
        dataset=dataset,
        variant=variant,
        resource_count=len(matrix.resources),
        role_count=len(matrix.roles),
        density=density(matrix.values),
        row_coherence=neighbor_similarity_coherence(matrix.values),
        column_coherence=neighbor_similarity_coherence(matrix.values.T),
        row_fragmentation=average_fragmentation(matrix.values),
        column_fragmentation=average_fragmentation(matrix.values.T),
        min_delta_e=color_distances["min_delta_e"],
        mean_delta_e=color_distances["mean_delta_e"],
        min_contrast_ratio=float(np.min(contrast_values)) if contrast_values else 0.0,
    )
```

---

## Step 6: Evaluate Matrix Variants

Create a helper that evaluates all relevant variants for one dataset.

```python
def evaluate_matrix_variants(
    dataset: str,
    matrix: ResourceRoleMatrix,
    palette: list[str],
    random_seed_count: int = 100,
) -> list[MatrixEvaluationResult]:
    results = []

    variants = {
        "current": matrix,
        "degree_based": degree_based_ordering(matrix),
        "similarity_based": similarity_based_ordering(matrix),
    }

    for variant_name, variant_matrix in variants.items():
        results.append(
            evaluate_resource_role_matrix(
                dataset=dataset,
                variant=variant_name,
                matrix=variant_matrix,
                palette=palette,
            )
        )

    for seed in range(random_seed_count):
        random_matrix = random_ordering(matrix, seed=seed)

        results.append(
            evaluate_resource_role_matrix(
                dataset=dataset,
                variant="random",
                matrix=random_matrix,
                palette=palette,
            )
        )

    return results
```

---

## Step 7: Export Results

Use Pandas to export the results.

```python
import pandas as pd


def export_results(
    results: list[MatrixEvaluationResult],
    csv_path: str,
    json_path: str,
) -> None:
    dataframe = pd.DataFrame([result.__dict__ for result in results])

    dataframe.to_csv(csv_path, index=False)
    dataframe.to_json(json_path, orient="records", indent=2)
```

Suggested output paths:

```text
evaluation/results/matrix_quality_results.csv
evaluation/results/matrix_quality_results.json
```

---

## Step 8: Generate Summary Tables

Create grouped summaries for thesis reporting.

```python
def summarize_results(results: list[MatrixEvaluationResult]) -> pd.DataFrame:
    dataframe = pd.DataFrame([result.__dict__ for result in results])

    return dataframe.groupby(["dataset", "variant"]).agg({
        "density": ["mean"],
        "row_coherence": ["mean", "std"],
        "column_coherence": ["mean", "std"],
        "row_fragmentation": ["mean", "std"],
        "column_fragmentation": ["mean", "std"],
        "min_delta_e": ["mean"],
        "mean_delta_e": ["mean"],
        "min_contrast_ratio": ["mean"],
    }).reset_index()
```

### Suggested Table Columns for Thesis

```text
Dataset
Ordering
Density
Row coherence ↑
Column coherence ↑
Row fragmentation ↓
Column fragmentation ↓
Min ΔE2000 ↑
Min contrast ratio ↑
```

Use arrows in the table headers to indicate whether higher or lower values are preferable.

---

## Step 9: Integrate With Existing Plotly Code

The existing Plotly visualization should consume the same `ResourceRoleMatrix` model.

Example:

```python
import plotly.express as px


def create_resource_role_matrix_figure(matrix: ResourceRoleMatrix):
    return px.imshow(
        matrix.values,
        x=matrix.roles,
        y=matrix.resources,
        color_continuous_scale=["#ffffff", "#1f77b4"],
    )
```

Evaluation should be callable before or after figure creation:

```python
matrix = build_resource_role_matrix(event_log)
results = evaluate_matrix_variants(
    dataset="example_log",
    matrix=matrix,
    palette=["#ffffff", "#1f77b4"],
)
fig = create_resource_role_matrix_figure(matrix)
```

---

## Step 10: Scientific Validity Requirements

The evaluation must be implemented and documented in a way that makes the results scientifically defensible.

### 10.1 Define Metric Claims Narrowly

Do not claim:

```text
The visualization is good because the metric is high.
```

Instead claim:

```text
A higher row coherence score indicates that neighboring resources have more similar role profiles, which may support visual identification of resource groups.
```

### 10.2 Use Baselines

Do not report only one metric value. Always compare against alternatives.

Minimum baselines:

1. Current ordering.
2. Degree-based ordering.
3. Similarity-based ordering.
4. Random ordering baseline with repeated seeds.

Optional:

5. Alphabetical ordering.

### 10.3 Use Repeated Random Baselines

Use at least 30 random orderings. Prefer 100 if runtime is acceptable.

Store or document:

```python
RANDOM_SEEDS = list(range(100))
```

### 10.4 Use Multiple Datasets if Possible

Recommended:

1. Small synthetic event log.
2. Medium or large synthetic event log.
3. Realistic or real event log.

For each dataset, report:

- number of resources,
- number of roles,
- number of filled cells,
- density.

### 10.5 Avoid Circular Interpretation

Similarity-based ordering is expected to improve row/column coherence because it is based on similarity.

This is acceptable, but document it.

Use fragmentation as an additional check to see whether the layout also becomes visually less scattered.

### 10.6 Separate Structural Metrics From User Performance

The implemented metrics are objective proxies. They do not prove that users complete tasks faster or more accurately.

Mention this limitation in the thesis.

Suggested wording:

```text
The evaluation is limited to computational quality metrics and does not directly measure user performance. The results should therefore be interpreted as indicators of potential readability and pattern visibility, not as direct evidence of improved human understanding.
```

### 10.7 Document Reproducibility Details

Document:

- dataset source,
- preprocessing,
- matrix construction,
- sorting methods,
- metric formulas,
- color palette,
- background color,
- Python version,
- package versions,
- random seeds.

---

## Final Deliverables

The coding agent should produce:

1. A clean `ResourceRoleMatrix` data model.
2. Metric functions for:
   - density,
   - row coherence,
   - column coherence,
   - row fragmentation,
   - column fragmentation.
3. Color metric functions for:
   - WCAG-style contrast ratio,
   - CIEDE2000 Delta E,
   - optional color-vision deficiency simulation.
4. Ordering functions for:
   - current ordering,
   - degree-based ordering,
   - similarity-based ordering,
   - random baseline ordering.
5. A combined evaluation function.
6. CSV and JSON result export.
7. A summary table generator.
8. Basic tests for all metric functions.
9. Documentation comments explaining what each metric measures and how it should be interpreted.

---

## Testing Requirements

Add tests for at least these cases:

### Density

- Empty matrix returns `0.0`.
- Fully empty matrix returns `0.0`.
- Fully filled matrix returns `1.0`.
- Half-filled matrix returns expected ratio.

### Jaccard Similarity

- Identical rows return `1.0`.
- Completely different rows return `0.0`.
- Partially overlapping rows return expected fraction.
- Two empty rows return `1.0`.

### Fragmentation

- `[0, 0, 0]` returns `0`.
- `[1, 1, 1]` returns `1`.
- `[1, 0, 1]` returns `2`.
- `[0, 1, 1, 0, 1]` returns `2`.

### Ordering

- Reordering preserves matrix shape.
- Reordering preserves the total number of filled cells.
- Random ordering is deterministic for a fixed seed.

### Color Metrics

- Contrast ratio between the same color is `1.0`.
- Contrast ratio between black and white is high.
- Delta E between the same color is `0.0` or approximately `0.0`.

---

## Suggested Thesis Interpretation

The implementation should support results that can be described like this:

```text
The evaluation compares different row and column orderings of the Resource × Role Matrix using lightweight quality metrics. Density describes the general complexity of the matrix, while coherence and fragmentation metrics evaluate whether the ordering supports compact and interpretable visual patterns. Color discriminability metrics evaluate whether the visual encoding provides sufficient perceptual separation between matrix states. The results are interpreted as indicators of potential readability and pattern visibility, rather than as direct proof of improved user performance.
```

---

## Non-Goals

Do not implement the following unless explicitly requested later:

- Full user study infrastructure.
- Pixel-level Plotly screenshot analysis.
- Complex optimal matrix reordering algorithms.
- Claims of universal visualization quality.
- Automatic thesis writing.

