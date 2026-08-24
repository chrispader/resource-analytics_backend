# Resource Analytics Backend

This FastAPI service processes event logs and creates the data, tables, and Plotly figures used by the Resource Analytics web interface. It is a fork of [`maxscho/resource-analytics_backend`](https://github.com/maxscho/resource-analytics_backend), which itself builds on the original [`MaxVidgof/resource-analytics`](https://github.com/MaxVidgof/resource-analytics) project.

## Resource × Role Matrix

The thesis extension adds an interactive matrix that shows which roles are assigned to each resource. Resources form the rows, roles form the columns, and colored cells mark existing assignments. The backend also provides row-degree, degree-based, similarity-based, alphabetical, and random orderings.

The matrix implementation was developed on `feat/visualization-resource-role-matrix` and merged through [pull request #1](https://github.com/chrispader/resource-analytics_backend/pull/1). Programmatic evaluation is kept in the separate `feat/evaluation-suite` branch and stacked on the current `@chrispader/new-visualizations` branch.

## Input format

Event logs can be uploaded as CSV or XES files. CSV files must contain these columns:

- `Case ID`
- `Start Timestamp`
- `Complete Timestamp`
- `Activity`
- `Resource`
- `Role`

The matrix uses the `Resource` and `Role` columns. Roles are read from the event log; the application does not discover them automatically.

## Local setup

Create a Python environment, install the dependencies, and start the API:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 9090
```

The API is then available at `http://localhost:9090`.

## Programmatic evaluation

The `feat/evaluation-suite` branch adds matrix-data metrics, Plotly design heuristics, and the `/resource_role_matrix_evaluation` endpoint. It also contains a reproducible factorial experiment with three resource counts, three role counts, three density levels, three profile-diversity levels, and five deterministic replicates per condition.

Generate and evaluate all 405 event logs with one command:

```bash
python -m experiments.role_matrix.workflow full \
  --output generated/role-matrix-experiment
```

The workflow records its configuration, input hashes, source hashes, environment information, and result tables. See [`experiments/role_matrix/README.md`](experiments/role_matrix/README.md) for the separate generation and evaluation commands.

Run the backend test suite with:

```bash
PYTHONPATH=. pytest -q
```

## Main files

- `main.py` defines the FastAPI endpoints and session handling.
- `pm.py` contains the event-log analysis and Plotly figure generation.
- `evaluation/` contains the reusable matrix metrics and heuristic checks.
- `experiments/role_matrix/` contains the dataset generator and experiment runner.
- `data/InfoPanel.json` contains the descriptions shown in the frontend.
- `hardcoded/` contains sample event logs.

## Branches used for the thesis

| Branch | Purpose |
| --- | --- |
| `@chrispader/new-visualizations` | Shared base containing the merged visualization work. |
| `feat/visualization-resource-role-matrix` | Matrix visualization implementation; merged into the base branch. |
| `feat/evaluation-suite` | Matrix metrics, heuristics, experiment workflow, and evaluation API. |

Older evaluation branches were consolidated into `feat/evaluation-suite`. They are kept only as historical references and should not be used for new work.
