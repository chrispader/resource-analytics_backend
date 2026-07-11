"""Export evaluation results for thesis documentation."""

from pathlib import Path

import pandas as pd

from evaluation.evaluate import MatrixEvaluationResult


def export_results(
    results: list[MatrixEvaluationResult],
    csv_path: str | Path,
    json_path: str | Path,
) -> None:
    dataframe = pd.DataFrame([result.to_dict() for result in results])
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(csv_path, index=False)
    dataframe.to_json(json_path, orient="records", indent=2)


def summarize_results(results: list[MatrixEvaluationResult]) -> pd.DataFrame:
    dataframe = pd.DataFrame([result.to_dict() for result in results])
    return dataframe.groupby(["dataset", "variant"]).agg(
        {
            "density": ["mean"],
            "row_coherence": ["mean", "std"],
            "column_coherence": ["mean", "std"],
            "row_fragmentation": ["mean", "std"],
            "column_fragmentation": ["mean", "std"],
            "blockiness": ["mean", "std"],
            "min_delta_e": ["mean"],
            "mean_delta_e": ["mean"],
            "min_contrast_ratio": ["mean"],
        }
    ).reset_index()
