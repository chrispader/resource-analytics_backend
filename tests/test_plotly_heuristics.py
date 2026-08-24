import json

import pytest

from evaluation.plotly_heuristics import (
    PAPER_URL,
    PUBLISHED_HEURISTIC_RULES,
    STATUS_ADVICE,
    STATUS_PASS,
    STATUS_WARNING,
    evaluate_plotly_figure,
    extract_plotly_features,
)


def findings_by_id(report):
    return {finding.rule_id: finding for finding in report.findings}


def test_catalog_contains_only_the_23_published_heuristics():
    assert len(PUBLISHED_HEURISTIC_RULES) == 23
    assert len({rule.rule_id for rule in PUBLISHED_HEURISTIC_RULES}) == 23


def test_evaluator_accepts_mapping_and_json_string():
    figure = {
        "data": [{"type": "scatter", "mode": "markers", "x": [1, 2], "y": [3, 4]}],
        "layout": {
            "title": {"text": "Example"},
            "xaxis": {"title": {"text": "X"}},
            "yaxis": {"title": {"text": "Y"}},
        },
    }
    from_mapping = evaluate_plotly_figure(figure).to_dict()
    from_json = evaluate_plotly_figure(json.dumps(figure)).to_dict()
    assert from_mapping == from_json


def test_missing_title_and_axis_labels_are_warnings():
    report = evaluate_plotly_figure(
        {"data": [{"type": "bar", "x": ["A", "B"], "y": [1, 2]}], "layout": {}}
    )
    findings = findings_by_id(report)
    assert findings["plot-title"].status == STATUS_WARNING
    assert findings["axis-labels"].status == STATUS_WARNING
    assert findings["axis-labels"].evidence["unlabeled_axes"] == ["xaxis", "yaxis"]


def test_title_and_axis_labels_can_pass():
    report = evaluate_plotly_figure(
        {
            "data": [{"type": "bar", "x": ["A", "B"], "y": [1, 2]}],
            "layout": {
                "title": "Amounts",
                "xaxis": {"title": "Category"},
                "yaxis": {"title": "Amount"},
            },
        }
    )
    findings = findings_by_id(report)
    assert findings["plot-title"].status == STATUS_PASS
    assert findings["axis-labels"].status == STATUS_PASS


def test_hidden_subplot_axes_do_not_require_visible_labels():
    report = evaluate_plotly_figure(
        {
            "data": [
                {"type": "scatter", "x": [0.5], "y": [1], "xaxis": "x", "yaxis": "y"},
                {"type": "heatmap", "z": [[1]], "x": [0.5], "y": ["A"], "xaxis": "x2", "yaxis": "y2"},
                {"type": "bar", "x": [1], "y": ["A"], "xaxis": "x3", "yaxis": "y3"},
            ],
            "layout": {
                "title": "Composite matrix",
                "xaxis": {"visible": False},
                "yaxis": {"showticklabels": False},
                "xaxis2": {"showticklabels": False},
                "yaxis2": {"showticklabels": False},
                "xaxis3": {"showticklabels": False},
                "yaxis3": {"showticklabels": False},
            },
        }
    )

    features, _ = extract_plotly_features(
        {
            "data": [
                {"type": "scatter", "x": [0.5], "y": [1]},
                {"type": "heatmap", "z": [[1]], "x": [0.5], "y": ["A"], "xaxis": "x2", "yaxis": "y2"},
            ],
            "layout": {
                "xaxis": {"visible": False},
                "yaxis": {"showticklabels": False},
                "xaxis2": {"showticklabels": False},
                "yaxis2": {"showticklabels": False},
            },
        }
    )

    findings = findings_by_id(report)
    assert findings["axis-labels"].status == STATUS_PASS
    assert "axis-reference-values" not in findings
    assert "log-scale" not in findings
    assert features.axis_names == ("xaxis", "yaxis", "xaxis2", "yaxis2")
    assert features.unlabeled_axes == ()


def test_numeric_features_respect_each_trace_axis_reference():
    figure = {
        "data": [
            {
                "type": "scatter",
                "x": [1, 100, 10_000],
                "y": [1, 2, 3],
            },
            {
                "type": "bar",
                "x": [1, 2, 3, 4, 5],
                "y": [10, 20, 30, 40, 50],
                "xaxis": "x2",
                "yaxis": "y2",
            },
        ],
        "layout": {
            "xaxis": {"title": "Wide range"},
            "yaxis": {"title": "First values"},
            "xaxis2": {"title": "Compact range"},
            "yaxis2": {"title": "Second values"},
        },
    }
    features, _ = extract_plotly_features(figure)
    findings = findings_by_id(evaluate_plotly_figure(figure))

    assert features.numeric_axes == ("xaxis", "yaxis", "xaxis2", "yaxis2")
    assert features.squeezed_axes == ("xaxis",)
    assert features.excessive_tick_axes == ("xaxis", "yaxis")
    assert findings["axis-reference-values"].status == STATUS_WARNING
    assert findings["log-scale"].status == STATUS_WARNING


def test_more_than_four_colors_is_warning():
    report = evaluate_plotly_figure(
        {
            "data": [
                {
                    "type": "scattergeo",
                    "marker": {"color": ["red", "blue", "green", "orange", "purple"]},
                }
            ],
            "layout": {"title": "Locations"},
        }
    )
    finding = findings_by_id(report)["color-count"]
    assert finding.status == STATUS_WARNING
    assert finding.evidence["count"] == 5


def test_pie_rule_matches_only_pie_charts():
    pie = findings_by_id(
        evaluate_plotly_figure(
            {"data": [{"type": "pie", "labels": ["A", "B"], "values": [1, 2]}], "layout": {}}
        )
    )
    bar = findings_by_id(
        evaluate_plotly_figure(
            {"data": [{"type": "bar", "x": ["A", "B"], "y": [1, 2]}], "layout": {}}
        )
    )
    assert pie["replace-pie"].status == STATUS_WARNING
    assert "replace-pie" not in bar


def test_line_rule_warns_for_categorical_connections_and_outliers():
    report = evaluate_plotly_figure(
        {
            "data": [
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": ["very low", "low", "medium", "high", "very high", "maximum"],
                    "y": [1, 1, 1, 1, 1, 100],
                }
            ],
            "layout": {"title": "Line"},
        }
    )
    findings = findings_by_id(report)
    assert findings["line-continuous"].status == STATUS_WARNING
    assert findings["line-outliers"].status == STATUS_WARNING


def test_human_judgment_rules_are_advice_not_unvalidated_scores():
    report = evaluate_plotly_figure(
        {"data": [{"type": "heatmap", "z": [[0, 1]], "colorscale": [[0, "white"], [1, "blue"]]}], "layout": {}}
    )
    findings = findings_by_id(report)
    assert findings["local-contrast"].status == STATUS_ADVICE
    assert findings["color-blind-palette"].status == STATUS_ADVICE
    assert "score" not in report.to_dict()


def test_report_summary_and_sources_are_serializable():
    report = evaluate_plotly_figure(
        {"data": [{"type": "pie", "labels": ["A", "B"], "values": [1, 2]}], "layout": {}}
    )
    payload = report.to_dict()
    assert payload["summary"][STATUS_WARNING] >= 1
    assert payload["summary"][STATUS_ADVICE] >= 1
    assert payload["summary"][STATUS_PASS] >= 0
    assert all(finding["source"] == PAPER_URL for finding in payload["findings"])
    json.dumps(payload)


def test_invalid_plotly_json_is_rejected():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        extract_plotly_features("[]")
