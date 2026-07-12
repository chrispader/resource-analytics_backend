"""Computer-assisted heuristic evaluation for declarative Plotly figures.

The rule catalog is limited to the heuristics published in Zhu and
Gumieniak (2021), Table 1 and Section 3.3. Rules are matched to extracted plot
features before they are evaluated, following the paper's seven-attribute
method. A rule becomes advice when its premise cannot be established reliably
from Plotly JSON alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
import math
from numbers import Real
from typing import Any, Callable, Mapping, Sequence


PAPER_URL = "https://par.nsf.gov/servlets/purl/10315473"

STATUS_PASS = "pass"
STATUS_WARNING = "warning"
STATUS_ADVICE = "advice"

ASSISTANCE_AUTOMATIC = "automatic-check"
ASSISTANCE_ADVICE = "advice"

CARTESIAN_TRACE_TYPES = {
    "bar",
    "box",
    "candlestick",
    "contour",
    "funnel",
    "heatmap",
    "histogram",
    "histogram2d",
    "histogram2dcontour",
    "image",
    "ohlc",
    "scatter",
    "scattergl",
    "violin",
    "waterfall",
}
LINE_TRACE_TYPES = {"scatter", "scattergl", "scatterpolar", "scattergeo"}
THREE_D_TRACE_TYPES = {"cone", "isosurface", "mesh3d", "scatter3d", "streamtube", "surface", "volume"}


@dataclass(frozen=True)
class RuleLabels:
    """The seven labels used by the paper to connect rules and plots."""

    visual_frames: tuple[str, ...] = ()
    visual_structures: tuple[str, ...] = ()
    visual_unities: tuple[str, ...] = ()
    visual_primitives: tuple[str, ...] = ()
    labeling: tuple[str, ...] = ()
    interaction: tuple[str, ...] = ()
    data_attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlotFeatures:
    frame_count: int
    trace_types: tuple[str, ...]
    cartesian: bool
    title: str | None
    axis_names: tuple[str, ...]
    unlabeled_axes: tuple[str, ...]
    colors: tuple[str, ...]
    categorical_colors: tuple[str, ...]
    uses_color: bool
    uses_size: bool
    uses_pattern: bool
    uses_symbol: bool
    uses_text: bool
    has_hover_details: bool
    has_zoom: bool
    has_filter: bool
    numeric_axes: tuple[str, ...]
    excessive_tick_axes: tuple[str, ...]
    squeezed_axes: tuple[str, ...]
    three_d_dimensions: tuple[int, ...]
    line_trace_indices: tuple[int, ...]
    discontinuous_line_indices: tuple[int, ...]
    outlier_line_indices: tuple[int, ...]
    pie_trace_indices: tuple[int, ...]
    stacked_bar: bool
    multivariate: bool


@dataclass(frozen=True)
class HeuristicFinding:
    rule_id: str
    heuristic: str
    assistance: str
    status: str
    message: str
    recommendation: str | None
    labels: RuleLabels
    evidence: dict[str, Any]
    source: str = PAPER_URL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HeuristicReport:
    findings: tuple[HeuristicFinding, ...]
    feature_summary: dict[str, Any]

    @property
    def summary(self) -> dict[str, int]:
        return {
            status: sum(finding.status == status for finding in self.findings)
            for status in (STATUS_WARNING, STATUS_ADVICE, STATUS_PASS)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "feature_summary": self.feature_summary,
            "findings": [finding.to_dict() for finding in self.findings],
        }


Evaluator = Callable[[PlotFeatures, Mapping[str, Any]], tuple[str, str, str | None, dict[str, Any]]]
Matcher = Callable[[PlotFeatures], bool]


@dataclass(frozen=True)
class HeuristicRule:
    rule_id: str
    heuristic: str
    assistance: str
    labels: RuleLabels
    matches: Matcher
    evaluate: Evaluator


def _as_figure_dict(figure: str | Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(figure, str):
        decoded = json.loads(figure)
    elif isinstance(figure, Mapping):
        decoded = dict(figure)
    elif hasattr(figure, "to_plotly_json"):
        decoded = figure.to_plotly_json()
    else:
        raise TypeError("figure must be Plotly JSON, a mapping, or a Plotly figure")

    if not isinstance(decoded, dict):
        raise ValueError("Plotly JSON must decode to an object")
    data = decoded.get("data", [])
    layout = decoded.get("layout", {})
    if not isinstance(data, list) or not isinstance(layout, Mapping):
        raise ValueError("Plotly JSON requires a data array and layout object")
    return decoded


def _nonempty_text(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, Mapping):
        return _nonempty_text(value.get("text"))
    return None


def _is_color(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    value = value.strip().lower()
    return (
        value.startswith("#")
        or value.startswith("rgb")
        or value.startswith("hsl")
        or value.startswith("hsv")
        or value.startswith("lab")
        or value.startswith("okl")
        or value in {
            "black", "blue", "brown", "cyan", "gray", "green", "grey",
            "magenta", "orange", "pink", "purple", "red", "white", "yellow",
        }
    )


def _collect_colors(value: Any, key: str | None = None) -> list[str]:
    colors: list[str] = []
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            lowered = str(child_key).lower()
            if "color" in lowered or lowered in {"colorscale", "fill"}:
                colors.extend(_collect_colors(child, lowered))
            elif isinstance(child, (Mapping, list, tuple)):
                colors.extend(_collect_colors(child, lowered))
    elif isinstance(value, (list, tuple)):
        for child in value:
            if isinstance(child, (list, tuple)) and len(child) >= 2 and _is_color(child[-1]):
                colors.append(str(child[-1]).strip())
            elif key and ("color" in key or key in {"colorscale", "fill"}):
                colors.extend(_collect_colors(child, key))
            elif isinstance(child, (Mapping, list, tuple)):
                colors.extend(_collect_colors(child, key))
    elif key and ("color" in key or key in {"colorscale", "fill"}) and _is_color(value):
        colors.append(str(value).strip())
    return colors


def _is_neutral_color(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"black", "gray", "grey", "white"}:
        return True
    if normalized.startswith("#"):
        hexadecimal = normalized[1:]
        if len(hexadecimal) in {3, 4}:
            hexadecimal = "".join(character * 2 for character in hexadecimal[:3])
        elif len(hexadecimal) in {6, 8}:
            hexadecimal = hexadecimal[:6]
        else:
            return False
        try:
            channels = [int(hexadecimal[index:index + 2], 16) for index in (0, 2, 4)]
        except ValueError:
            return False
        return max(channels) - min(channels) <= 12
    if normalized.startswith("rgb"):
        try:
            channels = [float(part.strip()) for part in normalized[normalized.index("(") + 1:normalized.index(")")].split(",")[:3]]
        except (ValueError, IndexError):
            return False
        return max(channels) - min(channels) <= 12
    return False


def _numeric_values(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result = []
    for item in value:
        if isinstance(item, Real) and not isinstance(item, bool) and math.isfinite(float(item)):
            result.append(float(item))
    return result


def _is_date_like(value: Any) -> bool:
    if isinstance(value, (date, datetime)):
        return True
    if not isinstance(value, str):
        return False
    candidate = value.strip().replace("Z", "+00:00")
    try:
        datetime.fromisoformat(candidate)
        return True
    except ValueError:
        return False


def _line_trace(trace: Mapping[str, Any]) -> bool:
    trace_type = str(trace.get("type", "scatter")).lower()
    if trace_type not in LINE_TRACE_TYPES:
        return False
    mode = str(trace.get("mode", "lines" if trace_type != "scatter" else "markers"))
    return "lines" in mode


def _has_iqr_outlier(values: list[float]) -> bool:
    if len(values) < 4:
        return False
    ordered = sorted(values)

    def median(items: list[float]) -> float:
        middle = len(items) // 2
        if len(items) % 2:
            return items[middle]
        return (items[middle - 1] + items[middle]) / 2

    midpoint = len(ordered) // 2
    lower = ordered[:midpoint]
    upper = ordered[midpoint + (len(ordered) % 2):]
    q1, q3 = median(lower), median(upper)
    iqr = q3 - q1
    if iqr == 0:
        return any(value != q1 for value in ordered)
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return any(value < lower_fence or value > upper_fence for value in ordered)


def extract_plotly_features(figure: str | Mapping[str, Any] | Any) -> tuple[PlotFeatures, dict[str, Any]]:
    """Extract the plot labels needed by the published heuristic catalog."""
    figure_dict = _as_figure_dict(figure)
    data = figure_dict.get("data", [])
    layout = dict(figure_dict.get("layout", {}))
    config = figure_dict.get("config", {})
    if not isinstance(config, Mapping):
        config = {}

    traces = [trace for trace in data if isinstance(trace, Mapping)]
    trace_types = tuple(str(trace.get("type", "scatter")).lower() for trace in traces)
    cartesian = any(trace_type in CARTESIAN_TRACE_TYPES for trace_type in trace_types)
    title = _nonempty_text(layout.get("title"))

    referenced_axes = []
    for trace, trace_type in zip(traces, trace_types):
        if trace_type not in CARTESIAN_TRACE_TYPES:
            continue
        for coordinate in ("x", "y"):
            reference = str(trace.get(f"{coordinate}axis", coordinate))
            axis_name = f"{coordinate}axis{reference[1:]}"
            if axis_name not in referenced_axes:
                referenced_axes.append(axis_name)
    axis_names = tuple(referenced_axes)
    unlabeled_axes = tuple(
        axis_name for axis_name in axis_names
        if not _nonempty_text((layout.get(axis_name) or {}).get("title"))
    )

    # Plotly serializes whole template palettes even when the figure does not use
    # them. Counting those would manufacture colors that are not rendered.
    explicit_layout = {key: value for key, value in layout.items() if key != "template"}
    colors = tuple(dict.fromkeys(_collect_colors({"data": traces, "layout": explicit_layout})))
    categorical_colors = tuple(color for color in colors if not _is_neutral_color(color))
    uses_color = bool(colors) or any("color" in json.dumps(trace, default=str).lower() for trace in traces)
    uses_size = any(isinstance(trace.get("marker"), Mapping) and "size" in trace["marker"] for trace in traces)
    uses_pattern = any(
        isinstance(trace.get("marker"), Mapping)
        and isinstance(trace["marker"].get("pattern"), Mapping)
        and bool(trace["marker"]["pattern"])
        for trace in traces
    )
    uses_symbol = any(isinstance(trace.get("marker"), Mapping) and "symbol" in trace["marker"] for trace in traces)
    uses_text = bool(title or layout.get("annotations")) or any(
        any(trace.get(key) is not None for key in ("text", "texttemplate", "labels"))
        for trace in traces
    )
    has_hover_details = any(
        trace.get("hovertemplate") not in (None, "")
        or trace.get("hovertext") not in (None, "")
        or trace.get("text") not in (None, "")
        or trace.get("customdata") is not None
        for trace in traces
    )
    static_plot = bool(config.get("staticPlot", False))
    has_zoom = not static_plot and not bool(config.get("scrollZoom") is False and layout.get("dragmode") is False)
    has_filter = any(
        isinstance(menu, Mapping)
        and any(
            isinstance(button, Mapping) and button.get("method") in {"filter", "restyle", "update"}
            for button in menu.get("buttons", [])
        )
        for menu in layout.get("updatemenus", [])
    )

    axis_values: dict[str, list[float]] = {}
    for axis_letter in ("x", "y"):
        values = [value for trace in traces for value in _numeric_values(trace.get(axis_letter))]
        if values:
            axis_values[f"{axis_letter}axis"] = values
    numeric_axes = tuple(axis_values)

    excessive_tick_axes = []
    for axis_name in numeric_axes:
        axis = layout.get(axis_name) or {}
        tickvals = axis.get("tickvals")
        if isinstance(tickvals, Sequence) and not isinstance(tickvals, (str, bytes)):
            tick_count = len(tickvals)
        else:
            tick_count = len(set(axis_values[axis_name]))
        if tick_count < 4 or tick_count > 12:
            excessive_tick_axes.append(axis_name)

    squeezed_axes = []
    for axis_name, values in axis_values.items():
        positive = [value for value in values if value > 0]
        axis = layout.get(axis_name) or {}
        if positive and max(positive) / min(positive) >= 100 and axis.get("type") != "log":
            squeezed_axes.append(axis_name)

    three_d_dimensions = []
    for trace, trace_type in zip(traces, trace_types):
        if trace_type not in THREE_D_TRACE_TYPES:
            continue
        varying = sum(len(set(_numeric_values(trace.get(axis)))) > 1 for axis in ("x", "y", "z"))
        three_d_dimensions.append(varying)

    line_indices = tuple(index for index, trace in enumerate(traces) if _line_trace(trace))
    discontinuous_lines = []
    outlier_lines = []
    for index in line_indices:
        trace = traces[index]
        x_values = trace.get("x")
        if isinstance(x_values, Sequence) and not isinstance(x_values, (str, bytes)):
            if x_values and not all(
                isinstance(value, Real) and not isinstance(value, bool) or _is_date_like(value)
                for value in x_values
            ):
                discontinuous_lines.append(index)
        if _has_iqr_outlier(_numeric_values(trace.get("y"))):
            outlier_lines.append(index)

    pie_indices = tuple(index for index, trace_type in enumerate(trace_types) if trace_type == "pie")
    bar_count = trace_types.count("bar")
    stacked_bar = bar_count > 1 and str(layout.get("barmode", "group")) in {"stack", "relative"}
    multivariate = any(
        sum(trace.get(key) is not None for key in ("x", "y", "z", "marker", "text", "customdata")) >= 3
        for trace in traces
    )

    features = PlotFeatures(
        frame_count=max(1, len(figure_dict.get("frames", []))),
        trace_types=trace_types,
        cartesian=cartesian,
        title=title,
        axis_names=axis_names,
        unlabeled_axes=unlabeled_axes,
        colors=colors,
        categorical_colors=categorical_colors,
        uses_color=uses_color,
        uses_size=uses_size,
        uses_pattern=uses_pattern,
        uses_symbol=uses_symbol,
        uses_text=uses_text,
        has_hover_details=has_hover_details,
        has_zoom=has_zoom,
        has_filter=has_filter,
        numeric_axes=numeric_axes,
        excessive_tick_axes=tuple(excessive_tick_axes),
        squeezed_axes=tuple(squeezed_axes),
        three_d_dimensions=tuple(three_d_dimensions),
        line_trace_indices=line_indices,
        discontinuous_line_indices=tuple(discontinuous_lines),
        outlier_line_indices=tuple(outlier_lines),
        pie_trace_indices=pie_indices,
        stacked_bar=stacked_bar,
        multivariate=multivariate,
    )
    return features, figure_dict


def _automatic(
    ok: bool,
    pass_message: str,
    warning_message: str,
    recommendation: str | None,
    evidence: dict[str, Any],
) -> tuple[str, str, str | None, dict[str, Any]]:
    return (STATUS_PASS if ok else STATUS_WARNING, pass_message if ok else warning_message, None if ok else recommendation, evidence)


def _advice(message: str, recommendation: str, evidence: dict[str, Any] | None = None):
    return STATUS_ADVICE, message, recommendation, evidence or {}


def _always(_: PlotFeatures) -> bool:
    return True


def _rule_catalog() -> tuple[HeuristicRule, ...]:
    all_primitives = RuleLabels(visual_primitives=("all",))
    color = RuleLabels(visual_primitives=("color",))
    return (
        HeuristicRule("plot-title", "A plot should have a title", ASSISTANCE_AUTOMATIC, RuleLabels(labeling=("title",)), _always,
            lambda f, _: _automatic(bool(f.title), "The plot has a title.", "The plot has no title.", "Add a concise, descriptive plot title.", {"title": f.title})),
        HeuristicRule("axis-labels", "Each axis should have a label", ASSISTANCE_AUTOMATIC, RuleLabels(labeling=("axis",)), lambda f: f.cartesian,
            lambda f, _: _automatic(not f.unlabeled_axes, "All detected axes have labels.", f"{len(f.unlabeled_axes)} detected axes have no label.", "Label every data-bearing axis; hide non-data subplot axes if they are intentionally decorative.", {"unlabeled_axes": list(f.unlabeled_axes)})),
        HeuristicRule("color-count", "Avoid using more than four different colors", ASSISTANCE_AUTOMATIC, color, lambda f: f.uses_color,
            lambda f, _: _automatic(len(f.categorical_colors) <= 4, f"The plot uses {len(f.categorical_colors)} detected categorical colors.", f"The plot uses {len(f.categorical_colors)} detected categorical colors, exceeding the paper's four-color heuristic.", "Reduce the palette or provide grouping and redundant encodings where many categories are necessary.", {"colors": list(f.categorical_colors), "count": len(f.categorical_colors), "threshold": 4})),
        HeuristicRule("ordinal-color", "Avoid using color to encode ordinal data", ASSISTANCE_AUTOMATIC, RuleLabels(visual_primitives=("color",), data_attributes=("ordinal",)), lambda f: f.uses_color,
            lambda f, _: _advice("Plotly JSON does not reliably identify whether a color-mapped variable is ordinal.", "Confirm that ordered values are encoded by position or size rather than hue alone.", {"uses_color": True})),
        HeuristicRule("local-contrast", "Local contrast affects color and gray perception", ASSISTANCE_AUTOMATIC, color, lambda f: f.uses_color,
            lambda f, _: _advice("Perceived color depends on neighboring marks and cannot be established from color tokens alone.", "Inspect adjacent cells and marks at the rendered size for misleading simultaneous-contrast effects.", {"colors": list(f.colors)})),
        HeuristicRule("color-item-size", "Color perception varies with the size of the colored item", ASSISTANCE_AUTOMATIC, RuleLabels(visual_primitives=("color", "size")), lambda f: f.uses_color and f.uses_size,
            lambda f, _: _advice("The plot uses both color and size encodings.", "Verify color distinctions on the smallest rendered marks; add a redundant encoding if needed.", {"uses_color": True, "uses_size": True})),
        HeuristicRule("color-blind-palette", "Use color blind friendly palettes", ASSISTANCE_AUTOMATIC, color, lambda f: f.uses_color,
            lambda f, _: _advice("Color tokens alone do not prove color-vision accessibility across rendered contexts.", "Test the rendered palette under common color-vision deficiencies and use redundant labels or shapes.", {"colors": list(f.colors)})),
        HeuristicRule("secondary-element-color", "Use a lighter color for secondary elements such as frames, grids, and axes", ASSISTANCE_AUTOMATIC, color, lambda f: f.cartesian and f.uses_color,
            lambda f, _: _advice("The plot contains data colors and cartesian framing elements.", "Ensure frames, grids, and axes are visually lighter than the data marks.", {"axis_count": len(f.axis_names)})),
        HeuristicRule("axis-reference-values", "A reasonable number of reference values on a coordinate axis might be between four and twelve", ASSISTANCE_AUTOMATIC, RuleLabels(labeling=("axis",)), lambda f: bool(f.numeric_axes),
            lambda f, _: _automatic(not f.excessive_tick_axes, "Detected numeric axes use between four and twelve reference values.", "Some numeric axes fall outside the suggested range of four to twelve reference values.", "Adjust tick values or tick spacing while retaining task-relevant reference points.", {"axes": list(f.excessive_tick_axes), "suggested_range": [4, 12]})),
        HeuristicRule("log-scale", "If data squeezes toward zero, use a log scale", ASSISTANCE_AUTOMATIC, RuleLabels(labeling=("axis",), data_attributes=("range",)), lambda f: bool(f.numeric_axes),
            lambda f, _: _automatic(not f.squeezed_axes, "No detected numeric axis spans two or more positive orders of magnitude on a linear scale.", "Data spans at least two positive orders of magnitude on a linear axis.", "Consider a logarithmic axis, and explain that transformation to readers.", {"axes": list(f.squeezed_axes), "detection_ratio": 100})),
        HeuristicRule("details-on-demand", "Zoom, filter, and details on demand", ASSISTANCE_AUTOMATIC, RuleLabels(visual_unities=("text",), interaction=("zoom", "filter", "tooltip")), _always,
            lambda f, _: _automatic(f.has_zoom and f.has_hover_details, "Zoom and details on demand are available.", "The JSON does not provide both zoom and details on demand.", "Provide appropriate zoom, filtering, and contextual tooltips for dense or exploratory plots.", {"zoom": f.has_zoom, "filter": f.has_filter, "details_on_demand": f.has_hover_details})),
        HeuristicRule("visual-variable-length", "Ensure visual variable has sufficient length. For example, it is difficult to tell small differences in size", ASSISTANCE_AUTOMATIC, all_primitives, _always,
            lambda f, _: _advice("Rendered mark size and display distance are required to judge discriminability.", "Inspect the rendered plot at its delivery size and avoid encodings whose differences are too small to compare.")),
        HeuristicRule("quantitative-position-size", "Quantitative assessment requires position or size variation", ASSISTANCE_AUTOMATIC, RuleLabels(visual_primitives=("position", "size")), _always,
            lambda f, _: _advice("JSON syntax alone cannot reliably distinguish identifiers from quantitative variables in every trace.", "Encode values that require quantitative comparison using aligned position or size.")),
        HeuristicRule("graphic-dimensionality", "Preserve data to graphic dimensionality. For example, avoid representing one or two-dimensional data in 3D visualizations", ASSISTANCE_AUTOMATIC, RuleLabels(visual_structures=("3D",), data_attributes=("dimension",)), lambda f: bool(f.three_d_dimensions),
            lambda f, _: _automatic(all(dimension >= 3 for dimension in f.three_d_dimensions), "Detected 3D traces vary across three data dimensions.", "A 3D trace represents fewer than three varying data dimensions.", "Use a 2D representation unless the third spatial dimension carries data.", {"varying_dimensions": list(f.three_d_dimensions)})),
        HeuristicRule("prefer-position", "In general, position is the most accurate and effective visual variable. Use position for visual encoding when possible", ASSISTANCE_AUTOMATIC, RuleLabels(visual_primitives=("position",)), _always,
            lambda f, _: _advice("The effectiveness of position depends on the analytical task and data semantics.", "Prefer aligned position for comparisons that require accurate magnitude judgments.")),
        HeuristicRule("integrate-text", "Integrate text wherever relevant", ASSISTANCE_AUTOMATIC, RuleLabels(visual_unities=("text",)), _always,
            lambda f, _: _automatic(f.uses_text, "The plot contains integrated text.", "No title, annotation, trace text, or labels were detected.", "Add concise text where it clarifies context, values, or unusual marks.", {"uses_text": f.uses_text})),
        HeuristicRule("replace-pie", "Consider replacing a pie chart with a bar chart", ASSISTANCE_AUTOMATIC, RuleLabels(visual_structures=("pie chart",)), lambda f: bool(f.pie_trace_indices),
            lambda f, _: (STATUS_WARNING, "The plot contains a pie chart.", "Use a bar chart when accurate comparison of values is important.", {"trace_indices": list(f.pie_trace_indices)})),
        HeuristicRule("line-outliers", "Do not use line plots if wild points are at all common", ASSISTANCE_AUTOMATIC, RuleLabels(visual_structures=("line plot",)), lambda f: bool(f.line_trace_indices),
            lambda f, _: _automatic(not f.outlier_line_indices, "No IQR outliers were detected in line-trace y values.", "One or more line traces contain IQR outliers.", "Inspect whether connecting extreme points implies continuity; consider points, intervals, or annotations.", {"trace_indices": list(f.outlier_line_indices), "method": "1.5 IQR"})),
        HeuristicRule("multivariate-representation", "Multivariate data calls for multivariate representation. Consider replacing a stacked bar chart with multiple bar charts or a grouped bar chart with a color-coded scatterplot", ASSISTANCE_AUTOMATIC, RuleLabels(visual_frames=("multiple views",), visual_structures=("scatterplot", "bar chart"), data_attributes=("multivariate",)), lambda f: f.multivariate and ("bar" in f.trace_types),
            lambda f, _: _automatic(not f.stacked_bar, "No stacked multivariate bar representation was detected.", "A stacked multivariate bar representation was detected.", "Consider small-multiple bars or a color-coded scatterplot, depending on the comparison task.", {"stacked_bar": f.stacked_bar})),
        HeuristicRule("line-continuous", "A line plot can only be used to express a continuous function", ASSISTANCE_AUTOMATIC, RuleLabels(visual_structures=("line plot",), data_attributes=("continuous",)), lambda f: bool(f.line_trace_indices),
            lambda f, _: _automatic(not f.discontinuous_line_indices, "Line-trace x values appear numeric, temporal, or implicit.", "A line trace connects non-temporal categorical x values.", "Use points or bars unless interpolation between adjacent categories is meaningful.", {"trace_indices": list(f.discontinuous_line_indices)})),
        HeuristicRule("data-ink-ratio", "Maximizing data-ink ratio, but within reason", ASSISTANCE_ADVICE, all_primitives, _always,
            lambda f, _: _advice("Data-ink balance requires judgment of the rendered figure.", "Remove decoration that does not support interpretation, while retaining useful structure and context.")),
        HeuristicRule("extraneous-ink", "Remove the extraneous ink", ASSISTANCE_ADVICE, all_primitives, _always,
            lambda f, _: _advice("Extraneous visual elements cannot be identified reliably from JSON attributes alone.", "Review borders, backgrounds, grids, and repeated labels for elements that can be removed without information loss.")),
        HeuristicRule("colormap-data-fit", "Choose colormaps that match the nature of the data", ASSISTANCE_ADVICE, RuleLabels(visual_primitives=("color",), data_attributes=("categorical", "ordinal", "continuous")), lambda f: f.uses_color,
            lambda f, _: _advice("The plot uses color, but Plotly JSON does not consistently declare the semantic type of the mapped variable.", "Use qualitative palettes for categories, sequential palettes for ordered magnitudes, and diverging palettes only around a meaningful midpoint.", {"colors": list(f.colors)})),
    )


PUBLISHED_HEURISTIC_RULES = _rule_catalog()


def evaluate_plotly_figure(figure: str | Mapping[str, Any] | Any) -> HeuristicReport:
    """Evaluate a Plotly figure using the paper's published heuristic set."""
    features, figure_dict = extract_plotly_features(figure)
    findings = []
    for rule in PUBLISHED_HEURISTIC_RULES:
        if not rule.matches(features):
            continue
        status, message, recommendation, evidence = rule.evaluate(features, figure_dict)
        findings.append(
            HeuristicFinding(
                rule_id=rule.rule_id,
                heuristic=rule.heuristic,
                assistance=rule.assistance,
                status=status,
                message=message,
                recommendation=recommendation,
                labels=rule.labels,
                evidence=evidence,
            )
        )

    return HeuristicReport(
        findings=tuple(findings),
        feature_summary={
            "frame_count": features.frame_count,
            "trace_types": list(features.trace_types),
            "axis_count": len(features.axis_names),
            "detected_colors": len(features.categorical_colors),
        },
    )
