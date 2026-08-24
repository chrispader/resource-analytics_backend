"""Color discriminability metrics for matrix visual encoding."""

from dataclasses import asdict, dataclass

from itertools import combinations

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

DEFAULT_FULL_DISCRIMINATION_DELTA_E = 20.0


@dataclass(frozen=True)
class ColorDiscriminabilityResult:
    """Order-independent color checks for one matrix visualization."""

    score: float
    min_delta_e: float
    mean_delta_e: float
    max_delta_e: float
    min_contrast_ratio: float
    mean_contrast_ratio: float
    max_contrast_ratio: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    normalized = hex_color.lstrip("#")
    return tuple(
        int(normalized[index : index + 2], 16) / 255
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


def hex_to_rgb_array(hex_color: str) -> np.ndarray:
    return np.array(hex_to_rgb01(hex_color), dtype=float).reshape(1, 1, 3)


def delta_e_2000(hex_a: str, hex_b: str) -> float:
    lab_a = rgb2lab(hex_to_rgb_array(hex_a))
    lab_b = rgb2lab(hex_to_rgb_array(hex_b))
    return float(deltaE_ciede2000(lab_a, lab_b)[0, 0])


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


def color_discriminability_score(
    role_colors: list[str],
    empty_cell_color: str,
    background_color: str,
    *,
    full_discrimination_delta_e: float = DEFAULT_FULL_DISCRIMINATION_DELTA_E,
) -> float:
    """Return a 0..1 total score for separation between all visible colors.

    Pairwise CIEDE2000 distances are normalized so the configured target and
    above receive full credit. Their harmonic mean makes the score sensitive
    to the weakest color pair instead of allowing it to be hidden by many
    strongly separated pairs.
    """
    if not role_colors or full_discrimination_delta_e <= 0:
        return 0.0

    visible_colors = [*role_colors, empty_cell_color, background_color]
    normalized_distances = [
        min(
            delta_e_2000(color_a, color_b) / full_discrimination_delta_e,
            1.0,
        )
        for color_a, color_b in combinations(visible_colors, 2)
    ]
    if not normalized_distances or any(
        distance <= 0 for distance in normalized_distances
    ):
        return 0.0

    return float(
        len(normalized_distances)
        / sum(1.0 / distance for distance in normalized_distances)
    )


def evaluate_color_discriminability(
    role_colors: list[str],
    empty_cell_color: str,
    background_color: str,
) -> ColorDiscriminabilityResult:
    """Evaluate visible-color differences and background contrast once.

    CIEDE2000 distances cover all colors visible in the matrix. WCAG contrast
    ratios compare every cell color with the configured background. Contrast
    is used as an accessibility-oriented design proxy, not as a claim that
    categorical matrix colors meet WCAG text requirements.
    """
    visible_colors = [*role_colors, empty_cell_color, background_color]
    color_distances = palette_discriminability(visible_colors)
    cell_colors = [*role_colors, empty_cell_color]
    contrast_values = [
        contrast_ratio(color, background_color) for color in cell_colors
    ]

    return ColorDiscriminabilityResult(
        score=color_discriminability_score(
            role_colors,
            empty_cell_color,
            background_color,
        ),
        **color_distances,
        min_contrast_ratio=(
            float(np.min(contrast_values)) if contrast_values else 0.0
        ),
        mean_contrast_ratio=(
            float(np.mean(contrast_values)) if contrast_values else 0.0
        ),
        max_contrast_ratio=(
            float(np.max(contrast_values)) if contrast_values else 0.0
        ),
    )
