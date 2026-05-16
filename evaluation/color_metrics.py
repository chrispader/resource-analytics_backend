"""Color discriminability metrics for matrix visual encoding."""

from itertools import combinations

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab


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
