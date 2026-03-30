"""Color palette extraction with luminance balance and perceptual dedup.

Palette rules (from prompts):
  - 24-30 colors
  - Maintain highlight / midtone / shadow balance
  - Ensure perceptual distinction between all colors
  - Avoid redundant near-identical colors
  - Colors must be easy to differentiate when painting
"""

import cv2
import numpy as np


def extract_palette(image_bgr: np.ndarray, label_map: np.ndarray,
                    n_colors: int) -> tuple:
    """Extract a color palette and map regions to palette colors.

    Returns:
        palette_rgb: np.ndarray of shape (n_colors, 3), dtype uint8, sorted by hue
        region_to_color: dict mapping region label -> 1-based palette index
        palette_lab: np.ndarray of shape (n_colors, 3), float64
    """
    # Stage 1: Per-region mean colors in LAB
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    unique_labels = np.unique(label_map)

    region_colors = []
    region_areas = []
    region_ids = []

    flat_lab = image_lab.reshape(-1, 3)
    flat_labels = label_map.ravel()

    for lab in unique_labels:
        mask = flat_labels == lab
        mean_color = flat_lab[mask].mean(axis=0)
        area = int(mask.sum())
        region_colors.append(mean_color)
        region_areas.append(area)
        region_ids.append(lab)

    region_colors = np.array(region_colors, dtype=np.float32)
    region_areas = np.array(region_areas, dtype=np.float32)

    # Stage 2: K-Means clustering weighted by area
    max_repeats = 10
    norm_areas = (region_areas / region_areas.max() * max_repeats).astype(int)
    norm_areas = np.clip(norm_areas, 1, max_repeats)

    samples = []
    for color, repeats in zip(region_colors, norm_areas):
        for _ in range(repeats):
            samples.append(color)
    samples = np.array(samples, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.5)
    _, _, centers = cv2.kmeans(
        samples, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    palette_lab = centers.astype(np.float64)

    # Stage 3: Deduplicate similar palette colors
    palette_lab = _deduplicate_palette(palette_lab, min_delta=5.0)

    # Stage 4: Luminance balance check - ensure highlight/midtone/shadow coverage
    palette_lab = _ensure_luminance_balance(palette_lab, region_colors, region_areas)

    n_colors = len(palette_lab)

    # Stage 5: Sort palette by hue
    palette_lab_uint8 = np.clip(palette_lab, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    palette_bgr = cv2.cvtColor(palette_lab_uint8, cv2.COLOR_LAB2BGR)
    palette_hsv = cv2.cvtColor(palette_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    palette_rgb_arr = cv2.cvtColor(palette_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    sort_keys = []
    for i, hsv in enumerate(palette_hsv):
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        is_achromatic = s < 30
        sort_keys.append((1 if is_achromatic else 0, h, -v))

    sort_order = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
    palette_lab = palette_lab[sort_order]
    palette_rgb = palette_rgb_arr[sort_order]

    # Stage 6: Map each region to nearest palette color (1-based index)
    region_to_color = {}
    for i, (rid, rc) in enumerate(zip(region_ids, region_colors)):
        dists = np.sqrt(np.sum((palette_lab - rc.astype(np.float64)) ** 2, axis=1))
        best_idx = int(np.argmin(dists)) + 1
        region_to_color[rid] = best_idx

    return palette_rgb.astype(np.uint8), region_to_color, palette_lab


def _deduplicate_palette(palette_lab: np.ndarray, min_delta: float) -> np.ndarray:
    """Remove near-duplicate palette colors (perceptually too similar)."""
    keep = list(range(len(palette_lab)))

    while True:
        merged = False
        n = len(keep)
        for i in range(n):
            for j in range(i + 1, n):
                ci = palette_lab[keep[i]]
                cj = palette_lab[keep[j]]
                delta = np.sqrt(np.sum((ci - cj) ** 2))
                if delta < min_delta:
                    palette_lab[keep[i]] = (ci + cj) / 2
                    keep.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break

    return palette_lab[keep]


def _ensure_luminance_balance(palette_lab: np.ndarray,
                              region_colors: np.ndarray,
                              region_areas: np.ndarray) -> np.ndarray:
    """Ensure the palette covers highlight, midtone, and shadow luminance bands.

    LAB L channel: 0 = black, 100 = white (stored as 0-255 in OpenCV LAB)
    Highlights: L > 170  (~67% brightness)
    Midtones:   L 85-170 (~33-67%)
    Shadows:    L < 85   (~0-33%)

    If any band is missing, find the best representative from region colors
    and add it to the palette.
    """
    L_values = palette_lab[:, 0]  # L channel

    has_highlight = np.any(L_values > 170)
    has_midtone = np.any((L_values >= 85) & (L_values <= 170))
    has_shadow = np.any(L_values < 85)

    additions = []

    if not has_highlight:
        # Find brightest region color
        bright = region_colors[region_colors[:, 0] > 170]
        if len(bright) > 0:
            # Pick the one with largest total area
            bright_mask = region_colors[:, 0] > 170
            areas_bright = region_areas[bright_mask]
            best = bright[np.argmax(areas_bright)]
            additions.append(best.astype(np.float64))

    if not has_midtone:
        mid = region_colors[(region_colors[:, 0] >= 85) & (region_colors[:, 0] <= 170)]
        if len(mid) > 0:
            mid_mask = (region_colors[:, 0] >= 85) & (region_colors[:, 0] <= 170)
            areas_mid = region_areas[mid_mask]
            best = mid[np.argmax(areas_mid)]
            additions.append(best.astype(np.float64))

    if not has_shadow:
        dark = region_colors[region_colors[:, 0] < 85]
        if len(dark) > 0:
            dark_mask = region_colors[:, 0] < 85
            areas_dark = region_areas[dark_mask]
            best = dark[np.argmax(areas_dark)]
            additions.append(best.astype(np.float64))

    if additions:
        palette_lab = np.vstack([palette_lab] + [a.reshape(1, 3) for a in additions])

    return palette_lab
