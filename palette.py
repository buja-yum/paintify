"""Color palette extraction and region-to-palette mapping."""

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
    # Repeat samples proportionally to area for weighted K-means
    # Normalize areas to reasonable repeat counts
    max_repeats = 10
    norm_areas = (region_areas / region_areas.max() * max_repeats).astype(int)
    norm_areas = np.clip(norm_areas, 1, max_repeats)

    samples = []
    for color, repeats in zip(region_colors, norm_areas):
        for _ in range(repeats):
            samples.append(color)
    samples = np.array(samples, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.5)
    _, best_labels, centers = cv2.kmeans(
        samples, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    palette_lab = centers.astype(np.float64)

    # Stage 3: Deduplicate similar palette colors
    palette_lab = _deduplicate_palette(palette_lab, min_delta=5.0)
    n_colors = len(palette_lab)

    # Stage 4: Sort palette by hue
    palette_lab_uint8 = np.clip(palette_lab, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    palette_bgr = cv2.cvtColor(palette_lab_uint8, cv2.COLOR_LAB2BGR)
    palette_hsv = cv2.cvtColor(palette_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    palette_rgb_arr = cv2.cvtColor(palette_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    # Sort: low saturation (grays) at end, rest by hue
    sort_keys = []
    for i, hsv in enumerate(palette_hsv):
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        is_achromatic = s < 30
        sort_keys.append((1 if is_achromatic else 0, h, -v))

    sort_order = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
    palette_lab = palette_lab[sort_order]
    palette_rgb = palette_rgb_arr[sort_order]

    # Stage 5: Map each region to nearest palette color (1-based index)
    region_to_color = {}
    for i, (rid, rc) in enumerate(zip(region_ids, region_colors)):
        dists = np.sqrt(np.sum((palette_lab - rc.astype(np.float64)) ** 2, axis=1))
        best_idx = int(np.argmin(dists)) + 1  # 1-based
        region_to_color[rid] = best_idx

    return palette_rgb.astype(np.uint8), region_to_color, palette_lab


def _deduplicate_palette(palette_lab: np.ndarray, min_delta: float) -> np.ndarray:
    """Remove near-duplicate palette colors."""
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
                    # Merge j into i (average)
                    palette_lab[keep[i]] = (ci + cj) / 2
                    keep.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break

    return palette_lab[keep]
