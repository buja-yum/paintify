"""Semantic-aware color palette extraction.

Design principles (from 11_color_priority.md):
  Priority: semantic importance > visual saliency > hue distinctiveness > tonal range > area
  - Subject colors preserved first, background simplified
  - Accent colors (high saturation, distinctive hue) MUST be kept even if small area
  - Neutrals limited: don't let grays/whites dominate the palette
  - Each hue family gets tonal spread (highlight/midtone/shadow)
  - Perceptual dedup: LAB distance, not RGB
"""

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000

TIER_BG = 0
TIER_SECONDARY = 1
TIER_SUBJECT = 2

# Neutral = low saturation in HSV
NEUTRAL_SAT_THRESHOLD = 35
# Max percentage of palette that neutrals can occupy
NEUTRAL_MAX_RATIO = 0.25


def extract_palette(image_bgr: np.ndarray, label_map: np.ndarray,
                    n_colors: int, region_tiers: dict = None,
                    vibrancy: float = 1.3) -> tuple:
    """Extract a semantically-prioritized color palette.

    Returns:
        palette_rgb: np.ndarray of shape (n_colors, 3), dtype uint8
        region_to_color: dict mapping region label -> 1-based palette index
        palette_lab: np.ndarray of shape (n_colors, 3), float64
    """
    # Stage 1: Per-region color statistics
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    unique_labels = np.unique(label_map)

    region_colors_lab = []  # LAB mean
    region_colors_hsv = []  # HSV mean (for saturation/hue analysis)
    region_areas = []
    region_ids = []

    flat_lab = image_lab.reshape(-1, 3)
    flat_hsv = image_hsv.reshape(-1, 3)
    flat_labels = label_map.ravel()

    for lab in unique_labels:
        mask = flat_labels == lab
        pixels_lab = flat_lab[mask]
        pixels_hsv = flat_hsv[mask]

        # Saturation-weighted color: vivid pixels contribute more to the
        # representative color, preventing dull averaging
        sat_weights = pixels_hsv[:, 1]  # S channel as weight
        sat_weights = sat_weights + 10  # floor so unsaturated regions still work
        sw_sum = sat_weights.sum()
        weighted_lab = (pixels_lab * sat_weights[:, np.newaxis]).sum(axis=0) / sw_sum
        weighted_hsv = (pixels_hsv * sat_weights[:, np.newaxis]).sum(axis=0) / sw_sum

        region_colors_lab.append(weighted_lab)
        region_colors_hsv.append(weighted_hsv)
        region_areas.append(int(mask.sum()))
        region_ids.append(lab)

    region_colors_lab = np.array(region_colors_lab, dtype=np.float32)
    region_colors_hsv = np.array(region_colors_hsv, dtype=np.float32)
    region_areas = np.array(region_areas, dtype=np.float32)

    # Stage 2: Classify each region as accent vs neutral
    saturations = region_colors_hsv[:, 1]  # S channel (0-255)
    is_accent = saturations > NEUTRAL_SAT_THRESHOLD

    # Stage 3: Compute semantic weight per region
    # Subject regions get 3x weight, secondary 1.5x, background 1x
    # Accents get additional 2x boost
    weights = np.ones(len(region_ids), dtype=np.float32)
    if region_tiers:
        for i, rid in enumerate(region_ids):
            tier = region_tiers.get(rid, TIER_BG)
            if tier == TIER_SUBJECT:
                weights[i] = 3.0
            elif tier == TIER_SECONDARY:
                weights[i] = 1.5
    # Accent boost
    weights[is_accent] *= 2.0

    # Stage 4: Split into accent and neutral groups
    accent_mask = is_accent
    neutral_mask = ~is_accent

    accent_colors = region_colors_lab[accent_mask]
    accent_weights = weights[accent_mask] * region_areas[accent_mask]
    accent_ids = [region_ids[i] for i in range(len(region_ids)) if accent_mask[i]]

    neutral_colors = region_colors_lab[neutral_mask]
    neutral_weights = weights[neutral_mask] * region_areas[neutral_mask]
    neutral_ids = [region_ids[i] for i in range(len(region_ids)) if neutral_mask[i]]

    # Stage 5: Allocate palette slots
    # Accents get at least 60% if they exist, neutrals capped at NEUTRAL_MAX_RATIO
    n_accent_regions = len(accent_colors)
    n_neutral_regions = len(neutral_colors)

    if n_accent_regions > 0 and n_neutral_regions > 0:
        max_neutral_slots = max(3, int(n_colors * NEUTRAL_MAX_RATIO))
        n_accent_slots = n_colors - max_neutral_slots
        # But don't allocate more accent slots than accent regions
        n_accent_slots = min(n_accent_slots, max(n_accent_regions, n_colors - 3))
        n_neutral_slots = n_colors - n_accent_slots
    elif n_accent_regions > 0:
        n_accent_slots = n_colors
        n_neutral_slots = 0
    else:
        n_accent_slots = 0
        n_neutral_slots = n_colors

    # Stage 6: K-means within each group
    palette_parts = []

    if n_accent_slots > 0 and len(accent_colors) > 0:
        actual_accent = min(n_accent_slots, len(accent_colors))
        accent_palette = _weighted_kmeans(accent_colors, accent_weights, actual_accent)
        palette_parts.append(accent_palette)
        # If we got fewer than allocated, give remainder to neutrals
        n_neutral_slots += (n_accent_slots - actual_accent)

    if n_neutral_slots > 0 and len(neutral_colors) > 0:
        actual_neutral = min(n_neutral_slots, len(neutral_colors))
        neutral_palette = _weighted_kmeans(neutral_colors, neutral_weights, actual_neutral)
        palette_parts.append(neutral_palette)

    if palette_parts:
        palette_lab = np.vstack(palette_parts)
    else:
        # Fallback: shouldn't happen, but just in case
        palette_lab = _weighted_kmeans(region_colors_lab,
                                       weights * region_areas, n_colors)

    # Stage 7: Deduplicate (CIEDE2000 ~3 = just noticeable difference)
    palette_lab = _deduplicate_palette(palette_lab, min_delta=3.0)

    # Stage 8: Luminance balance check
    palette_lab = _ensure_luminance_balance(palette_lab, region_colors_lab, region_areas)

    # Stage 8.5: Vibrancy boost - make colors more vivid and slightly brighter
    palette_lab = _boost_vibrancy(palette_lab,
                                  saturation_boost=1.0 + 0.3 * vibrancy,
                                  brightness_boost=1.0 + 0.08 * vibrancy)

    n_final = len(palette_lab)

    # Stage 9: Sort by hue (chromatic first, then achromatic)
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

    # Stage 10: Map each region to nearest palette color (CIEDE2000)
    # Batch convert palette to standard LAB for vectorized deltaE
    palette_std = np.zeros((len(palette_lab), 3), dtype=np.float64)
    palette_std[:, 0] = palette_lab[:, 0] * 100.0 / 255.0
    palette_std[:, 1] = palette_lab[:, 1] - 128.0
    palette_std[:, 2] = palette_lab[:, 2] - 128.0

    region_to_color = {}
    for i, (rid, rc) in enumerate(zip(region_ids, region_colors_lab)):
        rc_std = np.array([[rc[0] * 100.0 / 255.0, rc[1] - 128.0, rc[2] - 128.0]])
        # Broadcast: compare this region color against all palette colors
        rc_broadcast = np.tile(rc_std, (len(palette_std), 1))
        dists = deltaE_ciede2000(rc_broadcast, palette_std)
        best_idx = int(np.argmin(dists)) + 1
        region_to_color[rid] = best_idx

    return palette_rgb.astype(np.uint8), region_to_color, palette_lab


def _weighted_kmeans(colors: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """K-means clustering with sample weights via repetition."""
    if len(colors) <= k:
        return colors.astype(np.float64)

    # Normalize weights to repeat counts
    max_repeats = 15
    w_norm = weights / (weights.max() + 1e-8)
    repeats = np.clip((w_norm * max_repeats).astype(int), 1, max_repeats)

    samples = []
    for color, r in zip(colors, repeats):
        for _ in range(r):
            samples.append(color)
    samples = np.array(samples, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.5)
    _, _, centers = cv2.kmeans(
        samples, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    return centers.astype(np.float64)


def _boost_vibrancy(palette_lab: np.ndarray,
                    saturation_boost: float = 1.25,
                    brightness_boost: float = 1.08) -> np.ndarray:
    """Boost palette saturation and brightness for a more vivid painting result.

    Works in HSV space:
      - Saturation: multiplied by saturation_boost (clamped to 255)
      - Value (brightness): multiplied by brightness_boost (clamped to 255)

    Neutrals (very low saturation) get only brightness boost, not saturation,
    to keep whites/grays clean instead of tinting them.
    """
    # Convert LAB -> BGR -> HSV
    lab_uint8 = np.clip(palette_lab, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

    for i in range(len(hsv)):
        s = hsv[i, 1]
        v = hsv[i, 2]

        if s > 20:
            # Chromatic: boost both saturation and brightness
            hsv[i, 1] = min(255, s * saturation_boost)
            hsv[i, 2] = min(255, v * brightness_boost)
        else:
            # Neutral: only brighten slightly, don't add color tint
            hsv[i, 2] = min(255, v * brightness_boost)

    # Convert back: HSV -> BGR -> LAB
    hsv_uint8 = np.clip(hsv, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    bgr_out = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2BGR)
    lab_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)

    return lab_out


def _cv_lab_to_std(lab_cv):
    """Convert OpenCV LAB (L:0-255, a:0-255, b:0-255) to standard LAB."""
    return np.array([lab_cv[0] * 100.0 / 255.0,
                     lab_cv[1] - 128.0,
                     lab_cv[2] - 128.0])


def _ciede2000_pair(lab_cv_a, lab_cv_b):
    """CIEDE2000 distance between two OpenCV LAB colors."""
    a = _cv_lab_to_std(lab_cv_a).reshape(1, 3)
    b = _cv_lab_to_std(lab_cv_b).reshape(1, 3)
    return float(deltaE_ciede2000(a, b)[0])


def _deduplicate_palette(palette_lab: np.ndarray, min_delta: float) -> np.ndarray:
    """Remove near-duplicate palette colors using CIEDE2000 perceptual distance."""
    keep = list(range(len(palette_lab)))

    while True:
        merged = False
        n = len(keep)
        for i in range(n):
            for j in range(i + 1, n):
                ci = palette_lab[keep[i]]
                cj = palette_lab[keep[j]]
                delta = _ciede2000_pair(ci, cj)
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


def _ensure_luminance_balance(palette_lab, region_colors, region_areas):
    """Ensure palette covers highlight/midtone/shadow bands."""
    L_values = palette_lab[:, 0]

    has_highlight = np.any(L_values > 170)
    has_midtone = np.any((L_values >= 85) & (L_values <= 170))
    has_shadow = np.any(L_values < 85)

    additions = []

    if not has_highlight:
        bright = region_colors[region_colors[:, 0] > 170]
        if len(bright) > 0:
            bright_mask = region_colors[:, 0] > 170
            best = bright[np.argmax(region_areas[bright_mask])]
            additions.append(best.astype(np.float64))

    if not has_midtone:
        mid_mask = (region_colors[:, 0] >= 85) & (region_colors[:, 0] <= 170)
        mid = region_colors[mid_mask]
        if len(mid) > 0:
            best = mid[np.argmax(region_areas[mid_mask])]
            additions.append(best.astype(np.float64))

    if not has_shadow:
        dark_mask = region_colors[:, 0] < 85
        dark = region_colors[dark_mask]
        if len(dark) > 0:
            best = dark[np.argmax(region_areas[dark_mask])]
            additions.append(best.astype(np.float64))

    if additions:
        palette_lab = np.vstack([palette_lab] + [a.reshape(1, 3) for a in additions])

    return palette_lab
