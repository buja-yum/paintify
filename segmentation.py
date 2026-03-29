"""Semantic-aware adaptive segmentation engine.

3-tier approach:
  - SUBJECT (tier 2): highest detail, preserve anatomy/structure
  - SECONDARY (tier 1): moderate detail, preserve recognizable shape
  - BACKGROUND (tier 0): aggressive simplification, large regions allowed
"""

import cv2
import numpy as np
from skimage.segmentation import slic

# Tier constants
TIER_BG = 0
TIER_SECONDARY = 1
TIER_SUBJECT = 2


def segment_image(image_bgr: np.ndarray, n_segments: int, difficulty: str) -> np.ndarray:
    """Segment image with semantic understanding of subject, secondary, and background.

    Returns label_map of shape (H, W) with integer region labels.
    """
    h, w = image_bgr.shape[:2]
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # Stage 1: Build semantic tier map (0=bg, 1=secondary, 2=subject)
    print("    Computing semantic tier map...")
    tier_map = _compute_semantic_tiers(image_bgr, image_lab)

    # Stage 2: Preprocessing - lighter smoothing to preserve edges
    image_smoothed = cv2.bilateralFilter(image_lab, d=7, sigmaColor=40, sigmaSpace=40)

    # Stage 3: SLIC with high initial count (we'll merge down per-tier)
    compactness_map = {"easy": 25, "medium": 18, "hard": 12, "expert": 8}
    compactness = compactness_map.get(difficulty, 18)
    overshoot = {"easy": 1.8, "medium": 1.7, "hard": 1.6, "expert": 1.5}
    initial_segments = int(n_segments * overshoot.get(difficulty, 1.6))

    image_rgb_smoothed = cv2.cvtColor(image_smoothed, cv2.COLOR_LAB2RGB)

    print(f"    SLIC: {initial_segments} initial segments, compactness={compactness}...")
    label_map = slic(
        image_rgb_smoothed,
        n_segments=initial_segments,
        compactness=compactness,
        sigma=1,
        enforce_connectivity=True,
        convert2lab=True,
        start_label=0,
    ).astype(np.int32)

    # Stage 4: Classify each region into a tier
    region_tiers = _classify_regions(label_map, tier_map)

    # Stage 5: Compute mean colors
    mean_colors = _compute_mean_colors(image_lab, label_map)

    # Stage 6: Per-tier merging with different thresholds
    print("    Per-tier adaptive merging...")
    total_pixels = h * w

    tier_params = _get_tier_params(difficulty, total_pixels)

    # Merge small regions (per-tier min area)
    label_map, mean_colors, region_tiers = _merge_small_by_tier(
        label_map, mean_colors, region_tiers, tier_params
    )

    # Merge similar adjacent regions (per-tier color threshold)
    label_map, mean_colors, region_tiers = _merge_similar_by_tier(
        label_map, mean_colors, region_tiers, tier_params
    )

    # Relabel sequentially
    label_map = _relabel_sequential(label_map)

    return label_map


def _compute_semantic_tiers(image_bgr: np.ndarray, image_lab: np.ndarray) -> np.ndarray:
    """Build a 3-tier semantic map using saliency, edges, and GrabCut.

    Returns tier_map of shape (H, W): 0=background, 1=secondary, 2=subject.
    """
    h, w = image_bgr.shape[:2]

    # --- Saliency map (spectral residual + fine-grained) ---
    saliency_sr = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal_sr = saliency_sr.computeSaliency(image_bgr)
    sal_sr = (sal_sr * 255).astype(np.uint8) if ok else np.zeros((h, w), np.uint8)

    saliency_fg = cv2.saliency.StaticSaliencyFineGrained_create()
    ok2, sal_fg = saliency_fg.computeSaliency(image_bgr)
    sal_fg = (sal_fg * 255).astype(np.uint8) if ok2 else np.zeros((h, w), np.uint8)

    # Combine saliency maps
    saliency = cv2.addWeighted(sal_sr, 0.4, sal_fg, 0.6, 0)
    saliency = cv2.GaussianBlur(saliency, (31, 31), 0)

    # --- Edge density map ---
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.GaussianBlur(edges.astype(np.float32), (51, 51), 0)
    edge_density = (edge_density / (edge_density.max() + 1e-8) * 255).astype(np.uint8)

    # --- Color distance from mean (original approach, still useful) ---
    mean_color = image_lab.mean(axis=(0, 1)).astype(np.float32)
    diff = image_lab.astype(np.float32) - mean_color
    color_dist = np.sqrt(np.sum(diff ** 2, axis=2))
    color_dist = (color_dist / (color_dist.max() + 1e-8) * 255).astype(np.uint8)
    color_dist = cv2.GaussianBlur(color_dist, (31, 31), 0)

    # --- Combined score ---
    combined = (saliency.astype(np.float32) * 0.45 +
                edge_density.astype(np.float32) * 0.25 +
                color_dist.astype(np.float32) * 0.30)
    combined = (combined / (combined.max() + 1e-8) * 255).astype(np.uint8)

    # --- GrabCut refinement for subject extraction ---
    subject_mask = _grabcut_refine(image_bgr, combined)

    # --- Build 3-tier map ---
    # Tier thresholds on combined score
    tier_map = np.full((h, w), TIER_BG, dtype=np.uint8)

    # Secondary: moderate combined score
    p40 = np.percentile(combined, 40)
    p70 = np.percentile(combined, 70)

    tier_map[combined > p40] = TIER_SECONDARY
    tier_map[combined > p70] = TIER_SUBJECT

    # GrabCut-confirmed foreground boosts to subject
    tier_map[subject_mask > 0] = np.maximum(tier_map[subject_mask > 0], TIER_SUBJECT)

    # Clean up with morphological operations
    kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    # Clean subject tier
    subject_binary = (tier_map == TIER_SUBJECT).astype(np.uint8)
    subject_binary = cv2.morphologyEx(subject_binary, cv2.MORPH_CLOSE, kernel_lg)
    subject_binary = cv2.morphologyEx(subject_binary, cv2.MORPH_OPEN, kernel_sm)

    # Clean secondary tier
    secondary_binary = (tier_map >= TIER_SECONDARY).astype(np.uint8)
    secondary_binary = cv2.morphologyEx(secondary_binary, cv2.MORPH_CLOSE, kernel_lg)

    # Rebuild tier map from cleaned masks
    tier_map = np.full((h, w), TIER_BG, dtype=np.uint8)
    tier_map[secondary_binary > 0] = TIER_SECONDARY
    tier_map[subject_binary > 0] = TIER_SUBJECT

    return tier_map


def _grabcut_refine(image_bgr: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    """Use GrabCut seeded by saliency to get a clean foreground mask."""
    h, w = image_bgr.shape[:2]

    # Seed mask for GrabCut
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # High saliency = probable foreground
    high = np.percentile(saliency, 80)
    low = np.percentile(saliency, 20)

    gc_mask[saliency > high] = cv2.GC_PR_FGD
    gc_mask[saliency < low] = cv2.GC_BGD

    # Definite background at image borders (common heuristic)
    border = max(h, w) // 20
    gc_mask[:border, :] = cv2.GC_BGD
    gc_mask[-border:, :] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD

    # Run GrabCut (limited iterations for speed)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        # Downscale for speed if image is large
        max_gc_dim = 600
        scale = min(max_gc_dim / max(h, w), 1.0)
        if scale < 1.0:
            small_img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
            small_mask = cv2.resize(gc_mask, (int(w * scale), int(h * scale)),
                                    interpolation=cv2.INTER_NEAREST)
            cv2.grabCut(small_img, small_mask, None, bgd_model, fgd_model, 3,
                        cv2.GC_INIT_WITH_MASK)
            gc_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            cv2.grabCut(image_bgr, gc_mask, None, bgd_model, fgd_model, 3,
                        cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # GrabCut can fail on some images; fall back to saliency-only
        return (saliency > np.percentile(saliency, 70)).astype(np.uint8)

    foreground = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(np.uint8)
    return foreground


def _classify_regions(label_map: np.ndarray, tier_map: np.ndarray) -> dict:
    """Assign each region to a tier based on majority vote of its pixels."""
    region_tiers = {}
    for lab in np.unique(label_map):
        region_pixels = tier_map[label_map == lab]
        # Use the tier that covers the most pixels in this region
        counts = np.bincount(region_pixels, minlength=3)
        region_tiers[lab] = int(np.argmax(counts))
    return region_tiers


def _get_tier_params(difficulty: str, total_pixels: int) -> dict:
    """Get per-tier merging parameters based on difficulty."""
    params = {
        "easy": {
            TIER_SUBJECT:   {"min_area_pct": 0.0004, "color_threshold": 8,  "similar_threshold": 10},
            TIER_SECONDARY: {"min_area_pct": 0.0008, "color_threshold": 12, "similar_threshold": 14},
            TIER_BG:        {"min_area_pct": 0.0020, "color_threshold": 18, "similar_threshold": 22},
        },
        "medium": {
            TIER_SUBJECT:   {"min_area_pct": 0.0002, "color_threshold": 6,  "similar_threshold": 8},
            TIER_SECONDARY: {"min_area_pct": 0.0005, "color_threshold": 10, "similar_threshold": 12},
            TIER_BG:        {"min_area_pct": 0.0015, "color_threshold": 15, "similar_threshold": 18},
        },
        "hard": {
            TIER_SUBJECT:   {"min_area_pct": 0.00008, "color_threshold": 4,  "similar_threshold": 6},
            TIER_SECONDARY: {"min_area_pct": 0.00020, "color_threshold": 7,  "similar_threshold": 9},
            TIER_BG:        {"min_area_pct": 0.00100, "color_threshold": 12, "similar_threshold": 16},
        },
        "expert": {
            TIER_SUBJECT:   {"min_area_pct": 0.00004, "color_threshold": 3,  "similar_threshold": 5},
            TIER_SECONDARY: {"min_area_pct": 0.00012, "color_threshold": 5,  "similar_threshold": 7},
            TIER_BG:        {"min_area_pct": 0.00060, "color_threshold": 10, "similar_threshold": 12},
        },
    }

    tier_params = params.get(difficulty, params["medium"])

    # Convert pct to pixel counts
    for tier in tier_params:
        tier_params[tier]["min_area"] = max(
            15, int(total_pixels * tier_params[tier]["min_area_pct"])
        )

    return tier_params


def _compute_mean_colors(image_lab: np.ndarray, label_map: np.ndarray) -> dict:
    """Compute mean LAB color for each region."""
    mean_colors = {}
    flat_labels = label_map.ravel()
    flat_pixels = image_lab.reshape(-1, 3).astype(np.float64)

    for lab in np.unique(label_map):
        mask = flat_labels == lab
        mean_colors[lab] = flat_pixels[mask].mean(axis=0)

    return mean_colors


def _merge_small_by_tier(label_map, mean_colors, region_tiers, tier_params):
    """Merge regions below per-tier minimum area into nearest similar neighbor."""
    kernel = np.ones((3, 3), dtype=np.uint8)

    for _ in range(50):
        areas = {}
        for lab in np.unique(label_map):
            areas[lab] = int(np.sum(label_map == lab))

        small_regions = []
        for lab, area in areas.items():
            if area == 0:
                continue
            tier = region_tiers.get(lab, TIER_BG)
            min_area = tier_params[tier]["min_area"]
            if area < min_area:
                small_regions.append(lab)

        if not small_regions:
            break

        small_regions.sort(key=lambda x: areas.get(x, 0))
        merged_any = False

        for region_id in small_regions:
            if np.sum(label_map == region_id) == 0:
                continue

            mask = (label_map == region_id).astype(np.uint8)
            dilated = cv2.dilate(mask, kernel)
            neighbor_mask = (dilated > 0) & (mask == 0)
            neighbor_labels = np.unique(label_map[neighbor_mask])
            neighbor_labels = neighbor_labels[neighbor_labels != region_id]

            if len(neighbor_labels) == 0:
                continue

            region_color = mean_colors.get(region_id)
            if region_color is None:
                continue

            # Prefer merging into same-tier neighbor
            region_tier = region_tiers.get(region_id, TIER_BG)
            same_tier = [n for n in neighbor_labels if region_tiers.get(n, TIER_BG) == region_tier]
            candidates = same_tier if same_tier else neighbor_labels

            best_dist = float("inf")
            best_neighbor = candidates[0]
            for n in candidates:
                nc = mean_colors.get(n)
                if nc is None:
                    continue
                d = np.sqrt(np.sum((region_color - nc) ** 2))
                if d < best_dist:
                    best_dist = d
                    best_neighbor = n

            # Merge
            old_area = areas.get(region_id, 0)
            neighbor_area = areas.get(best_neighbor, 0)
            nc = mean_colors.get(best_neighbor, region_color)
            total = old_area + neighbor_area
            if total > 0:
                mean_colors[best_neighbor] = (nc * neighbor_area + region_color * old_area) / total

            label_map[label_map == region_id] = best_neighbor
            if region_id in mean_colors:
                del mean_colors[region_id]
            if region_id in region_tiers:
                del region_tiers[region_id]
            merged_any = True

        if not merged_any:
            break

    return label_map, mean_colors, region_tiers


def _merge_similar_by_tier(label_map, mean_colors, region_tiers, tier_params):
    """Merge similar adjacent regions using per-tier color thresholds.

    Subject regions have strict thresholds (preserve detail).
    Background regions have relaxed thresholds (simplify aggressively).
    """
    kernel = np.ones((3, 3), dtype=np.uint8)

    merged = True
    iterations = 0
    while merged and iterations < 40:
        merged = False
        iterations += 1
        current_labels = list(np.unique(label_map))

        for region_id in current_labels:
            if np.sum(label_map == region_id) == 0:
                continue

            tier = region_tiers.get(region_id, TIER_BG)
            threshold = tier_params[tier]["similar_threshold"]

            mask = (label_map == region_id).astype(np.uint8)
            dilated = cv2.dilate(mask, kernel)
            neighbor_mask = (dilated > 0) & (mask == 0)
            neighbor_labels = np.unique(label_map[neighbor_mask])
            neighbor_labels = neighbor_labels[neighbor_labels != region_id]

            # Only merge with same-tier neighbors
            same_tier_neighbors = [n for n in neighbor_labels
                                   if region_tiers.get(n, TIER_BG) == tier]
            if not same_tier_neighbors:
                continue

            region_color = mean_colors.get(region_id)
            if region_color is None:
                continue

            for n in same_tier_neighbors:
                nc = mean_colors.get(n)
                if nc is None:
                    continue
                delta = np.sqrt(np.sum((region_color - nc) ** 2))
                if delta < threshold:
                    area_r = int(np.sum(label_map == region_id))
                    area_n = int(np.sum(label_map == n))
                    total = area_r + area_n
                    if total > 0:
                        mean_colors[region_id] = (
                            region_color * area_r + nc * area_n
                        ) / total
                    label_map[label_map == n] = region_id
                    if n in mean_colors:
                        del mean_colors[n]
                    if n in region_tiers:
                        del region_tiers[n]
                    merged = True
                    break

    return label_map, mean_colors, region_tiers


def _relabel_sequential(label_map: np.ndarray) -> np.ndarray:
    """Relabel regions sequentially starting from 0."""
    unique_labels = np.unique(label_map)
    new_map = np.zeros_like(label_map)
    for new_id, old_id in enumerate(unique_labels):
        new_map[label_map == old_id] = new_id
    return new_map
