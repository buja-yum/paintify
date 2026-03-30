"""Semantic-aware adaptive segmentation engine.

Design principles (from prompt2.md):
  Priority: recognizability > paintability > texture > region count
  Difficulty increases detail INSIDE important regions only.
  Background stays simple at ALL difficulty levels.

3-tier approach:
  SUBJECT (tier 2)   - highest density, preserve anatomy/texture/identity
  SECONDARY (tier 1) - moderate density, preserve recognizable shape
  BACKGROUND (tier 0) - aggressively simplified, large regions

Performance: uses vectorized adjacency graph and bincount for O(pixels) merging.
"""

import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import deltaE_ciede2000

TIER_BG = 0
TIER_SECONDARY = 1
TIER_SUBJECT = 2

TARGET_RANGES = {
    "easy": (150, 300),
    "medium": (300, 600),
    "hard": (600, 1000),
    "expert": (1000, 5000),
}


def segment_image(image_bgr: np.ndarray, n_segments: int, difficulty: str) -> np.ndarray:
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # Stage 1: Semantic tier map
    print("    Computing semantic tier map...")
    tier_map = _compute_semantic_tiers(image_bgr, image_lab)

    # Stage 2: Edge strength for merge protection (boosted at high-detail features)
    edge_strength = _compute_edge_strength(image_bgr, image_lab, tier_map)

    # Stage 3: Texture-aware smoothing before SLIC
    # Anisotropic smoothing follows texture flow so SLIC segments align with fur/hair
    image_smoothed = _texture_aware_smooth(image_lab, tier_map)
    image_rgb_smoothed = cv2.cvtColor(image_smoothed, cv2.COLOR_LAB2RGB)

    # Subject-focused density: allocate segments proportionally but weighted
    compactness_map = {"easy": 25, "medium": 18, "hard": 12, "expert": 8}
    compactness = compactness_map.get(difficulty, 18)

    # Overshoot: start with 2x target, merge down
    initial_segments = int(n_segments * 2.0)

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

    # Stage 4: Build efficient data structures
    print("    Building region graph...")
    mean_colors = _compute_mean_colors_fast(image_lab, label_map)
    region_tiers = _classify_regions(label_map, tier_map)
    adjacency = _build_adjacency(label_map)
    areas = _compute_areas(label_map)

    # Stage 5: Compute per-boundary edge strength (cached)
    print("    Computing boundary strengths...")
    boundary_edges = _compute_boundary_edges(label_map, edge_strength)

    # Stage 6: Tiered merging
    target_min, target_max = TARGET_RANGES.get(difficulty, (300, 600))
    target_mid = (target_min + target_max) // 2

    tier_params = _get_tier_params(difficulty)

    # Phase 1: Merge tiny unpaintable regions
    print("    Phase 1: Merging tiny regions...")
    label_map, mean_colors, region_tiers, adjacency, areas, boundary_edges = _merge_tiny(
        label_map, mean_colors, region_tiers, adjacency, areas,
        boundary_edges, tier_params, target_mid
    )
    print(f"      → {len(areas)} regions")

    # Phase 2: Aggressive background simplification
    print("    Phase 2: Simplifying background...")
    label_map, mean_colors, region_tiers, adjacency, areas, boundary_edges = _merge_by_tier(
        label_map, mean_colors, region_tiers, adjacency, areas,
        boundary_edges, tier_params, target_mid, TIER_BG
    )
    print(f"      → {len(areas)} regions")

    # Phase 3: Moderate secondary simplification
    if len(areas) > target_mid:
        print("    Phase 3: Simplifying secondary regions...")
        label_map, mean_colors, region_tiers, adjacency, areas, boundary_edges = _merge_by_tier(
            label_map, mean_colors, region_tiers, adjacency, areas,
            boundary_edges, tier_params, target_mid, TIER_SECONDARY
        )
        print(f"      → {len(areas)} regions")

    # Phase 4: Very conservative subject merging (only if way over target)
    if len(areas) > target_max:
        print("    Phase 4: Light subject cleanup...")
        label_map, mean_colors, region_tiers, adjacency, areas, boundary_edges = _merge_by_tier(
            label_map, mean_colors, region_tiers, adjacency, areas,
            boundary_edges, tier_params, target_max, TIER_SUBJECT
        )
        print(f"      → {len(areas)} regions")

    label_map = _relabel_sequential(label_map)

    # Rebuild region_tiers for the relabeled map
    final_tiers = {}
    # Map old region_tiers to new sequential labels
    old_labels = np.unique(label_map)
    # region_tiers still uses pre-relabel IDs, but _relabel_sequential
    # created a new map. We need tier_map to reclassify.
    for lab in np.unique(label_map):
        pixels = tier_map[label_map == lab]
        counts = np.bincount(pixels, minlength=3)
        final_tiers[int(lab)] = int(np.argmax(counts))

    return label_map, final_tiers


# =============================================================================
# Semantic tier detection
# =============================================================================

def _compute_semantic_tiers(image_bgr, image_lab):
    """Build 3-tier map using rembg (U2-Net) for precise subject detection,
    with edge density for secondary region identification.

    Falls back to saliency + GrabCut if rembg is unavailable.
    """
    h, w = image_bgr.shape[:2]

    # --- Primary subject detection via rembg (deep learning) ---
    subject_mask = _rembg_subject_mask(image_bgr)

    if subject_mask is None:
        # Fallback to legacy saliency + GrabCut
        print("      rembg unavailable, falling back to saliency+GrabCut...")
        subject_mask = _legacy_subject_mask(image_bgr, image_lab)

    # --- Edge density for secondary region detection ---
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.GaussianBlur(edges.astype(np.float32), (51, 51), 0)
    edge_density = (edge_density / (edge_density.max() + 1e-8) * 255).astype(np.uint8)

    # Color distance from mean for secondary detection
    mean_color = image_lab.mean(axis=(0, 1)).astype(np.float32)
    diff = image_lab.astype(np.float32) - mean_color
    color_dist = np.sqrt(np.sum(diff ** 2, axis=2))
    color_dist = (color_dist / (color_dist.max() + 1e-8) * 255).astype(np.uint8)
    color_dist = cv2.GaussianBlur(color_dist, (31, 31), 0)

    # Secondary score: areas with moderate edge density or color distinctiveness
    secondary_score = (edge_density.astype(np.float32) * 0.4 +
                       color_dist.astype(np.float32) * 0.6)
    secondary_score = (secondary_score / (secondary_score.max() + 1e-8) * 255).astype(np.uint8)

    # --- Build 3-tier map ---
    tier_map = np.full((h, w), TIER_BG, dtype=np.uint8)

    # Secondary: moderate score and NOT already subject
    p50 = np.percentile(secondary_score, 50)
    tier_map[secondary_score > p50] = TIER_SECONDARY

    # Subject: from rembg mask (overrides secondary)
    tier_map[subject_mask > 0] = TIER_SUBJECT

    # Morphological cleanup
    kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    subject_binary = (tier_map == TIER_SUBJECT).astype(np.uint8)
    subject_binary = cv2.morphologyEx(subject_binary, cv2.MORPH_CLOSE, kernel_lg)
    subject_binary = cv2.morphologyEx(subject_binary, cv2.MORPH_OPEN, kernel_sm)

    secondary_binary = (tier_map >= TIER_SECONDARY).astype(np.uint8)
    secondary_binary = cv2.morphologyEx(secondary_binary, cv2.MORPH_CLOSE, kernel_lg)

    tier_map = np.full((h, w), TIER_BG, dtype=np.uint8)
    tier_map[secondary_binary > 0] = TIER_SECONDARY
    tier_map[subject_binary > 0] = TIER_SUBJECT

    return tier_map


def _rembg_subject_mask(image_bgr):
    """Use rembg (U2-Net) for accurate foreground/background separation."""
    try:
        from rembg import remove, new_session
        from PIL import Image as PILImage
    except ImportError:
        return None

    try:
        # Convert BGR to RGB PIL image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(image_rgb)

        # rembg returns RGBA with alpha = foreground mask
        result = remove(pil_img, only_mask=True)
        mask = np.array(result)

        # Threshold the soft mask to binary
        _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        return binary_mask.astype(np.uint8)
    except Exception as e:
        print(f"      rembg failed ({e}), falling back...")
        return None


def _legacy_subject_mask(image_bgr, image_lab):
    """Fallback: saliency + GrabCut for subject detection."""
    h, w = image_bgr.shape[:2]

    saliency_sr = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal_sr = saliency_sr.computeSaliency(image_bgr)
    sal_sr = (sal_sr * 255).astype(np.uint8) if ok else np.zeros((h, w), np.uint8)

    saliency_fg = cv2.saliency.StaticSaliencyFineGrained_create()
    ok2, sal_fg = saliency_fg.computeSaliency(image_bgr)
    sal_fg = (sal_fg * 255).astype(np.uint8) if ok2 else np.zeros((h, w), np.uint8)

    saliency = cv2.addWeighted(sal_sr, 0.4, sal_fg, 0.6, 0)
    saliency = cv2.GaussianBlur(saliency, (31, 31), 0)

    # GrabCut refinement
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    high = np.percentile(saliency, 80)
    low = np.percentile(saliency, 20)
    gc_mask[saliency > high] = cv2.GC_PR_FGD
    gc_mask[saliency < low] = cv2.GC_BGD

    border = max(h, w) // 20
    gc_mask[:border, :] = cv2.GC_BGD
    gc_mask[-border:, :] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
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
        return (saliency > np.percentile(saliency, 70)).astype(np.uint8)

    return ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(np.uint8)


# =============================================================================
# Texture flow detection & anisotropic smoothing
# =============================================================================

def _texture_aware_smooth(image_lab, tier_map):
    """Apply different smoothing per tier to guide SLIC along texture flow.

    Subject: light anisotropic smoothing (preserves fur/hair direction)
    Secondary: moderate bilateral smoothing
    Background: strong smoothing (encourages large uniform regions)
    """
    h, w = image_lab.shape[:2]
    result = image_lab.copy()

    # Background: strong smoothing to encourage large flat regions
    bg_mask = (tier_map == TIER_BG)
    bg_smoothed = cv2.bilateralFilter(image_lab, d=11, sigmaColor=60, sigmaSpace=60)

    # Secondary: moderate smoothing
    sec_smoothed = cv2.bilateralFilter(image_lab, d=9, sigmaColor=45, sigmaSpace=45)

    # Subject: minimal smoothing to preserve subtle gradients (lighting, shading)
    # Critical: do NOT destroy brightness variation within same-color fur/skin
    subj_smoothed = cv2.bilateralFilter(image_lab, d=3, sigmaColor=15, sigmaSpace=15)

    # Blend per tier
    bg_3d = np.stack([bg_mask] * 3, axis=-1)
    sec_mask = (tier_map == TIER_SECONDARY)
    sec_3d = np.stack([sec_mask] * 3, axis=-1)
    subj_mask = (tier_map == TIER_SUBJECT)
    subj_3d = np.stack([subj_mask] * 3, axis=-1)

    result[bg_3d] = bg_smoothed[bg_3d]
    result[sec_3d] = sec_smoothed[sec_3d]
    result[subj_3d] = subj_smoothed[subj_3d]

    return result


# =============================================================================
# Edge strength with feature-aware boost
# =============================================================================

def _compute_edge_strength(image_bgr, image_lab, tier_map):
    """Compute edge strength with boosted protection at high-detail subject features.

    High edge density within subject tier (eyes, nose, facial contours) gets
    extra protection to prevent merging across critical features.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    lab_float = image_lab.astype(np.float64)
    lab_grad = np.zeros_like(grad_mag)
    for c in range(3):
        lx = cv2.Sobel(lab_float[:, :, c], cv2.CV_64F, 1, 0, ksize=3)
        ly = cv2.Sobel(lab_float[:, :, c], cv2.CV_64F, 0, 1, ksize=3)
        lab_grad += lx ** 2 + ly ** 2
    lab_grad = np.sqrt(lab_grad)

    combined = grad_mag * 0.4 + lab_grad * 0.6
    combined = combined / (combined.max() + 1e-8)

    # Feature-aware boost: edges within subject tier get 1.5x strength
    # This protects eyes, nose, mouth, facial contours from being merged
    subject_mask = (tier_map == TIER_SUBJECT).astype(np.float32)
    # Find high-edge-density areas within subject (likely facial features)
    subject_edges = combined * subject_mask
    # Localized edge density within subject
    feature_density = cv2.GaussianBlur(subject_edges.astype(np.float32), (21, 21), 0)
    feature_density = feature_density / (feature_density.max() + 1e-8)
    # Boost factor: 1.0 (no boost) to 1.5 (max boost at feature-dense areas)
    boost = 1.0 + 0.5 * feature_density * subject_mask
    combined = combined * boost

    # Re-normalize
    combined = combined / (combined.max() + 1e-8)
    return combined.astype(np.float32)


# =============================================================================
# Fast data structure builders (vectorized, no per-region loops for adjacency)
# =============================================================================

def _compute_mean_colors_fast(image_lab, label_map):
    """Compute mean LAB color per region using vectorized ops."""
    flat_lab = image_lab.reshape(-1, 3).astype(np.float64)
    flat_labels = label_map.ravel()
    max_label = flat_labels.max() + 1

    sums = np.zeros((max_label, 3), dtype=np.float64)
    counts = np.zeros(max_label, dtype=np.float64)

    np.add.at(sums, flat_labels, flat_lab)
    np.add.at(counts, flat_labels, 1)

    counts[counts == 0] = 1  # avoid division by zero
    means = sums / counts[:, np.newaxis]

    result = {}
    for lab in np.unique(flat_labels):
        result[lab] = means[lab]
    return result


def _compute_areas(label_map):
    """Compute area per region using bincount."""
    flat = label_map.ravel()
    bc = np.bincount(flat)
    result = {}
    for lab in np.unique(flat):
        result[lab] = int(bc[lab])
    return result


def _build_adjacency(label_map):
    """Build adjacency dict from label_map using vectorized boundary detection."""
    adjacency = {}

    # Horizontal neighbors
    h_left = label_map[:, :-1].ravel()
    h_right = label_map[:, 1:].ravel()
    diff_h = h_left != h_right
    pairs_h_a = h_left[diff_h]
    pairs_h_b = h_right[diff_h]

    # Vertical neighbors
    v_top = label_map[:-1, :].ravel()
    v_bot = label_map[1:, :].ravel()
    diff_v = v_top != v_bot
    pairs_v_a = v_top[diff_v]
    pairs_v_b = v_bot[diff_v]

    all_a = np.concatenate([pairs_h_a, pairs_v_a])
    all_b = np.concatenate([pairs_h_b, pairs_v_b])

    for a, b in zip(all_a.tolist(), all_b.tolist()):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    # Ensure all labels have an entry
    for lab in np.unique(label_map):
        adjacency.setdefault(int(lab), set())

    return adjacency


def _compute_boundary_edges(label_map, edge_strength):
    """Compute mean edge strength at each region boundary. Vectorized."""
    boundary_edges = {}

    # Horizontal boundaries
    h_left = label_map[:, :-1]
    h_right = label_map[:, 1:]
    diff_h = h_left != h_right
    # Edge strength at boundary = max of left and right pixel
    es_h = np.maximum(edge_strength[:, :-1], edge_strength[:, 1:])

    rows_h, cols_h = np.where(diff_h)
    labels_a = h_left[rows_h, cols_h]
    labels_b = h_right[rows_h, cols_h]
    strengths_h = es_h[rows_h, cols_h]

    # Vertical boundaries
    v_top = label_map[:-1, :]
    v_bot = label_map[1:, :]
    diff_v = v_top != v_bot
    es_v = np.maximum(edge_strength[:-1, :], edge_strength[1:, :])

    rows_v, cols_v = np.where(diff_v)
    labels_c = v_top[rows_v, cols_v]
    labels_d = v_bot[rows_v, cols_v]
    strengths_v = es_v[rows_v, cols_v]

    all_a = np.concatenate([labels_a, labels_c])
    all_b = np.concatenate([labels_b, labels_d])
    all_s = np.concatenate([strengths_h, strengths_v])

    # Accumulate per pair
    pair_sums = {}
    pair_counts = {}

    for a, b, s in zip(all_a.tolist(), all_b.tolist(), all_s.tolist()):
        key = (min(a, b), max(a, b))
        pair_sums[key] = pair_sums.get(key, 0.0) + s
        pair_counts[key] = pair_counts.get(key, 0) + 1

    for key in pair_sums:
        boundary_edges[key] = pair_sums[key] / pair_counts[key]

    return boundary_edges


def _ciede2000(lab_cv_a, lab_cv_b):
    """Compute CIEDE2000 perceptual color distance between two OpenCV LAB colors.

    OpenCV LAB: L=0-255, a=0-255, b=0-255
    scikit-image expects: L=0-100, a=-128..127, b=-128..127
    """
    # Convert OpenCV LAB to standard LAB
    lab_a = np.array([[lab_cv_a[0] * 100.0 / 255.0,
                       lab_cv_a[1] - 128.0,
                       lab_cv_a[2] - 128.0]])
    lab_b = np.array([[lab_cv_b[0] * 100.0 / 255.0,
                       lab_cv_b[1] - 128.0,
                       lab_cv_b[2] - 128.0]])
    return float(deltaE_ciede2000(lab_a, lab_b)[0])


def _classify_regions(label_map, tier_map):
    region_tiers = {}
    for lab in np.unique(label_map):
        pixels = tier_map[label_map == lab]
        counts = np.bincount(pixels, minlength=3)
        region_tiers[lab] = int(np.argmax(counts))
    return region_tiers


def _get_tier_params(difficulty):
    """Per-tier merge parameters.

    Key design:
      - similar_threshold is now CIEDE2000 scale (~2=identical, ~5=noticeable, ~10=obvious)
      - background: high threshold for aggressive simplification
      - subject: very low threshold to preserve detail
      - lightness_protect: minimum L-channel difference that blocks merging
    """
    params = {
        "easy": {
            TIER_SUBJECT:   {"min_area_pct": 0.0005, "similar_threshold": 5,  "edge_protect": 0.30, "lightness_protect": 6},
            TIER_SECONDARY: {"min_area_pct": 0.0010, "similar_threshold": 8,  "edge_protect": 0.22, "lightness_protect": 8},
            TIER_BG:        {"min_area_pct": 0.0025, "similar_threshold": 15, "edge_protect": 0.12, "lightness_protect": 15},
        },
        "medium": {
            TIER_SUBJECT:   {"min_area_pct": 0.0003, "similar_threshold": 4,  "edge_protect": 0.28, "lightness_protect": 4},
            TIER_SECONDARY: {"min_area_pct": 0.0006, "similar_threshold": 6,  "edge_protect": 0.20, "lightness_protect": 6},
            TIER_BG:        {"min_area_pct": 0.0018, "similar_threshold": 12, "edge_protect": 0.10, "lightness_protect": 12},
        },
        "hard": {
            TIER_SUBJECT:   {"min_area_pct": 0.00010, "similar_threshold": 3,  "edge_protect": 0.25, "lightness_protect": 3},
            TIER_SECONDARY: {"min_area_pct": 0.00025, "similar_threshold": 5,  "edge_protect": 0.18, "lightness_protect": 5},
            TIER_BG:        {"min_area_pct": 0.00120, "similar_threshold": 10, "edge_protect": 0.08, "lightness_protect": 10},
        },
        "expert": {
            TIER_SUBJECT:   {"min_area_pct": 0.00005, "similar_threshold": 2,  "edge_protect": 0.22, "lightness_protect": 2},
            TIER_SECONDARY: {"min_area_pct": 0.00015, "similar_threshold": 4,  "edge_protect": 0.15, "lightness_protect": 4},
            TIER_BG:        {"min_area_pct": 0.00080, "similar_threshold": 8,  "edge_protect": 0.06, "lightness_protect": 8},
        },
    }
    return params.get(difficulty, params["medium"])


# =============================================================================
# Merge operations (graph-based, no pixel scanning per merge)
# =============================================================================

def _do_merge(label_map, a, b, mean_colors, region_tiers, adjacency, areas, boundary_edges):
    """Merge region b into region a. Updates all data structures in-place."""
    # Update label map
    label_map[label_map == b] = a

    # Update mean color (area-weighted average)
    area_a = areas.get(a, 0)
    area_b = areas.get(b, 0)
    total = area_a + area_b
    if total > 0 and a in mean_colors and b in mean_colors:
        mean_colors[a] = (mean_colors[a] * area_a + mean_colors[b] * area_b) / total

    # Update area
    areas[a] = total

    # Update adjacency: b's neighbors become a's neighbors
    b_neighbors = adjacency.pop(b, set())
    b_neighbors.discard(a)
    for n in b_neighbors:
        if n in adjacency:
            adjacency[n].discard(b)
            adjacency[n].add(a)
    adjacency.setdefault(a, set()).update(b_neighbors)
    adjacency[a].discard(a)

    # Update boundary edges: transfer b's edges to a
    new_edges = {}
    keys_to_remove = []
    for key, val in boundary_edges.items():
        if b in key:
            keys_to_remove.append(key)
            other = key[0] if key[1] == b else key[1]
            if other == a:
                continue  # internal boundary, discard
            new_key = (min(a, other), max(a, other))
            # Average with existing if present
            if new_key in new_edges:
                new_edges[new_key] = (new_edges[new_key] + val) / 2
            elif new_key in boundary_edges:
                new_edges[new_key] = (boundary_edges[new_key] + val) / 2
            else:
                new_edges[new_key] = val

    for key in keys_to_remove:
        del boundary_edges[key]
    boundary_edges.update(new_edges)

    # Cleanup
    mean_colors.pop(b, None)
    region_tiers.pop(b, None)
    areas.pop(b, None)


def _merge_tiny(label_map, mean_colors, region_tiers, adjacency, areas,
                boundary_edges, tier_params, target_stop):
    """Merge regions below per-tier minimum paintable size."""
    total_pixels = sum(areas.values())

    for _ in range(100):
        if len(areas) <= target_stop:
            break

        # Find tiny regions
        tiny = []
        for lab, area in list(areas.items()):
            tier = region_tiers.get(lab, TIER_BG)
            tp = tier_params.get(tier, tier_params[TIER_BG])
            min_area = max(20, int(total_pixels * tp["min_area_pct"]))
            if area < min_area:
                tiny.append((area, lab))

        if not tiny:
            break

        tiny.sort()  # smallest first

        merged_any = False
        for _, region_id in tiny:
            if region_id not in areas:
                continue
            if len(areas) <= target_stop:
                break

            neighbors = adjacency.get(region_id, set())
            if not neighbors:
                continue

            region_color = mean_colors.get(region_id)
            if region_color is None:
                continue

            # Prefer same-tier neighbor with closest color,
            # weighted to also consider lightness similarity (preserve shading)
            region_tier = region_tiers.get(region_id, TIER_BG)
            same = [n for n in neighbors if region_tiers.get(n, TIER_BG) == region_tier and n in mean_colors]
            candidates = same if same else [n for n in neighbors if n in mean_colors]
            if not candidates:
                continue

            # Use Euclidean LAB for fast neighbor selection (close enough for tiny merges)
            best_n = min(candidates,
                         key=lambda n: np.sum((region_color - mean_colors[n]) ** 2))

            _do_merge(label_map, best_n, region_id, mean_colors, region_tiers,
                      adjacency, areas, boundary_edges)
            merged_any = True

        if not merged_any:
            break

    return label_map, mean_colors, region_tiers, adjacency, areas, boundary_edges


def _merge_by_tier(label_map, mean_colors, region_tiers, adjacency, areas,
                   boundary_edges, tier_params, target_stop, target_tier):
    """Merge similar adjacent regions of a specific tier.

    Uses color threshold and edge protection from tier_params.
    """
    tp = tier_params.get(target_tier, tier_params[TIER_BG])
    threshold = tp["similar_threshold"]
    edge_protect = tp["edge_protect"]
    lightness_protect = tp["lightness_protect"]

    for _ in range(30):
        if len(areas) <= target_stop:
            break

        # Build merge candidates: pairs of same-tier neighbors below threshold
        candidates = []
        for region_id in list(areas.keys()):
            if region_tiers.get(region_id, TIER_BG) != target_tier:
                continue
            color_a = mean_colors.get(region_id)
            if color_a is None:
                continue

            for n in adjacency.get(region_id, set()):
                if n <= region_id:  # avoid duplicates
                    continue
                if region_tiers.get(n, TIER_BG) != target_tier:
                    continue
                color_b = mean_colors.get(n)
                if color_b is None:
                    continue

                # Fast pre-filter with Euclidean LAB (avoids expensive CIEDE2000
                # for obviously dissimilar pairs). CIEDE2000 threshold * 3 as cutoff.
                euclidean_fast = np.sqrt(np.sum((color_a - color_b) ** 2))
                if euclidean_fast >= threshold * 3:
                    continue

                # CIEDE2000 perceptual color distance (accurate)
                delta = _ciede2000(color_a, color_b)
                if delta >= threshold:
                    continue

                # Lightness protection: don't merge if brightness differs
                # even if overall color is similar (preserves shading/gradients)
                delta_L = abs(float(color_a[0]) - float(color_b[0]))
                if delta_L >= lightness_protect:
                    continue

                # Edge protection check
                key = (min(region_id, n), max(region_id, n))
                edge_str = boundary_edges.get(key, 0.0)
                if edge_str > edge_protect:
                    continue

                candidates.append((delta, region_id, n))

        if not candidates:
            break

        # Sort by color distance (merge most similar first)
        candidates.sort()

        merged_any = False
        merged_set = set()  # track which regions already merged this round

        for delta, a, b in candidates:
            if len(areas) <= target_stop:
                break
            if a in merged_set or b in merged_set:
                continue
            if a not in areas or b not in areas:
                continue

            # Merge smaller into larger
            if areas.get(a, 0) >= areas.get(b, 0):
                _do_merge(label_map, a, b, mean_colors, region_tiers,
                          adjacency, areas, boundary_edges)
            else:
                _do_merge(label_map, b, a, mean_colors, region_tiers,
                          adjacency, areas, boundary_edges)

            merged_set.add(a)
            merged_set.add(b)
            merged_any = True

        if not merged_any:
            break

    return label_map, mean_colors, region_tiers, adjacency, areas, boundary_edges


def _relabel_sequential(label_map):
    unique_labels = np.unique(label_map)
    new_map = np.zeros_like(label_map)
    for new_id, old_id in enumerate(unique_labels):
        new_map[label_map == old_id] = new_id
    return new_map
