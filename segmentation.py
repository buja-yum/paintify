"""Adaptive, edge-aware segmentation engine using SLIC superpixels."""

import cv2
import numpy as np
from skimage.segmentation import slic


def segment_image(image_bgr: np.ndarray, n_segments: int, difficulty: str) -> np.ndarray:
    """Segment image into paint-by-numbers regions using SLIC superpixels.

    Returns label_map of shape (H, W) with integer region labels.
    """
    # Stage 1: Preprocessing
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    image_smoothed = cv2.bilateralFilter(image_lab, d=7, sigmaColor=40, sigmaSpace=40)

    # Stage 2: SLIC Superpixel Segmentation
    compactness_map = {"easy": 25, "medium": 18, "hard": 12, "expert": 8}
    compactness = compactness_map.get(difficulty, 20)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb_smoothed = cv2.cvtColor(image_smoothed, cv2.COLOR_LAB2RGB)

    label_map = slic(
        image_rgb_smoothed,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1,
        enforce_connectivity=True,
        convert2lab=True,
        start_label=0,
    )
    label_map = label_map.astype(np.int32)

    # Stage 3: Saliency-based subject detection
    subject_mask = _compute_saliency_mask(image_lab)

    # Stage 4: Compute per-region mean colors in LAB
    mean_colors = _compute_mean_colors(image_lab, label_map)

    # Stage 5: Small region merging
    min_area_pct = {"easy": 0.0006, "medium": 0.0003, "hard": 0.00012, "expert": 0.00006}
    min_area = int(image_bgr.shape[0] * image_bgr.shape[1] * min_area_pct.get(difficulty, 0.0004))
    min_area = max(min_area, 20)

    label_map, mean_colors = _merge_small_regions(label_map, mean_colors, min_area)

    # Stage 6: Background simplification
    label_map, mean_colors = _simplify_background(label_map, mean_colors, subject_mask, difficulty)

    # Relabel sequentially
    label_map = _relabel_sequential(label_map)

    return label_map


def _compute_saliency_mask(image_lab: np.ndarray) -> np.ndarray:
    """Simple saliency detection based on color distance from mean."""
    mean_color = image_lab.mean(axis=(0, 1))
    diff = image_lab.astype(np.float32) - mean_color.astype(np.float32)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    dist = (dist / (dist.max() + 1e-8) * 255).astype(np.uint8)
    dist = cv2.GaussianBlur(dist, (41, 41), 0)
    threshold = np.median(dist)
    mask = (dist > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _compute_mean_colors(image_lab: np.ndarray, label_map: np.ndarray) -> dict:
    """Compute mean LAB color for each region."""
    unique_labels = np.unique(label_map)
    mean_colors = {}
    flat_labels = label_map.ravel()
    flat_pixels = image_lab.reshape(-1, 3).astype(np.float64)

    for lab in unique_labels:
        mask = flat_labels == lab
        mean_colors[lab] = flat_pixels[mask].mean(axis=0)

    return mean_colors


def _merge_small_regions(label_map: np.ndarray, mean_colors: dict, min_area: int):
    """Merge regions smaller than min_area into their most similar neighbor."""
    kernel = np.ones((3, 3), dtype=np.uint8)

    for _ in range(50):  # safety limit
        areas = {}
        for lab in np.unique(label_map):
            areas[lab] = int(np.sum(label_map == lab))

        small_regions = [lab for lab, area in areas.items() if area < min_area and area > 0]
        if not small_regions:
            break

        # Sort by size so smallest get merged first
        small_regions.sort(key=lambda x: areas[x])

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

            best_dist = float("inf")
            best_neighbor = neighbor_labels[0]
            for n in neighbor_labels:
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
            # Weighted average color
            total = old_area + neighbor_area
            if total > 0:
                mean_colors[best_neighbor] = (nc * neighbor_area + region_color * old_area) / total

            label_map[label_map == region_id] = best_neighbor
            if region_id in mean_colors:
                del mean_colors[region_id]
            merged_any = True

        if not merged_any:
            break

    return label_map, mean_colors


def _simplify_background(label_map: np.ndarray, mean_colors: dict,
                         subject_mask: np.ndarray, difficulty: str):
    """Merge similar background regions more aggressively."""
    merge_threshold = {"easy": 18, "medium": 13, "hard": 8, "expert": 6}
    threshold = merge_threshold.get(difficulty, 15)

    kernel = np.ones((3, 3), dtype=np.uint8)
    unique_labels = np.unique(label_map)

    # Determine which regions are "background" (majority of pixels in non-salient area)
    bg_regions = set()
    for lab in unique_labels:
        region_mask = label_map == lab
        if subject_mask[region_mask].mean() < 0.3:
            bg_regions.add(lab)

    # Merge similar adjacent background regions
    merged = True
    iterations = 0
    while merged and iterations < 30:
        merged = False
        iterations += 1
        bg_list = list(bg_regions & set(np.unique(label_map)))

        for region_id in bg_list:
            if np.sum(label_map == region_id) == 0:
                bg_regions.discard(region_id)
                continue

            mask = (label_map == region_id).astype(np.uint8)
            dilated = cv2.dilate(mask, kernel)
            neighbor_mask = (dilated > 0) & (mask == 0)
            neighbor_labels = np.unique(label_map[neighbor_mask])
            neighbor_labels = [n for n in neighbor_labels if n != region_id and n in bg_regions]

            if not neighbor_labels:
                continue

            region_color = mean_colors.get(region_id)
            if region_color is None:
                continue

            for n in neighbor_labels:
                nc = mean_colors.get(n)
                if nc is None:
                    continue
                delta = np.sqrt(np.sum((region_color - nc) ** 2))
                if delta < threshold:
                    # Merge n into region_id
                    area_r = int(np.sum(label_map == region_id))
                    area_n = int(np.sum(label_map == n))
                    total = area_r + area_n
                    if total > 0:
                        mean_colors[region_id] = (region_color * area_r + nc * area_n) / total
                    label_map[label_map == n] = region_id
                    if n in mean_colors:
                        del mean_colors[n]
                    bg_regions.discard(n)
                    merged = True
                    break

    return label_map, mean_colors


def _relabel_sequential(label_map: np.ndarray) -> np.ndarray:
    """Relabel regions sequentially starting from 0."""
    unique_labels = np.unique(label_map)
    new_map = np.zeros_like(label_map)
    for new_id, old_id in enumerate(unique_labels):
        new_map[label_map == old_id] = new_id
    return new_map
