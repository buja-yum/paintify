"""Paintify - Professional Paint-by-Numbers Kit Generator.

Usage:
    python main.py input.jpg --difficulty medium --colors 24 --output ./output
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from segmentation import segment_image
from palette import extract_palette
from renderer import render_all


DIFFICULTY_SEGMENTS = {
    "easy": 500,
    "medium": 1000,
    "hard": 1800,
    "expert": 2800,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a professional paint-by-numbers kit from an image."
    )
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("--difficulty", default="medium",
                        choices=["easy", "medium", "hard", "expert"],
                        help="Difficulty level (default: medium)")
    parser.add_argument("--colors", type=int, default=30,
                        help="Number of palette colors, 24-30 (default: 30)")
    parser.add_argument("--output", default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--max-dimension", type=int, default=2400,
                        help="Max image dimension for processing (default: 2400)")
    parser.add_argument("--segment-count", type=int, default=None,
                        help="Override segment count (ignores difficulty)")
    return parser.parse_args()


RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".rw2", ".dng", ".pef"}


def load_and_resize(path: str, max_dim: int) -> np.ndarray:
    """Load image and resize if needed. Supports RAW formats via rawpy."""
    ext = os.path.splitext(path)[1].lower()

    if ext in RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError:
            print("Error: rawpy is required for RAW files. Install with: pip install rawpy")
            sys.exit(1)
        print(f"  Detected RAW format ({ext}), processing with rawpy...")
        raw = rawpy.imread(path)
        rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path)

    if image is None:
        print(f"Error: Could not load image: {path}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"  Original size: {w}x{h}")

    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"  Resized to: {new_w}x{new_h}")

    return image


def validate_result(label_map: np.ndarray, difficulty: str, segment_count_override: int = None):
    """Validate the segmentation result."""
    n_regions = len(np.unique(label_map))
    h, w = label_map.shape

    target_ranges = {
        "easy": (150, 300),
        "medium": (300, 600),
        "hard": (600, 1000),
        "expert": (1000, 5000),
    }

    if segment_count_override:
        target_min = int(segment_count_override * 0.5)
        target_max = int(segment_count_override * 1.5)
    else:
        target_min, target_max = target_ranges[difficulty]

    issues = []

    # Check segment count (warning only - SLIC + merging makes exact control hard)
    if n_regions < target_min:
        issues.append(f"Region count ({n_regions}) below target minimum ({target_min})")
    elif n_regions > target_max:
        issues.append(f"Region count ({n_regions}) above target maximum ({target_max})")

    # Check for grid-like patterns
    # Sample horizontal and vertical boundary positions
    boundaries_h = np.where(label_map[:-1, :] != label_map[1:, :])
    boundaries_v = np.where(label_map[:, :-1] != label_map[:, 1:])

    if len(boundaries_h[0]) > 0:
        row_counts = np.bincount(boundaries_h[0], minlength=h - 1)
        # If any single row has boundaries across >80% of width, suspicious
        if row_counts.max() > w * 0.8:
            issues.append("Possible grid-like horizontal pattern detected")

    if len(boundaries_v[1]) > 0:
        col_counts = np.bincount(boundaries_v[1], minlength=w - 1)
        if col_counts.max() > h * 0.8:
            issues.append("Possible grid-like vertical pattern detected")

    # Check for tiny regions
    areas = np.bincount(label_map.ravel())
    tiny_count = np.sum(areas < 20)
    if tiny_count > n_regions * 0.1:
        issues.append(f"Too many tiny regions: {tiny_count}")

    return issues, n_regions


def main():
    args = parse_args()

    colors = max(24, min(30, args.colors))
    n_segments = args.segment_count or DIFFICULTY_SEGMENTS[args.difficulty]
    input_name = os.path.splitext(os.path.basename(args.input))[0]

    print(f"Paintify - Paint-by-Numbers Kit Generator")
    print(f"=" * 45)
    print(f"  Input: {args.input}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Target segments: {n_segments}")
    print(f"  Palette colors: {colors}")
    print(f"  Output: {args.output}")
    print()

    # Step 1: Load image
    print("[1/4] Loading image...")
    image = load_and_resize(args.input, args.max_dimension)

    # Step 2: Segment
    print("[2/4] Segmenting image (this may take a moment)...")
    t0 = time.time()
    label_map = segment_image(image, n_segments, args.difficulty)
    t1 = time.time()
    print(f"  Segmentation done in {t1 - t0:.1f}s")

    # Validate
    issues, n_regions = validate_result(label_map, args.difficulty, args.segment_count)
    print(f"  Final region count: {n_regions}")
    if issues:
        print("  Validation warnings:")
        for issue in issues:
            print(f"    - {issue}")

    # Step 3: Extract palette
    print("[3/4] Extracting color palette...")
    palette_rgb, region_to_color, _ = extract_palette(image, label_map, colors)
    print(f"  Palette: {len(palette_rgb)} colors")

    # Step 4: Render outputs
    print("[4/4] Rendering outputs...")
    outline, color_ref, palette_chart, pdf = render_all(
        image, label_map, palette_rgb, region_to_color, args.output, input_name
    )

    print()
    print("Done! Output files:")
    print(f"  Outline:    {outline}")
    print(f"  Color ref:  {color_ref}")
    print(f"  Palette:    {palette_chart}")
    print(f"  PDF kit:    {pdf}")


if __name__ == "__main__":
    main()
