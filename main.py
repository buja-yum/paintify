"""Paintify - Professional Paint-by-Numbers Kit Generator.

Usage:
    python main.py input.jpg --difficulty medium --colors 24 --output ./output
    python main.py photo.arw --difficulty hard --paper-size a3
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from segmentation import segment_image, TARGET_RANGES
from palette import extract_palette
from renderer import render_all


DIFFICULTY_SEGMENTS = {
    "easy": 500,
    "medium": 1000,
    "hard": 1800,
    "expert": 2800,
}

RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".rw2", ".dng", ".pef"}

MAX_RETRIES = 2


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
    parser.add_argument("--paper-size", default="a4",
                        choices=["letter", "a4", "a3"],
                        help="Paper size for PDF output (default: a4)")
    return parser.parse_args()


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


def validate_result(label_map: np.ndarray, difficulty: str,
                    segment_count_override: int = None):
    """Comprehensive validation per prompt specifications.

    Checks:
      1. Region count within difficulty range
      2. No grid-like patterns
      3. No excessive micro-regions (unpaintable)
      4. No excessively large regions dominating the image

    Returns (issues list, n_regions, is_critical).
    is_critical=True means auto-regeneration should be attempted.
    """
    n_regions = len(np.unique(label_map))
    h, w = label_map.shape
    total_pixels = h * w

    if segment_count_override:
        target_min = int(segment_count_override * 0.5)
        target_max = int(segment_count_override * 1.5)
    else:
        target_min, target_max = TARGET_RANGES[difficulty]

    issues = []
    is_critical = False

    # 1. Region count check
    if n_regions < target_min:
        issues.append(f"Region count ({n_regions}) below minimum ({target_min})")
    elif n_regions > target_max:
        issues.append(f"Region count ({n_regions}) above maximum ({target_max})")

    # 2. Grid-like pattern detection
    boundaries_h = np.where(label_map[:-1, :] != label_map[1:, :])
    boundaries_v = np.where(label_map[:, :-1] != label_map[:, 1:])

    if len(boundaries_h[0]) > 0:
        row_counts = np.bincount(boundaries_h[0], minlength=h - 1)
        if row_counts.max() > w * 0.8:
            issues.append("CRITICAL: Grid-like horizontal pattern detected")
            is_critical = True

    if len(boundaries_v[1]) > 0:
        col_counts = np.bincount(boundaries_v[1], minlength=w - 1)
        if col_counts.max() > h * 0.8:
            issues.append("CRITICAL: Grid-like vertical pattern detected")
            is_critical = True

    # 3. Micro-region check (too small to paint or label)
    areas = np.bincount(label_map.ravel())
    min_paintable = max(30, int(total_pixels * 0.00005))
    tiny_count = int(np.sum((areas > 0) & (areas < min_paintable)))
    if tiny_count > n_regions * 0.15:
        issues.append(f"Too many micro-regions: {tiny_count}/{n_regions} "
                      f"(below {min_paintable}px)")

    # 4. Dominant region check (single region > 30% of image is suspicious)
    max_region_pct = areas.max() / total_pixels
    if max_region_pct > 0.30:
        issues.append(f"Dominant region covers {max_region_pct:.0%} of image")

    return issues, n_regions, is_critical


def main():
    args = parse_args()

    colors = max(24, min(30, args.colors))
    n_segments = args.segment_count or DIFFICULTY_SEGMENTS[args.difficulty]
    input_name = os.path.splitext(os.path.basename(args.input))[0]

    print("Paintify - Paint-by-Numbers Kit Generator")
    print("=" * 45)
    print(f"  Input: {args.input}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Target segments: {n_segments}")
    print(f"  Palette colors: {colors}")
    print(f"  Paper size: {args.paper_size.upper()}")
    print(f"  Output: {args.output}")
    print()

    # Step 1: Load image
    print("[1/4] Loading image...")
    image = load_and_resize(args.input, args.max_dimension)

    # Step 2: Segment (with auto-retry on critical failure)
    label_map = None
    current_segments = n_segments

    for attempt in range(1, MAX_RETRIES + 2):
        print(f"[2/4] Segmenting image (attempt {attempt})...")
        t0 = time.time()
        label_map = segment_image(image, current_segments, args.difficulty)
        t1 = time.time()
        print(f"  Segmentation done in {t1 - t0:.1f}s")

        issues, n_regions, is_critical = validate_result(
            label_map, args.difficulty, args.segment_count
        )
        print(f"  Final region count: {n_regions}")

        if issues:
            print("  Validation:")
            for issue in issues:
                print(f"    - {issue}")

        if is_critical and attempt <= MAX_RETRIES:
            print(f"  Critical issue detected. Retrying with adjusted parameters...")
            # Increase segments and compactness to break grid patterns
            current_segments = int(current_segments * 1.3)
            continue
        else:
            break

    # Step 3: Extract palette
    print("[3/4] Extracting color palette...")
    palette_rgb, region_to_color, _ = extract_palette(image, label_map, colors)
    print(f"  Palette: {len(palette_rgb)} colors")

    # Step 4: Render outputs
    print("[4/4] Rendering outputs...")
    outline, color_ref, palette_chart, pdf = render_all(
        image, label_map, palette_rgb, region_to_color,
        args.output, input_name, args.paper_size
    )

    print()
    print("Done! Output files:")
    print(f"  Outline:    {outline}")
    print(f"  Color ref:  {color_ref}")
    print(f"  Palette:    {palette_chart}")
    print(f"  PDF kit:    {pdf}")


if __name__ == "__main__":
    main()
