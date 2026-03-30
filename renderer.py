"""Render paint-by-numbers outputs: outline, color reference, palette chart, PDF.

Number placement rules (from prompts):
  - One number per region, fully inside, never touching boundaries
  - No overlapping numbers
  - Dynamic font sizing based on region size
  - If region too small for readable number → it should have been merged already
"""

import io
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import letter, A4, A3
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

PAPER_SIZES = {
    "letter": letter,
    "a4": A4,
    "a3": A3,
}


def render_all(image_bgr: np.ndarray, label_map: np.ndarray,
               palette_rgb: np.ndarray, region_to_color: dict,
               output_dir: str, input_name: str, paper_size: str = "a4"):
    """Generate all paint-by-numbers output files."""
    os.makedirs(output_dir, exist_ok=True)

    print("  Rendering outline image...")
    outline_path = os.path.join(output_dir, f"{input_name}_outline.png")
    _render_outline(label_map, region_to_color, palette_rgb, outline_path)

    print("  Rendering color reference...")
    color_ref_path = os.path.join(output_dir, f"{input_name}_color_reference.png")
    _render_color_reference(label_map, region_to_color, palette_rgb, color_ref_path)

    print("  Rendering palette chart...")
    palette_path = os.path.join(output_dir, f"{input_name}_palette.png")
    _render_palette_chart(palette_rgb, palette_path)

    print("  Generating PDF...")
    pdf_path = os.path.join(output_dir, f"{input_name}_kit.pdf")
    _render_pdf(image_bgr, outline_path, color_ref_path, palette_path,
                pdf_path, input_name, len(palette_rgb),
                len(np.unique(label_map)), paper_size)

    return outline_path, color_ref_path, palette_path, pdf_path


def _find_boundaries(label_map: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Find region boundaries with controllable thickness for print readability."""
    h, w = label_map.shape
    boundary = np.zeros((h, w), dtype=bool)
    boundary[:-1, :] |= label_map[:-1, :] != label_map[1:, :]
    boundary[1:, :] |= label_map[:-1, :] != label_map[1:, :]
    boundary[:, :-1] |= label_map[:, :-1] != label_map[:, 1:]
    boundary[:, 1:] |= label_map[:, :-1] != label_map[:, 1:]

    if thickness > 1:
        kernel = np.ones((thickness, thickness), dtype=np.uint8)
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel).astype(bool)

    return boundary


def _find_label_positions(label_map: np.ndarray) -> dict:
    """Find the best interior position for number placement.

    Uses distance transform to find the most interior point, then ensures
    minimum margin from boundary so numbers never touch edges.
    """
    positions = {}
    unique_labels = np.unique(label_map)

    for lab in unique_labels:
        mask = (label_map == lab).astype(np.uint8)
        area = int(mask.sum())

        if area < 10:
            ys, xs = np.where(mask > 0)
            positions[lab] = (int(xs.mean()), int(ys.mean()), area, 0.0)
            continue

        # Distance transform: value = distance from nearest boundary
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_dist = dist.max()
        _, _, _, max_loc = cv2.minMaxLoc(dist)

        # max_dist tells us how much interior space we have for the number
        positions[lab] = (max_loc[0], max_loc[1], area, float(max_dist))

    return positions


def _get_font(size: int):
    """Try to load a clean font, fall back to default."""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _place_numbers(draw, positions, region_to_color, h, w, style="outline"):
    """Place numbers inside regions with overlap detection.

    style: "outline" (dark text on white) or "color" (adaptive text on colored bg)
    palette_rgb is needed only for "color" style.
    """
    base_size = max(8, min(16, int(min(h, w) / 80)))

    # Collect all planned text placements first for overlap detection
    placements = []

    for lab, (x, y, area, max_dist) in positions.items():
        color_idx = region_to_color.get(lab, 1)
        text = str(color_idx)

        # Skip regions too small to paint or label
        if area < 40:
            continue

        # Dynamic font size: based on both area and interior distance
        region_size = int(np.sqrt(area))
        font_size = max(6, min(base_size, int(region_size * 0.35)))

        # Further limit by available interior space
        if max_dist > 0:
            max_font_by_dist = max(6, int(max_dist * 0.8))
            font_size = min(font_size, max_font_by_dist)

        font = _get_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Ensure text fits within the interior (margin from boundary)
        margin = max(2, int(max_dist * 0.15)) if max_dist > 0 else 2
        if tw > max_dist * 1.6 or th > max_dist * 1.6:
            # Try smaller font
            font_size = max(6, font_size - 2)
            font = _get_font(font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

        # Center on interior point
        tx = x - tw // 2
        ty = y - th // 2
        tx = max(margin, min(w - tw - margin, tx))
        ty = max(margin, min(h - th - margin, ty))

        placements.append((tx, ty, tw, th, text, font, lab))

    # Overlap detection: remove overlapping numbers (keep the one for larger region)
    # Sort by area descending so larger regions' numbers take priority
    placements.sort(key=lambda p: positions[p[6]][2], reverse=True)
    occupied = []
    final_placements = []

    for tx, ty, tw, th, text, font, lab in placements:
        rect = (tx - 1, ty - 1, tx + tw + 1, ty + th + 1)
        overlaps = False
        for ox1, oy1, ox2, oy2 in occupied:
            if not (rect[2] < ox1 or rect[0] > ox2 or rect[3] < oy1 or rect[1] > oy2):
                overlaps = True
                break
        if not overlaps:
            final_placements.append((tx, ty, text, font, lab))
            occupied.append(rect)

    return final_placements


def _render_outline(label_map: np.ndarray, region_to_color: dict,
                    palette_rgb: np.ndarray, output_path: str):
    """Render numbered outline on white background."""
    h, w = label_map.shape
    boundary = _find_boundaries(label_map)
    positions = _find_label_positions(label_map)

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    canvas[boundary] = [0, 0, 0]

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    placements = _place_numbers(draw, positions, region_to_color, h, w)

    for tx, ty, text, font, lab in placements:
        draw.text((tx, ty), text, fill=(60, 60, 60), font=font)

    img.save(output_path, dpi=(300, 300))


def _render_color_reference(label_map: np.ndarray,
                            region_to_color: dict, palette_rgb: np.ndarray,
                            output_path: str):
    """Render color-filled regions with numbered overlay."""
    h, w = label_map.shape
    boundary = _find_boundaries(label_map)
    positions = _find_label_positions(label_map)

    # Fill regions with palette colors
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for lab in np.unique(label_map):
        color_idx = region_to_color.get(lab, 1) - 1
        color_idx = max(0, min(len(palette_rgb) - 1, color_idx))
        canvas[label_map == lab] = palette_rgb[color_idx]

    canvas[boundary] = [40, 40, 40]

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    placements = _place_numbers(draw, positions, region_to_color, h, w)

    for tx, ty, text, font, lab in placements:
        color_idx = region_to_color.get(lab, 1) - 1
        color_idx = max(0, min(len(palette_rgb) - 1, color_idx))
        rgb = palette_rgb[color_idx]
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

        text_color = (255, 255, 255) if luminance < 128 else (0, 0, 0)
        outline_color = (0, 0, 0) if luminance >= 128 else (255, 255, 255)

        # Text outline for readability
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                draw.text((tx + dx, ty + dy), text, fill=outline_color, font=font)
        draw.text((tx, ty), text, fill=text_color, font=font)

    img.save(output_path, dpi=(300, 300))


def _render_palette_chart(palette_rgb: np.ndarray, output_path: str):
    """Render color palette reference chart with RGB/HEX values."""
    n_colors = len(palette_rgb)
    cols = 6
    rows = (n_colors + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 1.5))
    fig.suptitle("Color Palette Reference", fontsize=16, fontweight="bold", y=0.98)

    if rows == 1:
        axes = [axes]

    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row][col] if rows > 1 else axes[col]

        if i < n_colors:
            color = palette_rgb[i]
            rgb_norm = color / 255.0
            hex_color = "#{:02X}{:02X}{:02X}".format(color[0], color[1], color[2])

            rect = mpatches.FancyBboxPatch((0.1, 0.25), 0.8, 0.65,
                                           boxstyle="round,pad=0.05",
                                           facecolor=rgb_norm, edgecolor="black",
                                           linewidth=1.5)
            ax.add_patch(rect)

            luminance = 0.299 * rgb_norm[0] + 0.587 * rgb_norm[1] + 0.114 * rgb_norm[2]
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(0.5, 0.58, str(i + 1), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=text_color)

            ax.text(0.5, 0.08, hex_color, ha="center", va="center",
                    fontsize=7, color="black")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def _render_pdf(image_bgr: np.ndarray, outline_path: str, color_ref_path: str,
                palette_path: str, pdf_path: str, name: str,
                n_colors: int, n_regions: int, paper_size: str = "a4"):
    """Generate a multi-page printable PDF kit (A4/A3/Letter)."""
    pagesize = PAPER_SIZES.get(paper_size, A4)
    page_w, page_h = pagesize
    margin = 0.6 * inch
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin

    doc = SimpleDocTemplate(pdf_path, pagesize=pagesize,
                            leftMargin=margin, rightMargin=margin,
                            topMargin=margin, bottomMargin=margin)
    styles = getSampleStyleSheet()
    story = []

    # Title page
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("Paint by Numbers Kit", styles["Title"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(f"<i>{name}</i>", styles["Heading2"]))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        f"Colors: {n_colors} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Regions: {n_regions} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Paper: {paper_size.upper()}",
        styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    orig_buf = _image_to_buffer(image_bgr, max_dim=400)
    story.append(RLImage(orig_buf, width=4 * inch, height=4 * inch,
                         kind="proportional"))

    # Outline page
    story.append(PageBreak())
    story.append(Paragraph("Numbered Outline", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(RLImage(outline_path, width=usable_w, height=usable_h - 0.4 * inch,
                         kind="proportional"))

    # Palette page
    story.append(PageBreak())
    story.append(Paragraph("Color Palette Reference", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(RLImage(palette_path, width=usable_w, height=usable_h - 0.4 * inch,
                         kind="proportional"))

    # Color reference page
    story.append(PageBreak())
    story.append(Paragraph("Color Reference (Answer Key)", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(RLImage(color_ref_path, width=usable_w, height=usable_h - 0.4 * inch,
                         kind="proportional"))

    doc.build(story)


def _image_to_buffer(image_bgr: np.ndarray, max_dim: int = 800) -> io.BytesIO:
    h, w = image_bgr.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf
