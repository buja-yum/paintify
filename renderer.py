"""Render paint-by-numbers outputs: outline, color reference, palette chart, PDF."""

import io
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet


def render_all(image_bgr: np.ndarray, label_map: np.ndarray,
               palette_rgb: np.ndarray, region_to_color: dict,
               output_dir: str, input_name: str):
    """Generate all paint-by-numbers output files."""
    os.makedirs(output_dir, exist_ok=True)

    print("  Rendering outline image...")
    outline_path = os.path.join(output_dir, f"{input_name}_outline.png")
    _render_outline(label_map, region_to_color, palette_rgb, outline_path)

    print("  Rendering color reference...")
    color_ref_path = os.path.join(output_dir, f"{input_name}_color_reference.png")
    _render_color_reference(image_bgr, label_map, region_to_color, palette_rgb, color_ref_path)

    print("  Rendering palette chart...")
    palette_path = os.path.join(output_dir, f"{input_name}_palette.png")
    _render_palette_chart(palette_rgb, palette_path)

    print("  Generating PDF...")
    pdf_path = os.path.join(output_dir, f"{input_name}_kit.pdf")
    _render_pdf(image_bgr, outline_path, color_ref_path, palette_path,
                pdf_path, input_name, len(palette_rgb), len(np.unique(label_map)))

    return outline_path, color_ref_path, palette_path, pdf_path


def _find_boundaries(label_map: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Find region boundaries using neighbor comparison with controllable thickness."""
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
    """Find the best position for placing a number inside each region.

    Uses distance transform to find the most interior point.
    """
    positions = {}
    unique_labels = np.unique(label_map)

    for lab in unique_labels:
        mask = (label_map == lab).astype(np.uint8)
        area = int(mask.sum())

        if area < 10:
            # Too small, just use centroid
            ys, xs = np.where(mask > 0)
            positions[lab] = (int(xs.mean()), int(ys.mean()), area)
            continue

        # Distance transform to find most interior point
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist)
        positions[lab] = (max_loc[0], max_loc[1], area)

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


def _render_outline(label_map: np.ndarray, region_to_color: dict,
                    palette_rgb: np.ndarray, output_path: str):
    """Render the numbered outline image on white background."""
    h, w = label_map.shape
    boundary = _find_boundaries(label_map)
    positions = _find_label_positions(label_map)

    # Create white canvas with black boundaries
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    canvas[boundary] = [0, 0, 0]

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    # Determine font size range based on image size
    base_size = max(8, min(16, int(min(h, w) / 80)))

    for lab, (x, y, area) in positions.items():
        color_idx = region_to_color.get(lab, 1)
        text = str(color_idx)

        # Dynamic font size based on region area
        region_size = int(np.sqrt(area))
        font_size = max(6, min(base_size, int(region_size * 0.35)))
        font = _get_font(font_size)

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Center text on position
        tx = x - tw // 2
        ty = y - th // 2

        # Clamp to image bounds
        tx = max(0, min(w - tw - 1, tx))
        ty = max(0, min(h - th - 1, ty))

        # Skip if region too small to fit any number
        if area < 30:
            continue

        draw.text((tx, ty), text, fill=(80, 80, 80), font=font)

    img.save(output_path, dpi=(300, 300))


def _render_color_reference(image_bgr: np.ndarray, label_map: np.ndarray,
                            region_to_color: dict, palette_rgb: np.ndarray,
                            output_path: str):
    """Render color-filled regions with numbers overlay."""
    h, w = label_map.shape
    boundary = _find_boundaries(label_map)
    positions = _find_label_positions(label_map)

    # Fill regions with palette colors
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for lab in np.unique(label_map):
        color_idx = region_to_color.get(lab, 1) - 1  # 0-based for array
        color_idx = max(0, min(len(palette_rgb) - 1, color_idx))
        color = palette_rgb[color_idx]
        canvas[label_map == lab] = color

    # Draw boundaries
    canvas[boundary] = [40, 40, 40]

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    base_size = max(8, min(14, int(min(h, w) / 90)))

    for lab, (x, y, area) in positions.items():
        color_idx = region_to_color.get(lab, 1)
        text = str(color_idx)

        region_size = int(np.sqrt(area))
        font_size = max(6, min(base_size, int(region_size * 0.3)))
        font = _get_font(font_size)

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x - tw // 2
        ty = y - th // 2
        tx = max(0, min(w - tw - 1, tx))
        ty = max(0, min(h - th - 1, ty))

        if area < 30:
            continue

        # Choose text color based on region luminance
        rgb = palette_rgb[max(0, min(len(palette_rgb) - 1, color_idx - 1))]
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = (255, 255, 255) if luminance < 128 else (0, 0, 0)

        # Draw text with outline for readability
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                outline_color = (0, 0, 0) if luminance >= 128 else (255, 255, 255)
                draw.text((tx + dx, ty + dy), text, fill=outline_color, font=font)
        draw.text((tx, ty), text, fill=text_color, font=font)

    img.save(output_path, dpi=(300, 300))


def _render_palette_chart(palette_rgb: np.ndarray, output_path: str):
    """Render a color palette reference chart."""
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

            # Number on swatch
            luminance = 0.299 * rgb_norm[0] + 0.587 * rgb_norm[1] + 0.114 * rgb_norm[2]
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(0.5, 0.58, str(i + 1), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=text_color)

            # Hex code below
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
                n_colors: int, n_regions: int):
    """Generate a multi-page printable PDF kit."""
    page_w, page_h = letter
    margin = 0.75 * inch
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin

    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            leftMargin=margin, rightMargin=margin,
                            topMargin=margin, bottomMargin=margin)
    styles = getSampleStyleSheet()
    story = []

    # Title page
    title_style = styles["Title"]
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("Paint by Numbers Kit", title_style))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(f"<i>{name}</i>", styles["Heading2"]))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(f"Colors: {n_colors} &nbsp;&nbsp;|&nbsp;&nbsp; Regions: {n_regions}",
                           styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # Original image thumbnail
    orig_buf = _image_to_buffer(image_bgr, max_dim=400)
    story.append(RLImage(orig_buf, width=4 * inch, height=4 * inch,
                         kind="proportional"))

    # Outline page(s)
    story.append(PageBreak())
    story.append(Paragraph("Numbered Outline", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(RLImage(outline_path, width=usable_w, height=usable_h - 0.5 * inch,
                         kind="proportional"))

    # Palette page
    story.append(PageBreak())
    story.append(Paragraph("Color Palette Reference", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(RLImage(palette_path, width=usable_w, height=usable_h - 0.5 * inch,
                         kind="proportional"))

    # Color reference page
    story.append(PageBreak())
    story.append(Paragraph("Color Reference (Answer Key)", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(RLImage(color_ref_path, width=usable_w, height=usable_h - 0.5 * inch,
                         kind="proportional"))

    doc.build(story)


def _image_to_buffer(image_bgr: np.ndarray, max_dim: int = 800) -> io.BytesIO:
    """Convert OpenCV image to a PNG in-memory buffer."""
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
