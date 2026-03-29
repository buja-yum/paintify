# Paintify

Professional paint-by-numbers kit generator. Transforms any photo into a printable paint-by-numbers kit with adaptive, edge-aware segmentation.

## Features

- **Adaptive SLIC segmentation** - follows natural color/edge boundaries, never grid-based
- **Saliency-aware merging** - subject stays detailed, background gets simplified
- **RAW format support** - ARW, CR2, CR3, NEF, DNG, and more via rawpy
- **Smart number placement** - positioned at the most interior point of each region using distance transform
- **Dynamic font sizing** - adapts to region size automatically
- **Hue-sorted palette** - 24-30 colors organized like a real paint set

## Output

| File | Description |
|------|-------------|
| `*_outline.png` | Numbered outline for painting |
| `*_color_reference.png` | Color-filled reference (answer key) |
| `*_palette.png` | Color palette chart with hex codes |
| `*_kit.pdf` | Printable PDF with all pages combined |

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py <image> --difficulty <level> [options]
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `input` | *(required)* | Path to input image (JPG, PNG, RAW, etc.) |
| `--difficulty` | `medium` | `easy` / `medium` / `hard` / `expert` |
| `--colors` | `24` | Number of palette colors (24-30) |
| `--output` | `./output` | Output directory |
| `--max-dimension` | `2400` | Max image dimension for processing |
| `--segment-count` | - | Override segment count directly |

### Difficulty Levels

| Level | Segments | Best for |
|-------|----------|----------|
| easy | 150-300 | Beginners, simple compositions |
| medium | 300-600 | Intermediate painters |
| hard | 600-1000 | Experienced painters, detailed subjects |
| expert | 1000+ | Advanced, maximum detail |

### Examples

```bash
# Standard usage
python main.py photo.jpg --difficulty hard

# RAW file with custom colors
python main.py DSC05523.ARW --difficulty medium --colors 28

# Custom segment count
python main.py portrait.png --segment-count 500 --output ./my_kit
```

## How It Works

1. **Preprocessing** - Bilateral filtering in LAB color space to reduce noise while preserving edges
2. **SLIC Superpixels** - Adaptive segmentation that follows color and texture boundaries
3. **Saliency Detection** - Identifies subject vs background for differential detail levels
4. **Region Merging** - Eliminates tiny fragments, simplifies background regions
5. **Palette Extraction** - K-means clustering weighted by region area, with deduplication
6. **Rendering** - Outline generation, number placement, palette chart, and combined PDF

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt) for dependencies
