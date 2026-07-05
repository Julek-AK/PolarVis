# Builtins
from math import pi
from dataclasses import dataclass

# External Libraries
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from colorsys import hsv_to_rgb



# =============================================
# CONFIG
# =============================================

BACKGROUND = (0, 0, 0, 180)
TEXT_COLOR = (255, 255, 255, 255)
BORDER_COLOR = (255, 255, 255, 180)


@dataclass
class LegendStyle:
    scale: float
    box_padding: int
    inner_padding: int
    header_height: int
    font: ImageFont.FreeTypeFont
    text_padding: int


LEGEND_STYLES = {
    'small': LegendStyle(
        scale=1.0,
        box_padding=12,
        inner_padding=8,
        header_height=22,
        font=ImageFont.truetype("arial.ttf", 12),
        text_padding=3,
    ),

    'large': LegendStyle(
        scale=1.75,
        box_padding=21,
        inner_padding=14,
        header_height=38,
        font=ImageFont.truetype("arial.ttf", 21),
        text_padding=5,
    ),
}


# =============================================
# HELPERS
# =============================================

def get_legend_style(style: str = 'small') -> LegendStyle:
    try:
        return LEGEND_STYLES[style]
    except KeyError:
        raise ValueError(f"[Rendering] Unknown legend style: {style}")


def px(value, scale):
    return int(round(value * scale))


def _create_overlay_box(width: int, height: int) -> Image.Image:
    return Image.new(
        "RGBA",
        (width, height),
        BACKGROUND
    )


def _paste_box(
    base: Image.Image,
    overlay: Image.Image,
    style: LegendStyle,
    position: str = "bottom_right"
) -> Image.Image:

    base = base.convert("RGBA")

    bw, bh = base.size
    ow, oh = overlay.size

    if position == "bottom_right":
        x = bw - ow - style.box_padding
        y = bh - oh - style.box_padding

    elif position == "bottom_left":
        x = style.box_padding
        y = bh - oh - style.box_padding

    elif position == "top_right":
        x = bw - ow - style.box_padding
        y = style.box_padding

    elif position == "top_left":
        x = style.box_padding
        y = style.box_padding

    else:
        raise ValueError(f"[Visualisation] Unknown legend position: {position}")

    base.alpha_composite(overlay, (x, y))
    return base.convert("RGB")


def _draw_vertical_gradient(height: int, width: int, cmap_func) -> Image.Image:
    gradient = np.linspace(1, 0, height)

    img = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        c = cmap_func(float(gradient[y]))
        img[y, :, :] = np.array(c[:3]) * 255

    return Image.fromarray(img)


def _hsv_color(h, s=1.0, v=1.0):
    r, g, b = hsv_to_rgb(h, s, v)
    return (
        int(r * 255),
        int(g * 255),
        int(b * 255),
        255
    )


def label(draw, x, y, text, font, align="left", padding=3):
    bbox = draw.textbbox((0, 0), text, font=font)
    left, top, right, bottom = bbox

    w = right - left
    h = bottom - top

    if align == "center":
        x -= w // 2
        y -= h // 2
    elif align == "right":
        x -= w

    rect = [
        x + left - padding,
        y + top - padding,
        x + right + padding,
        y + bottom + padding,
    ]

    draw.rectangle(rect, fill=(0, 0, 0, 160))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))


# =============================================
# SCALAR LEGEND
# =============================================

def scalar_legend(image, result, position="bottom_right", size="large"):
    cmap = plt.get_cmap(result.cmap)

    style = get_legend_style(size)
    scale = style.scale
    font = style.font

    bar_w = px(28, scale)
    bar_h = px(220, scale)
    label_w = px(30, scale)

    box_w = bar_w + label_w + style.inner_padding * 3
    box_h = bar_h + style.inner_padding * 2 + style.header_height

    overlay = _create_overlay_box(box_w, box_h)
    draw = ImageDraw.Draw(overlay)

    # Header
    label(
        draw,
        style.inner_padding,
        style.inner_padding,
        result.label,
        font,
        align="left",
        padding=style.text_padding
    )

    y0 = style.inner_padding + style.header_height

    # Gradient
    grad = _draw_vertical_gradient(bar_h, bar_w, cmap)
    overlay.paste(grad, (style.inner_padding, y0))

    # Border
    draw.rectangle(
        [
            style.inner_padding,
            y0,
            style.inner_padding + bar_w,
            y0 + bar_h
        ],
        outline=BORDER_COLOR,
        width=1
    )

    # Labels
    bx = style.inner_padding + bar_w + 8
    by = y0 - 2

    label(draw, bx, by             , "1.00", font, align="center", padding=style.text_padding)
    label(draw, bx, by + bar_h*0.25, "0.75", font, align="center", padding=style.text_padding)
    label(draw, bx, by + bar_h*0.5 , "0.50", font, align="center", padding=style.text_padding)
    label(draw, bx, by + bar_h*0.75, "0.25", font, align="center", padding=style.text_padding)
    label(draw, bx, by + bar_h     , "0.00", font, align="center", padding=style.text_padding)

    return _paste_box(image, overlay, style, position)


# =============================================
# ANGLE LEGEND (AoP)
# =============================================

def angle_legend(image, result, position="bottom_right", size="large"):
    style = get_legend_style(size)
    scale = style.scale
    font = style.font

    size = px(180, scale)

    overlay = _create_overlay_box(size, size)
    draw = ImageDraw.Draw(overlay)

    cx = size // 2
    cy = size // 2

    r_outer = px(65, scale)
    r_inner = px(40, scale)

    # Header
    label(
        draw,
        style.inner_padding,
        style.inner_padding,
        "AoP",
        font,
        align="left",
        padding=style.text_padding
    )

    # Hue wheel
    for y in range(size):
        for x in range(size):

            dx = x - cx
            dy = y - cy

            r = np.sqrt(dx**2 + dy**2)
            if r < r_inner or r > r_outer:
                continue
            
            angle = (np.arctan2(-dy, dx) % pi) / pi
            color = _hsv_color(angle)
            overlay.putpixel((x, y), color)

    # Angular labels
    angle_labels = [(0,  "0°"), (45, "45°"), (90, "90°"), (135, "135°")]
    for deg, text in angle_labels:
        theta = np.deg2rad(deg)

        # place slightly outside the circle
        r_text = r_outer + px(14, scale)

        x = cx + r_text * np.cos(theta)
        y = cy - r_text * np.sin(theta)

        label(draw, x, y, text, font, align="center", padding=style.text_padding)

    return _paste_box(image, overlay, style, position)


# =============================================
# POLARIMETRIC LEGEND (AoP + DoLP + Intensity)
# =============================================

def polarimetric_legend(image, result, position="bottom_right", size="large"):
    style = get_legend_style(size)
    scale = style.scale
    font = style.font

    size = px(240, scale)

    overlay = _create_overlay_box(size, size)
    draw = ImageDraw.Draw(overlay)

    cx = size // 2
    cy = size // 2

    radius = px(85, scale)

    # Header
    label(
        draw,
        style.inner_padding,
        style.inner_padding,
        "AoP DoLP Intensity",
        font,
        align="left",
        padding=style.text_padding
    )

    # Polar wheel
    for y in range(size):
        for x in range(size):

            dx = x - cx
            dy = y - cy

            r = np.sqrt(dx**2 + dy**2)
            if r > radius:
                continue

            angle = (np.arctan2(-dy, dx) % pi) / pi

            sat = r / radius

            color = _hsv_color(angle, sat, 1.0)
            overlay.putpixel((x, y), color)
    
    # DoLP guide rings
    rings = 4
    for i in range(1, rings + 1):
        r = int(radius * i / rings)
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=(180, 180, 180)
        )

    # Angular labels
    angle_labels = [(0,  "0°"), (45, "45°"), (90, "90°"), (135, "135°")]
    for deg, text in angle_labels:
        theta = np.deg2rad(deg)

        # place slightly outside the circle
        r_text = radius + 14

        x = cx + r_text * np.cos(theta)
        y = cy - r_text * np.sin(theta)

        label(draw, x, y, text, font, align="center", padding=style.text_padding)

    # DoLP labels
    dolp_labels = [0.25, 0.50, 0.75, 1.00]
    for v in dolp_labels:
        r = radius * v

        x = cx
        y = cy + r

        label(draw, x, y, f"{v:.2f}", font, align="center", padding=style.text_padding)

    # Intensity bar
    bar_w = px(14, scale)
    bar_h = px(120, scale)

    grad = np.linspace(1, 0, bar_h)[:, None]
    grad = np.repeat(grad, bar_w, axis=1)
    intensity_bar = Image.fromarray((grad * 255).astype(np.uint8)).convert("RGBA")

    bx = cx + radius + px(12, scale)
    by = cy - bar_h // 2

    overlay.paste(intensity_bar, (bx, by))

    label(draw, bx, by + bar_h, "0.0", font, align="center", padding=style.text_padding)
    label(draw, bx, by + bar_h/2, "0.5", font, align="center", padding=style.text_padding)
    label(draw, bx, by, "1.0", font, align="center", padding=style.text_padding)

    return _paste_box(image, overlay, style, position)


# =============================================
# POLAR-ONLY LEGEND (AoP + DoLP only)
# =============================================

def polar_only_legend(image, result, position="bottom_right", size="large"):
    style = get_legend_style(size)
    scale = style.scale
    font = style.font

    size = px(200, scale)

    overlay = _create_overlay_box(size, size)
    draw = ImageDraw.Draw(overlay)

    cx = size // 2
    cy = size // 2

    radius = px(80, scale)

    # Header
    label(
        draw,
        style.inner_padding,
        style.inner_padding,
        "AoP DoLP",
        font,
        align="left",
        padding=style.text_padding
    )

    # Polar encoding
    for y in range(size):
        for x in range(size):

            dx = x - cx
            dy = y - cy

            r = np.sqrt(dx**2 + dy**2)
            if r > radius:
                continue

            angle = (np.arctan2(-dy, dx) % pi) / pi
            sat = r / radius

            color = _hsv_color(angle, sat, 1.0)
            overlay.putpixel((x, y), color)

    # DoLP rings
    rings = 4
    for i in range(1, rings + 1):
        r = int(radius * i / rings)
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=(180, 180, 180)
        )

    # Angular labels
    angle_labels = [(0,  "0°"), (45, "45°"), (90, "90°"), (135, "135°")]
    for deg, text in angle_labels:
        theta = np.deg2rad(deg)

        # place slightly outside the circle
        r_text = radius + px(14, scale)

        x = cx + r_text * np.cos(theta)
        y = cy - r_text * np.sin(theta)

        label(draw, x, y, text, font, align="center", padding=style.text_padding)

    # DoLP labels
    dolp_labels = [0.25, 0.50, 0.75, 1.00]
    for v in dolp_labels:
        r = radius * v

        x = cx
        y = cy + r

        label(draw, x, y, f"{v:.2f}", font, align="center", padding=style.text_padding)

    return _paste_box(image, overlay, style, position)



if __name__ == "__main__":
    from polarvis.app.paths import TEST_OUT_DIR
    from polarvis.processing.image_visualisation import VisualisationResult

    SCALE = 5

    legends = {
        "polar_only": polar_only_legend,
        "polarimetric": polarimetric_legend,
        "angle": angle_legend,
        "intensity": scalar_legend,
        "dolp": scalar_legend,
    }

    print("Exporting legends...")

    for name, func in legends.items():
        img = Image.new("RGB", (500, 500), (0, 0, 0))
        if name == "intensity":
            result = VisualisationResult(img, 'gray', "Intensity")
        else:
            result = VisualisationResult(img, "viridis", "DoLP")

        out = func(img, result)

        out = out.resize(
            (out.width * SCALE, out.height * SCALE),
            Image.Resampling.NEAREST
        )

        out.save(TEST_OUT_DIR / f"legend_{name}.png", dpi=(300, 300))

    print("Export complete")