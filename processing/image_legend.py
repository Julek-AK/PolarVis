# Builtins
from math import pi

# External Libraries
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from colorsys import hsv_to_rgb



# =============================================
# CONFIG
# =============================================

BOX_PADDING = 12
INNER_PADDING = 8
HEADER_HEIGHT = 22

BACKGROUND = (0, 0, 0, 180)
TEXT_COLOR = (255, 255, 255, 255)
BORDER_COLOR = (255, 255, 255, 180)

FONT = ImageFont.load_default()


# =============================================
# HELPERS
# =============================================

def _create_overlay_box(width: int, height: int) -> Image.Image:
    return Image.new(
        "RGBA",
        (width, height),
        BACKGROUND
    )


def _paste_box(
    base: Image.Image,
    overlay: Image.Image,
    position: str = "bottom_right"
) -> Image.Image:

    base = base.convert("RGBA")

    bw, bh = base.size
    ow, oh = overlay.size

    if position == "bottom_right":
        x = bw - ow - BOX_PADDING
        y = bh - oh - BOX_PADDING

    elif position == "bottom_left":
        x = BOX_PADDING
        y = bh - oh - BOX_PADDING

    elif position == "top_right":
        x = bw - ow - BOX_PADDING
        y = BOX_PADDING

    elif position == "top_left":
        x = BOX_PADDING
        y = BOX_PADDING

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
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    if align == "center":
        x -= w // 2
        y -= h // 2
    elif align == "right":
        x -= w

    rect = [
        x - padding,
        y - padding,
        x + w + padding,
        y + h + padding
    ]

    draw.rectangle(rect, fill=(0, 0, 0, 160))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))


# =============================================
# SCALAR LEGEND
# =============================================

def scalar_legend(image, result, position="bottom_right"):
    cmap = plt.get_cmap(result.cmap)

    bar_w = 28
    bar_h = 220
    label_w = 30

    box_w = bar_w + label_w + INNER_PADDING * 3
    box_h = bar_h + INNER_PADDING * 2 + HEADER_HEIGHT

    overlay = _create_overlay_box(box_w, box_h)
    draw = ImageDraw.Draw(overlay)

    # Header
    label(
        draw,
        INNER_PADDING,
        INNER_PADDING,
        result.label,
        FONT,
        align="left"
    )

    y0 = INNER_PADDING + HEADER_HEIGHT

    # Gradient
    grad = _draw_vertical_gradient(bar_h, bar_w, cmap)
    overlay.paste(grad, (INNER_PADDING, y0))

    # Border
    draw.rectangle(
        [
            INNER_PADDING,
            y0,
            INNER_PADDING + bar_w,
            y0 + bar_h
        ],
        outline=BORDER_COLOR,
        width=1
    )

    # Labels
    bx = INNER_PADDING + bar_w + 8
    by = y0 - 2

    label(draw, bx, by             , "1.00", FONT, align="center")
    label(draw, bx, by + bar_h*0.25, "0.75", FONT, align="center")
    label(draw, bx, by + bar_h*0.5 , "0.50", FONT, align="center")
    label(draw, bx, by + bar_h*0.75, "0.25", FONT, align="center")
    label(draw, bx, by + bar_h     , "0.00", FONT, align="center")

    return _paste_box(image, overlay, position)


# =============================================
# ANGLE LEGEND (AoP)
# =============================================

def angle_legend(image, result, position="bottom_right"):
    size = 180

    overlay = _create_overlay_box(size, size)
    draw = ImageDraw.Draw(overlay)

    cx = size // 2
    cy = size // 2

    r_outer = 65
    r_inner = 40

    # Header
    label(
        draw,
        INNER_PADDING,
        INNER_PADDING,
        "AoP",
        FONT,
        align="left"
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
        r_text = r_outer + 14

        x = cx + r_text * np.cos(theta)
        y = cy - r_text * np.sin(theta)

        label(draw, x, y, text, FONT, align="center")

    return _paste_box(image, overlay, position)


# =============================================
# POLARIMETRIC LEGEND (AoP + DoLP + Intensity)
# =============================================

def polarimetric_legend(image, result, position="bottom_right"):
    size = 240

    overlay = _create_overlay_box(size, size)
    draw = ImageDraw.Draw(overlay)

    cx = size // 2
    cy = size // 2

    radius = 85

    # Header
    label(
        draw,
        INNER_PADDING,
        INNER_PADDING,
        "AoP DoLP Intensity",
        FONT,
        align="left"
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

        label(draw, x, y, text, FONT, align="center")

    # DoLP labels
    dolp_labels = [0.25, 0.50, 0.75, 1.00]
    for v in dolp_labels:
        r = radius * v

        x = cx
        y = cy + r

        label(draw, x, y, f"{v:.2f}", FONT, align="center")

    # Intensity bar
    bar_w = 14
    bar_h = 120

    grad = np.linspace(1, 0, bar_h)[:, None]
    grad = np.repeat(grad, bar_w, axis=1)
    intensity_bar = Image.fromarray((grad * 255).astype(np.uint8)).convert("RGBA")

    bx = cx + radius + 12
    by = cy - bar_h // 2

    overlay.paste(intensity_bar, (bx, by))

    label(draw, bx, by + bar_h, "0.0", FONT, align="center")
    label(draw, bx, by + bar_h/2, "0.5", FONT, align="center")
    label(draw, bx, by, "1.0", FONT, align="center")

    return _paste_box(image, overlay, position)


# =============================================
# POLAR-ONLY LEGEND (AoP + DoLP only)
# =============================================

def polar_only_legend(image, result, position="bottom_right"):
    size = 200

    overlay = _create_overlay_box(size, size)
    draw = ImageDraw.Draw(overlay)

    cx = size // 2
    cy = size // 2

    radius = 80

    # Header
    label(
        draw,
        INNER_PADDING,
        INNER_PADDING,
        "AoP DoLP",
        FONT,
        align="left"
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
        r_text = radius + 14

        x = cx + r_text * np.cos(theta)
        y = cy - r_text * np.sin(theta)

        label(draw, x, y, text, FONT, align="center")

    # DoLP labels
    dolp_labels = [0.25, 0.50, 0.75, 1.00]
    for v in dolp_labels:
        r = radius * v

        x = cx
        y = cy + r

        label(draw, x, y, f"{v:.2f}", FONT, align="center")

    return _paste_box(image, overlay, position)



if __name__ == "__main__":
    from paths import TEST_OUT_DIR
    from processing.image_visualisation import VisualisationResult

    SCALE = 5

    legends = {
        "polar_only": polar_only_legend,
        "polarimetric": polarimetric_legend,
        "angle": angle_legend,
        "scalar": scalar_legend,
    }

    for name, func in legends.items():
        img = Image.new("RGB", (300, 300), (0, 0, 0))
        result = VisualisationResult(img, 'gist_gray', "Intensity")

        out = func(img, result)

        out = out.resize(
            (out.width * SCALE, out.height * SCALE),
            Image.Resampling.NEAREST
        )

        out.save(TEST_OUT_DIR / f"{name}_legend.png", dpi=(300, 300))

    print("Export complete.")