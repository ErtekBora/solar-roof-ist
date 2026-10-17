"""Gemini roof perception: surface, obstacles, roof type and image view.

Gemini docs: https://ai.google.dev/gemini-api/docs/image-understanding (box_2d = [ymin, xmin, ymax, xmax],
normalized 0-1000) and https://ai.google.dev/gemini-api/docs/structured-output (JSON schema output).

Coordinate order: Gemini gives [y, x]; Shapely and Pillow need (x, y). The two helpers `_point_to_px`
and `_box_to_px` are the only places where that conversion happens.
"""
import copy
import hashlib
import io
import json
import math
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from shapely.geometry import Point, Polygon
from shapely.geometry import box as shapely_box

from core.config import (
    MAX_IMAGE_SIDE_PX,
    OBSTACLE_CLASSES,
    VISION_MODEL,
    MissingApiKeyError,
    get_gemini_api_key,
)

OBSTACLE_LABELS_TR = {
    "water tank": "Su deposu",
    "air conditioner unit": "Klima",
    "chimney": "Baca",
    "solar panel": "Güneş paneli",
    "satellite dish": "Uydu çanağı",
    "skylight": "Çatı penceresi",
    "other": "Diğer",
}

MIN_ROOF_IMAGE_FRACTION = 0.02
MIN_POLYGON_BOX_FRACTION = 0.25
MIN_OBSTACLE_OVERLAP = 0.30
MAX_OBSTACLE_RATIO = 0.95
RETRY_WAIT_S = 3

_cache = {}
_sleep = time.sleep  # wait before the 429/503 retry; patched in tests

_MSG_UNREADABLE = "Dosya okunamadı; geçerli bir JPG/PNG yükleyin."
_MSG_NO_ROOF = "Fotoğrafta geçerli bir çatı yüzeyi tespit edilemedi."
_MSG_QUOTA = "Gemini kullanım kotası doldu; lütfen biraz sonra tekrar deneyin."
_MSG_BUSY = "Gemini şu anda yoğun; lütfen biraz sonra tekrar deneyin."
_MSG_API = "Fotoğraf analizi şu an yapılamadı; lütfen tekrar deneyin."
_MSG_BAD_JSON = "Gemini yanıtı okunamadı; lütfen tekrar deneyin."


class VisionError(RuntimeError):
    """Roof perception failed; the message is Turkish and UI-safe."""


def clear_cache():
    """Drop all cached detections."""
    _cache.clear()


# ---------------------------------------------------------------- coordinates

def _point_to_px(point, width, height):
    """Gemini [y, x] (0-1000) -> pixel (x, y)."""
    y, x = point
    return (x / 1000 * width, y / 1000 * height)


def _box_to_px(box, width, height):
    """Gemini [ymin, xmin, ymax, xmax] (0-1000) -> pixel (x1, y1, x2, y2)."""
    ymin, xmin, ymax, xmax = box
    return (xmin / 1000 * width, ymin / 1000 * height, xmax / 1000 * width, ymax / 1000 * height)


def _is_number(v):
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def _clip_box(raw):
    """Return [ymin, xmin, ymax, xmax] clipped to [0, 1000], or None if it is not 4 finite numbers."""
    try:
        values = list(raw)
    except TypeError:
        return None
    if len(values) != 4 or not all(_is_number(v) for v in values):
        return None
    return [min(1000.0, max(0.0, float(v))) for v in values]


def _normalize_confidence(value):
    """None/invalid -> 0.5; 1 < v <= 100 is read as a percentage; clipped to [0, 1]."""
    if value is None or not _is_number(value):
        return 0.5
    if 1 < value <= 100:
        value = value / 100
    return min(1.0, max(0.0, float(value)))


def _normalize_label(label):
    text = str(label).strip().lower() if label is not None else ""
    return text if text in OBSTACLE_CLASSES else "other"


# ------------------------------------------------------------------- image IO

def _load_image(image_bytes):
    """Open, orient, convert to RGB and downscale; every later step uses this image."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, TypeError) as exc:
        raise VisionError(_MSG_UNREADABLE) from exc
    if max(img.size) > MAX_IMAGE_SIDE_PX:
        img.thumbnail((MAX_IMAGE_SIDE_PX, MAX_IMAGE_SIDE_PX))
    return img


# --------------------------------------------------------------------- Gemini

def _prompt():
    classes = ", ".join(f'"{c}"' for c in OBSTACLE_CLASSES)
    return (
        "You analyse one photo of a building roof for a rooftop solar assessment. Return JSON only. "
        "All coordinates are normalized 0-1000 relative to the image.\n"
        "- roof_polygon: outline of the MAIN roof (the largest, most central roof) as an ordered list of "
        "[y, x] points, clockwise, no self-intersections.\n"
        "- roof_box: [ymin, xmin, ymax, xmax] of the same main roof.\n"
        f"- obstacles: objects standing on the main roof that block panels. Each: label (one of {classes} "
        'or "other"), box_2d [ymin, xmin, ymax, xmax], confidence 0-1. Use an empty list if there are none.\n'
        '- roof_type: "flat", "pitched" or "mixed".\n'
        '- image_view: "top-down" or "oblique".\n'
        "- multiple_buildings: true if more than one building's roof is substantially visible.\n"
        "Rules: shadows, roof texture, edges and parapets are NOT obstacles. Only objects on the main roof "
        'count. Solar water heaters count as "solar panel".\n'
        'Label rules: "solar panel" means a dark, rectangular panel with a visible grid of cells, or a '
        "flat-plate solar water heater (often with a tank). Awnings, pergolas, terrace canopies, "
        'winter-garden roofs and glass shelters are "other", NOT "solar panel". A rooftop penthouse, '
        'stair housing or elevator housing is "other".'
    )


def _response_schema():
    number_list = {"type": "ARRAY", "items": {"type": "NUMBER"}}
    return {
        "type": "OBJECT",
        "properties": {
            "roof_polygon": {"type": "ARRAY", "items": number_list},
            "roof_box": number_list,
            "obstacles": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {"type": "STRING"},
                        "box_2d": number_list,
                        "confidence": {"type": "NUMBER"},
                    },
                    "required": ["label", "box_2d", "confidence"],
                },
            },
            "roof_type": {"type": "STRING", "enum": ["flat", "pitched", "mixed"]},
            "image_view": {"type": "STRING", "enum": ["top-down", "oblique"]},
            "multiple_buildings": {"type": "BOOLEAN"},
        },
        "required": ["roof_polygon", "roof_box", "obstacles", "roof_type", "image_view", "multiple_buildings"],
    }


def _get_client():
    """Create the Gemini client lazily (heavy import); patched in tests."""
    from google import genai

    try:
        return genai.Client(api_key=get_gemini_api_key())
    except MissingApiKeyError as exc:
        raise VisionError(str(exc)) from exc


def _error_status(exc):
    """Return 429 or 503 if the SDK exception means that, else the int code or None."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and not isinstance(code, bool) and code not in (429, 503):
        return code
    if code in (429, 503):
        return code
    text = str(exc)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return 429
    if "503" in text or "UNAVAILABLE" in text:
        return 503
    return None


def _generate(client, jpeg_bytes):
    """One generate_content call; retry ONCE after a wait on HTTP 429/503."""
    from google.genai import types

    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=_response_schema(), temperature=0
    )
    contents = [types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"), _prompt()]
    for attempt in range(2):
        try:
            return client.models.generate_content(model=VISION_MODEL, contents=contents, config=config).text
        except Exception as exc:  # the SDK raises exceptions, not HTTP responses
            status = _error_status(exc)
            if status in (429, 503) and attempt == 0:
                _sleep(RETRY_WAIT_S)
                continue
            if status == 429:
                raise VisionError(_MSG_QUOTA) from exc
            if status == 503:
                raise VisionError(_MSG_BUSY) from exc
            raise VisionError(_MSG_API) from exc


def _ask_gemini(jpeg_bytes):
    """Return the parsed JSON dict; retry once on malformed JSON."""
    client = _get_client()
    for attempt in range(2):
        text = _generate(client, jpeg_bytes)
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            data = None
        if isinstance(data, dict):
            return data
    raise VisionError(_MSG_BAD_JSON)


# ------------------------------------------------------------- post-processing

def _clean_points(raw, width, height):
    """Raw [y, x] list -> distinct pixel (x, y) points (closing/consecutive duplicates dropped), or None."""
    try:
        pts = []
        for p in raw:
            p = list(p)
            if len(p) != 2 or not all(_is_number(v) for v in p):
                return None
            y, x = (min(1000.0, max(0.0, float(v))) for v in p)
            pts.append(_point_to_px([y, x], width, height))
    except TypeError:
        return None
    dedup = []
    for p in pts:
        if not dedup or p != dedup[-1]:
            dedup.append(p)
    if len(dedup) > 1 and dedup[0] == dedup[-1]:
        dedup.pop()
    return dedup if len(set(dedup)) >= 3 else None


def _validate_polygon(points, box_area_px, width, height):
    """Return a valid Shapely polygon or None; box_area_px is None when the box is invalid."""
    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
        if poly.geom_type != "Polygon" or poly.is_empty or not poly.is_valid:
            return None
        area = poly.area
    except Exception:  # shapely raises various errors on degenerate input
        return None
    if area < MIN_ROOF_IMAGE_FRACTION * width * height:
        return None
    if box_area_px is not None and area < MIN_POLYGON_BOX_FRACTION * box_area_px:
        return None
    return poly


def _extract_roof(data, width, height):
    """Return (roof_polygon_px, roof_area_px, roof_source); raise VisionError if no valid roof."""
    box = _clip_box(data.get("roof_box"))
    box_px = None
    if box is not None:
        x1, y1, x2, y2 = _box_to_px(box, width, height)
        if x2 - x1 > 0 and y2 - y1 > 0 and (x2 - x1) * (y2 - y1) >= MIN_ROOF_IMAGE_FRACTION * width * height:
            box_px = (x1, y1, x2, y2)
    box_area = None if box_px is None else (box_px[2] - box_px[0]) * (box_px[3] - box_px[1])

    points = _clean_points(data.get("roof_polygon"), width, height)
    poly = _validate_polygon(points, box_area, width, height) if points else None
    if poly is not None:
        return list(poly.exterior.coords)[:-1], poly.area, "polygon"
    if box_px is None:
        raise VisionError(_MSG_NO_ROOF)
    x1, y1, x2, y2 = box_px
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], box_area, "box"


def _filter_obstacles(raw, roof_polygon_px, width, height):
    """Keep obstacles whose centre is on the roof or that overlap it by >= 30% of their area."""
    roof = Polygon(roof_polygon_px)
    kept = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        box = _clip_box(item.get("box_2d"))
        if box is None:
            continue
        x1, y1, x2, y2 = _box_to_px(box, width, height)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        rect = shapely_box(x1, y1, x2, y2)
        centre_in = roof.contains(Point((x1 + x2) / 2, (y1 + y2) / 2))
        if not centre_in and roof.intersection(rect).area / rect.area < MIN_OBSTACLE_OVERLAP:
            continue
        kept.append(
            {
                "id": len(kept),
                "box_px": (x1, y1, x2, y2),
                "label": _normalize_label(item.get("label")),
                "confidence": _normalize_confidence(item.get("confidence")),
            }
        )
    return kept


def compute_obstacle_ratio(roof_polygon_px, obstacle_boxes_px, width, height):
    """Area of (union of obstacles ∩ roof) / roof area, clipped to [0, 0.95].

    Both shapes are rasterized on the pixel grid by testing pixel centres, so the result is exact for
    axis-aligned boxes (Pillow's polygon fill would include the boundary pixels and inflate every area).
    """
    xs = np.arange(width) + 0.5
    ys = np.arange(height) + 0.5
    roof_mask = _pixel_centre_mask(Polygon(roof_polygon_px), xs, ys)
    roof_area = int(roof_mask.sum())
    if roof_area == 0:
        raise ValueError("roof area is zero")
    if not obstacle_boxes_px:
        return 0.0
    obstacles = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in obstacle_boxes_px:
        obstacles |= (ys[:, None] >= y1) & (ys[:, None] < y2) & (xs[None, :] >= x1) & (xs[None, :] < x2)
    ratio = int((obstacles & roof_mask).sum()) / roof_area
    return min(MAX_OBSTACLE_RATIO, max(0.0, ratio))


def _pixel_centre_mask(poly, xs, ys):
    import shapely

    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    minx, miny, maxx, maxy = poly.bounds
    xi = np.nonzero((xs >= minx) & (xs <= maxx))[0]
    yi = np.nonzero((ys >= miny) & (ys <= maxy))[0]
    if len(xi) and len(yi):
        gx, gy = np.meshgrid(xs[xi], ys[yi])
        mask[np.ix_(yi, xi)] = shapely.contains_xy(poly, gx, gy)
    return mask


# ------------------------------------------------------------------ public API

def detect_roof_objects(image_bytes):
    """Detect the main roof, obstacles, roof type and image view in a roof photo.

    Returns {roof_type, image_view, multiple_buildings, width, height, roof_source, roof_polygon_px,
    roof_area_px, obstacles[{id, box_px, label, confidence}], obstacle_ratio, model, image_hash}.
    Raises VisionError (Turkish message) on any failure.
    """
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    if image_hash in _cache:
        return copy.deepcopy(_cache[image_hash])

    img = _load_image(image_bytes)
    width, height = img.size
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    data = _ask_gemini(buf.getvalue())

    roof_polygon_px, roof_area_px, roof_source = _extract_roof(data, width, height)
    obstacles = _filter_obstacles(data.get("obstacles"), roof_polygon_px, width, height)
    roof_type = data.get("roof_type") if data.get("roof_type") in ("flat", "pitched", "mixed") else "flat"
    image_view = data.get("image_view") if data.get("image_view") in ("top-down", "oblique") else "oblique"
    result = {
        "roof_type": roof_type,
        "image_view": image_view,
        "multiple_buildings": data.get("multiple_buildings") is True,
        "width": width,
        "height": height,
        "roof_source": roof_source,
        "roof_polygon_px": roof_polygon_px,
        "roof_area_px": roof_area_px,
        "obstacles": obstacles,
        "obstacle_ratio": compute_obstacle_ratio(roof_polygon_px, [o["box_px"] for o in obstacles], width, height),
        "model": VISION_MODEL,
        "image_hash": image_hash,
    }
    _cache[image_hash] = result
    return copy.deepcopy(result)


MIN_OVERLAY_SIDE_PX = 800
_ROOF_GREEN = (0, 200, 0)
_INCLUDED_RED = (230, 30, 30)
_EXCLUDED_GREY = (150, 150, 150)


def _badge_font(size):
    """TrueType font if the system has one, else Pillow's built-in scalable default."""
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def draw_overlay(image_bytes, result, included_ids=None):
    """Return PNG bytes: roof outline green, numbered obstacle boxes (red included, grey excluded).

    Images with a long side under 800 px are upscaled (LANCZOS) first; coordinates, line widths and
    the badge font scale with the image. The badge number is the obstacle id shown in the table.
    """
    img = _load_image(image_bytes)
    scale = max(1.0, MIN_OVERLAY_SIDE_PX / max(img.size))
    if scale > 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    long_side = max(img.size)
    line_w = max(2, round(long_side / 250))
    font = _badge_font(max(12, round(long_side / 45)))
    draw = ImageDraw.Draw(img)

    draw.polygon([(x * scale, y * scale) for x, y in result["roof_polygon_px"]], outline=_ROOF_GREEN, width=line_w + 1)
    for obstacle in result["obstacles"]:
        x1, y1, x2, y2 = (v * scale for v in obstacle["box_px"])
        included = included_ids is None or obstacle["id"] in included_ids
        colour = _INCLUDED_RED if included else _EXCLUDED_GREY
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=line_w if included else max(1, line_w // 2))
        text = str(obstacle["id"])
        tl, tt, tr, tb = draw.textbbox((0, 0), text, font=font)
        pad = max(2, line_w)
        bw, bh = tr - tl + 2 * pad, tb - tt + 2 * pad
        bx = min(max(0, x1), img.width - bw)
        by = y1 - bh if y1 - bh >= 0 else y1
        draw.rectangle([bx, by, bx + bw, by + bh], fill=colour)
        draw.text((bx + pad - tl, by + pad - tt), text, fill=(255, 255, 255), font=font)
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()
