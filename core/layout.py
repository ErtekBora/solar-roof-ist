"""Roof geometry and panel layout: pure, deterministic functions in metric EPSG:32635 (x = East, y = North).

- `roof_faces`: hip (4 faces) or gable (2 faces) built on the footprint's minimum rotated rectangle, each face
  intersected with the real setback polygon (L/U-shaped buildings never get area outside the footprint).
- `split_mixed`: Karma with a drawn terrace; a symmetric gap keeps flat and pitched panels apart.
- `place_panels`: panel grid search (2 orientations x 4 x 4 offsets) with vectorized containment.
- `decide_capacity_mode`: which number wins (drawn obstacles > photo ratio > average share).

The LLM never places panels. UTM grid convergence (~1.4 degrees in the pilot area) is ignored on purpose.

Docs: https://shapely.readthedocs.io/en/stable/reference/shapely.contains_xy.html
      https://shapely.readthedocs.io/en/stable/reference/shapely.prepare.html
      https://shapely.readthedocs.io/en/stable/strtree.html
      https://shapely.readthedocs.io/en/stable/reference/shapely.oriented_envelope.html
"""
import math

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon

from core.building_mode import compass_tr, to_pvgis_aspect
from core.config import (
    DEFAULT_ROOF_PITCH_DEG,
    OBSTACLE_BUFFER_M,
    PANEL_LENGTH_M,
    PANEL_WIDTH_M,
    SETBACK_M,
    TERRACE_GAP_M,
    WINTER_SOLSTICE_DECLINATION_DEG,
)
from core.solar import normalize_aspect

PITCHED_TYPES = ("hip", "gable")
FLAT_SEGMENT = "Düz teras"
MIN_PART_M2 = 0.01  # geometry pieces smaller than this are dropped
_OFFSET_STEPS = 4
_EPS = 1e-9

MSG_KARMA_NEEDS_TERRACE = "Geometriyle netleştirmek için teras alanını da çizin."
MSG_NO_TERRACE_DRAWN = "Teras konumu çizilmedi; teras panelleri yalnızca sayısal olarak hesaplandı"


def face_segment_name(face_name):
    """Segment label of a pitched face (same as core.building_mode)."""
    return f"{face_name} yüzü"


# ---------------------------------------------------------------------- geometry

def clean(geom):
    """Valid polygonal part of `geom` (Polygon or MultiPolygon), or None when empty, invalid or tiny. Never raises."""
    try:
        if geom is None or geom.is_empty:
            return None
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
        parts = [g for g in shapely.get_parts(geom) if g.geom_type == "Polygon" and g.area >= MIN_PART_M2]
        if not parts:
            return None
        out = parts[0] if len(parts) == 1 else shapely.MultiPolygon(parts)
        return out if out.is_valid and not out.is_empty else None
    except Exception:  # GEOS errors on degenerate input: the segment is simply empty
        return None


def area(geom):
    return 0.0 if geom is None else float(geom.area)


def setback_polygon(footprint):
    """footprint.buffer(-SETBACK_M), cleaned (None for a very small roof)."""
    return clean(footprint.buffer(-SETBACK_M))


def _azimuth(dx, dy):
    """North-based clockwise azimuth (degrees) of a metric vector."""
    return math.degrees(math.atan2(dx, dy)) % 360


def _rectangle_axes(footprint):
    """(centre, u, v, a, b): minimum rotated rectangle centre, unit long axis u, unit short axis v, long a, short b."""
    ring = np.asarray(footprint.minimum_rotated_rectangle.exterior.coords)[:4]
    e1, e2 = ring[1] - ring[0], ring[2] - ring[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    if l1 >= l2:
        u, a, v, b = e1 / l1, l1, e2 / l2, l2
    else:
        u, a, v, b = e2 / l2, l2, e1 / l1, l1
    return ring.mean(axis=0), u, v, a, b


def _face(name_vec, polygon_raw, setback):
    az = _azimuth(*name_vec)
    clipped = clean(polygon_raw.intersection(setback)) if setback is not None else None
    return {
        "name": compass_tr(az), "azimuth_deg": az, "aspect": normalize_aspect(to_pvgis_aspect(az)),
        "polygon": clipped, "plan_m2": area(clipped),
    }


def roof_faces(footprint, pitched_type="hip", setback=None):
    """Pitched faces as [{name, azimuth_deg, aspect, polygon, plan_m2}] (plan polygons clipped to the setback).

    Hip: two trapezoids on the long sides and two triangles on the short sides, 45 degree hips in plan, ridge
    length = long side - short side. Gable: the rectangle split along the ridge (long axis).
    """
    if pitched_type not in PITCHED_TYPES:
        raise ValueError(f"pitched_type must be one of {PITCHED_TYPES}")
    setback = setback_polygon(footprint) if setback is None else setback
    c, u, v, a, b = _rectangle_axes(footprint)
    ha, hb = u * a / 2, v * b / 2
    p1, p2, p3, p4 = c + ha + hb, c - ha + hb, c - ha - hb, c + ha - hb
    if pitched_type == "gable":
        return [
            _face(v, Polygon([p1, p2, c - ha, c + ha]), setback),
            _face(-v, Polygon([p3, p4, c + ha, c - ha]), setback),
        ]
    d = u * (a - b) / 2
    r0, r1 = c - d, c + d
    long_pos = Polygon([p1, p2, r0, r1]) if a > b else Polygon([p1, p2, r0])
    long_neg = Polygon([p3, p4, r1, r0]) if a > b else Polygon([p3, p4, r0])
    return [
        _face(v, long_pos, setback),
        _face(-v, long_neg, setback),
        _face(u, Polygon([p4, p1, r1]), setback),
        _face(-u, Polygon([p2, p3, r0]), setback),
    ]


def face_weights(faces):
    """Share of the pitched plan area per face (proportional to the intersected areas); equal when all are 0."""
    total = sum(f["plan_m2"] for f in faces)
    if total <= 0:
        return [1 / len(faces)] * len(faces) if faces else []
    return [f["plan_m2"] / total for f in faces]


def split_mixed(faces, setback, terrace):
    """Karma with a drawn terrace: flat = terrace.buffer(-gap/2) & setback; faces = face - terrace.buffer(+gap/2).

    Returns (flat_polygon or None, faces with updated polygon/plan_m2). Empty results count as 0 m2.
    """
    half = TERRACE_GAP_M / 2
    if terrace is None or setback is None:
        return None, faces
    flat = clean(terrace.buffer(-half).intersection(setback))
    grown = terrace.buffer(half)
    out = []
    for f in faces:
        poly = clean(f["polygon"].difference(grown)) if f["polygon"] is not None else None
        out.append({**f, "polygon": poly, "plan_m2": area(poly)})
    return flat, out


# ----------------------------------------------------------------------- layout

def winter_noon_elevation_deg(lat):
    """Solar elevation at noon on 21 December: 90 - lat - 23.44."""
    return 90.0 - lat - WINTER_SOLSTICE_DECLINATION_DEG


def row_pitch(length_m, tilt_deg, lat):
    """Flat-roof row pitch: L cos(b) + L sin(b) / tan(h_min) (no self-shading at winter-solstice noon)."""
    beta, h = math.radians(tilt_deg), math.radians(winter_noon_elevation_deg(lat))
    return length_m * math.cos(beta) + length_m * math.sin(beta) / math.tan(h)


def _rotate_xy(x, y, deg, ox, oy):
    r = math.radians(deg)
    cs, sn = math.cos(r), math.sin(r)
    dx, dy = x - ox, y - oy
    return ox + dx * cs - dy * sn, oy + dx * sn + dy * cs


def _empty_layout(mount, azimuth_deg, tilt_deg):
    return {
        "panels": np.zeros((0, 4, 2)), "count": 0, "orientation": None, "mount": mount, "azimuth_deg": azimuth_deg,
        "tilt_deg": tilt_deg, "row_pitch_m": None, "depth_m": None,
    }


def place_panels(target, *, mount, tilt_deg, azimuth_deg=180.0, lat=41.0, obstacles=()):
    """Most panels that fit inside `target`, as {panels: (n, 4, 2) metric corners in row order, count, ...}.

    mount "flat": rows run East-West (panels face South), tilted at `tilt_deg`, row pitch from `row_pitch`.
    mount "flush": panels lie on the face (plan depth L cos 30), aligned with the face whose downslope points to
    `azimuth_deg`; rows touch (no spacing). Search: portrait/landscape x 4 x-offsets x 4 y-offsets on a grid
    generated once per set as numpy arrays; a candidate is valid when its 4 corners AND centre are inside
    (shapely.contains_xy on the prepared polygon) and, as an exactness guard for concave roofs, the rectangle is
    covered by the polygon; candidates hitting an obstacle buffered by 0.5 m are dropped with one STRtree bulk
    query. The set with the most panels wins; ties keep the first set (orientation, x-offset, y-offset order).
    """
    target = clean(target)
    if target is None:
        return _empty_layout(mount, azimuth_deg, tilt_deg)
    # Local frame: rotate so that the panel's downslope direction points South (-y): rows run along x.
    rot = -(180.0 - azimuth_deg)
    ox, oy = target.centroid.x, target.centroid.y
    local = affinity.rotate(target, rot, origin=(ox, oy))
    shapely.prepare(local)
    obs = [clean(o.buffer(OBSTACLE_BUFFER_M)) for o in obstacles]
    obs = [affinity.rotate(o, rot, origin=(ox, oy)) for o in obs if o is not None]
    tree = shapely.STRtree(obs) if obs else None
    minx, miny, maxx, maxy = local.bounds

    best = None
    for orientation, (along, slope_len) in (("portrait", (PANEL_WIDTH_M, PANEL_LENGTH_M)),
                                            ("landscape", (PANEL_LENGTH_M, PANEL_WIDTH_M))):
        if mount == "flat":
            depth = slope_len * math.cos(math.radians(tilt_deg))
            pitch = row_pitch(slope_len, tilt_deg, lat)
        else:
            depth = pitch = slope_len * math.cos(math.radians(DEFAULT_ROOF_PITCH_DEG))
        for ix in range(_OFFSET_STEPS):
            xs = np.arange(minx + ix * along / _OFFSET_STEPS, maxx - along + _EPS, along)
            for iy in range(_OFFSET_STEPS):
                ys = np.arange(maxy - iy * pitch / _OFFSET_STEPS - depth, miny - _EPS, -pitch)  # north -> south
                if not len(xs) or not len(ys):
                    continue
                gx, gy = np.meshgrid(xs, ys)  # row-major: rows north to south, west to east inside a row
                x0, y0 = gx.ravel(), gy.ravel()
                x1, y1 = x0 + along, y0 + depth
                ok = (
                    shapely.contains_xy(local, x0, y0) & shapely.contains_xy(local, x1, y0)
                    & shapely.contains_xy(local, x1, y1) & shapely.contains_xy(local, x0, y1)
                    & shapely.contains_xy(local, (x0 + x1) / 2, (y0 + y1) / 2)
                )
                idx = np.flatnonzero(ok)
                if len(idx):
                    boxes = shapely.box(x0[idx], y0[idx], x1[idx], y1[idx])
                    keep = shapely.covers(local, boxes)
                    idx, boxes = idx[keep], boxes[keep]
                    if tree is not None and len(idx):
                        hit = np.zeros(len(idx), dtype=bool)
                        hit[np.unique(tree.query(boxes, predicate="intersects")[0])] = True
                        idx = idx[~hit]
                if best is None or len(idx) > best[0]:
                    best = (len(idx), orientation, x0[idx], y0[idx], x1[idx], y1[idx], pitch, depth)

    if best is None or best[0] == 0:
        return _empty_layout(mount, azimuth_deg, tilt_deg)
    n, orientation, x0, y0, x1, y1, pitch, depth = best
    corners_x = np.stack([x0, x1, x1, x0], axis=1)
    corners_y = np.stack([y0, y0, y1, y1], axis=1)
    wx, wy = _rotate_xy(corners_x, corners_y, -rot, ox, oy)
    return {
        "panels": np.stack([wx, wy], axis=2), "count": int(n), "orientation": orientation, "mount": mount,
        "azimuth_deg": azimuth_deg, "tilt_deg": tilt_deg, "row_pitch_m": pitch, "depth_m": depth,
    }


def panel_polygons(layout, limit=None):
    """Shapely polygons of the first `limit` panels (row order); all when limit is None."""
    panels = layout["panels"] if limit is None else layout["panels"][: max(0, int(limit))]
    return list(shapely.polygons(panels)) if len(panels) else []


def hatch_lines(polygon, spacing_m=1.0):
    """Diagonal hatch lines (metric LineStrings) clipped to `polygon`, used to mark dropped faces on the map."""
    polygon = clean(polygon)
    if polygon is None:
        return []
    minx, miny, maxx, maxy = polygon.bounds
    span = (maxx - minx) + (maxy - miny)
    lines = shapely.linestrings([[(minx + t, miny), (minx + t - span, miny + span)] for t in np.arange(0, 2 * span, spacing_m)])
    out = []
    for g in shapely.intersection(lines, polygon):
        out += [p for p in shapely.get_parts(g) if p.geom_type == "LineString" and not p.is_empty]
    return out


# ---------------------------------------------------------------- roof model

def roof_model(footprint, pitched_type="hip", terrace=None, structure="pitched"):
    """Segment polygons of one roof scenario: {"flat": polygon|None, "faces": [...], "setback": polygon|None}.

    structure "flat": the whole setback polygon is flat. "pitched": the faces only. "mixed": with a terrace, the
    terrace part is flat and the faces lose the terrace (symmetric gap); without a terrace, the faces only (the
    terrace share stays numeric).
    """
    setback = setback_polygon(footprint)
    faces = roof_faces(footprint, pitched_type, setback)
    if structure == "flat":
        return {"flat": setback, "faces": [], "setback": setback}
    if structure == "mixed" and terrace is not None:
        flat, faces = split_mixed(faces, setback, terrace)
        return {"flat": flat, "faces": faces, "setback": setback}
    return {"flat": None, "faces": faces, "setback": setback}


def layout_model(model, *, flat_tilt_deg, lat, obstacles=(), skip_faces=()):
    """Panel layout per segment: {segment name: layout}. Faces named in `skip_faces` (dropped or unavailable)
    get no layout."""
    out = {}
    if model["flat"] is not None:
        out[FLAT_SEGMENT] = place_panels(model["flat"], mount="flat", tilt_deg=flat_tilt_deg, lat=lat, obstacles=obstacles)
    for f in model["faces"]:
        name = face_segment_name(f["name"])
        if name in skip_faces or f["polygon"] is None:
            continue
        out[name] = place_panels(
            f["polygon"], mount="flush", tilt_deg=DEFAULT_ROOF_PITCH_DEG, azimuth_deg=f["azimuth_deg"], lat=lat,
            obstacles=obstacles,
        )
    return out


# --------------------------------------------------------- which number wins

def decide_capacity_mode(structure, n_obstacles, empty_confirmed, terrace_drawn):
    """(mode, message): "geometry" (panel count x 0.40 kWp), "numeric" (photo or average share) or
    "numeric_clean" (numeric with obstacle ratio 0).

    Karma needs a terrace for geometry: drawn obstacles without a terrace keep the numeric result (with a hint),
    and "Engel yok, çatı boş" without a terrace means numeric with ratio 0.
    """
    marked = n_obstacles > 0 or empty_confirmed
    if not marked:
        return "numeric", None
    if structure == "mixed" and not terrace_drawn:
        if n_obstacles > 0:
            return "numeric", MSG_KARMA_NEEDS_TERRACE
        return "numeric_clean", MSG_KARMA_NEEDS_TERRACE
    return "geometry", None
