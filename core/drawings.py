"""User drawings on the roof view (obstacles and one terrace): parsing, labelling and generation bookkeeping.

Pure functions over a plain dict kept in st.session_state (no Streamlit import).

Input: st_folium's `all_drawings`, i.e. Leaflet.draw's `drawnItems.toGeoJSON().features` (streamlit-folium 0.27.4
sets it on draw:created / draw:edited / draw:deleted; its default is None). Each item is a GeoJSON Feature
{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[lng, lat], ...]]}};
rectangles also arrive as Polygons. There is no stable feature id, so shapes are matched by geometry with a
tolerance (area within 0.1 m2 and centroid within 0.1 m), never by raw coordinate lists.
Docs: https://folium.streamlit.app/draw_support
      https://leaflet.github.io/Leaflet.draw/docs/leaflet-draw-latest.html
      https://pyproj4.github.io/pyproj/stable/api/transformer.html

Generations: the roof map's st_folium key is f"roof_view_{gen}". Leaflet only knows the shapes drawn since the
map was last built, so a shape missing from the returned set is a deletion ONLY if it was first seen in the
current generation. None or [] on a fresh generation is "no information". Any real change to the stored shapes
(deletion, terrace replacement, clear, undo) bumps `gen` so the map is rebuilt with only our stored shapes.

Imagery alignment offset (display only): tall buildings look displaced on off-nadir imagery, so the user can shift
the overlays by a metric (dx east, dy north) offset kept in `state["offset"]`. Stored shapes stay in the footprint
frame; shapes returned by the map are in the displayed frame and are translated by -offset on the way in, the
overlays by +offset on the way out. Changing the offset bumps `gen` (the map is rebuilt) but no stored shape changes.

Click-to-align: while `state["align_active"]` the roof map has no Draw toolbar and also returns `last_clicked`; the
next click sets the offset (see apply_click) and turns the mode off. The last handled click is kept in `last_click`.
"""
from functools import lru_cache

import shapely
from shapely.affinity import translate
from shapely.geometry import Polygon

from core.layout import clean

OBSTACLE, TERRACE = "obstacle", "terrace"
MATCH_AREA_M2 = 0.1
MATCH_CENTROID_M = 0.1
MIN_SHAPE_M2 = 0.05
OFFSET_MAX_M = 30.0
OFFSET_STEP_M = 1.0
OFFSET_STEP_BIG_M = 5.0
MSG_OUTSIDE = "Çizim bina sınırının dışında kaldığı için yok sayıldı."


@lru_cache(maxsize=1)
def _transformers():
    from pyproj import Transformer

    return (
        Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True),
        Transformer.from_crs("EPSG:32635", "EPSG:4326", always_xy=True),
    )


def to_metric(lons, lats):
    return _transformers()[0].transform(lons, lats)


def to_wgs84(xs, ys):
    return _transformers()[1].transform(xs, ys)


def shift_metric(geom, offset):
    """`geom` translated by offset (dx east, dy north) in metres; the same object when the offset is zero."""
    if not offset or (offset[0] == 0 and offset[1] == 0):
        return geom
    return translate(geom, xoff=offset[0], yoff=offset[1])


def polygon_to_latlon(poly, offset=(0.0, 0.0)):
    """Exterior rings of a metric (Multi)Polygon as [[[lat, lon], ...], ...] for folium (shifted by `offset`)."""
    rings = []
    for part in shapely.get_parts(shift_metric(poly, offset)):
        xs, ys = part.exterior.xy
        lons, lats = to_wgs84(list(xs), list(ys))
        rings.append([[float(la), float(lo)] for lo, la in zip(lons, lats)])
    return rings


def feature_to_metric(feature, offset=(0.0, 0.0)):
    """Metric Polygon of a GeoJSON Polygon feature, or None (other types, broken or tiny shapes).

    The feature is in the displayed frame; the result is translated by -offset into the footprint frame."""
    try:
        geom = feature["geometry"]
        if geom["type"] != "Polygon":
            return None
        ring = geom["coordinates"][0]
        lons, lats = zip(*[(float(p[0]), float(p[1])) for p in ring])
        xs, ys = to_metric(list(lons), list(lats))
        poly = clean(Polygon([(x - offset[0], y - offset[1]) for x, y in zip(xs, ys)]))
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if poly is None or poly.area < MIN_SHAPE_M2:
        return None
    return poly


def same_shape(a, b):
    """Tolerant geometric match (vertex order and start point do not matter)."""
    return abs(a.area - b.area) <= MATCH_AREA_M2 and a.centroid.distance(b.centroid) <= MATCH_CENTROID_M


def new_state(building_id):
    return {
        "building": building_id, "gen": 0, "next_id": 1, "obstacles": [], "terrace": None, "ignored": [],
        "empty_confirmed": False, "notice": None,
        "offset": (0.0, 0.0), "align_active": False, "last_click": None,
    }


def stored_shapes(state):
    return [*state["obstacles"], *([state["terrace"]] if state["terrace"] else [])]


def map_key(state):
    return f"roof_view_{state['gen']}"


def _bump(state):
    state["gen"] += 1
    state["ignored"] = []


def sync(state, features, mode, footprint):
    """Merge st_folium's returned drawings into `state`. Returns {"changed": bool, "bump": bool}.

    New shapes get the active `mode` (OBSTACLE or TERRACE) once and keep it; they are clipped to the footprint
    (a shape fully outside is ignored with a notice). A new terrace replaces the old one. A current-generation
    shape missing from the returned set was deleted with the Draw toolbar.
    """
    if features is None:
        return {"changed": False, "bump": False}
    offset = state.get("offset", (0.0, 0.0))
    polys = [p for p in (feature_to_metric(f, offset) for f in features) if p is not None]
    gen = state["gen"]
    current = [s for s in stored_shapes(state) if s["gen"] == gen]
    matched, changed, bump = set(), False, False
    for poly in polys:
        hit = next((s for s in stored_shapes(state) if same_shape(s["raw"], poly)), None)
        if hit is not None:
            matched.add(hit["id"])
            continue
        if any(same_shape(i, poly) for i in state["ignored"]):
            continue
        clipped = clean(poly.intersection(footprint)) if footprint is not None else None
        if clipped is None:
            state["ignored"].append(poly)
            state["notice"] = MSG_OUTSIDE
            continue
        shape = {"id": state["next_id"], "kind": mode, "raw": poly, "geom": clipped, "gen": gen}
        state["next_id"] += 1
        matched.add(shape["id"])
        if mode == TERRACE:
            if state["terrace"] is not None:
                bump = True  # the old terrace may still be on the Leaflet map: rebuild it
            state["terrace"] = shape
        else:
            state["obstacles"].append(shape)
            state["empty_confirmed"] = False
        changed = True
    for shape in current:
        if shape["id"] not in matched:
            _remove(state, shape["id"])
            changed = bump = True
    if bump:
        _bump(state)
    return {"changed": changed, "bump": bump}


def move_offset(state, dx, dy):
    """Shift the imagery alignment by (dx, dy) metres, clamped to +-OFFSET_MAX_M per axis. A real change rebuilds the
    map (new `gen`) without touching any stored shape, so the re-rendered shapes are never new or deleted drawings."""
    ox, oy = state.get("offset", (0.0, 0.0))
    new = (_clamp_offset(ox + dx), _clamp_offset(oy + dy))
    if new != (ox, oy):
        state["offset"] = new
        _bump(state)


def _clamp_offset(value):
    return max(-OFFSET_MAX_M, min(OFFSET_MAX_M, value))


def start_align(state):
    """Enter click-to-align mode. The map is rebuilt (new `gen`) without the Draw toolbar."""
    if not state.get("align_active"):
        state["align_active"] = True
        _bump(state)


def cancel_align(state):
    """Leave click-to-align mode without changing the offset; the map is rebuilt with the Draw toolbar."""
    if state.get("align_active"):
        state["align_active"] = False
        _bump(state)


def apply_click(state, click, footprint):
    """Click-to-align: offset = clicked point - footprint centroid (EPSG:32635), rounded to 1 m, clamped per axis.

    `click` is st_folium's last_clicked ({"lat", "lng"}). Does nothing unless the mode is active, when the click is
    missing/malformed, or when it repeats the last handled click. A handled click turns the mode off and rebuilds the
    map (new `gen`). Never touches the stored shapes. Returns True when the offset was set."""
    if not state.get("align_active") or footprint is None or not isinstance(click, dict):
        return False
    try:
        lat, lng = float(click["lat"]), float(click["lng"])
    except (KeyError, TypeError, ValueError):
        return False
    if state.get("last_click") == (lat, lng):
        return False
    x, y = to_metric(lng, lat)
    centroid = footprint.centroid
    state["offset"] = (_clamp_offset(float(round(x - centroid.x))), _clamp_offset(float(round(y - centroid.y))))
    state["last_click"], state["align_active"] = (lat, lng), False
    _bump(state)
    return True


def reset_offset(state):
    if state.get("offset", (0.0, 0.0)) != (0.0, 0.0):
        state["offset"] = (0.0, 0.0)
        _bump(state)


def offset_text(offset):
    """'Kaydırma: 3 m D, 5 m K' (D/B = east/west, K/G = north/south); 'Kaydırma: yok' at zero."""
    dx, dy = offset
    parts = []
    if dx:
        parts.append(f"{abs(dx):g} m {'D' if dx > 0 else 'B'}")
    if dy:
        parts.append(f"{abs(dy):g} m {'K' if dy > 0 else 'G'}")
    return "Kaydırma: " + (", ".join(parts) if parts else "yok")


def _remove(state, shape_id):
    state["obstacles"] = [s for s in state["obstacles"] if s["id"] != shape_id]
    if state["terrace"] is not None and state["terrace"]["id"] == shape_id:
        state["terrace"] = None


def clear_all(state):
    """'Engelleri temizle': every stored shape and the 'no obstacles' confirmation go; the map is rebuilt."""
    state["obstacles"], state["terrace"], state["empty_confirmed"] = [], None, False
    _bump(state)


def confirm_empty(state):
    """'Engel yok, çatı boş': no obstacles on the roof (drawn obstacles are removed; the terrace stays)."""
    had = bool(state["obstacles"])
    state["obstacles"], state["empty_confirmed"] = [], True
    if had:
        _bump(state)


def undo_last(state):
    """'Son çizimi geri al': remove only the most recently added shape. Returns True when one was removed."""
    shapes = stored_shapes(state)
    if not shapes:
        return False
    _remove(state, max(s["id"] for s in shapes))
    _bump(state)
    return True


def obstacle_geoms(state):
    return [s["geom"] for s in state["obstacles"]]


def terrace_geom(state):
    return state["terrace"]["geom"] if state["terrace"] else None


def summary(state):
    """{"count", "area_m2" (union, clipped to the footprint), "terrace", "marked"} for the result and explainer."""
    geoms = obstacle_geoms(state)
    area = float(shapely.union_all(geoms).area) if geoms else 0.0
    return {
        "count": len(geoms), "area_m2": area, "terrace": state["terrace"] is not None,
        "marked": bool(geoms) or state["empty_confirmed"],
    }


def fingerprint(state):
    """Hashable identity of the stored shapes (ids never change their geometry) for layout caching."""
    return (state["building"], tuple(s["id"] for s in state["obstacles"]), state["terrace"]["id"] if state["terrace"] else None)
