"""Local Overture buildings extract: load once, find the clicked building, neighbours and the height rule.

Data: GeoParquet from Overture Maps (OpenStreetMap + Microsoft ML Buildings, ODbL) with a `height_ml_m`
column added on Kaggle. The app only reads that column; it never runs the model.

Docs: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
      https://shapely.readthedocs.io/en/stable/strtree.html
Nothing is read at import time; `load_buildings()` returns None instead of raising.
"""
import json
import logging
import math
import statistics
from pathlib import Path

import numpy as np

from core import config

log = logging.getLogger(__name__)

_COLUMNS = ("id", "geometry", "height", "num_floors", "height_ml_m", "is_underground", "class", "sources")
_UTM = "EPSG:32635"
_MIN_AREA_M2 = 1.0
_TIE_EPS_M = 1e-6

MSG_OUT_OF_COVERAGE = (
    "Bu prototip yalnızca Erenköy, Suadiye ve Göztepe bölgesini kapsıyor; seçtiğiniz konum kapsam dışında. "
    "Elle girdiğiniz çatı alanı kullanılıyor."
)
MSG_NOT_FOUND = "Tıkladığınız noktanın 15 m yakınında bina bulunamadı. Elle girdiğiniz çatı alanı kullanılıyor."
MSG_NO_DATA = "Bina verisi yüklenemedi; çatı alanını elle girerek devam edebilirsiniz."


class BuildingIndex:
    """Metric (EPSG:32635) footprints, an STRtree over them, and the model card."""

    def __init__(self, records, polygons_m, transformers, default_height_m, report, stats):
        from shapely import STRtree

        self.records = records
        self.polygons_m = polygons_m
        self.tree = STRtree(polygons_m)  # built on metric geometries, once
        self._to_m, self._to_wgs = transformers
        self.default_height_m = default_height_m
        self.report = report
        self.stats = stats

    def to_metric(self, lon, lat):
        """Project a single WGS84 point (the only per-click projection)."""
        return self._to_m.transform(lon, lat)

    def polygon_wgs84(self, poly_m):
        """Exterior ring of a metric polygon as [[lon, lat], ...] (for drawing only)."""
        xs, ys = poly_m.exterior.xy
        lons, lats = self._to_wgs.transform(list(xs), list(ys))
        return [[float(lo), float(la)] for lo, la in zip(lons, lats)]


# ---------------------------------------------------------------------- loading

def _clean_geometry(geom):
    """Return a valid single metric Polygon, or (None, reason)."""
    if geom is None or geom.is_empty:
        return None, "empty"
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        return None, "invalid"
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        parts = [g for g in geom.geoms if g.geom_type == "Polygon" and not g.is_empty]
        if not parts:
            return None, "invalid"
        geom = max(parts, key=lambda g: g.area)
    if geom.geom_type != "Polygon" or not geom.is_valid or geom.is_empty:
        return None, "invalid"
    if geom.area < _MIN_AREA_M2:
        return None, "too_small"
    return geom, None


def _data_source(sources):
    """'OpenStreetMap' / 'Microsoft ML Buildings' from the Overture `sources` list (first entry)."""
    try:
        name = (sources or [])[0]["dataset"]
    except (IndexError, KeyError, TypeError):
        return "Bilinmiyor"
    return name or "Bilinmiyor"


def _num(value):
    """float(value) when it is a finite number, else None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _read_report(table_meta, json_path):
    """Model card: parquet schema metadata key b'height_ml', else the JSON file, else None."""
    raw = (table_meta or {}).get(b"height_ml")
    if raw:
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            log.warning("height_ml metadata is not valid JSON; trying %s", json_path)
    try:
        return json.loads(Path(json_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _known_height(rec):
    """(height_m, source) from the `height` or `num_floors` tags only; (None, None) otherwise."""
    h = rec["height"]
    if h is not None and h > 0:
        return h, "height_tag"
    n = rec["num_floors"]
    if n is not None and n > 0:
        return n * config.LEVEL_HEIGHT_M, "levels_tag"
    return None, None


def load_buildings(path=None, report_path=None):
    """Load and clean the buildings file. Returns a BuildingIndex, or None when missing/unreadable."""
    try:
        return _load(Path(path or config.BUILDINGS_PARQUET), Path(report_path or config.BUILDINGS_REPORT_JSON))
    except Exception:  # missing file, corrupt parquet, pyarrow/shapely errors: never reach the UI
        log.exception("Building data could not be loaded")
        return None


def _load(path, report_path):
    import pyarrow.parquet as pq
    import shapely
    from pyproj import Transformer

    names = set(pq.read_schema(path).names)
    table = pq.read_table(path, columns=[c for c in _COLUMNS if c in names])
    report = _read_report(table.schema.metadata, report_path)
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    n = table.num_rows
    get = lambda name: cols.get(name, [None] * n)

    to_m = Transformer.from_crs("EPSG:4326", _UTM, always_xy=True)
    to_wgs = Transformer.from_crs(_UTM, "EPSG:4326", always_xy=True)

    dropped = {"underground": 0, "empty": 0, "invalid": 0, "too_small": 0}
    keep = []
    for i, (wkb, underground) in enumerate(zip(get("geometry"), get("is_underground"))):
        if underground is True:
            dropped["underground"] += 1
        elif not wkb:
            dropped["empty"] += 1
        else:
            keep.append(i)

    geoms = shapely.from_wkb([cols["geometry"][i] for i in keep], on_invalid="ignore")
    metric = shapely.transform(geoms, lambda c: np.column_stack(to_m.transform(c[:, 0], c[:, 1])))

    records, polygons = [], []
    for i, geom in zip(keep, metric):
        poly, reason = _clean_geometry(geom)
        if poly is None:
            dropped[reason] += 1
            continue
        records.append({
            "id": cols["id"][i],
            "area_m2": float(poly.area),
            "height": _num(get("height")[i]),
            "num_floors": _num(get("num_floors")[i]),
            "height_ml_m": _num(get("height_ml_m")[i]),
            "data_source": _data_source(get("sources")[i]),
        })
        polygons.append(poly)

    known = [h for h, _ in map(_known_height, records) if h is not None]
    default_height = statistics.median(known) if known else config.FALLBACK_HEIGHT_M
    stats = {"loaded": len(records), "dropped": dropped, "known_heights": len(known)}
    log.info("Buildings loaded: %d kept, dropped %s", len(records), dropped)
    return BuildingIndex(records, polygons, (to_m, to_wgs), default_height, report, stats)


# ----------------------------------------------------------------------- heights

def _auto_height(index, i):
    """Height rule without the user override: height_tag > levels_tag > ml_estimate > neighbour_median > default."""
    rec = index.records[i]
    h, source = _known_height(rec)
    if h is not None:
        return h, source
    if rec["height_ml_m"] is not None and rec["height_ml_m"] > 0:
        return rec["height_ml_m"], "ml_estimate"
    near = index.tree.query(index.polygons_m[i], predicate="dwithin", distance=config.NEIGHBOUR_RADIUS_M)
    heights = [h for j in near if j != i for h, _ in [_known_height(index.records[j])] if h is not None]
    if heights:  # known = tags only, never estimates; never median() of an empty list
        return statistics.median(heights), "neighbour_median"
    return index.default_height_m, "default"


def _pick(index, candidates):
    """Deterministic choice among overlapping footprints: OpenStreetMap first, then the largest area."""
    return min(candidates, key=lambda i: (index.records[i]["data_source"] != "OpenStreetMap", -index.records[i]["area_m2"], i))


def with_user_floors(building, floors):
    """Copy of `building` with the target height from user-typed floors (x LEVEL_HEIGHT_M); None restores the data.

    Only the target changes; neighbours keep their own sources.
    """
    floors = _num(floors)
    if floors is not None and floors > 0:
        return {**building, "height_m": floors * config.LEVEL_HEIGHT_M, "height_source": "user"}
    return {**building, "height_m": building["auto_height_m"], "height_source": building["auto_height_source"]}


# ------------------------------------------------------------------------ lookup

def _result(status, message=None, building=None, neighbours=None):
    return {"status": status, "message": message, "building": building, "neighbours": neighbours or []}


def find_building(index, lat, lon, floors=None):
    """Find the building at a WGS84 click. Status: ok | kapsam_disi | bina_bulunamadi | veri_yok.

    Inside a footprint wins; otherwise the nearest footprint within NEAREST_MAX_M metres.
    """
    from shapely.geometry import Point

    if index is None:
        return _result("veri_yok", MSG_NO_DATA)
    min_lon, min_lat, max_lon, max_lat = config.BUILDINGS_BBOX
    if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
        return _result("kapsam_disi", MSG_OUT_OF_COVERAGE)

    pt = Point(*index.to_metric(lon, lat))
    inside = index.tree.query(pt, predicate="intersects")
    match, distance = None, 0.0
    if len(inside):
        match = _pick(index, [int(i) for i in inside])
    else:
        nearest, dists = index.tree.query_nearest(pt, max_distance=config.NEAREST_MAX_M, return_distance=True, all_matches=True)
        if len(dists):
            best = float(dists.min())
            ties = [int(t) for t, d in zip(nearest, dists) if d - best <= _TIE_EPS_M]
            match, distance = _pick(index, ties), best
    if match is None:
        return _result("bina_bulunamadi", MSG_NOT_FOUND)

    building = _building_dict(index, match, distance)
    if floors is not None:
        building = with_user_floors(building, floors)
    return _result("ok", None, building, _neighbours(index, match))


def ridge_azimuth_deg(poly_m):
    """Ridge bearing (0..180, North-based, clockwise) along the long side of the minimum rotated rectangle.

    `poly_m` is in a metric CRS (EPSG:32635, x = east, y = north).
    """
    import math

    ring = list(poly_m.minimum_rotated_rectangle.exterior.coords)
    (x0, y0), (x1, y1), (x2, y2) = ring[0], ring[1], ring[2]
    first, second = math.hypot(x1 - x0, y1 - y0), math.hypot(x2 - x1, y2 - y1)
    dx, dy = (x1 - x0, y1 - y0) if first >= second else (x2 - x1, y2 - y1)
    return math.degrees(math.atan2(dx, dy)) % 180


def _building_dict(index, i, distance_m):
    rec, poly = index.records[i], index.polygons_m[i]
    height, source = _auto_height(index, i)
    setback = poly.buffer(-config.SETBACK_M)
    parts = [] if setback.is_empty else [g for g in getattr(setback, "geoms", [setback]) if g.geom_type == "Polygon"]
    centroid = index._to_wgs.transform(poly.centroid.x, poly.centroid.y)
    return {
        "id": rec["id"],
        "polygon": index.polygon_wgs84(poly),
        "area_m2": rec["area_m2"],
        "setback_polygons": [index.polygon_wgs84(g) for g in parts],
        "setback_area_m2": float(sum(g.area for g in parts)),
        "height_m": height,
        "height_source": source,
        "auto_height_m": height,
        "auto_height_source": source,
        "num_floors": rec["num_floors"],
        "data_source": rec["data_source"],
        "match_distance_m": distance_m,
        "centroid": (centroid[1], centroid[0]),  # (lat, lon)
        "ridge_azimuth_deg": ridge_azimuth_deg(poly),
    }


def _neighbours(index, target):
    target_poly = index.polygons_m[target]
    out = []
    for j in index.tree.query(target_poly, predicate="dwithin", distance=config.NEIGHBOUR_RADIUS_M):
        j = int(j)
        if j == target:
            continue
        height, source = _auto_height(index, j)
        out.append({
            "id": index.records[j]["id"],
            "polygon": index.polygon_wgs84(index.polygons_m[j]),
            "height_m": height,
            "height_source": source,
            "distance_m": float(target_poly.distance(index.polygons_m[j])),
        })
    return sorted(out, key=lambda n: (n["distance_m"], str(n["id"])))
