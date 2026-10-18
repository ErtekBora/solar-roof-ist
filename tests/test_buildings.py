import importlib
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, box

from core import buildings, config
from core.buildings import find_building, load_buildings, with_user_floors

FIXTURE = Path(__file__).parent / "fixtures" / "buildings_sample.parquet"
TO_M = Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True)
TO_WGS = Transformer.from_crs("EPSG:32635", "EPSG:4326", always_xy=True)
X0, Y0 = TO_M.transform(29.076, 40.972)  # inside the prototype bbox
NO_REPORT = Path("does-not-exist.json")


def wgs(poly_m):
    """Metric polygon -> WGS84 polygon."""
    return Polygon([TO_WGS.transform(x, y) for x, y in poly_m.exterior.coords])


def sq(x, y, w=10.0, h=10.0):
    """Metric box with its lower-left corner at (X0 + x, Y0 + y)."""
    return box(X0 + x, Y0 + y, X0 + x + w, Y0 + y + h)


def click(x, y):
    """(lat, lon) of the metric offset (X0 + x, Y0 + y)."""
    lon, lat = TO_WGS.transform(X0 + x, Y0 + y)
    return lat, lon


def row(id, geom, height=None, floors=None, ml=None, underground=False, source="OpenStreetMap", cls=None):
    return {"id": id, "geom": geom, "height": height, "num_floors": floors, "height_ml_m": ml,
            "is_underground": underground, "source": source, "class": cls}


def write(tmp_path, rows, metadata=None, drop=(), name="b.parquet"):
    columns = {
        "id": pa.array([r["id"] for r in rows], pa.string()),
        "geometry": pa.array([None if r["geom"] is None else r["geom"].wkb for r in rows], pa.binary()),
        "height": pa.array([r["height"] for r in rows], pa.float64()),
        "num_floors": pa.array([r["num_floors"] for r in rows], pa.float64()),
        "height_ml_m": pa.array([r["height_ml_m"] for r in rows], pa.float64()),
        "is_underground": pa.array([r["is_underground"] for r in rows], pa.bool_()),
        "class": pa.array([r["class"] for r in rows], pa.string()),
        "sources": pa.array([[{"dataset": r["source"]}] for r in rows], pa.list_(pa.struct([("dataset", pa.string())]))),
    }
    for col in drop:
        columns.pop(col)
    table = pa.table(columns)
    if metadata is not None:
        table = table.replace_schema_metadata(metadata)
    path = tmp_path / name
    pq.write_table(table, path)
    return path


def load(tmp_path, rows, report_path=NO_REPORT, **kw):
    index = load_buildings(write(tmp_path, rows, **kw), report_path)
    assert index is not None
    return index


def target_at(index, x=5, y=5):
    result = find_building(index, *click(x, y))
    assert result["status"] == "ok", result
    return result


# ------------------------------------------------------------------ lookup statuses

def test_click_inside_a_footprint_returns_that_building(tmp_path):
    index = load(tmp_path, [row("A", wgs(sq(0, 0))), row("B", wgs(sq(40, 0)))])
    r = target_at(index)
    b = r["building"]
    assert b["id"] == "A" and b["match_distance_m"] == 0
    assert b["area_m2"] == pytest.approx(100.0, rel=0.01)
    assert [n["id"] for n in r["neighbours"]] == ["B"]


def test_projected_area_of_a_ten_metre_square_given_in_degrees(tmp_path):
    dlat, dlon = 9.0e-5, 1.1895e-4  # ~10 m x ~10 m at latitude 40.97
    lon0, lat0 = 29.076, 40.972
    poly = Polygon([(lon0, lat0), (lon0 + dlon, lat0), (lon0 + dlon, lat0 + dlat), (lon0, lat0 + dlat)])
    index = load(tmp_path, [row("A", poly)])
    assert index.records[0]["area_m2"] == pytest.approx(100.0, rel=0.01)
    assert find_building(index, lat0 + dlat / 2, lon0 + dlon / 2)["status"] == "ok"


def test_ten_metres_outside_picks_the_nearest_footprint(tmp_path):
    index = load(tmp_path, [row("A", wgs(sq(0, 0)))])
    r = find_building(index, *click(20, 5))  # 10 m east of the east wall
    assert r["status"] == "ok" and r["building"]["id"] == "A"
    assert r["building"]["match_distance_m"] == pytest.approx(10.0, abs=0.05)


def test_far_from_any_building_is_bina_bulunamadi(tmp_path):
    index = load(tmp_path, [row("A", wgs(sq(0, 0)))])
    r = find_building(index, *click(40, 5))  # 30 m from the wall: beyond 15 m
    assert r["status"] == "bina_bulunamadi" and r["building"] is None
    assert "15 m" in r["message"] and "kapsıyor" not in r["message"]


def test_outside_the_bbox_is_kapsam_disi_and_not_confused_with_not_found(tmp_path):
    index = load(tmp_path, [row("A", wgs(sq(0, 0)))])
    out = find_building(index, 41.015, 28.979)
    assert out["status"] == "kapsam_disi" and "Erenköy" in out["message"] and "Göztepe" in out["message"]
    near = find_building(index, *click(300, 300))
    assert near["status"] == "bina_bulunamadi" and near["message"] != out["message"]


def test_no_index_gives_veri_yok_with_the_turkish_message():
    r = find_building(None, 40.972, 29.076)
    assert r["status"] == "veri_yok" and r["message"] == buildings.MSG_NO_DATA
    assert "Bina verisi yüklenemedi" in r["message"]


# ------------------------------------------------------------------ overlapping footprints

def test_overlap_prefers_openstreetmap_over_a_larger_ml_building(tmp_path):
    rows = [row("ml", wgs(sq(0, 0, 20, 20)), source="Microsoft ML Buildings"), row("osm", wgs(sq(2, 2, 6, 6)))]
    assert target_at(load(tmp_path, rows))["building"]["id"] == "osm"


def test_overlap_among_equal_sources_prefers_the_largest_area(tmp_path):
    rows = [row("small", wgs(sq(2, 2, 6, 6))), row("big", wgs(sq(0, 0, 20, 20)))]
    assert target_at(load(tmp_path, rows))["building"]["id"] == "big"


def test_equal_distance_nearest_search_uses_the_same_tie_break(tmp_path):
    # click midway: both walls are 10 m away
    rows = [row("ml", wgs(sq(-20, 0, 10, 10)), source="Microsoft ML Buildings"), row("osm", wgs(sq(10, 0, 10, 10)))]
    r = find_building(load(tmp_path, rows), *click(0, 5))
    assert r["building"]["id"] == "osm" and r["building"]["match_distance_m"] == pytest.approx(10.0, abs=0.01)
    rows = [row("small", wgs(sq(-20, 0, 10, 10))), row("big", wgs(sq(10, -5, 10, 20)))]
    assert find_building(load(tmp_path, rows, name="c.parquet"), *click(0, 5))["building"]["id"] == "big"


# ------------------------------------------------------------------ height rule

def test_height_source_order(tmp_path):
    cases = [
        (row("a", wgs(sq(0, 0)), height=20, floors=5, ml=9), 20.0, "height_tag"),
        (row("a", wgs(sq(0, 0)), floors=5, ml=9), 15.0, "levels_tag"),
        (row("a", wgs(sq(0, 0)), ml=9.5), 9.5, "ml_estimate"),
    ]
    for i, (r, h, source) in enumerate(cases):
        b = target_at(load(tmp_path, [r], name=f"{i}.parquet"))["building"]
        assert (b["height_m"], b["height_source"]) == (h, source)


def test_user_floors_override_only_the_target(tmp_path):
    rows = [row("t", wgs(sq(0, 0)), height=20, ml=9), row("n", wgs(sq(30, 0)), floors=2, ml=9)]
    index = load(tmp_path, rows)
    base = target_at(index)
    user = find_building(index, *click(5, 5), floors=4)
    assert (user["building"]["height_m"], user["building"]["height_source"]) == (4 * config.LEVEL_HEIGHT_M, "user")
    assert user["neighbours"] == base["neighbours"]
    assert base["neighbours"][0]["height_source"] == "levels_tag"  # neighbours keep their own sources
    restored = with_user_floors(user["building"], None)
    assert (restored["height_m"], restored["height_source"]) == (20.0, "height_tag")
    assert with_user_floors(base["building"], 0)["height_source"] == "height_tag"  # 0 floors is ignored
    assert with_user_floors(base["building"], float("nan"))["height_source"] == "height_tag"


def test_neighbour_median_uses_tag_heights_only_never_estimates(tmp_path):
    rows = [
        row("t", wgs(sq(0, 0))),
        row("n1", wgs(sq(20, 0)), height=10),
        row("n2", wgs(sq(40, 0)), floors=7),  # 21 m
        row("n3", wgs(sq(60, 0)), height=40),
        row("n4", wgs(sq(80, 0)), ml=99),  # an estimate must not count as known
    ]
    b = target_at(load(tmp_path, rows))["building"]
    assert (b["height_m"], b["height_source"]) == (21.0, "neighbour_median")


def test_neighbour_median_only_looks_within_150_m(tmp_path):
    rows = [row("t", wgs(sq(0, 0))), row("far", wgs(sq(300, 0)), height=50)]
    b = target_at(load(tmp_path, rows))["building"]
    assert b["height_source"] == "default" and b["height_m"] == 50.0  # file-wide median of known heights


def test_data_derived_default_is_the_median_of_known_heights(tmp_path):
    rows = [
        row("t", wgs(sq(0, 0))),
        row("k1", wgs(sq(500, 0)), height=10),
        row("k2", wgs(sq(520, 0)), floors=10),  # 30 m
        row("k3", wgs(sq(540, 0)), height=20),
        row("est", wgs(sq(560, 0)), ml=200),  # estimates are not known heights
    ]
    index = load(tmp_path, rows)
    assert index.default_height_m == 20.0
    b = target_at(index)["building"]
    assert (b["height_m"], b["height_source"]) == (20.0, "default")


def test_fallback_height_when_no_height_is_known_anywhere(tmp_path):
    index = load(tmp_path, [row("t", wgs(sq(0, 0))), row("n", wgs(sq(30, 0)))])
    b = target_at(index)["building"]  # no exception from an empty median
    assert (b["height_m"], b["height_source"]) == (config.FALLBACK_HEIGHT_M, "default") and b["height_m"] == 15.0


def test_missing_height_ml_column_is_handled(tmp_path):
    rows = [row("t", wgs(sq(0, 0))), row("n", wgs(sq(30, 0)), height=12)]
    index = load(tmp_path, rows, drop=("height_ml_m",))
    b = target_at(index)["building"]
    assert (b["height_m"], b["height_source"]) == (12.0, "neighbour_median")


def test_roof_columns_are_never_read():
    assert not any(c.startswith("roof_") for c in buildings._COLUMNS)


# ------------------------------------------------------------------ cleaning at load

def test_underground_and_empty_geometries_are_dropped_with_counts(tmp_path):
    rows = [row("ok", wgs(sq(0, 0))), row("under", wgs(sq(30, 0)), underground=True), row("none", None),
            row("empty", Polygon())]
    index = load(tmp_path, rows)
    assert [r["id"] for r in index.records] == ["ok"]
    assert index.stats["dropped"]["underground"] == 1
    assert index.stats["dropped"]["empty"] == 2


def test_invalid_geometry_is_repaired(tmp_path):
    bowtie = Polygon([(X0, Y0), (X0 + 10, Y0 + 10), (X0 + 10, Y0), (X0, Y0 + 10)])
    assert not bowtie.is_valid
    index = load(tmp_path, [row("bow", Polygon([TO_WGS.transform(x, y) for x, y in bowtie.exterior.coords]))])
    assert len(index.records) == 1
    poly = index.polygons_m[0]
    assert poly.geom_type == "Polygon" and poly.is_valid and poly.area == pytest.approx(25.0, rel=0.02)


def test_multipolygon_is_reduced_to_its_largest_part(tmp_path):
    multi = MultiPolygon([wgs(sq(0, 0, 10, 10)), wgs(sq(30, 0, 5, 5))])
    index = load(tmp_path, [row("m", multi)])
    assert index.polygons_m[0].geom_type == "Polygon"
    assert index.records[0]["area_m2"] == pytest.approx(100.0, rel=0.01)
    assert all(g.geom_type == "Polygon" for g in index.polygons_m)  # downstream may use .exterior


def test_parts_under_one_square_metre_are_dropped_in_metres_not_degrees(tmp_path):
    rows = [row("tiny", wgs(sq(0, 0, 0.9, 1.0))), row("keep", wgs(sq(30, 0, 1.2, 1.0)))]
    index = load(tmp_path, rows)
    assert [r["id"] for r in index.records] == ["keep"]
    assert index.stats["dropped"]["too_small"] == 1


def test_data_source_is_parsed_from_sources(tmp_path):
    rows = [row("o", wgs(sq(0, 0))), row("m", wgs(sq(30, 0)), source="Microsoft ML Buildings")]
    index = load(tmp_path, rows)
    assert {r["id"]: r["data_source"] for r in index.records} == {"o": "OpenStreetMap", "m": "Microsoft ML Buildings"}


# ------------------------------------------------------------------ neighbours, setback

def test_neighbours_within_150_m_only_with_metric_distances(tmp_path):
    rows = [row("t", wgs(sq(0, 0))), row("near", wgs(sq(110, 0))), row("edge", wgs(sq(159, 0))),
            row("out", wgs(sq(161, 0)))]
    r = target_at(load(tmp_path, rows))
    got = {n["id"]: n["distance_m"] for n in r["neighbours"]}
    assert set(got) == {"near", "edge"}
    assert got["near"] == pytest.approx(100.0, abs=0.05) and got["edge"] == pytest.approx(149.0, abs=0.05)
    assert [n["id"] for n in r["neighbours"]] == ["near", "edge"]  # sorted by distance


def test_setback_polygon_and_area(tmp_path):
    b = target_at(load(tmp_path, [row("t", wgs(sq(0, 0)))]))["building"]
    assert b["setback_area_m2"] == pytest.approx(64.0, rel=0.01)  # 10 x 10 m with 1 m setback
    assert len(b["setback_polygons"]) == 1
    tiny = target_at(load(tmp_path, [row("t", wgs(sq(0, 0, 1.5, 1.5)))], name="t.parquet"), 0.75, 0.75)["building"]
    assert tiny["setback_area_m2"] == 0 and tiny["setback_polygons"] == []


def test_result_contains_no_shapely_objects(tmp_path):
    r = target_at(load(tmp_path, [row("t", wgs(sq(0, 0))), row("n", wgs(sq(30, 0)))]))
    assert json.dumps(r["building"]) and json.dumps(r["neighbours"])
    lon, lat = r["building"]["polygon"][0]
    assert 29 < lon < 30 and 40 < lat < 41  # WGS84 [lon, lat]


# ------------------------------------------------------------------ model card

REPORT = {"model": "LightGBM (L1)", "cv_mae_m": 6.78}


def test_model_card_is_read_from_schema_metadata_bytes(tmp_path):
    meta = {b"height_ml": json.dumps(REPORT).encode("utf-8")}
    assert load(tmp_path, [row("a", wgs(sq(0, 0)))], metadata=meta).report == REPORT


def test_invalid_metadata_falls_back_to_the_json_file(tmp_path):
    json_file = tmp_path / "report.json"
    json_file.write_text(json.dumps({"cv_mae_m": 1.5}), encoding="utf-8")
    rows = [row("a", wgs(sq(0, 0)))]
    assert load(tmp_path, rows, report_path=json_file, metadata={b"height_ml": b"{not json"}).report == {"cv_mae_m": 1.5}
    assert load(tmp_path, rows, report_path=json_file, metadata={}, name="2.parquet").report == {"cv_mae_m": 1.5}


def test_missing_everywhere_gives_report_none_without_crashing(tmp_path):
    assert load(tmp_path, [row("a", wgs(sq(0, 0)))], metadata={b"height_ml": b"\xff\xfe"}).report is None
    assert load(tmp_path, [row("a", wgs(sq(0, 0)))], name="2.parquet").report is None


# ------------------------------------------------------------------ loading failures and laziness

def test_missing_parquet_returns_none_without_raising(tmp_path):
    assert load_buildings(tmp_path / "nope.parquet", NO_REPORT) is None


def test_corrupt_parquet_returns_none_without_raising(tmp_path):
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not a parquet file")
    assert load_buildings(bad, NO_REPORT) is None


def test_importing_the_module_does_not_read_the_file(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("the buildings file was read at import time")

    monkeypatch.setattr(pq, "read_table", fail)
    monkeypatch.setattr(pq, "read_schema", fail)
    importlib.reload(buildings)
    assert not hasattr(buildings, "_INDEX")


# ------------------------------------------------------------------ performance contract

def test_strtree_is_built_on_metric_geometries(tmp_path):
    index = load(tmp_path, [row("a", wgs(sq(0, 0))), row("b", wgs(sq(30, 0)))])
    for geom in index.tree.geometries:
        x, y = geom.exterior.coords[0]
        assert 100_000 < x < 900_000 and 4_000_000 < y < 5_000_000  # UTM metres, not degrees
    assert all(math.isclose(a, p.area) for a, p in zip([r["area_m2"] for r in index.records], index.polygons_m))


def test_a_click_projects_one_point_only(tmp_path):
    index = load(tmp_path, [row(str(i), wgs(sq(30 * i, 0))) for i in range(5)])
    calls = []

    class Counter:
        def __init__(self, inner):
            self.inner = inner

        def transform(self, *a, **k):
            calls.append(1)
            return self.inner.transform(*a, **k)

    index._to_m = Counter(index._to_m)
    r = find_building(index, *click(5, 5))
    assert r["status"] == "ok" and len(calls) == 1  # no per-click reprojection of the dataset


# ------------------------------------------------------------------ committed fixture (real Overture rows)

@pytest.fixture(scope="module")
def sample():
    return load_buildings(FIXTURE, NO_REPORT)


def test_committed_fixture_loads_with_model_card_and_ml_heights(sample):
    assert sample is not None and 100 <= len(sample.records) <= 200
    assert sample.report["cv_mae_m"] == 6.78 and sample.report["uses_gba"] is False
    assert all(r["height_ml_m"] is not None for r in sample.records)
    assert sample.default_height_m > 0


def test_committed_fixture_lookup_uses_the_height_rule(sample):
    seen = set()
    for poly in sample.polygons_m[:60]:
        p = poly.representative_point()
        lon, lat = TO_WGS.transform(p.x, p.y)
        r = find_building(sample, lat, lon)
        assert r["status"] == "ok"
        seen.add(r["building"]["height_source"])
    assert seen <= {"height_tag", "levels_tag", "ml_estimate"} and "ml_estimate" in seen
