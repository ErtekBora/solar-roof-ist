import math
import time

import numpy as np
import pytest
import shapely
from shapely import affinity
from shapely.geometry import Polygon, box

from core import layout
from core.config import DEFAULT_ROOF_PITCH_DEG, PANEL_LENGTH_M, PANEL_WIDTH_M, SETBACK_M, TERRACE_GAP_M

LAT = 40.97
OVERLAP_M2 = 1e-6
L_SHAPE = Polygon([(0, 0), (20, 0), (20, 8), (8, 8), (8, 20), (0, 20)])


def _boxes(lay):
    return shapely.polygons(lay["panels"]) if lay["count"] else np.array([])


def _assert_inside_and_disjoint(target, lay, obstacles=()):
    panels = _boxes(lay)
    assert len(panels) == lay["count"]
    if not len(panels):
        return
    assert shapely.covers(target, panels).all()  # every rectangle fully inside, not just its centre
    tree = shapely.STRtree(panels)
    left, right = tree.query(panels, predicate="intersects")
    pairs = left < right
    overlap = shapely.area(shapely.intersection(panels[left[pairs]], panels[right[pairs]]))
    assert (overlap <= OVERLAP_M2).all()  # touching edges (flush rows) are fine, a positive area is not
    for obs in obstacles:
        grown = obs.buffer(0.5)
        assert (shapely.area(shapely.intersection(panels, grown)) <= OVERLAP_M2).all()


# ------------------------------------------------------------------ roof faces

@pytest.mark.parametrize("kind", ["hip", "gable"])
@pytest.mark.parametrize("footprint", [box(0, 0, 24, 14), affinity.rotate(box(0, 0, 24, 14), 30), L_SHAPE])
def test_face_plan_areas_sum_to_the_setback_area(kind, footprint):
    faces = layout.roof_faces(footprint, kind)
    setback = layout.setback_polygon(footprint)
    assert sum(f["plan_m2"] for f in faces) == pytest.approx(setback.area, rel=1e-6)


@pytest.mark.parametrize("kind", ["hip", "gable"])
def test_l_shaped_footprint_gives_no_face_area_outside_it(kind):
    setback = layout.setback_polygon(L_SHAPE)
    for face in layout.roof_faces(L_SHAPE, kind):
        if face["polygon"] is not None:
            assert face["polygon"].difference(setback).area < 1e-6
            assert L_SHAPE.covers(face["polygon"])


def test_hip_faces_azimuths_and_shapes_on_an_east_west_rectangle():
    faces = layout.roof_faces(box(0, 0, 24, 14), "hip", setback=box(0, 0, 24, 14))  # no setback: raw shapes
    by_name = {f["name"]: f for f in faces}
    assert {f["name"]: round(f["azimuth_deg"]) % 360 for f in faces} == {"Kuzey": 0, "Güney": 180, "Doğu": 90, "Batı": 270}
    assert {f["name"]: f["aspect"] for f in faces} == {"Kuzey": 180, "Güney": 0, "Doğu": -90, "Batı": 90}
    # 45 degree hips: ridge length = long - short = 10 m, triangles 14 x 7 / 2, trapezoids (24 + 10) / 2 x 7
    assert by_name["Doğu"]["plan_m2"] == pytest.approx(14 * 7 / 2)
    assert by_name["Güney"]["plan_m2"] == pytest.approx((24 + 10) / 2 * 7)


def test_square_hip_is_a_pyramid_and_gable_has_two_faces():
    faces = layout.roof_faces(box(0, 0, 10, 10), "hip", setback=box(0, 0, 10, 10))
    assert len(faces) == 4 and all(f["plan_m2"] == pytest.approx(25) for f in faces)
    assert len(layout.roof_faces(box(0, 0, 10, 10), "gable")) == 2


def test_face_weights_follow_the_intersected_areas():
    faces = layout.roof_faces(box(0, 0, 24, 14), "hip")
    weights = layout.face_weights(faces)
    assert sum(weights) == pytest.approx(1)
    total = sum(f["plan_m2"] for f in faces)
    assert weights == pytest.approx([f["plan_m2"] / total for f in faces])


def test_mixed_terrace_gap_is_symmetric_and_empty_pieces_never_raise():
    footprint = box(0, 0, 24, 14)
    setback = layout.setback_polygon(footprint)
    faces = layout.roof_faces(footprint, "hip", setback)
    terrace = box(2, 2, 10, 12)
    flat, rest = layout.split_mixed(faces, setback, terrace)
    assert flat.area == pytest.approx(terrace.buffer(-TERRACE_GAP_M / 2).intersection(setback).area)
    for f in rest:
        if f["polygon"] is not None:
            assert f["polygon"].distance(flat) >= TERRACE_GAP_M - 1e-6
    # a terrace covering the whole roof leaves empty faces (0 m2), never an exception
    flat_all, rest_all = layout.split_mixed(faces, setback, footprint.buffer(5))
    assert flat_all is not None and all(f["plan_m2"] == 0 and f["polygon"] is None for f in rest_all)
    assert layout.clean(Polygon()) is None and layout.clean(box(0, 0, 1, 1).buffer(-5)) is None


def test_mixed_flat_and_pitched_panels_never_touch():
    footprint = box(0, 0, 24, 14)
    model = layout.roof_model(footprint, "hip", box(1, 1, 11, 13), "mixed")
    lays = layout.layout_model(model, flat_tilt_deg=33, lat=LAT)
    flat = shapely.union_all(_boxes(lays[layout.FLAT_SEGMENT]))
    for name, lay in lays.items():
        if name != layout.FLAT_SEGMENT and lay["count"]:
            assert shapely.union_all(_boxes(lay)).distance(flat) >= TERRACE_GAP_M - 1e-6


# --------------------------------------------------------------------- layout

def test_row_pitch_formula():
    h_min = 90 - 41 - 23.44
    expected = 1.72 * math.cos(math.radians(30)) + 1.72 * math.sin(math.radians(30)) / math.tan(math.radians(h_min))
    assert layout.row_pitch(1.72, 30, 41) == pytest.approx(expected)
    assert layout.winter_noon_elevation_deg(41) == pytest.approx(25.56)


def test_flat_rows_run_east_west_regardless_of_building_orientation():
    target = affinity.rotate(box(0, 0, 30, 18), 33)
    lay = layout.place_panels(target, mount="flat", tilt_deg=33, lat=LAT)
    assert lay["count"] > 0
    edges = lay["panels"][:, 1] - lay["panels"][:, 0]  # first edge of every panel runs along its row
    assert np.allclose(edges[:, 1], 0, atol=1e-9)  # pure East-West (no North component)
    _assert_inside_and_disjoint(target, lay)


def test_flat_versus_flush_plan_lengths():
    target = box(0, 0, 40, 40)
    flat = layout.place_panels(target, mount="flat", tilt_deg=33, lat=LAT)
    flush = layout.place_panels(target, mount="flush", tilt_deg=DEFAULT_ROOF_PITCH_DEG, azimuth_deg=180, lat=LAT)
    slope_flat = PANEL_LENGTH_M if flat["orientation"] == "portrait" else PANEL_WIDTH_M
    slope_flush = PANEL_LENGTH_M if flush["orientation"] == "portrait" else PANEL_WIDTH_M
    assert flat["depth_m"] == pytest.approx(slope_flat * math.cos(math.radians(33)))
    assert flat["row_pitch_m"] == pytest.approx(layout.row_pitch(slope_flat, 33, LAT))
    assert flush["depth_m"] == pytest.approx(slope_flush * math.cos(math.radians(30)))
    assert flush["row_pitch_m"] == flush["depth_m"]  # flush rows: no spacing
    _assert_inside_and_disjoint(target, flush)


def test_flush_panels_follow_the_face_axes():
    target = affinity.rotate(box(0, 0, 20, 8), 20, origin=(0, 0))  # face rotated 20 degrees, downslope at 160
    lay = layout.place_panels(target, mount="flush", tilt_deg=30, azimuth_deg=160, lat=LAT)
    edge = lay["panels"][0, 1] - lay["panels"][0, 0]
    assert math.degrees(math.atan2(edge[1], edge[0])) == pytest.approx(20, abs=1e-6)
    _assert_inside_and_disjoint(target, lay)


def test_concave_l_roof_places_no_panel_across_the_notch():
    for mount in ("flat", "flush"):
        lay = layout.place_panels(L_SHAPE, mount=mount, tilt_deg=30, lat=LAT)
        assert lay["count"] > 0
        _assert_inside_and_disjoint(L_SHAPE, lay)
        notch = box(8, 8, 20, 20)
        assert (shapely.area(shapely.intersection(_boxes(lay), notch)) <= OVERLAP_M2).all()


def test_corner_test_alone_would_miss_a_reflex_vertex():
    # all 4 corners and the centre of [0, 2.2] x [0, 1.13] are inside, but a sliver crack cuts the rectangle
    roof = box(-5, -5, 5, 5).difference(Polygon([(0.9, 0.3), (1.0, 0.3), (1.0, 20), (0.9, 20)]))
    xs, ys = np.array([0, 2.2, 2.2, 0, 1.1]), np.array([0, 0, 1.13, 1.13, 0.565])
    assert shapely.contains_xy(roof, xs, ys).all() and not roof.covers(box(0, 0, 2.2, 1.13))
    lay = layout.place_panels(roof, mount="flush", tilt_deg=30, lat=LAT)
    _assert_inside_and_disjoint(roof, lay)


def test_obstacle_reduces_panel_count_and_is_kept_clear():
    target = box(0, 0, 20, 12)
    free = layout.place_panels(target, mount="flush", tilt_deg=30, lat=LAT)
    obstacle = box(9, 5, 11, 7)
    blocked = layout.place_panels(target, mount="flush", tilt_deg=30, lat=LAT, obstacles=[obstacle])
    assert blocked["count"] < free["count"]
    _assert_inside_and_disjoint(target, blocked, [obstacle])


def test_output_is_deterministic():
    target = affinity.rotate(L_SHAPE, 17)
    a = layout.place_panels(target, mount="flat", tilt_deg=33, lat=LAT, obstacles=[box(2, 2, 4, 4)])
    b = layout.place_panels(target, mount="flat", tilt_deg=33, lat=LAT, obstacles=[box(2, 2, 4, 4)])
    assert a["count"] == b["count"] and a["orientation"] == b["orientation"]
    assert np.array_equal(a["panels"], b["panels"])


def test_panels_are_in_row_order_north_to_south():
    lay = layout.place_panels(box(0, 0, 20, 20), mount="flat", tilt_deg=33, lat=LAT)
    tops = lay["panels"][:, 2, 1]
    assert (np.diff(tops) <= 1e-9).all()


def test_dropped_faces_get_no_panels():
    model = layout.roof_model(box(0, 0, 24, 14), "hip")
    lays = layout.layout_model(model, flat_tilt_deg=33, lat=LAT, skip_faces={"Kuzey yüzü"})
    assert "Kuzey yüzü" not in lays and lays["Güney yüzü"]["count"] > 0


def test_layout_on_a_400_m2_roof_is_fast():
    footprint = affinity.rotate(box(0, 0, 20, 20).buffer(SETBACK_M, join_style="mitre"), 25)  # 400 m2 after setback
    obstacles = [box(5, 5, 7, 6), box(12, 10, 13, 12)]
    model_flat = layout.roof_model(footprint, "hip", structure="flat")
    model_hip = layout.roof_model(footprint, "hip")
    assert model_flat["flat"].area == pytest.approx(400, rel=0.01)
    start = time.perf_counter()
    layout.layout_model(model_flat, flat_tilt_deg=33, lat=LAT, obstacles=obstacles)
    flat_s = time.perf_counter() - start
    start = time.perf_counter()
    layout.layout_model(model_hip, flat_tilt_deg=33, lat=LAT, obstacles=obstacles)
    hip_s = time.perf_counter() - start
    assert flat_s < 0.3 and hip_s < 0.3


def test_empty_target_gives_an_empty_layout():
    lay = layout.place_panels(None, mount="flat", tilt_deg=33, lat=LAT)
    assert lay["count"] == 0 and lay["panels"].shape == (0, 4, 2)
    assert layout.place_panels(box(0, 0, 1, 1), mount="flat", tilt_deg=33, lat=LAT)["count"] == 0


def test_hatch_lines_stay_inside_the_face():
    face = Polygon([(0, 0), (10, 0), (5, 5)])
    lines = layout.hatch_lines(face)
    assert lines and all(face.buffer(1e-6).covers(line) for line in lines)


# ------------------------------------------------------------ which number wins

@pytest.mark.parametrize(
    "structure, n_obstacles, empty, terrace, expected",
    [
        ("pitched", 0, False, False, ("numeric", None)),
        ("pitched", 2, False, False, ("geometry", None)),
        ("flat", 0, True, False, ("geometry", None)),
        ("unknown", 1, False, False, ("geometry", None)),
        ("mixed", 2, False, False, ("numeric", layout.MSG_KARMA_NEEDS_TERRACE)),
        ("mixed", 0, True, False, ("numeric_clean", layout.MSG_KARMA_NEEDS_TERRACE)),
        ("mixed", 2, False, True, ("geometry", None)),
        ("mixed", 0, True, True, ("geometry", None)),
    ],
)
def test_capacity_mode_priority(structure, n_obstacles, empty, terrace, expected):
    assert layout.decide_capacity_mode(structure, n_obstacles, empty, terrace) == expected
