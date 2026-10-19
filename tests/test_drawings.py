import pytest
from shapely.geometry import box

from core import drawings

# a 24 x 14 m footprint in the pilot area (metric EPSG:32635)
X0, Y0 = 676500.0, 4537500.0
FOOTPRINT = box(X0, Y0, X0 + 24, Y0 + 14)


def feature(poly, jitter_deg=0.0, rotate_start=0):
    """GeoJSON Feature as Leaflet.draw returns it ([lng, lat] ring), optionally jittered or with another start vertex."""
    xs, ys = poly.exterior.xy
    lons, lats = drawings.to_wgs84(list(xs), list(ys))
    ring = [[lo + jitter_deg, la - jitter_deg] for lo, la in zip(lons, lats)][:-1]
    ring = ring[rotate_start:] + ring[:rotate_start]
    ring.append(ring[0])
    return {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


def obstacle(dx=2.0, dy=2.0, w=1.5, h=1.0):
    return box(X0 + dx, Y0 + dy, X0 + dx + w, Y0 + dy + h)


def test_wgs84_utm_round_trip_of_a_drawn_shape():
    poly = obstacle()
    back = drawings.feature_to_metric(feature(poly))
    assert back.symmetric_difference(poly).area < 1e-4
    assert drawings.same_shape(back, poly)


def test_none_or_empty_return_changes_nothing_and_never_bumps():
    state = drawings.new_state("b1")
    assert drawings.sync(state, None, drawings.OBSTACLE, FOOTPRINT) == {"changed": False, "bump": False}
    assert drawings.sync(state, [], drawings.OBSTACLE, FOOTPRINT) == {"changed": False, "bump": False}
    assert state["gen"] == 0 and drawings.stored_shapes(state) == []


def test_fresh_generation_with_empty_or_none_return_keeps_older_shapes_and_does_not_bump():
    state = drawings.new_state("b1")
    drawings.sync(state, [feature(obstacle())], drawings.OBSTACLE, FOOTPRINT)
    drawings.clear_all(state)
    drawings.sync(state, [feature(obstacle(10))], drawings.OBSTACLE, FOOTPRINT)
    state["gen"] += 1  # e.g. a terrace replacement rebuilt the map: the new Leaflet map knows no shape yet
    gen = state["gen"]
    for empty in (None, []):
        assert drawings.sync(state, empty, drawings.OBSTACLE, FOOTPRINT) == {"changed": False, "bump": False}
    assert state["gen"] == gen and len(state["obstacles"]) == 1


def test_obstacle_then_mode_switch_then_terrace_keeps_stable_labels_across_reruns():
    state = drawings.new_state("b1")
    obs, terr = obstacle(), box(X0 + 12, Y0 + 2, X0 + 20, Y0 + 12)
    drawings.sync(state, [feature(obs)], drawings.OBSTACLE, FOOTPRINT)
    # the user switches to "Teras çiz" and draws; the obstacle is still in the returned set
    res = drawings.sync(state, [feature(obs), feature(terr)], drawings.TERRACE, FOOTPRINT)
    assert res == {"changed": True, "bump": False}
    for mode in (drawings.TERRACE, drawings.OBSTACLE):  # later reruns in either mode relabel nothing
        assert drawings.sync(state, [feature(obs), feature(terr)], mode, FOOTPRINT)["changed"] is False
    assert len(state["obstacles"]) == 1 and state["terrace"] is not None
    assert state["obstacles"][0]["kind"] == drawings.OBSTACLE and state["terrace"]["kind"] == drawings.TERRACE


def test_tiny_coordinate_jitter_or_another_start_vertex_does_not_duplicate():
    state = drawings.new_state("b1")
    drawings.sync(state, [feature(obstacle())], drawings.OBSTACLE, FOOTPRINT)
    res = drawings.sync(state, [feature(obstacle(), jitter_deg=2e-7, rotate_start=2)], drawings.OBSTACLE, FOOTPRINT)
    assert res["changed"] is False and len(state["obstacles"]) == 1


def test_deleting_a_current_generation_shape_removes_it_and_bumps():
    state = drawings.new_state("b1")
    a, b = obstacle(), obstacle(10)
    drawings.sync(state, [feature(a), feature(b)], drawings.OBSTACLE, FOOTPRINT)
    res = drawings.sync(state, [feature(b)], drawings.OBSTACLE, FOOTPRINT)  # deleted with the Draw toolbar
    assert res == {"changed": True, "bump": True} and state["gen"] == 1
    assert len(state["obstacles"]) == 1 and drawings.same_shape(state["obstacles"][0]["raw"], b)
    # the rebuilt map returns nothing yet: no further deletion, no further bump (no rerun loop)
    assert drawings.sync(state, [], drawings.OBSTACLE, FOOTPRINT) == {"changed": False, "bump": False}
    assert len(state["obstacles"]) == 1 and state["gen"] == 1


def test_new_terrace_replaces_the_old_one_and_bumps():
    state = drawings.new_state("b1")
    t1, t2 = box(X0 + 2, Y0 + 2, X0 + 8, Y0 + 8), box(X0 + 12, Y0 + 2, X0 + 20, Y0 + 10)
    drawings.sync(state, [feature(t1)], drawings.TERRACE, FOOTPRINT)
    res = drawings.sync(state, [feature(t1), feature(t2)], drawings.TERRACE, FOOTPRINT)
    assert res["bump"] and state["gen"] == 1
    assert drawings.same_shape(state["terrace"]["raw"], t2)


def test_drawing_outside_the_footprint_is_ignored_with_a_message_and_clipped_otherwise():
    state = drawings.new_state("b1")
    outside = box(X0 + 40, Y0, X0 + 45, Y0 + 5)
    res = drawings.sync(state, [feature(outside)], drawings.OBSTACLE, FOOTPRINT)
    assert res["changed"] is False and state["obstacles"] == [] and state["notice"] == drawings.MSG_OUTSIDE
    state["notice"] = None
    drawings.sync(state, [feature(outside)], drawings.OBSTACLE, FOOTPRINT)
    assert state["notice"] is None  # already ignored: no repeated notice
    half = box(X0 + 20, Y0 + 2, X0 + 30, Y0 + 4)  # half outside: clipped to the footprint
    drawings.sync(state, [feature(outside), feature(half)], drawings.OBSTACLE, FOOTPRINT)
    assert state["obstacles"][0]["geom"].area == pytest.approx(8, rel=1e-3)


def test_stored_shapes_are_counted_once_even_when_still_in_the_draw_output():
    state = drawings.new_state("b1")
    obs = obstacle(w=2, h=2)
    for _ in range(3):  # every rerun returns the same Leaflet layer while we also re-render it
        drawings.sync(state, [feature(obs)], drawings.OBSTACLE, FOOTPRINT)
    assert drawings.summary(state) == {"count": 1, "area_m2": pytest.approx(4, rel=1e-3), "terrace": False, "marked": True}
    assert len(drawings.stored_shapes(state)) == 1


def test_stored_shapes_are_re_rendered_after_a_rebuild():
    state = drawings.new_state("b1")
    drawings.sync(state, [feature(obstacle())], drawings.OBSTACLE, FOOTPRINT)
    drawings.sync(state, [feature(obstacle()), feature(obstacle(10))], drawings.OBSTACLE, FOOTPRINT)
    drawings.sync(state, [feature(obstacle(10))], drawings.OBSTACLE, FOOTPRINT)  # one deleted: map rebuilt
    rings = [ring for s in drawings.stored_shapes(state) for ring in drawings.polygon_to_latlon(s["geom"])]
    assert len(rings) == 1 and drawings.map_key(state) == "roof_view_1"


def test_undo_removes_only_the_most_recent_shape_and_bumps():
    state = drawings.new_state("b1")
    drawings.sync(state, [feature(obstacle())], drawings.OBSTACLE, FOOTPRINT)
    drawings.sync(state, [feature(obstacle()), feature(obstacle(10))], drawings.OBSTACLE, FOOTPRINT)
    assert drawings.undo_last(state) is True and state["gen"] == 1
    assert len(state["obstacles"]) == 1 and drawings.same_shape(state["obstacles"][0]["raw"], obstacle())
    assert drawings.undo_last(state) is True and drawings.undo_last(state) is False


def test_clear_and_confirm_empty():
    state = drawings.new_state("b1")
    drawings.sync(state, [feature(obstacle())], drawings.OBSTACLE, FOOTPRINT)
    drawings.confirm_empty(state)
    assert state["obstacles"] == [] and state["empty_confirmed"] and drawings.summary(state)["marked"]
    drawings.clear_all(state)
    assert not state["empty_confirmed"] and not drawings.summary(state)["marked"] and state["gen"] == 2


def test_non_polygon_features_are_ignored():
    point = {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [29.07, 40.97]}}
    assert drawings.feature_to_metric(point) is None and drawings.feature_to_metric({"broken": True}) is None
