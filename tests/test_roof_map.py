"""Roof map: static building-fitted view, enhancement control, alignment expander, drawings across rebuilds."""
import re

import pytest
import shapely
from streamlit_folium import _get_html, _get_map_string

import app
from core import drawings, explain, imagery_fx
from test_app import (  # noqa: F401  (fixtures and helpers shared with the app tests)
    _button, _centre_obstacle, _feature, _roof_draw, _roof_inputs, _run_with_point, _widget, building_point, map_calls,
    mocked, roof_figs, sample_index,
)

BOUNDS = [[40.969, 29.069], [40.971, 29.071]]


def _script(m):
    m.render()
    return re.sub(r"\s+", " ", _get_map_string(m))


def _obstacles(at):
    return [s["raw"].wkb for s in _roof_draw(at)["obstacles"]]


def _bounds(at):
    return at.session_state[app.KEY_ROOF_BOUNDS]["bounds"]


def _flat(bounds):
    return [v for corner in bounds for v in corner]


def _draw_one(at, box, sample_index, building_point):
    box["roof"] = {"all_drawings": [_feature(_centre_obstacle(_roof_inputs(sample_index, building_point)))]}
    at.run()
    box["roof"] = {"all_drawings": None}  # a rebuilt map has nothing yet
    assert len(_roof_draw(at)["obstacles"]) == 1


# ------------------------------------------------------------------ the builder

def test_rendered_roof_map_is_a_static_fitted_view():
    js = _script(app.build_roof_map([40.97, 29.07], bounds=BOUNDS))
    assert '.fitBounds( [[40.969, 29.069], [40.971, 29.071]], {"maxZoom": 22, "padding": [10, 10]} );' in js
    assert '"zoomSnap": 0.1' in js and '"zoomDelta": 0.1' in js and '"maxZoom": 22, } } );' in js
    for option in ("dragging", "scrollWheelZoom", "doubleClickZoom", "touchZoom", "boxZoom", "keyboard", "zoomControl"):
        assert f'"{option}": false' in js, option
    tile = re.search(r"L\.tileLayer\(.*?\);", js).group()
    assert '"className": "roof-esri"' in tile and '"maxNativeZoom": 19' in tile and '"maxZoom": 22' in tile
    assert "Tiles \\u00a9 Esri" in tile  # the Esri attribution is unchanged


def test_the_draw_toolbar_still_exists_on_the_static_map():
    js = _script(app.build_roof_map([40.97, 29.07], bounds=BOUNDS))
    assert "L.Control.Draw" in js and '"remove": true' in js


def test_fit_bounds_is_called_before_the_zoom_sigma_script():
    js = _script(app.build_roof_map([40.97, 29.07], bounds=BOUNDS, enhance=True))
    assert 0 <= js.index(".fitBounds(") < js.index("roofFxSigma")
    assert "whenReady(function () { setTimeout(roofFxSigma, 0); })" in js and '.on("zoomend", roofFxSigma)' in js
    m = app.build_roof_map([40.97, 29.07], bounds=BOUNDS, enhance=True)
    m.render()
    html = _get_html(m)
    assert 'id="roof-fx"' in html and ".roof-esri { filter: url(#roof-fx); }" in html


def test_enhancement_off_adds_no_filter_to_the_map():
    m = app.build_roof_map([40.97, 29.07], bounds=BOUNDS, enhance=False)
    m.render()
    html = _get_html(m)
    assert "<filter" not in html and "filter: url" not in html and "roofFxSigma" not in _get_map_string(m)


def test_the_first_map_gets_no_filter_and_stays_navigable():
    m = app.build_map()
    html = m.get_root().render()
    assert "roof-esri" not in html and "<filter" not in html and "fitBounds" not in html


# --------------------------------------------------------------- fitted bounds

def test_fit_bounds_is_the_footprint_bbox_plus_15_percent(sample_index, building_point):
    footprint = shapely.from_wkb(_roof_inputs(sample_index, building_point)["footprint_wkb_m"])
    rings = drawings.polygon_to_latlon(footprint)
    lats, lons = [p[0] for r in rings for p in r], [p[1] for r in rings for p in r]
    (s, w), (n, e) = app.roof_fit_bounds(footprint)
    d_lat, d_lon = (max(lats) - min(lats)) * 0.15, (max(lons) - min(lons)) * 0.15
    assert (s, w, n, e) == pytest.approx((min(lats) - d_lat, min(lons) - d_lon, max(lats) + d_lat, max(lons) + d_lon))


def test_fit_bounds_of_a_shifted_frame_moves_with_the_offset(sample_index, building_point):
    footprint = shapely.from_wkb(_roof_inputs(sample_index, building_point)["footprint_wkb_m"])
    (s0, w0), (n0, e0) = app.roof_fit_bounds(footprint)
    (s1, w1), (n1, e1) = app.roof_fit_bounds(footprint, (20.0, 10.0))
    assert s1 > s0 and n1 > n0 and w1 > w0 and e1 > e0  # 20 m east, 10 m north
    assert (n1 - s1) == pytest.approx(n0 - s0, rel=1e-3) and (e1 - w1) == pytest.approx(e0 - w0, rel=1e-3)


def test_rebuild_bumps_the_generation_and_keeps_every_shape():
    state = drawings.new_state("b")
    state["obstacles"] = [{"id": 1, "kind": drawings.OBSTACLE, "raw": shapely.box(0, 0, 1, 1), "geom": shapely.box(0, 0, 1, 1), "gen": 0}]
    drawings.rebuild(state)
    assert state["gen"] == 1 and len(state["obstacles"]) == 1 and state["offset"] == (0.0, 0.0)


# --------------------------------------------------------------- AppTest flows

def _fx_toggle(at):
    return next(t for t in at.toggle if t.label == "Görüntü iyileştirme")


def _has_filter(fig):
    fig.render()
    return "<filter" in _get_html(fig)


def test_enhancement_toggle_and_offset_changes_keep_drawings_and_the_fitted_view(mocked, roof_figs, sample_index, building_point):
    calls, box, figs = roof_figs
    at = _run_with_point(building_point)
    assert _fx_toggle(at).value is True and _has_filter(figs[-1][0])  # on by default
    assert imagery_fx.CAPTION in [c.value for c in at.caption]
    assert not [r for r in at.radio if r.label == "Görüntü iyileştirme"] and not [x for x in at.slider if x.label == "Şiddet"]
    _draw_one(at, box, sample_index, building_point)
    stored, bounds = _obstacles(at), _bounds(at)
    for change in (
        lambda: _fx_toggle(at).set_value(False).run(),
        lambda: _button(at, "5 m →").click().run(),
        lambda: _fx_toggle(at).set_value(True).run(),
        lambda: _button(at, "Sıfırla").click().run(),
        lambda: _fx_toggle(at).set_value(False).run(),
    ):
        gen = _roof_draw(at)["gen"]
        change()
        assert not at.exception
        assert _obstacles(at) == stored and _bounds(at) == bounds
        assert _roof_draw(at)["gen"] > gen  # a fresh map generation: never read as a deletion
    fig, kwargs = figs[-1]
    assert "center" not in kwargs and "zoom" not in kwargs
    assert at.session_state[app.KEY_FX_ON] is False and not _has_filter(fig)  # off: the original image, no filter


def test_fit_view_button_refits_to_the_shifted_frame_once(mocked, roof_figs, sample_index, building_point):
    calls, box, figs = roof_figs
    at = _run_with_point(building_point)
    footprint = shapely.from_wkb(_roof_inputs(sample_index, building_point)["footprint_wkb_m"])
    before = _bounds(at)
    assert _flat(before) == pytest.approx(_flat(app.roof_fit_bounds(footprint)))
    _button(at, "5 m →").click().run()
    _button(at, "5 m ↑").click().run()
    assert _bounds(at) == before  # the offset alone never moves the view
    _button(at, app.FIT_VIEW_LABEL).click().run()
    assert not at.exception
    after = _bounds(at)
    assert after != before and _flat(after) == pytest.approx(_flat(app.roof_fit_bounds(footprint, (5.0, 5.0))))
    assert f"{after[0][0]}" in _script(figs[-1][0])  # the next build fits the new bounds
    at.run()
    assert _bounds(at) == after


def test_alignment_expander_stays_open_across_alignment_clicks(mocked, building_point):
    at = _run_with_point(building_point)
    expander = lambda: next(e for e in at.expander if e.label == app.ALIGN_EXPANDER_LABEL)
    assert expander().proto.expanded is False  # collapsed until the user aligns
    for label in ("→", "5 m ↑", "Sıfırla", "←"):
        _button(at, label).click().run()
        assert not at.exception and expander().proto.expanded is True, label


def test_a_new_drawing_after_hesapla_marks_the_explanation_stale_until_hesapla(mocked, map_calls, sample_index, building_point):
    calls, box = map_calls
    at = _run_with_point(building_point)
    _button(at, "Hesapla").click().run()
    assert "Yapay Zekâ Değerlendirmesi" in [h.value for h in at.subheader]
    assert not any("açıklama güncel değil" in i.value for i in at.info)
    old_key = at.session_state["explanation"]["key"]
    _draw_one(at, box, sample_index, building_point)
    assert at.session_state["result"]["cizilen_engel_sayisi"] == 1
    assert any(i.value == "Değerler değişti; açıklama güncel değil." for i in at.info)  # never silently kept
    _button(at, "Hesapla").click().run()
    fresh = at.session_state["explanation"]
    assert fresh["key"] != old_key
    assert fresh["key"] == explain.payload_key(explain.build_payload(at.session_state["result"], None, False))
    assert not any("açıklama güncel değil" in i.value for i in at.info)


def test_flat_roof_notes_read_as_sentences():
    assert app.NOTE_FLAT_ASSUMED == "Değer düz çatı varsayımıyla hesaplandı."
    assert app.NOTE_FLAT_ASSUMED_GEOMETRY == (
        "Panel yerleşimi düz çatı varsayımıyla gösteriliyor; çatı yapısını seçerseniz sonuç tek değere iner."
    )


# ------------------------------------------------------ tile seam + Sıfırla restores the view

def test_tile_seam_css_is_on_the_roof_tiles_only_whether_or_not_the_filter_is_on():
    for enhance in (True, False):
        m = app.build_roof_map([40.97, 29.07], bounds=BOUNDS, enhance=enhance)
        m.render()
        assert ".roof-esri .leaflet-tile { width: 257px !important; height: 257px !important; }" in _get_html(m)
    assert "257px" not in app.build_map().get_root().render()  # the first map is untouched


def test_sifirla_restores_the_initial_fitted_view(mocked, roof_figs, sample_index, building_point):
    calls, box, figs = roof_figs
    at = _run_with_point(building_point)
    footprint = shapely.from_wkb(_roof_inputs(sample_index, building_point)["footprint_wkb_m"])
    initial = _bounds(at)
    assert _flat(initial) == pytest.approx(_flat(app.roof_fit_bounds(footprint)))
    _button(at, "5 m →").click().run()
    _button(at, app.FIT_VIEW_LABEL).click().run()
    assert _bounds(at) != initial
    _button(at, "Sıfırla").click().run()
    assert not at.exception
    assert _roof_draw(at)["offset"] == (0.0, 0.0) and _bounds(at) == initial
    assert f"{initial[0][0]}" in _script(figs[-1][0])  # the map is built with the initial view again
