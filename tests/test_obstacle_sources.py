"""Part B: obstacles come only from the map drawing, and no fake panels in ratio modes."""
import folium

import app
from core import config, report
from test_app import (  # noqa: F401  (fixtures and helpers shared with the app tests)
    _after_hesapla, _button, _centre_obstacle, _draw_state, _feature, _roof_analysis, _roof_draw, _roof_inputs,
    _roof_map_calls, _run_with_point, _widget, building_point, map_calls, mocked, roof_figs, sample_index,
    spy_capacity,
)


def _polygons(figs):
    fg = figs[-1][1]["feature_group_to_add"]
    return [c for c in fg._children.values() if isinstance(c, folium.Polygon)]


# ---------------------------------------------------------------- the selector

def test_hesapla_works_right_after_building_selection_with_the_drawing_source_active(mocked, building_point):
    at = _run_with_point(building_point)
    assert not [r for r in at.radio if r.label == "Çatı engelleri nereden gelsin?"]
    _button(at, "Hesapla").click().run()
    assert not at.exception and at.session_state["result"]["guc_kaynagi"] == "ortalama"


def test_only_the_drawing_widgets_are_shown(mocked, building_point):
    at = _run_with_point(building_point)
    assert [b for b in at.button if b.label == "Engel yok, çatı boş"] and not at.file_uploader
    assert "Engel kaynağı" not in [h.value for h in at.subheader]
    assert not [r for r in at.radio if r.label in ("Çatı engelleri nereden gelsin?", "Örnek fotoğraf")]


def test_the_sidebar_help_describes_the_new_flow(mocked, building_point):
    at = _run_with_point(building_point)
    help_md = [m.value for m in at.sidebar.markdown if m.value.startswith("1. ")][0]
    assert "3. Engelleri haritada çizin." in help_md
    assert help_md.endswith("5. İsteğe bağlı bilgileri girip Hesapla'ya basın.")


# ------------------------------------------------------------ one source is active

def test_ratio_modes_draw_no_panels_and_layout_mode_does(mocked, roof_figs, sample_index, building_point):
    calls, box, figs = roof_figs
    at = _run_with_point(building_point)
    _button(at, "Hesapla").click().run()
    assert at.session_state["result"]["guc_kaynagi"] == "ortalama"
    assert len(_polygons(figs)) == 2 and app.RATIO_NO_PANELS_CAPTION in [c.value for c in at.caption]
    label = [c for c in figs[-1][1]["feature_group_to_add"]._children.values() if isinstance(c, folium.Marker)][0]
    assert "≈" in label.icon.options["html"] and "(ortalama orana göre)" in label.icon.options["html"]
    box["roof"] = {"all_drawings": [_feature(_centre_obstacle(_roof_inputs(sample_index, building_point)))]}
    at.run()
    assert at.session_state["result"]["guc_kaynagi"] == "geometri"
    assert len(_polygons(figs)) > 3  # footprint, setback, the obstacle and the panels
    assert app.RATIO_NO_PANELS_CAPTION not in [c.value for c in at.caption]


def test_the_labels_name_the_ratio_source():
    sc = {"kwp": 12.34, "kwh": 15000.0, "paylasim": {"indirim_ay_ortalama_tl": [100.0, 200.0]}}
    roof = {"scenario": sc, "mode": "numeric"}
    assert app.roof_label(roof, {"guc_kaynagi": "ortalama"}).startswith("≈12,3 kWp (ortalama orana göre)")
    assert app.roof_label(roof, {"guc_kaynagi": "fotograf"}).startswith("≈12,3 kWp (fotoğraftaki engel oranına göre)")


# ---------------------------------------------------- "Engeller" and "Panel" rows follow the source

def _rows(r, inputs):
    return {row[0]: row for row in app.assumption_rows(r, inputs, None).itertuples(index=False, name=None)}


def test_obstacle_and_panel_rows_follow_the_active_source(mocked, sample_index, building_point):
    inputs = _roof_inputs(sample_index, building_point, structure="flat")
    analysis = _roof_analysis(inputs)
    avg, _, _ = app.evaluate(analysis, inputs, None, set(), None)
    rows = _rows(avg, inputs)
    assert rows["Engeller"][1] == "ortalama oran"
    assert rows["Panel"][1].startswith("≈") and rows["Panel"][1].endswith("panel (orandan tahmini; yerleşim için engelleri çizin)")
    lay, _, _ = app.evaluate(analysis, inputs, None, set(), _draw_state(inputs, [_centre_obstacle(inputs)]))
    rows = _rows(lay, inputs)
    assert rows["Engeller"][1].endswith("çizim, 4,0 m²") and "orandan" not in rows["Panel"][1]
    assert rows["Panel"][1] == "1,72 × 1,13 m, 0,40 kWp"
    detection = {
        "roof_type": "flat", "image_view": "top-down", "width": 100, "height": 100, "roof_polygon_px": [(0, 0), (100, 0), (100, 100), (0, 100)],
        "obstacles": [{"id": 0, "box_px": (0, 0, 50, 100), "label": "chimney", "confidence": 0.9}],
        "multiple_buildings": False, "roof_source": "polygon",
    }
    photo, _, _ = app.evaluate(analysis, inputs, detection, {0}, None)
    rows = _rows(photo, inputs)
    assert photo["guc_kaynagi"] == "fotograf" and rows["Engeller"][1] == "fotoğraf analizi, %50,0"
    assert rows["Panel"][1].startswith("≈")
    md = report.report_markdown(photo, [])
    assert "- Panel: ≈" in md and "orandan tahmini" in md
    assert "- Panel sayısı:" in report.report_markdown(lay, [])


def _offset_caption(at):
    return [c.value for c in at.caption if c.value.startswith("Kaydırma:")]


def test_drawing_source_has_a_collapsed_expander_with_a_constant_label(mocked, building_point):
    at = _run_with_point(building_point)
    [exp] = [e for e in at.expander if e.label == "Çerçeveyi çatıya hizala (yüksek binalarda gerekir)"]
    assert exp.proto.expanded is False
    inside = [b.label for b in exp.button]
    assert "Çatıya tıklayarak hizala" in inside and "↑" in inside and "5 m →" in inside and "Sıfırla" in inside
    assert app.ALIGN_CAPTION in [c.value for c in exp.caption] and app.ALIGN_APPROX_CAPTION in [c.value for c in exp.caption]
    assert not _offset_caption(at)  # no line at a zero offset
    _button(at, "→").click().run()
    _button(at, "5 m ↑").click().run()
    assert _offset_caption(at) == ["Kaydırma: 1 m D, 5 m K"]
    [again] = [e for e in at.expander if e.label.startswith("Çerçeveyi çatıya hizala")]
    assert again.label == exp.label and again.proto.expanded is False  # the label never changes
    assert _offset_caption(at) and all(c.value not in _offset_caption(at) for c in again.caption)  # the line is outside
    _button(at, "Sıfırla").click().run()
    assert not _offset_caption(at)


def test_click_to_align_shows_caption_and_cancel_outside_the_expander(mocked, roof_figs, sample_index, building_point):
    from test_app import _click_at

    calls, box, figs = roof_figs
    at = _run_with_point(building_point)
    _button(at, "Çatıya tıklayarak hizala").click().run()
    [exp] = [e for e in at.expander if e.label == app.ALIGN_EXPANDER_LABEL]
    assert app.ALIGN_CLICK_CAPTION in [c.value for c in at.caption]
    assert app.ALIGN_CLICK_CAPTION not in [c.value for c in exp.caption]
    assert "İptal" in [b.label for b in at.button] and "İptal" not in [b.label for b in exp.button]
    assert [b.label for b in exp.button].count("Çatıya tıklayarak hizala") == 0
    box["roof"] = {"all_drawings": None, "last_clicked": _click_at(_roof_inputs(sample_index, building_point), 6.0, -3.0)}
    at.run()
    assert _roof_draw(at)["offset"] == (6.0, -3.0) and not _roof_draw(at).get("align_active")
    assert "İptal" not in [b.label for b in at.button] and _offset_caption(at) == ["Kaydırma: 6 m D, 3 m G"]


