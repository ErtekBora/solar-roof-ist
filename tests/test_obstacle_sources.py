"""Part B: ONE obstacle source (map drawing / own photo / sample demo) and no fake panels in ratio modes."""
import folium
import pytest

import app
from core import config, report
from test_app import (  # noqa: F401  (fixtures and helpers shared with the app tests)
    _after_hesapla, _button, _centre_obstacle, _draw_state, _feature, _roof_analysis, _roof_draw, _roof_inputs,
    _roof_map_calls, _run_photo, _run_with_point, _widget, building_point, map_calls, mocked, roof_figs, sample_index,
    spy_capacity,
)

SOURCE = app.SOURCE_LABEL


def _select(at, source):
    _widget(at.radio, SOURCE).set_value(source).run()
    assert not at.exception
    return at


def _draw_one(at, box, sample_index, building_point):
    box["roof"] = {"all_drawings": [_feature(_centre_obstacle(_roof_inputs(sample_index, building_point)))]}
    at.run()


def _polygons(figs):
    fg = figs[-1][1]["feature_group_to_add"]
    return [c for c in fg._children.values() if isinstance(c, folium.Polygon)]


def _demo_labels(at):
    return list(_widget(at.radio, "Örnek fotoğraf").options)


@pytest.fixture
def images(monkeypatch):
    """Every st.image call of the app, in order: [caption] (the real call still runs)."""
    import streamlit as st

    shown, real = [], st.image

    def spy(image, *args, **kwargs):
        shown.append(kwargs.get("caption"))
        return real(image, *args, **kwargs)

    monkeypatch.setattr(st, "image", spy)
    return shown


# ---------------------------------------------------------------- the selector

def test_hesapla_works_right_after_building_selection_with_the_drawing_source_active(mocked, building_point):
    at = _run_with_point(building_point)
    assert _widget(at.radio, SOURCE).value == app.SRC_DRAW
    _button(at, "Hesapla").click().run()
    assert not at.exception and at.session_state["result"]["guc_kaynagi"] == "ortalama"


def test_only_the_selected_sources_widgets_are_shown(mocked, building_point):
    at = _run_with_point(building_point)
    assert [b for b in at.button if b.label == "Engel yok, çatı boş"] and not at.file_uploader
    _select(at, app.SRC_PHOTO)
    assert at.file_uploader and not [b for b in at.button if b.label == "Engel yok, çatı boş"]
    assert app.PHOTO_MODE_CAPTION in [c.value for c in at.caption]
    assert not [r for r in at.radio if r.label == "Örnek fotoğraf"]
    _select(at, app.SRC_DEMO)
    assert not at.file_uploader and [r for r in at.radio if r.label == "Örnek fotoğraf"]
    assert not [b for b in at.button if b.label == "Engel yok, çatı boş"]


def test_the_sidebar_help_describes_the_new_flow(mocked, building_point):
    at = _run_with_point(building_point)
    help_md = [m.value for m in at.sidebar.markdown if m.value.startswith("1. ")][0]
    assert "3. Engel kaynağını seçin: haritada çizin (önerilen) veya çatı fotoğrafı yükleyin." in help_md
    assert help_md.endswith("5. İsteğe bağlı bilgileri girip Hesapla'ya basın.")


# ------------------------------------------------------------ one source is active

def test_only_one_source_counts_after_any_sequence_of_switches(mocked, map_calls, sample_index, building_point):
    calls, box = map_calls
    at = _run_with_point(building_point)
    _draw_one(at, box, sample_index, building_point)
    _button(at, "Hesapla").click().run()
    drawn = at.session_state["result"]
    assert drawn["guc_kaynagi"] == "geometri"
    for source, expected in ((app.SRC_PHOTO, "ortalama"), (app.SRC_DEMO, "ortalama"), (app.SRC_DRAW, "geometri"),
                             (app.SRC_DEMO, "ortalama"), (app.SRC_PHOTO, "ortalama"), (app.SRC_DRAW, "geometri")):
        _select(at, source)
        assert at.session_state["result"]["guc_kaynagi"] == expected, source
    assert at.session_state["result"]["kurulabilir_guc_kwp_aralik"] == pytest.approx(drawn["kurulabilir_guc_kwp_aralik"])


def test_inactive_sources_keep_their_data_and_get_it_back(
    mocked, map_calls, fake_gemini, roof10_json, sample_index, building_point
):
    calls, box = map_calls
    fake_gemini([roof10_json])
    at = _run_photo("roof_10.jpg", building_point)
    _button(at, "Hesapla").click().run()
    photo_ratio = at.session_state["result"]["engel_orani"]
    assert photo_ratio is not None and at.session_state["result"]["guc_kaynagi"] == "fotograf"
    _select(at, app.SRC_DRAW)
    box["roof"] = {"all_drawings": [_feature(_centre_obstacle(_roof_inputs(sample_index, building_point)))]}
    at.run()
    assert [b for b in at.button if b.label == "Çatıya tıklayarak hizala"]  # alignment stays available
    at.session_state[app.KEY_DRAWINGS]["offset"] = (4.0, -2.0)
    at.run()
    result = at.session_state["result"]
    assert result["guc_kaynagi"] == "geometri" and result["engel_orani"] != photo_ratio  # photo ignored meanwhile
    assert at.session_state[app.KEY_PHOTO_RESULT]["detection"] is not None  # ... but kept
    _select(at, app.SRC_PHOTO)
    assert at.session_state["result"]["guc_kaynagi"] == "fotograf"
    assert at.session_state["result"]["engel_orani"] == pytest.approx(photo_ratio)
    draw = at.session_state[app.KEY_DRAWINGS]
    assert len(draw["obstacles"]) == 1 and draw["offset"] == (4.0, -2.0)  # drawings and offset are kept
    _select(at, app.SRC_DRAW)
    assert at.session_state["result"]["guc_kaynagi"] == "geometri" and at.session_state["result"]["cizilen_engel_sayisi"] == 1


def test_drawn_then_other_source_then_back_gives_the_same_drawings_and_kwp(mocked, map_calls, sample_index, building_point):
    calls, box = map_calls
    at = _run_with_point(building_point)
    _button(at, "Hesapla").click().run()
    _draw_one(at, box, sample_index, building_point)
    before = at.session_state["result"]
    stored = [s["raw"].wkb for s in at.session_state[app.KEY_DRAWINGS]["obstacles"]]
    gen = at.session_state[app.KEY_DRAWINGS]["gen"]
    keys = {c["key"] for c in _roof_map_calls(calls)}
    _select(at, app.SRC_PHOTO)
    # the map now returns nothing / something else: it must be ignored completely, however often it reruns
    for returned in ({"all_drawings": None}, {"all_drawings": []}, {"all_drawings": [_feature(_centre_obstacle(
            _roof_inputs(sample_index, building_point), 9.0))]}):
        box["roof"] = returned
        at.run()
        at.run()
        draw = at.session_state[app.KEY_DRAWINGS]
        assert [s["raw"].wkb for s in draw["obstacles"]] == stored and draw["gen"] == gen
    box["roof"] = {"all_drawings": None}  # a rebuilt map has nothing yet
    _select(at, app.SRC_DRAW)
    after = at.session_state["result"]
    draw = at.session_state[app.KEY_DRAWINGS]
    assert [s["raw"].wkb for s in draw["obstacles"]] == stored and draw["gen"] == gen
    assert after["guc_kaynagi"] == "geometri"
    assert after["kurulabilir_guc_kwp_aralik"] == pytest.approx(before["kurulabilir_guc_kwp_aralik"])
    assert {c["key"] for c in _roof_map_calls(calls)} == keys | {f"roof_view_{gen}"}  # one and the same key all along


def test_the_roof_map_keeps_one_key_and_loses_its_toolbar_outside_the_drawing_source(mocked, roof_figs, building_point):
    calls, box, figs = roof_figs
    at = _run_with_point(building_point)
    from test_app import _has_draw_toolbar

    assert _has_draw_toolbar(figs[-1][0])
    key = figs[-1][1]["key"]
    for source in (app.SRC_PHOTO, app.SRC_DEMO):
        _select(at, source)
        assert figs[-1][1]["key"] == key and not _has_draw_toolbar(figs[-1][0])
    _select(at, app.SRC_DRAW)
    assert _has_draw_toolbar(figs[-1][0])


# --------------------------------------------------------------- own photo source

def test_photo_mode_uses_the_photo_ratio_and_hides_the_drawing_tools(
    mocked, roof_figs, fake_gemini, roof10_json, spy_capacity, building_point
):
    calls, box, figs = roof_figs
    fake_gemini([roof10_json])
    at = _run_photo("roof_10.jpg", building_point)
    from test_app import _has_draw_toolbar

    assert not _has_draw_toolbar(figs[-1][0]) and app.PHOTO_MODE_CAPTION in [c.value for c in at.caption]
    assert not [b for b in at.button if b.label in ("Engelleri temizle", "Son çizimi geri al")]
    assert [b for b in at.button if b.label == "Çatıya tıklayarak hizala"]  # alignment stays (display only)
    _button(at, "Hesapla").click().run()
    r = at.session_state["result"]
    assert r["guc_kaynagi"] == "fotograf" and 0 < spy_capacity[-1]["obstacle_ratio"] < 0.95
    assert [k for k in at.session_state if k.startswith("obstacles_")]


def test_the_uploaded_photo_survives_a_trip_to_another_source(mocked, building_point):
    at = _run_photo("roof_10.jpg", building_point)
    _select(at, app.SRC_DRAW)
    _select(at, app.SRC_PHOTO)
    assert at.session_state[app.KEY_PHOTO]["name"] == "roof_10.jpg"
    assert any("roof_10.jpg" in c.value for c in at.caption)


# ------------------------------------------------------------------ sample demo

def test_the_selector_offers_exactly_the_five_curated_samples(mocked, images, building_point):
    at = _select(_run_with_point(building_point), app.SRC_DEMO)
    radio = _widget(at.radio, "Örnek fotoğraf")
    assert _demo_labels(at) == [
        "Örnek 1 – Düz teras, çok engelli (eski bina)", "Örnek 2 – Düz teras, çok engelli (yeni bina)",
        "Örnek 3 – Düz teras, az engelli", "Örnek 4 – Beşik çatı", "Örnek 5 – Kırma çatı",
        "Örnek 6 – Karma (teras + eğimli, yeni bina)", "Örnek 7 – Karma (teras + eğimli, eski bina)",
        "Örnek 8 – Gerçek fotoğraf",
    ]
    assert app.DEMO_SAMPLES == [f"synth_0{i}.jpg" for i in range(1, 8)] + ["roof_03.jpg"]
    assert radio.value is None  # nothing selected by default
    assert not images


def test_selecting_a_sample_shows_the_photo_at_once_without_a_gemini_call(mocked, fake_gemini, images, building_point):
    client = fake_gemini([])
    at = _select(_run_with_point(building_point), app.SRC_DEMO)
    _widget(at.radio, "Örnek fotoğraf").set_value(app.DEMO_LABELS[3]).run()
    assert not at.exception and client.calls == 0
    assert len(images) == 1 and images[0] is None  # the synthetic caption is a st.caption, not an image caption
    captions = [c.value for c in at.caption]
    assert "Dosya: synth_04.jpg" in captions and app.SYNTH_CAPTION in captions
    assert app.DEMO_NOT_YOURS in [i.value for i in at.info]
    assert [b for b in at.button if b.label == "YZ ile analiz et"]


def test_analyze_runs_one_call_only_when_pressed_and_is_cached_per_sample(
    mocked, fake_gemini, roof10_json, images, building_point
):
    client = fake_gemini([roof10_json])
    at = _select(_run_with_point(building_point), app.SRC_DEMO)
    _widget(at.radio, "Örnek fotoğraf").set_value(app.DEMO_LABELS[3]).run()
    _button(at, "Hesapla").click().run()  # Hesapla never analyses a sample
    assert client.calls == 0
    _button(at, "YZ ile analiz et").click().run()
    assert not at.exception and client.calls == 1
    assert len(images) == 4 and images[-1].startswith("Yeşil: çatı sınırı") and at.dataframe  # photo, then annotated photo + table
    assert any("YZ bu fotoğraftaki çatıyı eğimli olarak değerlendirdi" in w.value for w in at.markdown)
    at.run()
    images.clear()
    _widget(at.radio, "Örnek fotoğraf").set_value(app.DEMO_LABELS[1]).run()
    assert len(images) == 1 and images[0] is None and client.calls == 1  # no analysis for this sample yet
    images.clear()
    _widget(at.radio, "Örnek fotoğraf").set_value(app.DEMO_LABELS[3]).run()
    assert client.calls == 1 and images[-1].startswith("Yeşil: çatı sınırı")  # cached: shown again, no new call


def test_the_demo_never_changes_inputs_or_results(mocked, fake_gemini, roof10_json, building_point):
    fake_gemini([{**roof10_json, "roof_type": "flat"}])
    at = _run_with_point(building_point)
    _button(at, "Hesapla").click().run()
    before = at.session_state["result"]
    inputs = (
        at.session_state[app.KEY_STRUCTURE], at.session_state[app.KEY_SHARE], at.session_state[app.KEY_FLATS],
    )
    _select(at, app.SRC_DEMO)
    _widget(at.radio, "Örnek fotoğraf").set_value(app.DEMO_LABELS[3]).run()
    _button(at, "YZ ile analiz et").click().run()
    assert not at.exception
    assert (at.session_state[app.KEY_STRUCTURE], at.session_state[app.KEY_SHARE], at.session_state[app.KEY_FLATS]) == inputs
    assert _widget(at.radio, "Çatı yapısı").value == "Bilmiyorum"
    _button(at, "Hesapla").click().run()
    after = at.session_state["result"]
    assert after["guc_kaynagi"] == "ortalama" and after["engel_orani"] is None
    assert after["kurulabilir_guc_kwp_aralik"] == pytest.approx(before["kurulabilir_guc_kwp_aralik"])
    assert after["cati_yapisi"] == before["cati_yapisi"] and after["daire_sayisi"] == before["daire_sayisi"]


def test_a_failed_sample_analysis_is_shown_inside_the_demo_section(mocked, fake_gemini, building_point):
    fake_gemini(["{bad", "{bad"])
    at = _select(_run_with_point(building_point), app.SRC_DEMO)
    _widget(at.radio, "Örnek fotoğraf").set_value(app.DEMO_LABELS[3]).run()
    _button(at, "YZ ile analiz et").click().run()
    assert not at.exception and any("okunamadı" in w.value for w in at.warning)


# ------------------------------------------------------- no fake panels in ratio modes

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
    _button(at, "Engelleri temizle").click().run()
    for source in (app.SRC_PHOTO, app.SRC_DEMO):
        _select(at, source)
        assert len(_polygons(figs)) == 2 and app.RATIO_NO_PANELS_CAPTION in [c.value for c in at.caption]


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


def test_every_sample_has_a_clean_caption_and_synthetic_ones_say_so(mocked, images, building_point):
    at = _select(_run_with_point(building_point), app.SRC_DEMO)
    for label, name in zip(app.DEMO_LABELS, app.DEMO_SAMPLES):
        images.clear()
        _widget(at.radio, "Örnek fotoğraf").set_value(label).run()
        assert not at.exception and len(images) == 1
        texts = [c.value for c in at.caption] + [images[0] or ""] + [m.value for m in at.markdown]
        assert not any("TODO" in t or "eklenecek" in t for t in texts), name
        if name.startswith("synth_"):
            assert app.SYNTH_CAPTION in [c.value for c in at.caption] and "sentetik" in app.SYNTH_CAPTION.lower()
            assert "Sentetik" in " ".join(m.value for m in at.markdown)
            assert images[0] is None
        else:
            assert images[0].endswith("https://www.pexels.com/tr-tr/@drector/") and "Pexels License" in images[0]
    assert app.SYNTH_CAPTION == (
        "YZ ile üretilmiş sentetik örnek (Google Gemini görüntü modeli, 21.10.2026). Gerçek bir bina değildir; "
        "YZ'nin çatıda neleri bulduğunu göstermek içindir."
    )
