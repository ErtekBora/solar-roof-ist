import json
from pathlib import Path

import folium
import pytest
import shapely
import streamlit as st
from pyproj import Transformer
from streamlit.testing.v1 import AppTest

import app
from core import building_mode, buildings, calc, config, drawings, explain, layout, report, solar
from core.formatting import tr_num as tr
from core.solar import _parse_pvcalc

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "buildings_sample.parquet"
TO_WGS = Transformer.from_crs("EPSG:32635", "EPSG:4326", always_xy=True)
OUT_OF_COVERAGE = (41.015, 28.979)  # old map centre, outside the prototype bbox -> manual-area flow
BILL_LABEL = "Ortalama aylık daire faturası (TL)"
FLATS_LABEL = "Daire sayısı"
ROOF_LABEL = "Çatı yapısı"
SHARE_LABEL = "Düz teras payı (%)"
PITCHED_LABEL = "Eğimli çatı tipi"


def _fake_yield(lat, lon, horizon72=None):
    load = lambda n: _parse_pvcalc(json.loads((FIXTURES / n).read_text(encoding="utf-8")))
    ideal, shaded = load("pvgis_optimal.json"), load("pvgis_terrain.json")
    return {"ideal": ideal, "shaded": shaded, "angle": ideal["angle"], "aspect": ideal["aspect"]}


def _fake_face(lat, lon, aspect, angle=30, horizon72=None):
    """A pitched face: 93 % of the flat shaded yield, a north-ish face (|aspect| > 135) 55 %."""
    shaded = _parse_pvcalc(json.loads((FIXTURES / "pvgis_terrain.json").read_text(encoding="utf-8")))
    k = 0.55 if abs(aspect) > 135 else 0.93
    return {**shaded, "e_y": shaded["e_y"] * k, "e_m": [m * k for m in shaded["e_m"]], "angle": angle, "aspect": aspect}


def _bundle(ridge=0.0):
    specs = building_mode.face_specs(ridge)
    return {"solar": _fake_yield(41.0, 29.0), "faces": [{**s, "solar": _fake_face(41.0, 29.0, s["aspect"])} for s in specs]}


@pytest.fixture
def mocked(monkeypatch):
    # app.py fetches through core.solar.fetch_roof_yields, which looks these up at call time (also under AppTest)
    monkeypatch.setattr(solar, "get_solar_yield", _fake_yield)
    monkeypatch.setattr(solar, "get_face_yield", _fake_face)
    monkeypatch.setattr(config, "BUILDINGS_PARQUET", SAMPLE)
    monkeypatch.setattr(config, "get_gemini_api_key", lambda: "test-key")  # never read the real secrets file
    st.cache_resource.clear()  # the building index is cached per process
    explain.clear_cache()
    # default: the explanation model is unavailable -> template; tests wanting Gemini install fake_explain
    monkeypatch.setattr(explain, "_get_client", lambda api_key=None: (_ for _ in ()).throw(RuntimeError("offline")))
    yield
    st.cache_resource.clear()


@pytest.fixture(scope="module")
def sample_index():
    return buildings.load_buildings(SAMPLE, Path("none.json"))


@pytest.fixture
def building_point(sample_index):
    """(lat, lon) inside the first fixture building."""
    p = sample_index.polygons_m[0].representative_point()
    lon, lat = TO_WGS.transform(p.x, p.y)
    return lat, lon


def _widget(elements, label):
    return next(e for e in elements if e.label == label)


def test_calculate_builds_result(mocked):
    r = app.calculate(41.015, 28.979, 150.0, "flat", n_flats=3)
    assert r["uygunluk_puani_yuzde"] == pytest.approx(99.79, abs=0.01)
    assert r["yillik_isinim_kwh_m2"] == pytest.approx(1738.72)
    assert r["sicaklik_kaybi_yuzde"] == pytest.approx(6.49)
    assert len(r["senaryolar"][0]["aylik_kwh"]) == 12 and r["daire_sayisi"] == 3
    assert r["tasarruf_yontemi"] == "genis_tahmin"
    assert app.calculate(41.015, 28.979, 150.0, "flat", 1500)["tasarruf_yontemi"] == "fatura"


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _run_with_point(point=OUT_OF_COVERAGE, skip=True):
    """Open the app on a point; by default also press 'Fotoğrafsız devam et' so the later steps are visible."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    if point:
        at.session_state["selected_point"] = point
    at.run()
    if point and skip:
        _button(at, "Fotoğrafsız devam et").click().run()
    return at


def test_nothing_to_calculate_without_a_point(mocked):
    at = _run_with_point(point=None)
    assert not [b for b in at.button if b.label == "Hesapla"] and not at.exception
    assert "result" not in at.session_state


def test_hesapla_appears_only_after_the_photo_step(mocked, building_point):
    at = _run_with_point(building_point, skip=False)
    assert not [b for b in at.button if b.label == "Hesapla"]
    assert not [n for n in at.number_input if n.label == BILL_LABEL] and not [r for r in at.radio if r.label == ROOF_LABEL]
    titles = [h.value for h in at.subheader]
    assert titles[:2] == ["Daire sayısı", app.PHOTO_STEP_TITLE]  # building summary, then the flat count, then the photo
    assert app.PHOTO_GUIDANCE in [m.value for m in at.markdown] and app.PHOTO_BENEFIT in [c.value for c in at.caption]
    _button(at, "Fotoğrafsız devam et").click().run()
    assert [b for b in at.button if b.label == "Hesapla"] and [r for r in at.radio if r.label == ROOF_LABEL]
    assert any("Fotoğrafsız devam ediliyor" in c.value for c in at.caption)
    assert not [b for b in at.button if b.label == "Fotoğrafsız devam et"]


def test_building_summary_shows_footprint_setback_height_and_source(mocked, building_point):
    at = _run_with_point(building_point, skip=False)
    assert any("Bina bulundu: taban alanı" in s.value and "kenar boşluğu" in s.value for s in at.success)
    assert any(c.value.startswith(("Kat sayısı verisi", "Kat sayısı verisi yok")) and "Bina yüksekliği:" in c.value for c in at.caption)


def test_sidebar_keeps_only_global_things(mocked, building_point):
    at = _run_with_point(building_point, skip=False)
    assert [b.label for b in at.sidebar.button] == ["Sıfırla"]
    assert not at.sidebar.number_input and not at.sidebar.radio
    assert app.ATTRIBUTION in [c.value for c in at.sidebar.caption]


def test_no_sidebar_tariff_input_and_calculation_runs_without_a_bill(mocked):
    at = _run_with_point()
    assert not any("tarife" in n.label.lower() for n in at.number_input)
    assert not any("tarife" in w.value for w in at.warning)
    _button(at, "Hesapla").click().run()
    assert not at.exception and not at.error
    assert at.session_state["result"]["tasarruf_yontemi"] == "genis_tahmin"


def test_full_flow_shows_cards_and_keeps_result(mocked, monkeypatch):
    at = _run_with_point()
    _button(at, "Hesapla").click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Uygunluk (arazi gölgesi)" in labels and "Toplam kurulum maliyeti" in labels
    assert len(labels) == 13  # 4 building cards, 3 per-flat cards (no bill), 6 others
    # a rerun without the button must not call PVGIS again
    monkeypatch.setattr(solar, "get_solar_yield", lambda *a, **k: pytest.fail("PVGIS called on rerun"))
    monkeypatch.setattr(solar, "get_face_yield", lambda *a, **k: pytest.fail("PVGIS called on rerun"))
    at.run()
    assert len(at.metric) == 13


def test_attribution_text(mocked):
    at = _run_with_point()
    assert (
        "© OpenStreetMap katkıcıları · Bina verisi: Overture Maps Foundation (ODbL), Microsoft ML Buildings · "
        "Güneş verisi: PVGIS (JRC)"
    ) in [c.value for c in at.caption]


# ------------------------------------------------------------- building flow (P1)

@pytest.fixture
def spy_capacity(monkeypatch):
    calls = []
    real = building_mode.analyze_building

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(building_mode, "analyze_building", spy)  # app.py looks it up on every call
    return calls


def test_building_found_uses_the_footprint_and_setback_area(mocked, spy_capacity, sample_index, building_point):
    at = _run_with_point(building_point)
    assert any("Bina bulundu" in s.value for s in at.success)
    assert not [n for n in at.number_input if n.label.startswith("Bina bulunamazsa")]  # manual area only without a building
    _button(at, "Hesapla").click().run()
    assert not at.exception
    found = buildings.find_building(sample_index, *building_point)["building"]
    call = spy_capacity[-1]
    assert call["roof_area_m2"] == pytest.approx(found["area_m2"])
    assert call["usable_base_area_m2"] == pytest.approx(found["setback_area_m2"])
    assert call["structure"] == "unknown"  # nothing chosen: "Bilmiyorum"
    assert at.session_state["result"]["guven_seviyesi"] == "Düşük"  # polygon + unknown roof structure
    rows = at.dataframe[-1].value
    assert "Ölçüm/veri" in set(rows[rows["Değer"] == "Çatı alanı"]["Durum"])
    assert {"Kenar boşluğu", "Bina yüksekliği"} <= set(rows["Değer"])


def test_known_roof_type_with_a_building_is_orta(mocked, building_point):
    at = _run_with_point(building_point)
    _button(at, "Hesapla").click().run()
    _widget(at.radio, ROOF_LABEL).set_value("Tamamen düz").run()
    assert at.session_state["result"]["guven_seviyesi"] == "Orta"


def test_building_not_found_message_and_manual_area(mocked, spy_capacity):
    at = _run_with_point((40.9501, 29.0501))  # inside the bbox, far from the fixture buildings
    assert any("15 m" in i.value for i in at.info)
    _widget(at.number_input, "Bina bulunamazsa kullanılacak çatı alanı (m²)").set_value(222.0)
    _button(at, "Hesapla").click().run()
    call = spy_capacity[-1]
    assert call["roof_area_m2"] == 222.0 and call["usable_base_area_m2"] is None
    assert not at.exception


def test_out_of_coverage_message_and_manual_flow(mocked):
    at = _run_with_point(OUT_OF_COVERAGE)
    assert any("Erenköy" in i.value and "Göztepe" in i.value for i in at.info)
    _button(at, "Hesapla").click().run()
    assert not at.exception and len(at.metric) == 13


def test_loader_returning_none_warns_and_manual_flow_still_works(mocked, monkeypatch, building_point):
    monkeypatch.setattr(buildings, "load_buildings", lambda *a, **k: None)
    at = _run_with_point(building_point)
    assert not at.exception
    assert any("Bina verisi yüklenemedi; çatı alanını elle girerek devam edebilirsiniz" in w.value for w in at.warning)
    _button(at, "Hesapla").click().run()
    assert not at.exception and len(at.metric) == 13


def test_first_page_load_without_a_click_does_not_crash(mocked):
    at = _run_with_point(point=None)
    assert not at.exception and any("kapsama alanı" in i.value for i in at.info)


# ----------------------------------------------------- map: layers, click handling, one-shot centre

def test_base_map_holds_only_tiles_and_layer_control():
    kinds = [type(c).__name__ for c in app.build_map()._children.values()]
    assert kinds.count("TileLayer") == 2 and kinds.count("LayerControl") == 1
    assert set(kinds) <= {"TileLayer", "LayerControl", "Element"} or "Marker" not in kinds
    assert not any(k in kinds for k in ("Polygon", "Rectangle", "GeoJson", "Marker", "CircleMarker", "FeatureGroup"))


def test_all_dynamic_elements_live_in_one_feature_group(sample_index, building_point):
    lookup = buildings.find_building(sample_index, *building_point)
    fg = app.build_overlay(lookup, building_point)
    assert isinstance(fg, folium.FeatureGroup)
    kinds = [type(c).__name__ for c in fg._children.values()]
    assert kinds.count("Rectangle") == 1 and kinds.count("CircleMarker") == 1
    assert kinds.count("Polygon") == 1 + len(lookup["building"]["setback_polygons"]) + len(lookup["neighbours"])
    empty = [type(c).__name__ for c in app.build_overlay(None, None)._children.values()]
    assert empty == ["Rectangle"]  # just the coverage box before any click


def test_selected_point_is_only_replaced_by_a_different_click():
    stored = (40.97, 29.07)
    assert app.updated_point(stored, None) == (stored, False)
    assert app.updated_point(stored, {"lat": 40.97, "lng": 29.07}) == (stored, False)  # same click again
    assert app.updated_point(stored, {"lat": 40.98, "lng": 29.06}) == ((40.98, 29.06), True)
    assert app.updated_point(None, {"lat": 40.98, "lng": 29.06}) == ((40.98, 29.06), True)


def test_recenter_flag_is_consumed_after_one_use(sample_index, building_point):
    state = {}
    lookup = app.ensure_lookup(state, sample_index, building_point)
    assert lookup["status"] == "ok"
    taken = app.take_recenter(state)
    assert taken["zoom"] == 18 and taken["center"] == lookup["building"]["centroid"]
    assert app.take_recenter(state) is None
    app.ensure_lookup(state, sample_index, building_point)  # same point: cached, no new flag
    assert app.take_recenter(state) is None
    app.ensure_lookup(state, sample_index, OUT_OF_COVERAGE)  # no building -> no flag
    assert app.take_recenter(state) is None


@pytest.fixture
def map_calls(monkeypatch):
    import streamlit_folium

    # box["value"]: what the main map returns; box["roof"]: what the roof map (key roof_view_<gen>) returns
    calls, box = [], {"value": {"last_clicked": None}, "roof": {"all_drawings": None}}

    def fake(fig, **kwargs):
        calls.append(kwargs)
        return box["roof"] if str(kwargs.get("key", "")).startswith("roof_view") else box["value"]

    monkeypatch.setattr(streamlit_folium, "st_folium", fake)
    return calls, box


def _main_call(calls):
    return [c for c in calls if c.get("key") == "map"][-1]


def test_center_and_zoom_are_passed_only_on_the_rerun_after_a_building_is_found(mocked, map_calls, building_point):
    calls, _ = map_calls
    at = _run_with_point(building_point, skip=False)
    first = _main_call(calls)
    assert first["zoom"] == 18 and first["center"] is not None
    assert first["returned_objects"] == ["last_clicked"] and first["key"] == "map"
    assert isinstance(first["feature_group_to_add"], folium.FeatureGroup)
    at.run()
    assert "center" not in _main_call(calls) and "zoom" not in _main_call(calls)  # the map stays where the user left it
    at.run()
    assert "center" not in _main_call(calls) and "zoom" not in _main_call(calls)


def test_a_new_click_clears_the_previous_results(mocked, map_calls, building_point):
    calls, box = map_calls
    at = _run_with_point(building_point)
    _button(at, "Hesapla").click().run()
    assert "analysis" in at.session_state and "explanation" in at.session_state
    box["value"] = {"last_clicked": {"lat": building_point[0], "lng": building_point[1]}}
    at.run()  # the same point again is not a new click
    assert "analysis" in at.session_state
    box["value"] = {"last_clicked": {"lat": 41.0, "lng": 29.0}}
    at.run()
    assert "analysis" not in at.session_state and "explanation" not in at.session_state
    assert at.session_state["selected_point"] == (41.0, 29.0)


# ----------------------------------------------- recompute from the state: bill, floors, roof type

@pytest.fixture
def spy_gemini(monkeypatch, fake_explain):
    return fake_explain


def _after_hesapla(building_point):
    at = _run_with_point(building_point)
    _button(at, "Hesapla").click().run()
    assert not at.exception
    return at


def _no_pvgis(monkeypatch):
    fail = lambda *a, **k: pytest.fail("PVGIS called on a widget change")
    monkeypatch.setattr(solar, "get_solar_yield", fail)
    monkeypatch.setattr(solar, "get_face_yield", fail)
    monkeypatch.setattr(solar, "fetch_pvcalc", fail)


def test_changing_the_bill_recomputes_savings_without_pvgis_or_gemini_and_marks_the_explanation_stale(
    mocked, monkeypatch, fake_explain, building_point
):
    client = fake_explain([])  # an empty script: any Gemini call fails and falls back to the template
    at = _after_hesapla(building_point)
    calls_after_hesapla = client.calls
    assert at.session_state["result"]["tasarruf_yontemi"] == "genis_tahmin"
    assert any("Ortalama daire faturasını girin" in c.value for c in at.caption)
    before = at.session_state["result"]["daire_basi_tasarruf_tl_aralik"]
    _no_pvgis(monkeypatch)
    _widget(at.number_input, BILL_LABEL).set_value(1500.0).run()
    assert not at.exception
    r = at.session_state["result"]
    assert r["tasarruf_yontemi"] == "fatura" and r["daire_basi_tasarruf_tl_aralik"] != before
    assert r["daire_basi_tasarruf_tl_aralik"][1] <= 12 * 1500 + 0.01
    assert any("Daire başına faturanın yaklaşık" in i.value for i in at.info)
    assert any("her ay için ortalama tüketim" in c.value or "her ayı için ortalama tüketim" in c.value for c in at.caption)
    assert any("Değerler değişti; açıklama güncel değil." in i.value for i in at.info)  # stale, never auto-called
    assert client.calls == calls_after_hesapla


@pytest.mark.parametrize("bad", [0.0, -100.0])
def test_zero_or_negative_bill_counts_as_no_bill(mocked, building_point, bad):
    at = _after_hesapla(building_point)
    _widget(at.number_input, BILL_LABEL).set_value(bad).run()
    assert at.session_state["result"]["tasarruf_yontemi"] == "genis_tahmin"


def test_flat_count_is_prefilled_editable_and_changes_the_per_flat_numbers(mocked, monkeypatch, sample_index, building_point):
    at = _after_hesapla(building_point)
    found = buildings.find_building(sample_index, *building_point)["building"]
    flats = _widget(at.number_input, FLATS_LABEL)
    n0 = flats.value
    assert n0 == app.estimate_flats(found, 150.0) >= 1
    assert any("Tahmini — farklıysa düzeltin" in c.value for c in at.caption)
    _no_pvgis(monkeypatch)
    before = at.session_state["result"]["daire_basi_maliyet_tl_aralik"]
    _widget(at.number_input, FLATS_LABEL).set_value(n0 + 3).run()
    assert not at.exception
    r = at.session_state["result"]
    assert r["daire_sayisi"] == n0 + 3 and r["daire_sayisi_tahmini"] is False
    assert r["daire_basi_maliyet_tl_aralik"][0] < before[0]
    assert not any(c.value == "Tahmini — farklıysa düzeltin" for c in at.caption)  # an edited count is no longer an estimate


def test_roof_structure_and_slider_changes_recompute_with_no_pvgis_call(mocked, monkeypatch, building_point):
    at = _after_hesapla(building_point)
    _no_pvgis(monkeypatch)
    unknown = at.session_state["result"]
    assert len(unknown["senaryolar"]) == 2  # Bilmiyorum: both extremes
    _widget(at.radio, ROOF_LABEL).set_value("Tamamen eğimli").run()
    pitched = at.session_state["result"]
    assert pitched["cati_yapisi"] == "pitched" and pitched["senaryolar"][0]["duz_pay"] == 0.0
    assert not [s for s in at.slider if s.label == SHARE_LABEL]
    _widget(at.radio, ROOF_LABEL).set_value("Karma (teras + eğimli)").run()
    [slider] = [s for s in at.slider if s.label == SHARE_LABEL]
    assert slider.value == 50 and slider.step == 5 and (slider.min, slider.max) == (0, 100)
    slider.set_value(80).run()
    mixed = at.session_state["result"]
    assert mixed["cati_yapisi"] == "mixed" and mixed["duz_pay_yuzde"] == 80
    assert not at.exception


def test_confidence_card_lists_inputs_and_improvements(mocked, building_point):
    at = _after_hesapla(building_point)
    text = " ".join(m.value for m in at.markdown)
    assert "Ölçülen:" in text and "Varsayılan:" in text and "Çatı yapısı: bilinmiyor" in text
    assert "Tahmin edilen:" in text and "Daire sayısı: kat sayısı × taban alanı" in text
    assert any("Çatı yapısını seçin" in c.value for c in at.caption)


# ---------------------------------------------------------- demo photo vs a real building

def test_photo_applies_rule():
    assert app.photo_applies({"id": "x"}, is_demo=True) is False  # bundled demo photo on a real building
    assert app.photo_applies({"id": "x"}, is_demo=False) is True  # user upload is assumed to be that building
    assert app.photo_applies(None, is_demo=True) is True  # manual-area flow: the demo behaves as before
    assert app.photo_applies(None, is_demo=False) is True


def _run_demo_photo(name, point=OUT_OF_COVERAGE):
    at = _run_with_point(point, skip=False)
    at.selectbox[0].select(name)
    at.run()
    return at


def test_demo_photo_is_shown_but_not_applied_to_a_real_building(
    mocked, fake_gemini, roof10_json, spy_capacity, building_point
):
    client = fake_gemini([roof10_json])
    at = _run_demo_photo("roof_10.jpg", building_point)
    _button(at, "Hesapla").click().run()
    assert not at.exception and client.calls == 1
    assert any(app.DEMO_PHOTO_NOTE == i.value for i in at.info)
    call = spy_capacity[-1]
    assert call["obstacle_ratio"] is None and call["image_view"] is None  # no obstacle ratio applied
    assert call["structure"] == "unknown"  # Gemini's pitched is not pre-selected
    assert at.session_state[app.KEY_STRUCTURE] == "Bilmiyorum"
    r = at.session_state["result"]
    assert r["engel_orani"] is None and r["guven_seviyesi"] == "Düşük"  # confidence not raised
    assert at.session_state["analysis"]["detection"] is not None  # the vision result is still displayed
    assert not [m for m in at.metric if m.label == "Engel oranı"]
    assert not [k for k in at.session_state if k.startswith("obstacles_")]  # no table: obstacles do not change the result


def test_uploaded_style_photo_is_applied_to_the_building(sample_index, building_point):
    building = buildings.find_building(sample_index, *building_point)["building"]
    inputs = {**_inputs(), "area": building["area_m2"], "usable_base_area_m2": building["setback_area_m2"],
              "photo_applies": app.photo_applies(building, False)}
    det = _square_detection()
    result, ratio = app.recompute(_bundle(), inputs, det, {0, 1})
    assert ratio == pytest.approx(0.5) and result["engel_orani"] == pytest.approx(0.5)
    assert result["kullanilabilir_alan_m2"] == pytest.approx(building["setback_area_m2"] * 0.5, abs=0.01)
    assert result["guven_seviyesi"] == "Orta"  # polygon + top-down photo + known roof structure
    inputs["photo_applies"] = app.photo_applies(building, True)  # same photo, but a bundled demo
    result, ratio = app.recompute(_bundle(), inputs, det, {0, 1})
    assert ratio is None and result["engel_orani"] is None and result["goruntu_turu"] is None
    assert app.explained_detection(det, {0, 1}, inputs) is None


# ------------------------------------------------------------- texts

def test_height_text_per_source_and_mae_from_the_report():
    b = {"height_m": 24.0}
    rep = {"cv_mae_m": 6.78}
    assert "ortalama hata ±6,78 m" in app.height_text({**b, "height_source": "ml_estimate"}, rep)
    assert "YZ tahmini, LightGBM" in app.height_text({**b, "height_source": "ml_estimate"}, rep)
    no_mae = app.height_text({**b, "height_source": "ml_estimate"}, None)
    assert "YZ tahmini" in no_mae and "ortalama hata" not in no_mae  # report missing: the MAE sentence is omitted
    assert "ortalama hata" not in app.height_text({**b, "height_source": "ml_estimate"}, {})
    assert "varsayım" in app.height_text({**b, "height_source": "neighbour_median"}, rep)
    assert "varsayım" in app.height_text({**b, "height_source": "default"}, rep)
    assert "kat sayısı verisi" in app.height_text({**b, "height_source": "levels_tag"}, rep)
    assert "yükseklik verisi" in app.height_text({**b, "height_source": "height_tag"}, rep)
    assert "girdiğiniz kat sayısı" in app.height_text({**b, "height_source": "user"}, rep)


def test_estimate_flats_helper():
    assert app.estimate_flats(None, 120.0) == 1 and app.estimate_flats(None, 0.0) == 1
    assert app.estimate_flats({"num_floors": 4.0, "height_m": 99.0, "area_m2": 240.0}, 0) == 8
    assert app.estimate_flats({"num_floors": None, "height_m": 15.0, "area_m2": 240.0}, 0) == 10
    assert app.estimate_flats({"num_floors": None, "height_m": 1.0, "area_m2": 20.0}, 0) == 1


def test_clean_bill():
    assert app.clean_bill(1500) == 1500.0
    assert all(app.clean_bill(v) is None for v in (None, 0, -5, float("nan"), float("inf"), "abc", True))


def test_coverage_text_is_per_flat_and_never_single_flat_framing():
    with_bill = app.build_result(_bundle(), {**_inputs(), "bill": 1500.0, "usable_base_area_m2": 250.0})
    text = app.coverage_text(with_bill)
    assert text.startswith("Daire başına faturanın yaklaşık %") and text.endswith("kadarı karşılanır.")
    lo, hi = (round(v) for v in with_bill["fatura_karsilama_yuzde_aralik"])
    span = f"%{lo}" if lo == hi else f"%{lo}–%{hi}"
    assert text == f"Daire başına faturanın yaklaşık {span} kadarı karşılanır."
    assert app.remaining_bill_text(with_bill).startswith("Kalan ortalama aylık daire faturası")
    no_bill = app.build_result(_bundle(), _inputs())
    assert app.coverage_text(no_bill) is None and app.remaining_bill_text(no_bill) is None
    assert not hasattr(app, "shared_roof_text")


def test_no_single_flat_framing_anywhere_in_the_rendered_page(mocked, building_point):
    at = _after_hesapla(building_point)
    _widget(at.number_input, BILL_LABEL).set_value(1500.0).run()
    page = " ".join(
        [i.value for i in at.info] + [w.value for w in at.warning] + [c.value for c in at.caption]
        + [m.value for m in at.markdown] + [m.label for m in at.metric]
    )
    for phrase in ("Faturanızın", "faturanızın", "Tek daire faturanızın tamamını karşılar", "Tek hanenin", "tek hane"):
        assert phrase not in page, phrase
    assert config.TARIFF_SOURCE in page
    assert building_mode.LEGAL_NOTE in [w.value for w in at.warning]  # always visible


def test_tl_card_always_shows_tariff_source_and_method(mocked, building_point):
    at = _after_hesapla(building_point)
    assert any(config.TARIFF_SOURCE in c.value and "vergiler dahil yaklaşık" in c.value for c in at.caption)
    assert any("Geniş tahmin: kademe bilinmiyor" in c.value for c in at.caption)
    _widget(at.number_input, BILL_LABEL).set_value(2000.0).run()
    assert any("Ortalama daire faturasına göre hesaplandı" in c.value for c in at.caption)
    assert config.TARIFF_SOURCE in " ".join(c.value for c in at.caption)


# ------------------------------------------------------------- photo (Demo-2)

from core import vision  # noqa: E402



def _square_detection(obstacles=((0, 0, 50, 50), (50, 50, 100, 100)), roof_type="flat", image_view="top-down"):
    return {
        "roof_type": roof_type, "image_view": image_view, "multiple_buildings": False,
        "width": 100, "height": 100, "roof_source": "polygon",
        "roof_polygon_px": [(0, 0), (100, 0), (100, 100), (0, 100)], "roof_area_px": 10000.0,
        "obstacles": [{"id": i, "box_px": b, "label": "chimney", "confidence": 0.9} for i, b in enumerate(obstacles)],
        "obstacle_ratio": 0.5, "model": "m", "image_hash": "ab" * 32,
    }


def _inputs(structure="flat"):
    return {"lat": 41.0, "lon": 29.0, "area": 100.0, "structure": structure, "flat_share_pct": 50, "n_flats": 4,
            "n_flats_estimated": False, "roof_tag": "Varsayım"}


def test_deselecting_obstacle_lowers_ratio_without_gemini_or_pvgis(monkeypatch):
    monkeypatch.setattr(vision, "_get_client", lambda: pytest.fail("Gemini called"))
    solar_result = _bundle()
    det = _square_detection()
    _, both = app.recompute(solar_result, _inputs(), det, {0, 1})
    _, one = app.recompute(solar_result, _inputs(), det, {0})
    assert both == pytest.approx(0.5) and one == pytest.approx(0.25) and one < both


def test_all_obstacles_deselected_is_zero_ratio_not_no_photo(monkeypatch):
    solar_result = _bundle()
    r, ratio = app.recompute(solar_result, _inputs(), _square_detection(), set())
    assert ratio == 0.0 and r["engel_orani"] == 0.0
    assert r["kullanilabilir_alan_m2"] == pytest.approx(100.0 * 0.85)  # photo, clean roof: 0.85 (not 0.276)
    r_none, ratio_none = app.recompute(solar_result, _inputs(), None, set())
    assert ratio_none is None and r_none["kullanilabilir_alan_m2"] == pytest.approx(100.0 * 0.276)


def test_selected_ids_from_editor_dataframe():
    import pandas as pd

    df = pd.DataFrame({"Dahil": [True, False, True]}, index=[0, 1, 2])
    assert app.selected_ids(df) == {0, 2}


def _state(label, share=50, flats=3, estimate=3, bill=None):
    return {app.KEY_STRUCTURE: label, app.KEY_SHARE: share, app.KEY_FLATS: flats, app.KEY_FLATS_ESTIMATE: estimate, app.KEY_BILL: bill}


@pytest.mark.parametrize(
    "label, preselect, structure, tag",
    [
        ("Tamamen düz", None, "flat", "Kullanıcı girdisi"),
        ("Tamamen eğimli", ("Tamamen eğimli", 50), "pitched", "Ölçüm/veri"),  # unchanged from the photo's answer
        ("Tamamen düz", ("Tamamen eğimli", 50), "flat", "Kullanıcı girdisi"),  # the user overrode it
        ("Karma (teras + eğimli)", None, "mixed", "Kullanıcı girdisi"),
        ("Bilmiyorum", None, "unknown", "Varsayım"),
    ],
)
def test_collect_inputs_maps_the_roof_structure_and_its_tag(label, preselect, structure, tag):
    analysis = {"inputs": {"lat": 1, "lon": 2}, "building": None, "roof_preselect": preselect}
    got = app.collect_inputs(_state(label), analysis)
    assert (got["structure"], got["roof_tag"]) == (structure, tag)
    assert got["n_flats"] == 3 and got["n_flats_estimated"] is True and got["bill"] is None


def test_collect_inputs_clamps_flats_cleans_bill_and_marks_edits():
    analysis = {"inputs": {}, "building": None}
    got = app.collect_inputs(_state("Bilmiyorum", flats=0, estimate=4, bill=-5), analysis)
    assert got["n_flats"] == 1 and got["n_flats_estimated"] is False and got["bill"] is None
    assert app.collect_inputs(_state("Bilmiyorum", bill=900), analysis)["bill"] == 900.0
    override = app.collect_inputs(_state("Bilmiyorum"), analysis, override=("Karma (teras + eğimli)", 70))
    assert override["structure"] == "mixed" and override["flat_share_pct"] == 70


def test_photo_roof_choice_preselects_gemini_answer():
    assert app.photo_roof_choice({"roof_type": "pitched"}) == (("Tamamen eğimli", 50), None)
    assert app.photo_roof_choice({"roof_type": "flat"}) == (("Tamamen düz", 50), None)
    choice, note = app.photo_roof_choice({"roof_type": "mixed"})
    assert choice == ("Karma (teras + eğimli)", 50) and "karma" in note and "%50" in note
    assert "Bilmiyorum" not in {app.photo_roof_choice({"roof_type": t})[0][0] for t in ("flat", "pitched", "mixed")}


def test_parse_sources_and_caption():
    info = app.parse_sources()["roof_10.jpg"]
    assert info["license"] == "Pexels License"
    assert "Pexels License" in app.source_caption("roof_10.jpg")
    assert "kaynak bağlantısı eklenecek" in app.source_caption("roof_10.jpg")  # SOURCES.md link is a TODO
    assert "https://" in app.source_caption("roof_01.jpg")


def test_assumption_rows_tags_and_demo_row():
    solar_result = _bundle()
    inputs = dict(_inputs(), demo_photo=True)
    det = _square_detection()
    r, _ = app.recompute(solar_result, inputs, det, {0})
    df = app.assumption_rows(r, inputs, det)
    assert set(df["Durum"]) <= {"Ölçüm/veri", "Kullanıcı girdisi", "Varsayım", "Sonraki aşamada", "YZ tahmini"}
    assert "Sonraki aşamada" in set(df["Durum"])
    assert any("Demo fotoğrafı örnektir; seçilen binanın kendi çatısı değildir." in n for n in df["Kaynak / not"])
    assert {"Elektrik tarifesi", "Tasarruf aralığı", "Paylaşım", "Daire sayısı", "Kurulum maliyeti", "Geri ödeme"} <= set(df["Değer"])
    assert df.set_index("Değer").loc["Daire sayısı", "Durum"] == "Kullanıcı girdisi"
    assert any(config.TARIFF_SOURCE in n for n in df["Kaynak / not"])
    r2, _ = app.recompute(solar_result, _inputs(), None, set())
    assert not any("Demo fotoğrafı" in n for n in app.assumption_rows(r2, _inputs(), None)["Kaynak / not"])


def test_assumption_rows_for_a_building_include_setback_height_and_ml_note(sample_index, building_point):
    building = buildings.find_building(sample_index, *building_point)["building"]
    inputs = {**_inputs(), "building": building, "usable_base_area_m2": building["setback_area_m2"],
              "area": building["area_m2"], "photo_applies": True}
    r, _ = app.recompute(_bundle(), inputs, None, set())
    df = app.assumption_rows(r, inputs, None, report=sample_index.report)
    by = df.set_index("Değer")
    assert by.loc["Kenar boşluğu", "Durum"] == "Varsayım" and "1,0 m" in by.loc["Kenar boşluğu", "Sonuç"]
    assert by.loc["Çatı alanı", "Durum"] == "Ölçüm/veri"
    assert by.loc["Bina yüksekliği", "Durum"] == app.HEIGHT_SOURCE_TAG[building["height_source"]]
    assert by.loc["Komşu bina gölgesi", "Durum"] == "Sonraki aşamada"


@pytest.fixture
def spy_vision_capacity(spy_capacity):
    return spy_capacity


def test_photo_flow_ratio_reaches_the_analysis_and_preselects_detected_structure(
    mocked, fake_gemini, roof10_json, spy_vision_capacity
):
    client = fake_gemini([roof10_json])
    at = _run_demo_photo("roof_10.jpg")  # no building: the demo photo applies as before
    _button(at, "Hesapla").click().run()
    assert not at.exception
    assert client.calls == 1
    call = spy_vision_capacity[-1]
    assert 0 < call["obstacle_ratio"] < 0.95 and call["structure"] == "pitched"  # roof_10 fixture: pitched, oblique
    assert call["image_view"] == "oblique"
    assert _widget(at.radio, ROOF_LABEL).value == "Tamamen eğimli"  # pre-selected via the pending mechanism
    assert any("Çatı yapısı fotoğraf analizine göre seçildi" in c.value for c in at.caption)
    labels = [m.label for m in at.metric]
    assert "Engel oranı" in labels and len(labels) == 14
    assert any("Açılı fotoğraf" in i.value for i in at.info)
    assert at.dataframe  # assumptions table
    assert at.session_state["result"]["guven_seviyesi"] == "Düşük"
    row = at.dataframe[-1].value
    assert "Ölçüm/veri" in set(row[row["Değer"] == "Çatı yapısı"]["Durum"])
    # a plain rerun must not call Gemini again, and the explanation made after the pre-selection is not stale
    at.run()
    assert client.calls == 1
    assert not any("açıklama güncel değil" in i.value for i in at.info)


def test_mixed_photo_preselects_karma_at_fifty_percent(mocked, fake_gemini, roof10_json):
    fake_gemini([{**roof10_json, "roof_type": "mixed"}])
    at = _run_demo_photo("roof_10.jpg")
    _button(at, "Hesapla").click().run()
    assert not at.exception
    assert _widget(at.radio, ROOF_LABEL).value == "Karma (teras + eğimli)"
    [slider] = [x for x in at.slider if x.label == SHARE_LABEL]
    assert slider.value == 50
    assert any("karma" in i.value for i in at.info)


def test_switching_photos_gives_fresh_editor_keys(mocked, fake_gemini, roof10_json):
    fake_gemini([roof10_json, roof10_json])
    at = _run_demo_photo("roof_10.jpg")
    _button(at, "Hesapla").click().run()
    key_a = next(k for k in at.session_state if k.startswith("obstacles_"))
    at.selectbox[0].select("roof_11.jpg").run()
    _button(at, "Hesapla").click().run()
    keys = [k for k in at.session_state if k.startswith("obstacles_")]
    key_b = [k for k in keys if k != key_a]
    assert key_b, "a different photo must use a different data_editor key"


def test_no_photo_path_still_works_and_lists_assumption(mocked, spy_capacity):
    at = _run_with_point()
    _button(at, "Hesapla").click().run()
    assert not at.exception
    call = spy_capacity[-1]
    assert call["obstacle_ratio"] is None and call["structure"] == "unknown"
    assert len(at.metric) == 13
    assert at.session_state["result"]["guven_seviyesi"] == "Düşük"


def test_vision_error_falls_back_to_no_photo_calculation(mocked, fake_gemini, spy_capacity):
    fake_gemini(["{bad", "{bad"])
    at = _run_demo_photo("roof_10.jpg")
    _button(at, "Hesapla").click().run()
    assert not at.exception
    assert any("okunamadı" in w.value for w in at.warning)
    assert spy_capacity[-1]["obstacle_ratio"] is None
    assert at.session_state["result"]["guven_seviyesi"] == "Düşük"
    assert len(at.metric) == 13


# ------------------------------------------------------------- explanation (Demo-3)

EXPLAIN_BUTTON = "Açıklamayı YZ ile güncelle"


def _flow_text():
    """A valid Gemini-like text for the no-photo flow (template text passes the validator by construction)."""
    flow = app.calculate(41.015, 28.979, 150.0, "unknown", n_flats=1, n_flats_estimated=True)
    return explain.build_fallback_explanation(flow)


def _run_no_photo():
    at = _run_with_point()
    _button(at, "Hesapla").click().run()
    return at


def test_explanation_section_shows_gemini_text_and_source(mocked, fake_explain):
    client = fake_explain([_flow_text()])
    at = _run_no_photo()
    assert not at.exception and client.calls == 1
    assert "Yapay Zekâ Değerlendirmesi" in [h.value for h in at.subheader]
    assert any(m.value.startswith("**") and "kWp" in m.value for m in at.markdown)  # bold verdict
    assert any("statik taşıma" in w.value for w in at.warning)
    assert "Kaynak: Gemini (tüm sayılar doğrulandı)" in [c.value for c in at.caption]
    assert at.session_state["explanation"]["source"] == "gemini"
    at.run()  # a rerun must not call Gemini again
    assert client.calls == 1


def test_template_source_when_gemini_fails(mocked):
    at = _run_no_photo()
    assert not at.exception
    assert "Kaynak: Standart şablon (YZ yanıtı kullanılamadı)" in [c.value for c in at.caption]
    assert any("statik taşıma" in w.value for w in at.warning)
    assert at.session_state["explanation"]["source"] == "template"


def test_no_photo_flow_has_thirteen_cards_and_explanation(mocked):
    at = _run_no_photo()
    assert len(at.metric) == 13 and "explanation" in at.session_state
    _widget(at.number_input, BILL_LABEL).set_value(1500.0).run()
    assert len(at.metric) == 14 and "Faturanın karşılanan kısmı" in [m.label for m in at.metric]


def test_missing_key_skips_vision_and_uses_template(mocked, fake_gemini, monkeypatch):
    def no_key():
        raise config.MissingApiKeyError("Gemini API anahtarı bulunamadı. Test.")

    monkeypatch.setattr(config, "get_gemini_api_key", no_key)
    client = fake_gemini([])
    at = _run_demo_photo("roof_10.jpg")
    _button(at, "Hesapla").click().run()
    assert not at.exception and client.calls == 0
    assert any("anahtarı bulunamadı" in w.value for w in at.warning)
    assert at.session_state["explanation"]["source"] == "template"
    assert len(at.metric) == 13


def test_api_key_is_resolved_in_main_thread_and_passed_to_worker(mocked, monkeypatch, roof10_json):
    seen = []

    def spy(image_bytes, api_key=None):
        seen.append(api_key)
        raise vision.VisionError("Test hatası.")

    monkeypatch.setattr(vision, "detect_roof_objects", spy)
    at = _run_demo_photo("roof_10.jpg")
    _button(at, "Hesapla").click().run()
    assert not at.exception and seen == ["test-key"]
    assert any("Test hatası." in w.value for w in at.warning)


def test_obstacle_edit_makes_explanation_stale_without_gemini_then_regenerate_clears_it(
    mocked, fake_explain, monkeypatch
):
    solar_result = _bundle()
    det = _square_detection()
    inputs = dict(_inputs(), demo_photo=False)
    monkeypatch.setattr(app, "get_gemini_api_key", lambda: "test-key")
    result_all, _ = app.recompute(solar_result, inputs, det, {0, 1})
    client = fake_explain([explain.build_fallback_explanation(result_all, app.explained_detection(det, {0, 1}), False)] * 2)
    exp = app.generate_explanation(result_all, app.explained_detection(det, {0, 1}), False, "k")
    assert exp["source"] == "gemini" and client.calls == 1

    result_one, _ = app.recompute(solar_result, inputs, det, {0})  # obstacle 1 unticked
    det_one = app.explained_detection(det, {0})
    assert not app.explanation_is_stale(exp, result_all, app.explained_detection(det, {0, 1}), False)
    assert app.explanation_is_stale(exp, result_one, det_one, False)
    assert client.calls == 1  # detecting staleness never calls Gemini

    fake_explain([explain.build_fallback_explanation(result_one, det_one, False)])
    new = app.regenerate_explanation(result_one, det_one, False)
    assert new["source"] == "gemini" and not app.explanation_is_stale(new, result_one, det_one, False)


def test_a_new_bill_makes_the_stored_explanation_stale():
    solar_result = _bundle()
    r0, _ = app.recompute(solar_result, _inputs(), None, set())
    r1, _ = app.recompute(solar_result, {**_inputs(), "bill": 1500.0}, None, set())
    exp = {"key": explain.payload_key(explain.build_payload(r0, None, False))}
    assert not app.explanation_is_stale(exp, r0, None, False)
    assert app.explanation_is_stale(exp, r1, None, False)


def test_stale_notice_and_update_button_in_ui(mocked, fake_explain):
    client = fake_explain([_flow_text(), _flow_text()])
    at = _run_no_photo()
    assert client.calls == 1 and not any("açıklama güncel değil" in i.value for i in at.info)
    assert not [b for b in at.button if b.label == EXPLAIN_BUTTON]

    at.session_state["explanation"] = {**at.session_state["explanation"], "key": "old"}
    at.run()
    assert client.calls == 1  # the stale notice never triggers Gemini
    assert any("Değerler değişti; açıklama güncel değil." in i.value for i in at.info)
    assert any("kWp" in m.value for m in at.markdown)  # last explanation is kept
    [button] = [b for b in at.button if b.label == EXPLAIN_BUTTON]
    explain.clear_cache()  # same numbers as the first run would otherwise be an explain cache hit
    button.click().run()
    assert client.calls == 2 and not at.exception
    assert not any("açıklama güncel değil" in i.value for i in at.info)
    assert not [b for b in at.button if b.label == EXPLAIN_BUTTON]


def test_savings_short_format_and_tooltip(mocked):
    assert app.format_savings_short(27515, 55031) == "27,5 – 55,0 bin TL"
    assert app.format_savings_short(500, 900) == "500 – 900 TL"
    at = _run_no_photo()
    card = next(m for m in at.metric if m.label == "Daire başına sabit indirim (yıllık)")
    assert card.value.endswith("bin TL") and "Kesin aralık:" in card.help and card.help.endswith("TL")


def test_assumption_row_shows_active_explanation_source():
    solar_result = _bundle()
    r, _ = app.recompute(solar_result, _inputs(), None, set())
    row = lambda src: app.assumption_rows(r, _inputs(), None, {"source": src})
    gem = row("gemini")
    assert "YZ yorumu (sayılar doğrulandı)" in set(gem["Durum"])
    assert "Şablon" in set(row("template")["Durum"])
    assert "YZ açıklaması" not in set(app.assumption_rows(r, _inputs(), None)["Değer"])


def test_cards_and_explanation_payload_round_identically():
    from core.formatting import percent_from_ratio, tr_num

    r = app.build_result(_bundle(), {**_inputs("unknown"), "bill": 1500.0}, obstacle_ratio=0.2715, image_view="top-down")
    r["yillik_isinim_kwh_m2"] = 1738.5  # a half: f-format would truncate it to 1.738
    payload = explain.build_payload(r, _square_detection(), False)
    for key, decimals in (("yillik_isinim_kwh_m2", 0), ("sicaklik_kaybi_yuzde", 1),
                          ("kullanilabilir_alan_m2", 1), ("uygunluk_puani_yuzde", 1)):
        assert tr_num(r[key], decimals) == tr_num(payload[key], decimals), key
    ranges = (
        ("kurulabilir_guc_kwp_aralik", "kurulabilir_guc_kwp", 1), ("yillik_uretim_kwh_aralik", "yillik_uretim_kwh", 0),
        ("co2_ton_yil_aralik", "co2_ton_yil", 1), ("toplam_maliyet_tl_aralik", "kurulum_maliyeti_tl", 0),
        ("daire_basi_maliyet_tl_aralik", "daire_basi_maliyet_tl", 0), ("daire_basi_tasarruf_tl_aralik", "daire_basi_tasarruf_tl_yil", 0),
        ("fatura_karsilama_yuzde_aralik", "fatura_karsilama_yuzde_aralik", 0), ("kalan_fatura_tl_ay_aralik", "kalan_fatura_tl_ay", 0),
        ("ozgul_verim_bina_aralik", "ozgul_verim_kwh_kwp", 0),
    )
    for result_key, payload_key, decimals in ranges:
        shown = app.format_range(r[result_key], decimals).replace(" ", "")
        assert shown == explain._fmt(payload[payload_key], decimals), result_key
    assert tr_num(r["yillik_isinim_kwh_m2"]) == "1.739"
    assert tr_num(percent_from_ratio(r["engel_orani"]), 1) == tr_num(payload["engel_orani"] * 100, 1)


# --------------------------------------------------- building mode: inputs across clicks, reset, faces

def test_sync_building_inputs_resets_per_building_values_but_never_the_bill():
    state = {app.KEY_BILL: 1500.0}
    b1 = {"id": "b1", "num_floors": 4.0, "height_m": 12.0, "area_m2": 240.0}
    b2 = {"id": "b2", "num_floors": 2.0, "height_m": 6.0, "area_m2": 120.0}
    assert app.sync_building_inputs(state, b1, 150.0) is True
    assert state[app.KEY_FLATS] == state[app.KEY_FLATS_ESTIMATE] == 8
    assert state[app.KEY_STRUCTURE] == "Bilmiyorum" and state[app.KEY_SHARE] == config.FLAT_SHARE_DEFAULT_PCT
    state[app.KEY_FLATS], state[app.KEY_STRUCTURE], state[app.KEY_SHARE] = 12, "Karma (teras + eğimli)", 80
    assert app.sync_building_inputs(state, b1, 150.0) is False  # same building: user edits stay
    assert (state[app.KEY_FLATS], state[app.KEY_SHARE]) == (12, 80)
    assert app.sync_building_inputs(state, b2, 150.0) is True  # a different building id resets
    assert state[app.KEY_FLATS] == 2 and state[app.KEY_STRUCTURE] == "Bilmiyorum"
    assert state[app.KEY_SHARE] == config.FLAT_SHARE_DEFAULT_PCT
    assert state[app.KEY_BILL] == 1500.0  # only the bill survives
    app.sync_building_inputs(state, None, 120.0)  # the manual-area flow is its own "building"
    assert state[app.KEY_BUILDING] == "manual" and state[app.KEY_FLATS] == 1


def test_reset_clears_the_bill_and_forces_the_per_building_reset():
    state = {app.KEY_BILL: 900.0}
    b = {"id": "b1", "num_floors": 3.0, "height_m": 9.0, "area_m2": 240.0}
    app.sync_building_inputs(state, b, 100.0)
    state[app.KEY_FLATS], state[app.KEY_STRUCTURE] = 9, "Tamamen düz"
    app.reset_inputs(state)
    assert state[app.KEY_BILL] is None
    assert app.sync_building_inputs(state, b, 100.0) is True
    assert state[app.KEY_FLATS] == 6 and state[app.KEY_STRUCTURE] == "Bilmiyorum"


def test_pending_photo_preselection_is_applied_once():
    state = {app.KEY_PENDING_ROOF: ("Tamamen eğimli", 50)}
    app.apply_pending_roof(state)
    assert state[app.KEY_STRUCTURE] == "Tamamen eğimli" and app.KEY_PENDING_ROOF not in state
    state[app.KEY_STRUCTURE] = "Tamamen düz"
    app.apply_pending_roof(state)  # nothing pending: the user choice is untouched
    assert state[app.KEY_STRUCTURE] == "Tamamen düz"


def _second_building_point(sample_index):
    assert len(sample_index.polygons_m) >= 2
    p = sample_index.polygons_m[1].representative_point()
    lon, lat = TO_WGS.transform(p.x, p.y)
    return lat, lon


def test_bill_persists_across_clicks_and_per_building_inputs_reset(mocked, map_calls, sample_index, building_point):
    calls, box = map_calls
    at = _run_with_point(building_point)
    _widget(at.number_input, BILL_LABEL).set_value(1500.0).run()
    _widget(at.radio, ROOF_LABEL).set_value("Tamamen düz").run()
    n0 = _widget(at.number_input, FLATS_LABEL).value
    _widget(at.number_input, FLATS_LABEL).set_value(n0 + 5).run()
    _button(at, "Hesapla").click().run()
    assert at.session_state["result"]["fatura_tl_ay"] == 1500.0

    second = _second_building_point(sample_index)
    box["value"] = {"last_clicked": {"lat": second[0], "lng": second[1]}}
    at.run()
    assert not at.exception and "analysis" not in at.session_state  # results are cleared by a new click
    assert not [n for n in at.number_input if n.label == BILL_LABEL]  # the photo step starts again for the new building
    assert at.session_state[app.KEY_BILL_KEEP] == 1500.0
    _button(at, "Fotoğrafsız devam et").click().run()
    assert _widget(at.number_input, BILL_LABEL).value == 1500.0  # the bill survives
    assert at.session_state[app.KEY_BILL] == 1500.0
    assert _widget(at.radio, ROOF_LABEL).value == "Bilmiyorum"  # roof structure reset
    found = buildings.find_building(sample_index, *second)["building"]
    assert _widget(at.number_input, FLATS_LABEL).value == app.estimate_flats(found, 150.0)  # re-estimated

    _button(at, "Hesapla").click().run()  # the new building is calculated with the persisted bill
    assert at.session_state["result"]["fatura_tl_ay"] == 1500.0 and at.session_state["result"]["daire_sayisi_tahmini"]


def test_clicking_the_same_building_again_keeps_the_edited_inputs(mocked, map_calls, building_point, sample_index):
    calls, box = map_calls
    at = _run_with_point(building_point)
    _widget(at.radio, ROOF_LABEL).set_value("Tamamen eğimli").run()
    lon, lat = TO_WGS.transform(*sample_index.polygons_m[0].centroid.coords[0])
    box["value"] = {"last_clicked": {"lat": lat, "lng": lon}}
    at.run()
    assert at.session_state[app.KEY_STRUCTURE] == "Tamamen eğimli"  # same building id: no reset


def test_sifirla_button_clears_the_inputs_including_the_bill(mocked, building_point):
    at = _run_with_point(building_point)
    _widget(at.number_input, BILL_LABEL).set_value(1500.0).run()
    _widget(at.radio, ROOF_LABEL).set_value("Tamamen düz").run()
    n0 = _widget(at.number_input, FLATS_LABEL).value
    _widget(at.number_input, FLATS_LABEL).set_value(n0 + 4).run()
    [reset] = [b for b in at.sidebar.button if b.label == "Sıfırla"]
    reset.click().run()
    assert not at.exception
    assert at.session_state[app.KEY_BILL_KEEP] is None
    _button(at, "Fotoğrafsız devam et").click().run()  # a reset also restarts the photo step
    assert _widget(at.number_input, BILL_LABEL).value is None
    assert _widget(at.radio, ROOF_LABEL).value == "Bilmiyorum"
    assert _widget(at.number_input, FLATS_LABEL).value == n0
    assert at.session_state[app.KEY_BILL] is None


def test_sidebar_inputs_are_rendered_before_any_calculation(mocked, building_point):
    at = _run_with_point(building_point)
    labels = [n.label for n in at.number_input]
    assert FLATS_LABEL in labels and BILL_LABEL in labels and ROOF_LABEL in [r.label for r in at.radio]
    assert not [s for s in at.slider if s.label == SHARE_LABEL]  # only Karma shows the slider
    assert any(c.value == "Tahmini — farklıysa düzeltin" for c in at.caption)


def test_a_failed_pitched_face_still_produces_a_result_with_a_warning(mocked, monkeypatch, building_point):
    def flaky(lat, lon, aspect, angle=30, horizon72=None):
        if abs(aspect) > 135:  # the north-ish face fails, the south-ish one works
            raise solar.SolarDataError("PVGIS yok")
        return _fake_face(lat, lon, aspect, angle)

    monkeypatch.setattr(solar, "get_face_yield", flaky)
    at = _run_with_point(building_point)
    _button(at, "Hesapla").click().run()
    assert not at.exception and not at.error
    _widget(at.radio, ROOF_LABEL).set_value("Tamamen eğimli").run()
    assert any("Bir çatı yüzü için güneş verisi alınamadı; sonuç eksik olabilir" in w.value for w in at.warning)
    r = at.session_state["result"]
    assert r["senaryolar"][0]["alinamayan_yuz"] == 1 and r["senaryolar"][0]["kwp"] > 0
    assert r["guven_seviyesi"] == "Düşük"  # the unavailable face drops one level


def test_a_failed_flat_call_keeps_the_existing_error_behaviour(mocked, monkeypatch, building_point):
    def boom(lat, lon, horizon72=None):
        raise solar.SolarDataError("PVGIS sunucusuna ulaşılamadı.")

    monkeypatch.setattr(solar, "get_solar_yield", boom)
    at = _run_with_point(building_point)
    _button(at, "Hesapla").click().run()
    assert any("PVGIS sunucusuna ulaşılamadı" in e.value for e in at.error)
    assert "analysis" not in at.session_state


def test_hesapla_makes_four_unique_face_calls_and_structure_or_subtype_changes_make_none(
    mocked, monkeypatch, sample_index, building_point
):
    seen = []

    def spy(lat, lon, aspect, angle=30, horizon72=None):
        seen.append(aspect)
        return _fake_face(lat, lon, aspect, angle)

    monkeypatch.setattr(solar, "get_face_yield", spy)
    at = _run_with_point(building_point)
    _button(at, "Hesapla").click().run()
    found = buildings.find_building(sample_index, *building_point)["building"]
    footprint = shapely.from_wkb(found["footprint_wkb_m"])
    expected = sorted({f["aspect"] for f in layout.roof_faces(footprint, "hip")})
    assert sorted(seen) == expected and len(seen) == 4  # gable aspects are the hip long-side aspects
    n = len(seen)
    _widget(at.radio, ROOF_LABEL).set_value("Tamamen eğimli").run()
    assert len(seen) == n  # a roof-structure change never calls PVGIS again
    _widget(at.radio, PITCHED_LABEL).set_value("Beşik (2 yüz)").run()
    assert len(seen) == n and at.session_state["result"]["egimli_tip"] == "gable"  # nor does a sub-type change


def test_discount_notes_are_shown_under_the_savings_card(mocked, building_point):
    at = _after_hesapla(building_point)
    captions = [c.value for c in at.caption]
    assert building_mode.SEASONAL_NOTE in captions and building_mode.PHYSICAL_SHARING_NOTE in captions
    assert any(c.startswith("Aylık ortalama:") and "TL/ay" in c for c in captions)
    assert "Daire başına sabit indirim (yıllık)" in [m.label for m in at.metric]


def test_netting_note_is_shown_with_a_bill_and_not_without(mocked, building_point):
    at = _after_hesapla(building_point)
    assert building_mode.NETTING_NOTE not in [c.value for c in at.caption]
    _widget(at.number_input, BILL_LABEL).set_value(1500.0).run()
    assert building_mode.NETTING_NOTE in [c.value for c in at.caption]


# ------------------------------------------------- live-test feedback: ranges, notices

def test_ranges_collapse_to_one_value_when_equal_after_rounding():
    assert app.format_savings_short(6190, 6210) == "6,2 bin TL"
    assert app.format_savings_short(500.2, 500.4) == "500 TL"
    assert app.format_savings_short(6000, 7000) == "6,0 – 7,0 bin TL"
    assert app.format_range([21.04, 21.0], 1, "kWp") == "21,0 kWp"
    assert app.format_range([5.0, 6.0], 1, "kWp") == "5,0 – 6,0 kWp"
    assert app.payback_text([7.04, 7.0]) == "7,0 yıl"
    assert app.payback_text([None, None]) is None


def test_out_of_range_cost_flag_and_demo_notice_texts(mocked, building_point):
    assert app.DEMO_PHOTO_NOTE.startswith("Bu, uygulamayla birlikte gelen bir örnek fotoğraftır (Pexels / Roboflow")
    assert "hesaba katılmaz. Kendi çatınızın fotoğrafını yukarıdaki 'Çatı fotoğrafı yükleyin' adımından" in app.DEMO_PHOTO_NOTE
    assert app.DEMO_EXPANDER == "Örnek fotoğrafla dene (stok fotoğraf, hesaba katılmaz)"
    at = _run_with_point(building_point, skip=False)
    assert [e.label for e in at.expander] == [app.DEMO_EXPANDER]


def test_large_system_shows_the_labelled_cost_warning(mocked, monkeypatch, building_point):
    real = building_mode.analyze_building

    def big(**kwargs):
        return real(**{**kwargs, "usable_base_area_m2": 800.0, "roof_area_m2": 900.0})

    monkeypatch.setattr(building_mode, "analyze_building", big)
    at = _after_hesapla(building_point)
    assert any(
        w.value == "Maliyet: sistem 20 kWp'nin üzerinde; fiyat bandı kaynak aralığı dışında, tahmin daha belirsiz."
        for w in at.warning
    )
    assert not any(w.value == "Kaynak aralığı dışında, tahmin daha belirsiz." for w in at.warning)


def test_explanation_text_renders_verdict_paragraph_tips_and_warning(mocked):
    at = _run_no_photo()
    text = at.session_state["explanation"]["text"]
    assert len(text.splitlines()) <= 6
    assert any(m.value.startswith("**") for m in at.markdown)
    assert any("statik taşıma" in w.value for w in at.warning)
    assert not any("Bina toplamı yılda yaklaşık" in w.value for w in at.warning)  # the paragraph is plain text, not a warning


# ------------------------------------------------------------ roof view, layout, hip roofs

def _roof_inputs(sample_index, building_point, structure="pitched", pitched="hip", **extra):
    found = buildings.find_building(sample_index, *building_point)["building"]
    return {
        "lat": building_point[0], "lon": building_point[1], "area": found["area_m2"],
        "usable_base_area_m2": found["setback_area_m2"], "footprint_wkb_m": found["footprint_wkb_m"],
        "structure": structure, "flat_share_pct": 50, "n_flats": 4, "n_flats_estimated": False, "bill": None,
        "roof_tag": "Kullanıcı girdisi", "pitched_type": pitched, "photo_applies": True, "building": found, **extra,
    }


def _roof_analysis(inputs, north_factor=0.55):
    yields = {}
    for a in app.pvgis_aspects(inputs):
        face = _fake_face(inputs["lat"], inputs["lon"], a)
        yields[a] = face
    return {"solar": _fake_yield(inputs["lat"], inputs["lon"]), "face_yields": yields}


def _feature(poly_m):
    xs, ys = poly_m.exterior.xy
    lons, lats = drawings.to_wgs84(list(xs), list(ys))
    return {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[lo, la] for lo, la in zip(lons, lats)]]}}


def _draw_state(inputs, obstacles=(), terrace=None, empty=False):
    footprint = shapely.from_wkb(inputs["footprint_wkb_m"])
    state = drawings.new_state(inputs["building"]["id"])
    if obstacles:
        drawings.sync(state, [_feature(o) for o in obstacles], drawings.OBSTACLE, footprint)
    if terrace is not None:
        drawings.sync(state, [_feature(o) for o in obstacles] + [_feature(terrace)], drawings.TERRACE, footprint)
    if empty:
        drawings.confirm_empty(state)
    return state


def _centre_obstacle(inputs, size=2.0):
    c = shapely.from_wkb(inputs["footprint_wkb_m"]).centroid
    return shapely.box(c.x - size / 2, c.y - size / 2, c.x + size / 2, c.y + size / 2)


def test_without_drawings_the_numeric_kwp_stays_and_exactly_that_many_panels_are_drawn(mocked, sample_index, building_point):
    inputs = _roof_inputs(sample_index, building_point)
    r, _, roof = app.evaluate(_roof_analysis(inputs), inputs, None, set(), _draw_state(inputs))
    assert r["guc_kaynagi"] == "ortalama" and roof["mode"] == "numeric" and app.NOTE_AVERAGE in roof["notes"]
    expected = sum(int(s["kwp"] / config.PANEL_KWP + 1e-9) for s in r["senaryolar"][0]["segmentler"] if s["durum"] == "kullanildi")
    assert app.drawn_panel_count(roof) == expected == r["panel_sayisi_aralik"][0]
    assert r["senaryolar"][0]["kwp"] != pytest.approx(expected * config.PANEL_KWP)  # numeric kWp is not n x 0.40


def test_drawn_obstacles_switch_to_geometry_mode_with_panel_count_times_0_40(mocked, sample_index, building_point):
    inputs = _roof_inputs(sample_index, building_point)
    analysis = _roof_analysis(inputs)
    draw = _draw_state(inputs, [_centre_obstacle(inputs)])
    r, _, roof = app.evaluate(analysis, inputs, None, set(), draw)
    sc = r["senaryolar"][0]
    assert r["guc_kaynagi"] == "geometri" and roof["mode"] == "geometry" and roof["limits"] is None
    assert sc["kwp"] == pytest.approx(sc["panel_sayisi"] * config.PANEL_KWP)
    assert app.drawn_panel_count(roof) == sc["panel_sayisi"] > 0
    # kWh from the stored specific yields, cost from the cost tool: no new PVGIS call
    kwh = sum(seg["kwp"] * sum(analysis["face_yields"][seg["aspect"]]["e_m"]) for seg in sc["segmentler"] if seg["durum"] == "kullanildi")
    assert sc["kwh"] == pytest.approx(kwh)
    assert sc["maliyet_tl"] == pytest.approx(list(building_mode.cost.estimate_install_cost(sc["kwp"])[:2]))
    assert "Engeller kullanıcı tarafından işaretlendi." in r["guven_nedenleri"]
    assert r["guven_seviyesi"] == "Orta"  # never above Orta (neighbour shading still missing)
    assert r["cizilen_engel_sayisi"] == 1 and r["cizilen_engel_alani_m2"] == pytest.approx(4.0, rel=1e-3)
    empty, _, _ = app.evaluate(analysis, inputs, None, set(), _draw_state(inputs, empty=True))
    assert empty["guc_kaynagi"] == "geometri" and empty["senaryolar"][0]["panel_sayisi"] > sc["panel_sayisi"]


def test_drawn_panels_never_overlap_obstacles_and_dropped_faces_stay_empty(mocked, sample_index, building_point):
    inputs = _roof_inputs(sample_index, building_point)
    obstacle = _centre_obstacle(inputs, 3.0)
    r, _, roof = app.evaluate(_roof_analysis(inputs), inputs, None, set(), _draw_state(inputs, [obstacle]))
    panels = shapely.polygons(next(p for p in app.drawn_panels(roof).values() if len(p)))
    assert (shapely.area(shapely.intersection(panels, obstacle.buffer(0.5))) <= 1e-6).all()
    dropped = {s["ad"] for s in r["senaryolar"][0]["segmentler"] if s["durum"] == "elendi"}
    assert dropped and dropped == set(roof["skip"]) and not dropped & set(roof["layouts"])


def test_unknown_structure_in_geometry_mode_gives_a_range_and_draws_the_flat_layout(mocked, sample_index, building_point):
    inputs = _roof_inputs(sample_index, building_point, structure="unknown")
    r, _, roof = app.evaluate(_roof_analysis(inputs), inputs, None, set(), _draw_state(inputs, [_centre_obstacle(inputs)]))
    kwps = [sc["panel_sayisi"] * config.PANEL_KWP for sc in r["senaryolar"]]
    assert len(kwps) == 2 and kwps[0] != pytest.approx(kwps[1])
    assert r["kurulabilir_guc_kwp_aralik"] == pytest.approx([min(kwps), max(kwps)])
    assert r["panel_sayisi_aralik"] == sorted(sc["panel_sayisi"] for sc in r["senaryolar"])
    assert r["yillik_uretim_kwh_aralik"] == pytest.approx(sorted(sc["kwh"] for sc in r["senaryolar"]))
    assert app.NOTE_FLAT_ASSUMED_GEOMETRY in roof["notes"] and list(roof["layouts"]) == [layout.FLAT_SEGMENT]


def test_karma_with_obstacles_but_no_terrace_stays_numeric_with_a_hint(mocked, sample_index, building_point):
    inputs = _roof_inputs(sample_index, building_point, structure="mixed")
    analysis = _roof_analysis(inputs)
    plain, _, _ = app.evaluate(analysis, inputs, None, set(), _draw_state(inputs))
    r, _, roof = app.evaluate(analysis, inputs, None, set(), _draw_state(inputs, [_centre_obstacle(inputs)]))
    assert r["guc_kaynagi"] == "ortalama" and r["kurulabilir_guc_kwp_aralik"] == pytest.approx(plain["kurulabilir_guc_kwp_aralik"])
    assert layout.MSG_KARMA_NEEDS_TERRACE in roof["notes"] and layout.MSG_NO_TERRACE_DRAWN in roof["notes"]
    assert "Engeller kullanıcı tarafından işaretlendi." not in r["guven_nedenleri"]  # the drawing is not used yet
    assert layout.FLAT_SEGMENT not in roof["layouts"]  # only the pitched faces are drawn


def test_karma_empty_roof_confirmed_without_terrace_is_numeric_with_ratio_zero(mocked, sample_index, building_point):
    inputs = _roof_inputs(sample_index, building_point, structure="mixed")
    r, ratio, roof = app.evaluate(_roof_analysis(inputs), inputs, None, set(), _draw_state(inputs, empty=True))
    assert ratio == 0.0 and r["guc_kaynagi"] == "engel_yok" and r["engel_orani"] == 0.0
    assert r["kullanilabilir_alan_m2"] == pytest.approx(inputs["usable_base_area_m2"], abs=0.01)  # photo rule, ratio 0
    assert roof["mode"] == "numeric_clean" and layout.MSG_KARMA_NEEDS_TERRACE in roof["notes"]


def test_karma_with_a_terrace_uses_geometry_for_both_parts(mocked, sample_index, building_point):
    inputs = _roof_inputs(sample_index, building_point, structure="mixed")
    fp = shapely.from_wkb(inputs["footprint_wkb_m"])
    minx, miny, maxx, maxy = fp.bounds
    terrace = shapely.box(minx, miny, (minx + maxx) / 2, maxy).intersection(fp)
    r, _, roof = app.evaluate(_roof_analysis(inputs), inputs, None, set(), _draw_state(inputs, [_centre_obstacle(inputs)], terrace))
    segs = {s["ad"]: s for s in r["senaryolar"][0]["segmentler"]}
    assert r["guc_kaynagi"] == "geometri" and r["teras_cizildi"] and segs[layout.FLAT_SEGMENT]["panel_sayisi"] > 0
    assert any(s["tur"] == "face" and s["panel_sayisi"] > 0 for s in segs.values())
    assert r["duz_pay_yuzde"] == round(100 * segs[layout.FLAT_SEGMENT]["plan_m2"] / r["kullanilabilir_alan_m2"])


def test_gable_and_hip_give_different_kwp_for_the_same_building(mocked, sample_index, building_point):
    hip = _roof_inputs(sample_index, building_point, pitched="hip")
    gable = _roof_inputs(sample_index, building_point, pitched="gable")
    analysis = _roof_analysis(hip)
    r_hip, _, _ = app.evaluate(analysis, hip, None, set(), _draw_state(hip))
    r_gable, _, _ = app.evaluate(analysis, gable, None, set(), _draw_state(gable))
    assert len(r_hip["senaryolar"][0]["segmentler"]) == 5 and len(r_gable["senaryolar"][0]["segmentler"]) == 3
    assert r_hip["kurulabilir_guc_kwp_aralik"][0] != pytest.approx(r_gable["kurulabilir_guc_kwp_aralik"][0])


def test_pvgis_links_are_the_exact_urls_of_the_calls(mocked, sample_index, building_point):
    inputs = _roof_inputs(sample_index, building_point)
    r, _, _ = app.evaluate(_roof_analysis(inputs), inputs, None, set(), _draw_state(inputs))
    links = dict(report.pvgis_links(r))
    assert len(links) == 2 + 4
    face = next(url for label, url in links.items() if label.endswith("yüzü"))
    assert "optimalangles=0" in face and "angle=30" in face and "usehorizon=1" in face and "outputformat=json" in face


def _roof_call(calls):
    return [c for c in calls if str(c.get("key", "")).startswith("roof_view")][-1]


def test_roof_map_is_a_separate_drawing_only_map(mocked, map_calls, building_point):
    calls, _ = map_calls
    _after_hesapla(building_point)
    roof = _roof_call(calls)
    assert roof["key"] == "roof_view_0" and roof["returned_objects"] == ["all_drawings"]
    main = [c for c in calls if c.get("key") == "map"][-1]
    assert main["returned_objects"] == ["last_clicked"]


def test_drawing_or_clicking_on_the_roof_map_never_changes_the_selected_building(mocked, map_calls, sample_index, building_point):
    calls, box = map_calls
    at = _after_hesapla(building_point)
    flats = _widget(at.number_input, FLATS_LABEL).value
    inputs = _roof_inputs(sample_index, building_point)
    # the roof map returns a drawing AND (if it ever leaked) a click somewhere else
    box["roof"] = {"all_drawings": [_feature(_centre_obstacle(inputs))], "last_clicked": {"lat": 41.0, "lng": 29.0}}
    at.run()
    assert not at.exception
    assert at.session_state["selected_point"] == building_point and "analysis" in at.session_state
    assert _widget(at.number_input, FLATS_LABEL).value == flats
    r = at.session_state["result"]
    assert r["guc_kaynagi"] == "geometri" and r["cizilen_engel_sayisi"] == 1
    at.run()  # the same returned set again: no duplicate, no rerun loop
    assert at.session_state["result"]["cizilen_engel_sayisi"] == 1
    assert at.session_state[app.KEY_DRAWINGS]["gen"] == 0


def test_undo_and_clear_buttons_update_the_drawings_and_rebuild_the_map(mocked, map_calls, sample_index, building_point):
    calls, box = map_calls
    at = _after_hesapla(building_point)
    inputs = _roof_inputs(sample_index, building_point)
    box["roof"] = {"all_drawings": [_feature(_centre_obstacle(inputs))]}
    at.run()
    assert at.session_state["result"]["guc_kaynagi"] == "geometri"
    box["roof"] = {"all_drawings": None}  # a rebuilt map returns nothing yet
    _button(at, "Son çizimi geri al").click().run()
    assert not at.exception and at.session_state[app.KEY_DRAWINGS]["obstacles"] == []
    assert _roof_call(calls)["key"] == "roof_view_1" and at.session_state["result"]["guc_kaynagi"] == "ortalama"
    _button(at, "Engel yok, çatı boş").click().run()
    assert at.session_state["result"]["guc_kaynagi"] == "geometri"
    _button(at, "Engelleri temizle").click().run()
    assert at.session_state["result"]["guc_kaynagi"] == "ortalama" and _roof_call(calls)["key"] == "roof_view_2"


def test_drawings_reset_for_a_new_building(mocked, map_calls, sample_index, building_point):
    calls, box = map_calls
    at = _after_hesapla(building_point)
    box["roof"] = {"all_drawings": [_feature(_centre_obstacle(_roof_inputs(sample_index, building_point)))]}
    at.run()
    assert at.session_state[app.KEY_DRAWINGS]["obstacles"]
    second = _second_building_point(sample_index)
    box["roof"] = {"all_drawings": None}
    box["value"] = {"last_clicked": {"lat": second[0], "lng": second[1]}}
    at.run()
    assert at.session_state[app.KEY_DRAWINGS]["obstacles"] == []


def test_pitched_subtype_selector_defaults_to_hip_and_shows_only_for_pitched_or_karma(mocked, building_point):
    at = _run_with_point(building_point)
    assert not [r for r in at.radio if r.label == PITCHED_LABEL]
    _widget(at.radio, ROOF_LABEL).set_value("Karma (teras + eğimli)").run()
    assert _widget(at.radio, PITCHED_LABEL).value == "Kırma (4 yüz)"
    _widget(at.radio, PITCHED_LABEL).set_value("Beşik (2 yüz)").run()
    _widget(at.radio, ROOF_LABEL).set_value("Bilmiyorum").run()  # hidden: the choice is kept for the extremes
    _button(at, "Hesapla").click().run()
    assert at.session_state["result"]["egimli_tip"] == "gable"


def test_trust_features_render(mocked, building_point):
    at = _after_hesapla(building_point)
    assert "Hesap dökümü" in [e.label for e in at.expander]
    assert any(c.value == report.PVGIS_CAPTION for c in at.caption)
    assert any(c.value.startswith(app.ROOF_VIEW_CAPTION) and "Esri" in c.value for c in at.caption)
    assert at.get("download_button")


def test_small_fixes_help_list_bill_minimum_and_high_share_note(mocked, monkeypatch, building_point):
    at = _after_hesapla(building_point)
    help_md = [m.value for m in at.sidebar.markdown if m.value.startswith("1. ")]
    assert help_md and help_md[0].count("\n") >= 3 and "\n2. " in help_md[0]
    assert _widget(at.number_input, BILL_LABEL).min == 0
    _widget(at.number_input, FLATS_LABEL).set_value(1).run()  # one flat gets the whole production
    r = at.session_state["result"]
    assert r["daire_payi_yuksek"] and r["daire_payi_kwh_ay_aralik"][1] > 240
    note = building_mode.high_share_note(r)
    assert note.startswith("Daire payı (≈") and note in [c.value for c in at.caption]
    _widget(at.number_input, BILL_LABEL).set_value(1500.0).run()
    assert not at.session_state["result"]["daire_payi_yuksek"]


# ------------------------------------------- pitched-face fix: zero result, labels, texts, roof map

def test_zero_capacity_replaces_only_the_card_block_and_keeps_the_rest(mocked, monkeypatch, building_point):
    def weak(lat, lon, aspect, angle=30, horizon72=None):
        face = _fake_face(lat, lon, aspect, angle)
        return {**face, "e_y": face["e_y"] * 0.5, "e_m": [m * 0.5 for m in face["e_m"]]}  # every face < 80 %

    monkeypatch.setattr(solar, "get_face_yield", weak)
    at = _run_with_point(building_point)
    _widget(at.radio, ROOF_LABEL).set_value("Tamamen eğimli").run()
    _widget(at.radio, PITCHED_LABEL).set_value("Beşik (2 yüz)").run()
    _button(at, "Hesapla").click().run()
    assert not at.exception
    assert app.ZERO_PITCHED_WARNING in [w.value for w in at.warning]
    labels = [m.label for m in at.metric]
    for gone in ("Kurulabilir güç", "Toplam kurulum maliyeti", "Tek seferlik maliyet", "Daire başına sabit indirim (yıllık)"):
        assert gone not in labels
    assert not any(w.value == "Kurulabilir panel yok." for w in at.warning)
    assert "Güven seviyesi" in labels  # the rest of the page is still there
    assert any(m.value == "**Çatı yüzleri**" for m in at.markdown)
    assert any("yüzü: elendi" in m.value for m in at.markdown)
    assert any(s.value == "Çatı görünümü" for s in at.subheader)
    assert "Hesap dökümü" in [e.label for e in at.expander] and at.get("link_button")
    assert any(i.value.startswith("Üretim yapan çatı bölümü yok") for i in at.info)


def test_face_status_list_is_shown_for_every_face(mocked, building_point):
    at = _run_with_point(building_point)
    _widget(at.radio, ROOF_LABEL).set_value("Tamamen eğimli").run()
    _button(at, "Hesapla").click().run()
    r = at.session_state["result"]
    status = next(m.value for m in at.markdown if m.value.startswith("- ") and "yüzü:" in m.value)
    for seg in r["senaryolar"][0]["segmentler"]:
        if seg["tur"] == "face":
            assert f"- {seg['ad']}: " in status


def test_obstacle_line_follows_the_drawing_or_the_no_obstacle_statement(mocked, map_calls, sample_index, building_point):
    calls, box = map_calls
    at = _after_hesapla(building_point)
    texts = lambda: " ".join(m.value for m in at.markdown)
    assert "Engeller: fotoğraf yok, ortalama kullanılabilir alan oranı" in texts()
    box["roof"] = {"all_drawings": [_feature(_centre_obstacle(_roof_inputs(sample_index, building_point)))]}
    at.run()
    assert "Engeller: kullanıcı çizimi" in texts() and "ortalama kullanılabilir alan oranı" not in texts()
    box["roof"] = {"all_drawings": None}
    _button(at, "Engel yok, çatı boş").click().run()
    assert "Engel yok (kullanıcı beyanı)" in texts()


def test_roof_label_numeric_mode_has_no_panel_count_geometry_mode_has_n_times_0_40(mocked, sample_index, building_point):
    inputs = _roof_inputs(sample_index, building_point)
    analysis = _roof_analysis(inputs)
    r, _, roof = app.evaluate(analysis, inputs, None, set(), _draw_state(inputs))
    label = app.roof_label(roof, r)
    assert label.startswith(f"≈{tr(r['senaryolar'][0]['kwp'], 1)} kWp (ortalama orana göre; paneller temsilidir)")
    assert " panel ·" not in label
    r, _, roof = app.evaluate(analysis, inputs, None, set(), _draw_state(inputs, [_centre_obstacle(inputs)]))
    n = app.drawn_panel_count(roof)
    assert app.roof_label(roof, r).startswith(f"{n} panel · {tr(n * config.PANEL_KWP, 1)} kWp")


def test_roof_map_has_no_scroll_wheel_zoom_and_explains_the_trash_button(mocked, building_point):
    m = app.build_roof_map([41.0, 29.0])
    assert m.options["scroll_wheel_zoom"] is False
    assert "scrollWheelZoom" in m.get_root().render()
    at = _after_hesapla(building_point)
    assert app.TRASH_NOTE in [c.value for c in at.caption]
    assert app.TRASH_NOTE.startswith("Haritadaki çöp kutusu yalnızca son yeniden çizimden sonraki şekilleri siler")


# ------------------------------------------------------------ imagery alignment offset (display only)

def test_offset_round_trip_maps_a_displayed_shape_back_to_the_footprint_frame(sample_index, building_point):
    inputs = _roof_inputs(sample_index, building_point)
    footprint = shapely.from_wkb(inputs["footprint_wkb_m"])
    poly = _centre_obstacle(inputs)
    state = drawings.new_state(inputs["building"]["id"])
    drawings.move_offset(state, 3.0, -5.0)
    displayed = _feature(shapely.affinity.translate(poly, 3.0, -5.0))  # what the shifted map returns
    drawings.sync(state, [displayed], drawings.OBSTACLE, footprint)
    stored = state["obstacles"][0]["raw"]
    assert stored.centroid.distance(poly.centroid) < 0.01 and abs(stored.area - poly.area) < 0.01
    # the overlay goes out by +offset: rendered back at the displayed position
    ring = drawings.polygon_to_latlon(stored, state["offset"])[0]
    xs, ys = drawings.to_metric([p[1] for p in ring], [p[0] for p in ring])
    assert abs(sum(xs[:-1]) / (len(xs) - 1) - (poly.centroid.x + 3.0)) < 0.01


def test_offset_is_clamped_and_described(sample_index, building_point):
    state = drawings.new_state("b")
    for _ in range(40):
        drawings.move_offset(state, 1.0, 0.0)
    drawings.move_offset(state, 0.0, 5.0)
    assert state["offset"] == (30.0, 5.0) and drawings.offset_text(state["offset"]) == "Kaydırma: 30 m D, 5 m K"
    drawings.reset_offset(state)
    assert state["offset"] == (0.0, 0.0) and drawings.offset_text(state["offset"]) == "Kaydırma: yok"


def test_offset_change_keeps_results_and_drawings_and_rebuilds_the_map(mocked, map_calls, sample_index, building_point):
    calls, box = map_calls
    at = _after_hesapla(building_point)
    inputs = _roof_inputs(sample_index, building_point)
    box["roof"] = {"all_drawings": [_feature(_centre_obstacle(inputs))]}
    at.run()
    before = at.session_state["result"]
    ids = [s["id"] for s in at.session_state[app.KEY_DRAWINGS]["obstacles"]]
    gen = at.session_state[app.KEY_DRAWINGS]["gen"]
    box["roof"] = {"all_drawings": None}  # the rebuilt map returns nothing yet
    _button(at, "→").click().run()
    assert not at.exception
    draw = at.session_state[app.KEY_DRAWINGS]
    assert draw["offset"] == (1.0, 0.0) and draw["gen"] == gen + 1 and _roof_call(calls)["key"] == f"roof_view_{gen + 1}"
    assert [s["id"] for s in draw["obstacles"]] == ids  # neither deleted nor re-added
    after = at.session_state["result"]
    assert after["kurulabilir_guc_kwp_aralik"] == before["kurulabilir_guc_kwp_aralik"]
    assert after["panel_sayisi_aralik"] == before["panel_sayisi_aralik"]
    # the map now returns the same shape in the displayed frame: still the same single obstacle
    box["roof"] = {"all_drawings": [_feature(shapely.affinity.translate(_centre_obstacle(inputs), 1.0, 0.0))]}
    at.run()
    assert [s["id"] for s in at.session_state[app.KEY_DRAWINGS]["obstacles"]] == ids
    assert at.session_state[app.KEY_DRAWINGS]["gen"] == gen + 1
    assert any(c.value == app.ALIGN_CAPTION for c in at.caption)


def test_offset_resets_for_a_new_building(sample_index, building_point):
    state = {}
    first = _roof_inputs(sample_index, building_point)["building"]
    draw = app.drawing_state(state | {app.KEY_DRAWINGS: None}, first)
    draw["offset"] = (4.0, 2.0)
    other = {"id": "another"}
    state2 = {app.KEY_DRAWINGS: draw}
    assert app.drawing_state(state2, other)["offset"] == (0.0, 0.0)


def test_report_mentions_a_non_zero_offset_only(sample_index):
    assert report.alignment_row((0.0, 0.0)) is None and report.alignment_row(None) is None
    row = report.alignment_row((3.0, -2.0))
    assert row["Hesap"] == "3 m doğu, 2 m güney (yalnızca gösterim)"


def test_segment_table_and_chart_use_turkish_number_format(sample_index, building_point, mocked):
    from core.formatting import tr_num
    inputs = _roof_inputs(sample_index, building_point, structure="flat")
    r, _, _ = app.evaluate(_roof_analysis(inputs), inputs, None, set(), _draw_state(inputs))
    table = app.segment_table(r)
    plan = table["Plan alanı (m²)"].tolist()
    assert all("," in v and "." not in v.split(",")[-1] for v in plan)  # 83,8 and never 83.8
    assert tr_num(r["kullanilabilir_alan_m2"], 1) in plan  # the same rounding as the caption (no 83,9 vs 83,8)
    assert app.TR_AXIS_LABEL_EXPR == "replace(format(datum.value, ',.0f'), ',', '.')"


# ------------------------------------- click-to-align and the always-visible roof map (display only)

@pytest.fixture
def roof_figs(map_calls, monkeypatch):
    """Wraps the stubbed st_folium so the roof map figures are kept next to the calls: [(fig, kwargs)]."""
    import streamlit_folium

    calls, box = map_calls
    inner, figs = streamlit_folium.st_folium, []

    def fake(fig, **kwargs):
        if str(kwargs.get("key", "")).startswith("roof_view"):
            figs.append((fig, kwargs))
        return inner(fig, **kwargs)

    monkeypatch.setattr(streamlit_folium, "st_folium", fake)
    return calls, box, figs


def _has_draw_toolbar(fig):
    from folium.plugins import Draw

    return any(isinstance(c, Draw) for c in fig._children.values())


def _click_at(inputs, dx, dy):
    """A st_folium last_clicked dict dx/dy metres from the footprint centroid."""
    c = shapely.from_wkb(inputs["footprint_wkb_m"]).centroid
    lon, lat = drawings.to_wgs84(c.x + dx, c.y + dy)
    return {"lat": lat, "lng": lon}


def _flat_elements(block):
    for child in block.children.values():
        if hasattr(child, "children") and child.children:
            yield from _flat_elements(child)
        else:
            yield child


def _roof_draw(at):
    return at.session_state[app.KEY_DRAWINGS]


def _roof_map_calls(calls):
    return [c for c in calls if str(c.get("key", "")).startswith("roof_view")]


def test_click_to_align_sets_the_offset_from_the_click_and_turns_itself_off(mocked, roof_figs, sample_index, building_point):
    calls, box, figs = roof_figs
    inputs = _roof_inputs(sample_index, building_point)
    at = _run_with_point(building_point)
    _button(at, "Çatıya tıklayarak hizala").click().run()
    assert _roof_draw(at)["align_active"] and app.ALIGN_CLICK_CAPTION in [c.value for c in at.caption]
    assert not [b for b in at.button if b.label == "Çatıya tıklayarak hizala"] and _button(at, "İptal")
    box["roof"] = {"all_drawings": None, "last_clicked": _click_at(inputs, 4.3, -6.6)}
    at.run()
    draw = _roof_draw(at)
    assert not at.exception and draw["offset"] == (4.0, -7.0) and not draw["align_active"]
    assert app.ALIGN_CLICK_CAPTION not in [c.value for c in at.caption]
    assert draw["obstacles"] == [] and draw["terrace"] is None  # a click never creates drawings
    assert "analysis" not in at.session_state
    assert _button(at, "Çatıya tıklayarak hizala")


def test_click_to_align_clamps_to_30_m_and_cancel_keeps_the_offset(mocked, roof_figs, sample_index, building_point):
    calls, box, figs = roof_figs
    inputs = _roof_inputs(sample_index, building_point)
    at = _run_with_point(building_point)
    _button(at, "Çatıya tıklayarak hizala").click().run()
    box["roof"] = {"all_drawings": None, "last_clicked": _click_at(inputs, 80, -80)}
    at.run()
    assert _roof_draw(at)["offset"] == (30.0, -30.0)
    _button(at, "Çatıya tıklayarak hizala").click().run()
    _button(at, "İptal").click().run()
    assert not _roof_draw(at)["align_active"] and _roof_draw(at)["offset"] == (30.0, -30.0)


def test_an_identical_repeated_click_is_ignored(mocked, roof_figs, sample_index, building_point):
    calls, box, figs = roof_figs
    inputs = _roof_inputs(sample_index, building_point)
    click = _click_at(inputs, 3.0, 2.0)
    at = _run_with_point(building_point)
    _button(at, "Çatıya tıklayarak hizala").click().run()
    box["roof"] = {"all_drawings": None, "last_clicked": click}
    at.run()
    assert _roof_draw(at)["offset"] == (3.0, 2.0)
    _button(at, "→").click().run()  # fine tuning after the click
    _button(at, "Çatıya tıklayarak hizala").click().run()
    at.run()
    draw = _roof_draw(at)
    assert draw["offset"] == (4.0, 2.0) and draw["align_active"]  # the stale identical click did nothing


def test_clicks_while_the_mode_is_inactive_change_nothing(mocked, roof_figs, sample_index, building_point):
    calls, box, figs = roof_figs
    inputs = _roof_inputs(sample_index, building_point)
    at = _run_with_point(building_point)
    gen = _roof_draw(at)["gen"]
    box["roof"] = {"all_drawings": None, "last_clicked": _click_at(inputs, 9.0, 9.0)}
    at.run()
    draw = _roof_draw(at)
    assert draw["offset"] == (0.0, 0.0) and draw["gen"] == gen and not draw["align_active"] and draw["last_click"] is None
    assert at.session_state["selected_point"] == building_point


def test_last_clicked_is_returned_and_the_draw_toolbar_shown_only_outside_the_mode(mocked, roof_figs, building_point):
    calls, box, figs = roof_figs
    at = _run_with_point(building_point)
    fig, kwargs = figs[-1]
    assert kwargs["returned_objects"] == ["all_drawings"] and _has_draw_toolbar(fig)
    _button(at, "Çatıya tıklayarak hizala").click().run()
    fig, kwargs = figs[-1]
    assert "last_clicked" in kwargs["returned_objects"] and not _has_draw_toolbar(fig)
    assert kwargs["key"] == "roof_view_1"  # toggling the mode bumps the map generation
    _button(at, "İptal").click().run()
    fig, kwargs = figs[-1]
    assert kwargs["returned_objects"] == ["all_drawings"] and _has_draw_toolbar(fig) and kwargs["key"] == "roof_view_2"


def test_drawings_returned_while_aligning_are_not_read(mocked, roof_figs, sample_index, building_point):
    calls, box, figs = roof_figs
    inputs = _roof_inputs(sample_index, building_point)
    at = _run_with_point(building_point)
    _button(at, "Çatıya tıklayarak hizala").click().run()
    box["roof"] = {"all_drawings": [_feature(_centre_obstacle(inputs))], "last_clicked": None}
    at.run()
    assert _roof_draw(at)["obstacles"] == []


def test_offset_steps_of_1_m_and_5_m_clamp_at_30_m(mocked, roof_figs, building_point):
    at = _run_with_point(building_point)
    assert {"↑", "↓", "←", "→", "5 m ↑", "5 m ↓", "5 m ←", "5 m →"} <= {b.label for b in at.button}
    _button(at, "5 m →").click().run()
    _button(at, "5 m ↑").click().run()
    _button(at, "←").click().run()
    assert _roof_draw(at)["offset"] == (4.0, 5.0)
    for _ in range(8):
        _button(at, "5 m →").click().run()
    assert _roof_draw(at)["offset"][0] == 30.0
    for _ in range(8):
        _button(at, "5 m ↓").click().run()
    assert _roof_draw(at)["offset"][1] == -30.0
    state = drawings.new_state("b")
    drawings.move_offset(state, 100.0, -100.0)
    assert state["offset"] == (30.0, -30.0)


def test_drawings_and_offset_made_before_hesapla_are_used_by_the_result(mocked, roof_figs, monkeypatch, sample_index, building_point):
    calls, box, figs = roof_figs
    pvgis = []
    real_yield, real_face = solar.get_solar_yield, solar.get_face_yield
    monkeypatch.setattr(solar, "get_solar_yield", lambda *a, **k: pvgis.append(a) or real_yield(*a, **k))
    monkeypatch.setattr(solar, "get_face_yield", lambda *a, **k: pvgis.append(a) or real_face(*a, **k))
    inputs = _roof_inputs(sample_index, building_point)
    at = _run_with_point(building_point)
    _button(at, "5 m →").click().run()
    box["roof"] = {"all_drawings": [_feature(shapely.affinity.translate(_centre_obstacle(inputs), 5.0, 0.0))]}
    at.run()
    draw = _roof_draw(at)
    assert not at.exception and draw["offset"] == (5.0, 0.0) and len(draw["obstacles"]) == 1
    assert pvgis == [] and "analysis" not in at.session_state and "result" not in at.session_state  # nothing evaluated
    _button(at, "Hesapla").click().run()
    assert not at.exception and pvgis
    r = at.session_state["result"]
    assert r["guc_kaynagi"] == "geometri" and r["cizilen_engel_sayisi"] == 1
    assert _roof_draw(at)["offset"] == (5.0, 0.0)


def test_after_hesapla_drawing_changes_still_recompute_without_pvgis(mocked, roof_figs, monkeypatch, sample_index, building_point):
    calls, box, figs = roof_figs
    at = _after_hesapla(building_point)
    _no_pvgis(monkeypatch)
    box["roof"] = {"all_drawings": [_feature(_centre_obstacle(_roof_inputs(sample_index, building_point)))]}
    at.run()
    assert at.session_state["result"]["guc_kaynagi"] == "geometri"


def test_before_hesapla_the_roof_map_shows_outline_and_setback_but_no_panels(mocked, roof_figs, building_point):
    calls, box, figs = roof_figs
    at = _run_with_point(building_point)
    _, kwargs = figs[-1]
    fg = kwargs["feature_group_to_add"]
    assert len([c for c in fg._children.values() if isinstance(c, folium.Polygon)]) == 2  # footprint + setback
    assert not [c for c in fg._children.values() if isinstance(c, folium.Marker)]  # no label yet
    assert app.ROOF_VIEW_PRE_CAPTION in [c.value for c in at.caption]
    _button(at, "Hesapla").click().run()
    fg = figs[-1][1]["feature_group_to_add"]
    assert len([c for c in fg._children.values() if isinstance(c, folium.Polygon)]) > 2
    assert [c for c in fg._children.values() if isinstance(c, folium.Marker)]
    assert app.ROOF_VIEW_PRE_CAPTION not in [c.value for c in at.caption]


def test_exactly_one_roof_map_per_run_and_it_sits_above_hesapla(mocked, roof_figs, building_point):
    calls, box, figs = roof_figs
    at = _run_with_point(building_point, skip=False)
    assert _roof_map_calls(calls)  # visible as soon as a building is selected
    _button(at, "Fotoğrafsız devam et").click().run()
    calls.clear()
    at.run()
    assert len(_roof_map_calls(calls)) == 1

    def order():
        marks = [
            "roof" if e.type == "subheader" and e.value == "Çatı görünümü" else "hesapla" if e.type == "button" and e.label == "Hesapla" else None
            for e in _flat_elements(at.main)
        ]
        return [m for m in marks if m]

    assert order() == ["roof", "hesapla"]
    _button(at, "Hesapla").click().run()
    assert not at.exception and order() == ["roof", "hesapla"]  # still one roof view, still above the button
    calls.clear()
    at.run()
    assert len(_roof_map_calls(calls)) == 1
