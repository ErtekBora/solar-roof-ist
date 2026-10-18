import json
from pathlib import Path

import folium
import pytest
import streamlit as st
from pyproj import Transformer
from streamlit.testing.v1 import AppTest

import app
from core import buildings, calc, config, explain, solar
from core.solar import _parse_pvcalc

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "buildings_sample.parquet"
TO_WGS = Transformer.from_crs("EPSG:32635", "EPSG:4326", always_xy=True)
OUT_OF_COVERAGE = (41.015, 28.979)  # old map centre, outside the prototype bbox -> manual-area flow
BILL_LABEL = "Aylık elektrik faturanız yaklaşık kaç TL?"
FLOORS_LABEL = "Bina kat sayısı (isteğe bağlı)"
ROOF_LABEL = "Çatı tipi"


def _fake_yield(lat, lon, horizon72=None):
    load = lambda n: _parse_pvcalc(json.loads((FIXTURES / n).read_text(encoding="utf-8")))
    ideal, shaded = load("pvgis_optimal.json"), load("pvgis_terrain.json")
    return {"ideal": ideal, "shaded": shaded, "angle": ideal["angle"], "aspect": ideal["aspect"]}


@pytest.fixture
def mocked(monkeypatch):
    monkeypatch.setattr(app, "get_solar_yield", _fake_yield)
    monkeypatch.setattr(solar, "get_solar_yield", _fake_yield)  # AppTest re-executes app.py
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
    r = app.calculate(41.015, 28.979, 150.0, "flat")
    assert r["uygunluk_puani_yuzde"] == pytest.approx(99.79, abs=0.01)
    assert r["yillik_isinim_kwh_m2"] == pytest.approx(1738.72)
    assert r["sicaklik_kaybi_yuzde"] == pytest.approx(6.49)
    assert len(r["aylik_uretim_kwh"]) == 12 and len(r["aylik_ideal_kwh"]) == 12
    assert r["tasarruf_yontemi"] == "genis_tahmin"
    assert app.calculate(41.015, 28.979, 150.0, "flat", 1500)["tasarruf_yontemi"] == "fatura"


def _run_with_point(point=OUT_OF_COVERAGE):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    if point:
        at.session_state["selected_point"] = point
    return at.run()


def test_button_without_point_shows_error(mocked):
    at = _run_with_point(point=None)
    at.button[0].click().run()
    assert any("konum" in e.value for e in at.error)
    assert "result" not in at.session_state


def test_no_sidebar_tariff_input_and_calculation_runs_without_a_bill(mocked):
    at = _run_with_point()
    assert not any("tarife" in n.label.lower() for n in at.number_input)
    assert not any("tarife" in w.value for w in at.warning)
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert at.session_state["result"]["tasarruf_yontemi"] == "genis_tahmin"


def test_full_flow_shows_cards_and_keeps_result(mocked, monkeypatch):
    at = _run_with_point()
    at.button[0].click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Uygunluk (arazi gölgesi)" in labels
    assert len(labels) == 9
    # a rerun without the button must not call PVGIS again
    monkeypatch.setattr(solar, "get_solar_yield", lambda *a, **k: pytest.fail("PVGIS called on rerun"))
    at.run()
    assert len(at.metric) == 9


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
    real = calc.compute_capacity

    def spy(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return real(*args, **kwargs)

    monkeypatch.setattr(calc, "compute_capacity", spy)  # app.py re-imports it on every AppTest run
    return calls


def test_building_found_uses_the_footprint_and_setback_area(mocked, spy_capacity, sample_index, building_point):
    at = _run_with_point(building_point)
    assert any("Bina bulundu" in s.value for s in at.success)
    manual = _widget(at.number_input, "Bina bulunamazsa kullanılacak çatı alanı (m²)")
    manual.set_value(999.0)
    at.button[0].click().run()
    assert not at.exception
    found = buildings.find_building(sample_index, *building_point)["building"]
    call = spy_capacity[-1]
    assert call["args"][0] == pytest.approx(found["area_m2"]) and call["args"][0] != 999.0
    assert call["kwargs"]["usable_base_area_m2"] == pytest.approx(found["setback_area_m2"])
    assert call["kwargs"]["roof_type_known"] is False  # no photo: "Bilmiyorum"
    assert at.session_state["result"]["guven_seviyesi"] == "Düşük"  # polygon + unknown roof type
    rows = at.dataframe[0].value
    assert "Ölçüm/veri" in set(rows[rows["Değer"] == "Çatı alanı"]["Durum"])
    assert {"Kenar boşluğu", "Bina yüksekliği"} <= set(rows["Değer"])


def test_known_roof_type_with_a_building_is_orta(mocked, building_point):
    at = _run_with_point(building_point)
    at.button[0].click().run()
    _widget(at.radio, ROOF_LABEL).set_value("Düz").run()
    assert at.session_state["result"]["guven_seviyesi"] == "Orta"


def test_building_not_found_message_and_manual_area(mocked, spy_capacity):
    at = _run_with_point((40.9501, 29.0501))  # inside the bbox, far from the fixture buildings
    assert any("15 m" in i.value for i in at.info)
    _widget(at.number_input, "Bina bulunamazsa kullanılacak çatı alanı (m²)").set_value(222.0)
    at.button[0].click().run()
    call = spy_capacity[-1]
    assert call["args"][0] == 222.0 and call["kwargs"]["usable_base_area_m2"] is None
    assert not at.exception


def test_out_of_coverage_message_and_manual_flow(mocked):
    at = _run_with_point(OUT_OF_COVERAGE)
    assert any("Erenköy" in i.value and "Göztepe" in i.value for i in at.info)
    at.button[0].click().run()
    assert not at.exception and len(at.metric) == 9


def test_loader_returning_none_warns_and_manual_flow_still_works(mocked, monkeypatch, building_point):
    monkeypatch.setattr(buildings, "load_buildings", lambda *a, **k: None)
    at = _run_with_point(building_point)
    assert not at.exception
    assert any("Bina verisi yüklenemedi; çatı alanını elle girerek devam edebilirsiniz" in w.value for w in at.warning)
    at.button[0].click().run()
    assert not at.exception and len(at.metric) == 9


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

    calls, box = [], {"value": {"last_clicked": None}}

    def fake(fig, **kwargs):
        calls.append(kwargs)
        return box["value"]

    monkeypatch.setattr(streamlit_folium, "st_folium", fake)
    return calls, box


def test_center_and_zoom_are_passed_only_on_the_rerun_after_a_building_is_found(mocked, map_calls, building_point):
    calls, _ = map_calls
    at = _run_with_point(building_point)
    first = calls[-1]
    assert first["zoom"] == 18 and first["center"] is not None
    assert first["returned_objects"] == ["last_clicked"] and first["key"] == "map"
    assert isinstance(first["feature_group_to_add"], folium.FeatureGroup)
    at.run()
    assert "center" not in calls[-1] and "zoom" not in calls[-1]  # the map stays where the user left it
    at.run()
    assert "center" not in calls[-1] and "zoom" not in calls[-1]


def test_a_new_click_clears_the_previous_results(mocked, map_calls, building_point):
    calls, box = map_calls
    at = _run_with_point(building_point)
    at.button[0].click().run()
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
    at.button[0].click().run()
    assert not at.exception
    return at


def test_changing_the_bill_recomputes_savings_without_pvgis_or_gemini_and_marks_the_explanation_stale(
    mocked, monkeypatch, fake_explain, building_point
):
    client = fake_explain([])  # an empty script: any Gemini call fails and falls back to the template
    at = _after_hesapla(building_point)
    calls_after_hesapla = client.calls
    assert at.session_state["result"]["tasarruf_yontemi"] == "genis_tahmin"
    assert any("Faturanızı girin" in c.value for c in at.caption)
    before = at.session_state["result"]["yillik_tasarruf_tl_aralik"]
    monkeypatch.setattr(solar, "get_solar_yield", lambda *a, **k: pytest.fail("PVGIS called on a widget change"))
    _widget(at.number_input, BILL_LABEL).set_value(1500.0).run()
    assert not at.exception
    r = at.session_state["result"]
    assert r["tasarruf_yontemi"] == "fatura" and r["yillik_tasarruf_tl_aralik"] != before
    assert r["yillik_tasarruf_tl_aralik"][1] <= 12 * 1500 + 0.01
    assert any("Faturanızın yaklaşık" in i.value for i in at.info)
    assert any("Değerler değişti; açıklama güncel değil." in i.value for i in at.info)  # stale, never auto-called
    assert client.calls == calls_after_hesapla


@pytest.mark.parametrize("bad", [0.0, -100.0])
def test_zero_or_negative_bill_counts_as_no_bill(mocked, building_point, bad):
    at = _after_hesapla(building_point)
    _widget(at.number_input, BILL_LABEL).set_value(bad).run()
    assert at.session_state["result"]["tasarruf_yontemi"] == "genis_tahmin"


def test_changing_floors_updates_the_height_without_pvgis_or_gemini(mocked, monkeypatch, fake_explain, sample_index, building_point):
    client = fake_explain([])
    at = _after_hesapla(building_point)
    calls_after_hesapla = client.calls
    found = buildings.find_building(sample_index, *building_point)["building"]
    floors = _widget(at.number_input, FLOORS_LABEL)
    assert floors.value == app.floors_prefill(found)
    monkeypatch.setattr(solar, "get_solar_yield", lambda *a, **k: pytest.fail("PVGIS called on a widget change"))
    typed = (app.floors_prefill(found) or 0) + 2
    floors.set_value(typed).run()
    assert not at.exception and client.calls == calls_after_hesapla
    assert not any("açıklama güncel değil" in i.value for i in at.info)  # floors do not enter the explanation payload
    assert any("girdiğiniz kat sayısı" in c.value and f"{typed * 3},0 m" in c.value for c in at.caption)
    rows = at.dataframe[0].value
    assert "Kullanıcı girdisi" in set(rows[rows["Değer"] == "Bina yüksekliği"]["Durum"])


def test_roof_type_change_recomputes_immediately_with_no_pvgis_call(mocked, monkeypatch, building_point):
    at = _after_hesapla(building_point)
    monkeypatch.setattr(solar, "get_solar_yield", lambda *a, **k: pytest.fail("PVGIS called on a widget change"))
    flat = at.session_state["result"]
    _widget(at.radio, ROOF_LABEL).set_value("Eğimli").run()
    pitched = at.session_state["result"]
    # TODO(P2): pitched roofs get their own PVGIS orientation; then this change needs a new solar run
    assert pitched["cati_tipi"] == "pitched" and pitched["kurulabilir_guc_kwp"] > flat["kurulabilir_guc_kwp"]


def test_confidence_card_lists_inputs_and_improvements(mocked, building_point):
    at = _after_hesapla(building_point)
    text = " ".join(m.value for m in at.markdown)
    assert "Ölçülen:" in text and "Varsayılan:" in text and "Çatı tipi: bilinmiyor" in text
    assert any("Çatı tipini seçin" in c.value for c in at.caption)


# ---------------------------------------------------------- demo photo vs a real building

def test_photo_applies_rule():
    assert app.photo_applies({"id": "x"}, is_demo=True) is False  # bundled demo photo on a real building
    assert app.photo_applies({"id": "x"}, is_demo=False) is True  # user upload is assumed to be that building
    assert app.photo_applies(None, is_demo=True) is True  # manual-area flow: the demo behaves as before
    assert app.photo_applies(None, is_demo=False) is True


def _run_demo_photo(name, point=OUT_OF_COVERAGE):
    at = _run_with_point(point)
    _widget(at.radio, "Fotoğraf").set_value(PHOTO_RADIO_DEMO)
    at.run()
    at.selectbox[0].select(name)
    at.run()
    return at


def test_demo_photo_is_shown_but_not_applied_to_a_real_building(
    mocked, fake_gemini, roof10_json, spy_capacity, building_point
):
    client = fake_gemini([roof10_json])
    at = _run_demo_photo("roof_10.jpg", building_point)
    at.button[0].click().run()
    assert not at.exception and client.calls == 1
    assert any(app.DEMO_PHOTO_NOTE == i.value for i in at.info)
    call = spy_capacity[-1]
    assert call["args"][1] is None and call["kwargs"]["image_view"] is None  # no obstacle ratio applied
    assert call["args"][2] == "flat" and call["kwargs"]["roof_type_known"] is False  # Gemini's pitched not applied
    r = at.session_state["result"]
    assert r["engel_orani"] is None and r["guven_seviyesi"] == "Düşük"  # confidence not raised
    assert at.session_state["analysis"]["detection"] is not None  # the vision result is still displayed
    assert not [m for m in at.metric if m.label == "Engel oranı"]
    assert not [k for k in at.session_state if k.startswith("obstacles_")]  # no table: obstacles do not change the result


def test_uploaded_style_photo_is_applied_to_the_building(sample_index, building_point):
    solar_result = _fake_yield(41.0, 29.0)
    building = buildings.find_building(sample_index, *building_point)["building"]
    inputs = {"lat": 41.0, "lon": 29.0, "area": building["area_m2"], "usable_base_area_m2": building["setback_area_m2"],
              "roof_type": "flat", "roof_known": True, "photo_applies": app.photo_applies(building, False)}
    det = _square_detection()
    result, ratio = app.recompute(solar_result, inputs, det, {0, 1})
    assert ratio == pytest.approx(0.5) and result["engel_orani"] == pytest.approx(0.5)
    assert result["kullanilabilir_alan_m2"] == pytest.approx(building["setback_area_m2"] * 0.5, abs=0.01)
    assert result["guven_seviyesi"] == "Orta"  # polygon + top-down photo + known roof type
    inputs["photo_applies"] = app.photo_applies(building, True)  # same photo, but a bundled demo
    result, ratio = app.recompute(solar_result, inputs, det, {0, 1})
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


def test_floors_helpers():
    assert app.floors_prefill({"num_floors": 4.0}) == 4 and app.floors_prefill({"num_floors": None}) is None
    assert app.floors_prefill({"num_floors": 400}) == 100 and app.floors_prefill(None) is None
    b = {"num_floors": 4.0}
    assert app.effective_floors(4, b) is None  # unchanged from the data: the data height rule stays
    assert app.effective_floors(6, b) == 6 and app.effective_floors(None, b) is None


def test_clean_bill():
    assert app.clean_bill(1500) == 1500.0
    assert all(app.clean_bill(v) is None for v in (None, 0, -5, float("nan"), float("inf"), "abc", True))


def test_coverage_and_shared_roof_texts():
    monthly = [50, 60, 90, 110, 140, 160, 170, 160, 120, 90, 55, 45]
    base = dict(roof_area_m2=300, obstacle_ratio=None, roof_type="flat", yield_ideal=1300, yield_shaded=sum(monthly),
                monthly_shaded=monthly, usable_base_area_m2=250.0)
    big = calc.compute_capacity(**base, monthly_bill_tl=300)
    assert app.coverage_text(big) == "Faturanızın yaklaşık %100 kadarını karşılar."
    shared = app.shared_roof_text(big)
    assert shared.startswith("Tek daire faturanızın tamamını karşılar; üretimin yalnızca %")
    assert "apartman ortak alanı veya diğer daireler için değerlendirilebilir." in shared
    small = calc.compute_capacity(**{**base, "usable_base_area_m2": 20.0}, monthly_bill_tl=6000)
    assert app.shared_roof_text(small) is None
    lo, hi = small["fatura_karsilama_yuzde_aralik"]
    assert app.coverage_text(small) == f"Faturanızın yaklaşık %{round(lo)}–%{round(hi)} kadarını karşılar."
    assert app.coverage_text(calc.compute_capacity(**base)) is None  # no bill


def test_tl_card_always_shows_tariff_source_and_method(mocked, building_point):
    at = _after_hesapla(building_point)
    assert any(config.TARIFF_SOURCE in c.value and "vergiler dahil yaklaşık" in c.value for c in at.caption)
    assert any("Geniş tahmin (tüketim bilinmiyor)" in c.value for c in at.caption)
    _widget(at.number_input, BILL_LABEL).set_value(2000.0).run()
    assert any("Faturanıza göre hesaplandı" in c.value for c in at.caption)
    assert config.TARIFF_SOURCE in " ".join(c.value for c in at.caption)


# ------------------------------------------------------------- photo (Demo-2)

from core import vision  # noqa: E402

PHOTO_RADIO_DEMO = "Örnek fotoğraf seç"


def _square_detection(obstacles=((0, 0, 50, 50), (50, 50, 100, 100)), roof_type="flat", image_view="top-down"):
    return {
        "roof_type": roof_type, "image_view": image_view, "multiple_buildings": False,
        "width": 100, "height": 100, "roof_source": "polygon",
        "roof_polygon_px": [(0, 0), (100, 0), (100, 100), (0, 100)], "roof_area_px": 10000.0,
        "obstacles": [{"id": i, "box_px": b, "label": "chimney", "confidence": 0.9} for i, b in enumerate(obstacles)],
        "obstacle_ratio": 0.5, "model": "m", "image_hash": "ab" * 32,
    }


def _inputs(roof_type="flat"):
    return {"lat": 41.0, "lon": 29.0, "area": 100.0, "roof_type": roof_type, "roof_known": True, "roof_tag": "Varsayım"}


def test_deselecting_obstacle_lowers_ratio_without_gemini_or_pvgis(monkeypatch):
    monkeypatch.setattr(vision, "_get_client", lambda: pytest.fail("Gemini called"))
    solar_result = _fake_yield(41.0, 29.0)
    det = _square_detection()
    _, both = app.recompute(solar_result, _inputs(), det, {0, 1})
    _, one = app.recompute(solar_result, _inputs(), det, {0})
    assert both == pytest.approx(0.5) and one == pytest.approx(0.25) and one < both


def test_all_obstacles_deselected_is_zero_ratio_not_no_photo(monkeypatch):
    solar_result = _fake_yield(41.0, 29.0)
    r, ratio = app.recompute(solar_result, _inputs(), _square_detection(), set())
    assert ratio == 0.0 and r["engel_orani"] == 0.0
    assert r["kullanilabilir_alan_m2"] == pytest.approx(100.0 * 0.85)  # photo, clean roof: 0.85 (not 0.276)
    r_none, ratio_none = app.recompute(solar_result, _inputs(), None, set())
    assert ratio_none is None and r_none["kullanilabilir_alan_m2"] == pytest.approx(100.0 * 0.276)


def test_selected_ids_from_editor_dataframe():
    import pandas as pd

    df = pd.DataFrame({"Dahil": [True, False, True]}, index=[0, 1, 2])
    assert app.selected_ids(df) == {0, 2}


@pytest.mark.parametrize(
    "choice, preselected, expected",
    [
        ("Düz", None, ("flat", True, "Kullanıcı girdisi")),
        ("Eğimli", None, ("pitched", True, "Kullanıcı girdisi")),
        ("Eğimli", "Eğimli", ("pitched", True, "Ölçüm/veri")),  # unchanged from the photo's answer
        ("Düz", "Eğimli", ("flat", True, "Kullanıcı girdisi")),  # the user overrode it
        ("Bilmiyorum", None, ("flat", False, "Varsayım")),
    ],
)
def test_resolve_roof_type(choice, preselected, expected):
    assert app.resolve_roof_type(choice, preselected) == expected


def test_photo_roof_choice_preselects_gemini_answer():
    assert app.photo_roof_choice({"roof_type": "pitched"}) == ("Eğimli", None)
    assert app.photo_roof_choice({"roof_type": "flat"}) == ("Düz", None)
    choice, note = app.photo_roof_choice({"roof_type": "mixed"})
    assert choice == "Düz" and "karma" in note


def test_parse_sources_and_caption():
    info = app.parse_sources()["roof_10.jpg"]
    assert info["license"] == "Pexels License"
    assert "Pexels License" in app.source_caption("roof_10.jpg")
    assert "kaynak bağlantısı eklenecek" in app.source_caption("roof_10.jpg")  # SOURCES.md link is a TODO
    assert "https://" in app.source_caption("roof_01.jpg")


def test_assumption_rows_tags_and_demo_row():
    solar_result = _fake_yield(41.0, 29.0)
    inputs = dict(_inputs(), demo_photo=True)
    det = _square_detection()
    r, _ = app.recompute(solar_result, inputs, det, {0})
    df = app.assumption_rows(r, inputs, det)
    assert set(df["Durum"]) <= {"Ölçüm/veri", "Kullanıcı girdisi", "Varsayım", "Sonraki aşamada", "YZ tahmini"}
    assert "Sonraki aşamada" in set(df["Durum"])
    assert any("Demo fotoğrafı örnektir; seçilen binanın kendi çatısı değildir." in n for n in df["Kaynak / not"])
    assert {"Elektrik tarifesi", "Tasarruf aralığı", "Tasarruf kapsamı"} <= set(df["Değer"])
    assert any(config.TARIFF_SOURCE in n for n in df["Kaynak / not"])
    r2, _ = app.recompute(solar_result, _inputs(), None, set())
    assert not any("Demo fotoğrafı" in n for n in app.assumption_rows(r2, _inputs(), None)["Kaynak / not"])


def test_assumption_rows_for_a_building_include_setback_height_and_ml_note(sample_index, building_point):
    building = buildings.find_building(sample_index, *building_point)["building"]
    inputs = {**_inputs(), "building": building, "usable_base_area_m2": building["setback_area_m2"],
              "area": building["area_m2"], "photo_applies": True}
    r, _ = app.recompute(_fake_yield(41.0, 29.0), inputs, None, set())
    df = app.assumption_rows(r, inputs, None, report=sample_index.report)
    by = df.set_index("Değer")
    assert by.loc["Kenar boşluğu", "Durum"] == "Varsayım" and "1,0 m" in by.loc["Kenar boşluğu", "Sonuç"]
    assert by.loc["Çatı alanı", "Durum"] == "Ölçüm/veri"
    assert by.loc["Bina yüksekliği", "Durum"] == app.HEIGHT_SOURCE_TAG[building["height_source"]]
    assert by.loc["Komşu bina gölgesi", "Durum"] == "Sonraki aşamada"


@pytest.fixture
def spy_vision_capacity(monkeypatch):
    calls = []
    real = calc.compute_capacity

    def spy(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return real(*args, **kwargs)

    monkeypatch.setattr(calc, "compute_capacity", spy)
    return calls


def test_photo_flow_ratio_reaches_compute_capacity_and_preselects_detected_type(
    mocked, fake_gemini, roof10_json, spy_vision_capacity
):
    client = fake_gemini([roof10_json])
    at = _run_demo_photo("roof_10.jpg")  # no building: the demo photo applies as before
    at.button[0].click().run()
    assert not at.exception
    assert client.calls == 1
    call = spy_vision_capacity[-1]
    ratio, roof_type = call["args"][1], call["args"][2]
    assert 0 < ratio < 0.95 and roof_type == "pitched"  # roof_10 fixture: pitched, oblique
    assert call["kwargs"]["image_view"] == "oblique"
    assert _widget(at.radio, ROOF_LABEL).value == "Eğimli"
    assert any("Fotoğraf analizine göre seçildi" in c.value for c in at.caption)
    labels = [m.label for m in at.metric]
    assert "Engel oranı" in labels and len(labels) == 10
    assert any("Açılı fotoğraf" in i.value for i in at.info)
    assert at.dataframe  # assumptions table
    assert at.session_state["result"]["guven_seviyesi"] == "Düşük"
    # a plain rerun must not call Gemini again
    at.run()
    assert client.calls == 1


def test_switching_photos_gives_fresh_editor_keys(mocked, fake_gemini, roof10_json):
    fake_gemini([roof10_json, roof10_json])
    at = _run_demo_photo("roof_10.jpg")
    at.button[0].click().run()
    key_a = next(k for k in at.session_state if k.startswith("obstacles_"))
    at.selectbox[0].select("roof_11.jpg").run()
    at.button[0].click().run()
    keys = [k for k in at.session_state if k.startswith("obstacles_")]
    key_b = [k for k in keys if k != key_a]
    assert key_b, "a different photo must use a different data_editor key"


def test_no_photo_path_still_works_and_lists_assumption(mocked, spy_capacity):
    at = _run_with_point()
    at.button[0].click().run()
    assert not at.exception
    call = spy_capacity[-1]
    assert call["args"][1] is None and call["args"][2] == "flat"
    assert len(at.metric) == 9
    assert at.session_state["result"]["guven_seviyesi"] == "Düşük"


def test_vision_error_falls_back_to_no_photo_calculation(mocked, fake_gemini, spy_capacity):
    fake_gemini(["{bad", "{bad"])
    at = _run_demo_photo("roof_10.jpg")
    at.button[0].click().run()
    assert not at.exception
    assert any("okunamadı" in w.value for w in at.warning)
    assert spy_capacity[-1]["args"][1] is None
    assert at.session_state["result"]["guven_seviyesi"] == "Düşük"
    assert len(at.metric) == 9


# ------------------------------------------------------------- explanation (Demo-3)

EXPLAIN_BUTTON = "Açıklamayı YZ ile güncelle"


def _flow_text():
    """A valid Gemini-like text for the no-photo flow (template text passes the validator by construction)."""
    return explain.build_fallback_explanation(app.calculate(41.015, 28.979, 150.0, "flat"))


def _run_no_photo():
    at = _run_with_point()
    at.button[0].click().run()
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


def test_no_photo_flow_has_nine_cards_and_explanation(mocked):
    at = _run_no_photo()
    assert len(at.metric) == 9 and "explanation" in at.session_state


def test_missing_key_skips_vision_and_uses_template(mocked, fake_gemini, monkeypatch):
    def no_key():
        raise config.MissingApiKeyError("Gemini API anahtarı bulunamadı. Test.")

    monkeypatch.setattr(config, "get_gemini_api_key", no_key)
    client = fake_gemini([])
    at = _run_demo_photo("roof_10.jpg")
    at.button[0].click().run()
    assert not at.exception and client.calls == 0
    assert any("anahtarı bulunamadı" in w.value for w in at.warning)
    assert at.session_state["explanation"]["source"] == "template"
    assert len(at.metric) == 9


def test_api_key_is_resolved_in_main_thread_and_passed_to_worker(mocked, monkeypatch, roof10_json):
    seen = []

    def spy(image_bytes, api_key=None):
        seen.append(api_key)
        raise vision.VisionError("Test hatası.")

    monkeypatch.setattr(vision, "detect_roof_objects", spy)
    at = _run_demo_photo("roof_10.jpg")
    at.button[0].click().run()
    assert not at.exception and seen == ["test-key"]
    assert any("Test hatası." in w.value for w in at.warning)


def test_obstacle_edit_makes_explanation_stale_without_gemini_then_regenerate_clears_it(
    mocked, fake_explain, monkeypatch
):
    solar_result = _fake_yield(41.0, 29.0)
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
    solar_result = _fake_yield(41.0, 29.0)
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
    card = next(m for m in at.metric if m.label == "Yıllık tasarruf (aralık)")
    assert card.value.endswith("bin TL") and "Kesin aralık:" in card.help and card.help.endswith("TL")


def test_assumption_row_shows_active_explanation_source():
    solar_result = _fake_yield(41.0, 29.0)
    r, _ = app.recompute(solar_result, _inputs(), None, set())
    row = lambda src: app.assumption_rows(r, _inputs(), None, {"source": src})
    gem = row("gemini")
    assert "YZ yorumu (sayılar doğrulandı)" in set(gem["Durum"])
    assert "Şablon" in set(row("template")["Durum"])
    assert "YZ açıklaması" not in set(app.assumption_rows(r, _inputs(), None)["Değer"])


def test_cards_and_explanation_payload_round_identically():
    from core.formatting import percent_from_ratio, tr_num

    solar_result = _fake_yield(41.0, 29.0)
    r, _ = app.recompute(solar_result, _inputs(), _square_detection(), {0, 1})
    r["yillik_isinim_kwh_m2"] = 1738.5  # a half: f-format would truncate it to 1.738
    payload = explain.build_payload(r, _square_detection(), False)
    for key, decimals in (("yillik_isinim_kwh_m2", 0), ("golgeli_uretim_kwh_kwp", 0), ("yillik_uretim_kwh", 0),
                          ("sicaklik_kaybi_yuzde", 1), ("kurulabilir_guc_kwp", 1), ("kullanilabilir_alan_m2", 1),
                          ("co2_ton_yil", 1), ("uygunluk_puani_yuzde", 1)):
        assert tr_num(r[key], decimals) == tr_num(payload[key], decimals), key
    assert tr_num(r["yillik_isinim_kwh_m2"]) == "1.739"
    assert tr_num(percent_from_ratio(r["engel_orani"]), 1) == tr_num(payload["engel_orani"] * 100, 1)
