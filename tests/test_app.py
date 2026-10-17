import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app
from core import config, explain, solar
from core.solar import _parse_pvcalc

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"


def _fake_yield(lat, lon, horizon72=None):
    load = lambda n: _parse_pvcalc(json.loads((FIXTURES / n).read_text(encoding="utf-8")))
    ideal, shaded = load("pvgis_optimal.json"), load("pvgis_terrain.json")
    return {"ideal": ideal, "shaded": shaded, "angle": ideal["angle"], "aspect": ideal["aspect"]}


@pytest.fixture
def mocked(monkeypatch):
    monkeypatch.setattr(app, "get_solar_yield", _fake_yield)
    monkeypatch.setattr(solar, "get_solar_yield", _fake_yield)  # AppTest re-executes app.py
    monkeypatch.setattr(config, "TARIFF_TL_PER_KWH", 0.0)
    monkeypatch.setattr(config, "get_gemini_api_key", lambda: "test-key")  # never read the real secrets file
    explain.clear_cache()
    # default: the explanation model is unavailable -> template; tests wanting Gemini install fake_explain
    monkeypatch.setattr(explain, "_get_client", lambda api_key=None: (_ for _ in ()).throw(RuntimeError("offline")))


def test_calculate_builds_result(mocked):
    r = app.calculate(41.015, 28.979, 150.0, "flat", 3.0)
    assert r["uygunluk_puani_yuzde"] == pytest.approx(99.79, abs=0.01)
    assert r["yillik_isinim_kwh_m2"] == pytest.approx(1738.72)
    assert r["sicaklik_kaybi_yuzde"] == pytest.approx(6.49)
    assert len(r["aylik_uretim_kwh"]) == 12 and len(r["aylik_ideal_kwh"]) == 12


def _run_with_point(point=True):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    if point:
        at.session_state["point"] = (41.015, 28.979)
    return at.run()


def test_button_without_point_shows_error(mocked):
    at = _run_with_point(point=False)
    at.button[0].click().run()
    assert any("konum" in e.value for e in at.error)
    assert "result" not in at.session_state


def test_placeholder_tariff_asks_user(mocked):
    at = _run_with_point()
    assert any("tarife" in w.value for w in at.warning)
    at.button[0].click().run()
    assert any("tarifesini" in e.value for e in at.error)


def test_full_flow_shows_cards_and_keeps_result(mocked, monkeypatch):
    at = _run_with_point()
    at.number_input[1].set_value(3.0)
    at.button[0].click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Uygunluk (arazi gölgesi)" in labels
    assert len(labels) == 9
    # a rerun without the button must not call PVGIS again
    monkeypatch.setattr(solar, "get_solar_yield", lambda *a, **k: pytest.fail("PVGIS called on rerun"))
    at.run()
    assert len(at.metric) == 9


# ------------------------------------------------------------- photo (Demo-2)

from core import calc, vision  # noqa: E402

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
    return {"lat": 41.0, "lon": 29.0, "area": 100.0, "tariff": 3.0, "roof_type": roof_type, "roof_tag": "Varsayım"}


def test_deselecting_obstacle_lowers_ratio_without_gemini_or_pvgis(monkeypatch):
    monkeypatch.setattr(vision, "_get_client", lambda: pytest.fail("Gemini called"))
    solar = _fake_yield(41.0, 29.0)
    det = _square_detection()
    _, both = app.recompute(solar, _inputs(), det, {0, 1})
    _, one = app.recompute(solar, _inputs(), det, {0})
    assert both == pytest.approx(0.5) and one == pytest.approx(0.25) and one < both


def test_all_obstacles_deselected_is_zero_ratio_not_no_photo(monkeypatch):
    solar = _fake_yield(41.0, 29.0)
    r, ratio = app.recompute(solar, _inputs(), _square_detection(), set())
    assert ratio == 0.0 and r["engel_orani"] == 0.0
    assert r["kullanilabilir_alan_m2"] == pytest.approx(100.0 * 0.85)  # photo, clean roof: 0.85 (not 0.276)
    r_none, ratio_none = app.recompute(solar, _inputs(), None, set())
    assert ratio_none is None and r_none["kullanilabilir_alan_m2"] == pytest.approx(100.0 * 0.276)


def test_selected_ids_from_editor_dataframe():
    import pandas as pd

    df = pd.DataFrame({"Dahil": [True, False, True]}, index=[0, 1, 2])
    assert app.selected_ids(df) == {0, 2}


@pytest.mark.parametrize(
    "choice, detection, expected",
    [
        ("Düz", {"roof_type": "pitched"}, ("flat", "Kullanıcı girdisi")),
        ("Eğimli", None, ("pitched", "Kullanıcı girdisi")),
        (app.AUTO_LABEL, {"roof_type": "pitched"}, ("pitched", "Ölçüm/veri")),
        (app.AUTO_LABEL, {"roof_type": "mixed"}, ("flat", "Ölçüm/veri")),
        (app.AUTO_LABEL, None, ("flat", "Varsayım")),
    ],
)
def test_resolve_roof_type(choice, detection, expected):
    roof_type, tag, note = app.resolve_roof_type(choice, detection)
    assert (roof_type, tag) == expected
    assert (note is not None) == (detection is not None and detection["roof_type"] == "mixed" and choice == app.AUTO_LABEL)


def test_parse_sources_and_caption():
    info = app.parse_sources()["roof_10.jpg"]
    assert info["license"] == "Pexels License"
    assert "Pexels License" in app.source_caption("roof_10.jpg")
    assert "kaynak bağlantısı eklenecek" in app.source_caption("roof_10.jpg")  # SOURCES.md link is a TODO
    assert "https://" in app.source_caption("roof_01.jpg")


def test_assumption_rows_tags_and_demo_row():
    solar = _fake_yield(41.0, 29.0)
    inputs = dict(_inputs(), demo_photo=True)
    det = _square_detection()
    r, _ = app.recompute(solar, inputs, det, {0})
    df = app.assumption_rows(r, inputs, det)
    assert set(df["Durum"]) <= {"Ölçüm/veri", "Kullanıcı girdisi", "Varsayım", "Sonraki aşamada"}
    assert "Sonraki aşamada" in set(df["Durum"])
    assert any("Demo fotoğrafı örnektir; seçilen binanın kendi çatısı değildir." in n for n in df["Kaynak / not"])
    r2, _ = app.recompute(solar, _inputs(), None, set())
    assert not any("Demo fotoğrafı" in n for n in app.assumption_rows(r2, _inputs(), None)["Kaynak / not"])


@pytest.fixture
def spy_capacity(monkeypatch):
    calls = []
    real = calc.compute_capacity

    def spy(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return real(*args, **kwargs)

    monkeypatch.setattr(calc, "compute_capacity", spy)  # app.py re-imports it on every AppTest run
    return calls


def _run_demo_photo(name, tariff=3.0):
    at = _run_with_point()
    at.number_input[1].set_value(tariff)
    at.radio[0].set_value(PHOTO_RADIO_DEMO)
    at.run()
    at.selectbox[1].select(name)
    at.run()
    return at


def test_photo_flow_ratio_reaches_compute_capacity_and_auto_uses_detected_type(
    mocked, fake_gemini, roof10_json, spy_capacity
):
    client = fake_gemini([roof10_json])
    at = _run_demo_photo("roof_10.jpg")
    at.button[0].click().run()
    assert not at.exception
    assert client.calls == 1
    call = spy_capacity[-1]
    ratio, roof_type = call["args"][1], call["args"][2]
    assert 0 < ratio < 0.95 and roof_type == "pitched"  # roof_10 fixture: pitched, oblique
    assert call["kwargs"]["image_view"] == "oblique"
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
    at.selectbox[1].select("roof_11.jpg").run()
    at.button[0].click().run()
    keys = [k for k in at.session_state if k.startswith("obstacles_")]
    key_b = [k for k in keys if k != key_a]
    assert key_b, "a different photo must use a different data_editor key"


def test_no_photo_path_still_works_and_lists_assumption(mocked, spy_capacity):
    at = _run_with_point()
    at.number_input[1].set_value(3.0)
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
    return explain.build_fallback_explanation(app.calculate(41.015, 28.979, 150.0, "flat", 3.0))


def _run_no_photo():
    at = _run_with_point()
    at.number_input[1].set_value(3.0)
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
    solar = _fake_yield(41.0, 29.0)
    det = _square_detection()
    inputs = dict(_inputs(), demo_photo=False)
    monkeypatch.setattr(app, "get_gemini_api_key", lambda: "test-key")
    result_all, _ = app.recompute(solar, inputs, det, {0, 1})
    client = fake_explain([explain.build_fallback_explanation(result_all, app.explained_detection(det, {0, 1}), False)] * 2)
    exp = app.generate_explanation(result_all, app.explained_detection(det, {0, 1}), False, "k")
    assert exp["source"] == "gemini" and client.calls == 1

    result_one, _ = app.recompute(solar, inputs, det, {0})  # obstacle 1 unticked
    det_one = app.explained_detection(det, {0})
    assert not app.explanation_is_stale(exp, result_all, app.explained_detection(det, {0, 1}), False)
    assert app.explanation_is_stale(exp, result_one, det_one, False)
    assert client.calls == 1  # detecting staleness never calls Gemini

    fake_explain([explain.build_fallback_explanation(result_one, det_one, False)])
    new = app.regenerate_explanation(result_one, det_one, False)
    assert new["source"] == "gemini" and not app.explanation_is_stale(new, result_one, det_one, False)


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
    solar = _fake_yield(41.0, 29.0)
    r, _ = app.recompute(solar, _inputs(), None, set())
    row = lambda src: app.assumption_rows(r, _inputs(), None, {"source": src})
    gem = row("gemini")
    assert "YZ yorumu (sayılar doğrulandı)" in set(gem["Durum"])
    assert "Şablon" in set(row("template")["Durum"])
    assert "YZ açıklaması" not in set(app.assumption_rows(r, _inputs(), None)["Değer"])


def test_cards_and_explanation_payload_round_identically():
    from core.formatting import percent_from_ratio, tr_num

    solar = _fake_yield(41.0, 29.0)
    r, _ = app.recompute(solar, _inputs(), _square_detection(), {0, 1})
    r["yillik_isinim_kwh_m2"] = 1738.5  # a half: f-format would truncate it to 1.738
    payload = explain.build_payload(r, _square_detection(), False)
    for key, decimals in (("yillik_isinim_kwh_m2", 0), ("golgeli_uretim_kwh_kwp", 0), ("yillik_uretim_kwh", 0),
                          ("sicaklik_kaybi_yuzde", 1), ("kurulabilir_guc_kwp", 1), ("kullanilabilir_alan_m2", 1),
                          ("co2_ton_yil", 1), ("uygunluk_puani_yuzde", 1)):
        assert tr_num(r[key], decimals) == tr_num(payload[key], decimals), key
    assert tr_num(r["yillik_isinim_kwh_m2"]) == "1.739"
    assert tr_num(percent_from_ratio(r["engel_orani"]), 1) == tr_num(payload["engel_orani"] * 100, 1)
