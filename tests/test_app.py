import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app
from core import config, solar
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
