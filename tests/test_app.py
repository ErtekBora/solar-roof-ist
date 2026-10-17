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
