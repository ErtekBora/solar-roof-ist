"""Regression: every pitched face gets a PVGIS yield and a visible status (browser repro at 40.96988, 29.07833).

PVGIS is mocked at the HTTP level, so the whole chain runs: face aspects -> de-duplicated parallel requests ->
per-key cache -> response parsing -> face mapping -> segments and statuses.
"""
import copy
import json
import logging
import math
import threading
from pathlib import Path

import pytest
from shapely import affinity
from shapely.geometry import box

import app
from core import drawings, solar

FIXTURES = Path(__file__).parent / "fixtures"
OPTIMAL = json.loads((FIXTURES / "pvgis_optimal.json").read_text(encoding="utf-8"))
TERRAIN = json.loads((FIXTURES / "pvgis_terrain.json").read_text(encoding="utf-8"))
FLAT_E_Y = TERRAIN["outputs"]["totals"]["fixed"]["E_y"]
X0, Y0 = 676650.0, 4536550.0  # pilot area, EPSG:32635


def face_factor(aspect):
    """Face yield relative to the flat segment: 1.0 facing South, falling towards North (80 % rule cuts |aspect| > ~68)."""
    return 0.68 + 0.32 * math.cos(math.radians(aspect))


class Resp:
    def __init__(self, data):
        self.status_code, self._data = 200, data

    def json(self):
        return self._data


@pytest.fixture
def pvgis_http(monkeypatch):
    """requests.get stand-in: optimal / terrain fixtures for the flat runs, an aspect-scaled answer for a face."""
    solar.clear_cache()
    monkeypatch.setattr(solar, "_throttle_sleep", lambda s: None)
    lock, calls = threading.Lock(), []

    def fake_get(url, params=None, headers=None, timeout=None):
        with lock:
            calls.append(dict(params))
        if params["optimalangles"] == 1:
            return Resp(OPTIMAL)
        data = copy.deepcopy(TERRAIN)
        if params["angle"] == 30:  # a pitched face
            k = face_factor(params["aspect"])
            data["outputs"]["totals"]["fixed"]["E_y"] *= k
            for m in data["outputs"]["monthly"]["fixed"]:
                m["E_m"] *= k
            data["inputs"]["mounting_system"]["fixed"]["azimuth"]["value"] = params["aspect"]
        return Resp(data)

    monkeypatch.setattr(solar.requests, "get", fake_get)
    yield calls
    solar.clear_cache()


def footprint(rotation_deg):
    """182 m2 rectangle (15.6 x 11.67 m); rotation -23.13 gives the repro building's ridge bearing of 113.13 deg."""
    return affinity.rotate(box(X0, Y0, X0 + 15.6, Y0 + 11.67), rotation_deg)


def inputs_for(fp, pitched):
    c = fp.centroid
    lon, lat = drawings.to_wgs84(c.x, c.y)
    setback = fp.buffer(-1.0)
    return {
        "lat": lat, "lon": lon, "area": fp.area, "usable_base_area_m2": setback.area, "footprint_wkb_m": fp.wkb_hex,
        "structure": "pitched", "flat_share_pct": 50, "n_flats": 12, "n_flats_estimated": False, "bill": None,
        "roof_tag": "Kullanıcı girdisi", "pitched_type": pitched, "photo_applies": True, "building": None,
    }


def hesapla(inputs):
    """What 'Hesapla' stores: flat + every unique face aspect (hip and gable) in one parallel fetch."""
    return solar.fetch_roof_yields(inputs["lat"], inputs["lon"], app.pvgis_aspects(inputs))


@pytest.mark.parametrize("rotation", [-23.13, 66.87, 0.4, -0.4, 89.6, 90.4, 0.0, 180.0])
def test_every_face_gets_a_yield_and_the_80_percent_rule_decides(pvgis_http, rotation):
    fp = footprint(rotation)
    analysis = hesapla(inputs_for(fp, "hip"))
    assert len(pvgis_http) == 6  # flat ideal + flat terrain + 4 unique face aspects
    assert all(v is not None for v in analysis["face_yields"].values())
    for pitched, n_faces in (("hip", 4), ("gable", 2)):
        result, _, _ = app.evaluate(analysis, inputs_for(fp, pitched), None, set(), None)
        faces = [s for s in result["senaryolar"][0]["segmentler"] if s["tur"] == "face"]
        assert len(faces) == n_faces
        for seg in faces:
            assert seg["durum"] in ("kullanildi", "elendi"), (pitched, seg["ad"], seg["durum"])
            expected = "kullanildi" if face_factor(seg["aspect"]) >= 0.8 else "elendi"
            assert seg["durum"] == expected and (seg["kwp"] > 0) == (expected == "kullanildi")
        assert "Bir çatı yüzü için güneş verisi alınamadı" not in result["uyarilar"]
    assert len(pvgis_http) == 6  # switching hip <-> gable made no new call


def test_repro_building_hip_and_gable_use_the_south_west_face(pvgis_http):
    fp = footprint(-23.13)
    analysis = hesapla(inputs_for(fp, "hip"))
    hip, _, _ = app.evaluate(analysis, inputs_for(fp, "hip"), None, set(), None)
    gable, _, _ = app.evaluate(analysis, inputs_for(fp, "gable"), None, set(), None)
    status = lambda r: {s["ad"]: s["durum"] for s in r["senaryolar"][0]["segmentler"] if s["tur"] == "face"}
    assert status(hip) == {
        "Kuzeydoğu yüzü": "elendi", "Güneybatı yüzü": "kullanildi", "Kuzeybatı yüzü": "elendi", "Güneydoğu yüzü": "kullanildi",
    }
    assert status(gable) == {"Kuzeydoğu yüzü": "elendi", "Güneybatı yüzü": "kullanildi"}
    assert gable["kurulabilir_guc_kwp_aralik"][0] > 0 and hip["kurulabilir_guc_kwp_aralik"][0] > 0
    sw = {s["aspect"] for s in hip["senaryolar"][0]["segmentler"] if s["ad"] == "Güneybatı yüzü"}
    assert sw == {23} and analysis["face_yields"][23]["e_y"] == pytest.approx(FLAT_E_Y * face_factor(23))


def test_aspects_near_plus_minus_180_and_0_map_to_one_request_each(pvgis_http):
    fp = footprint(-0.4)  # faces at 0.4 and 180.4 degrees: PVGIS aspects -179.6 -> 180 and 0.4 -> 0
    inputs = inputs_for(fp, "gable")
    assert app.pvgis_aspects(inputs) == [-90, 0, 90, 180]
    analysis = hesapla(inputs)
    assert sorted(c["aspect"] for c in pvgis_http if c["optimalangles"] == 0 and c["angle"] == 30) == [-90, 0, 90, 180]
    result, _, _ = app.evaluate(analysis, inputs, None, set(), None)
    assert {s["aspect"]: s["durum"] for s in result["senaryolar"][0]["segmentler"] if s["tur"] == "face"} == {
        180: "elendi", 0: "kullanildi",
    }


def test_a_failed_face_is_logged_with_its_key_and_listed_not_hidden(pvgis_http, monkeypatch, caplog):
    real = solar.requests.get

    def flaky(url, params=None, **kw):
        if params["optimalangles"] == 0 and params["angle"] == 30 and params["aspect"] == 23:
            return type("R", (), {"status_code": 400, "json": lambda self: {"message": "bad"}})()
        return real(url, params=params, **kw)

    monkeypatch.setattr(solar.requests, "get", flaky)
    fp = footprint(-23.13)
    with caplog.at_level(logging.WARNING, logger="core.solar"):
        analysis = hesapla(inputs_for(fp, "hip"))
    assert analysis["face_yields"][23] is None
    assert any("aspect=23" in r.getMessage() and "SolarDataError" in r.getMessage() for r in caplog.records)
    result, _, _ = app.evaluate(analysis, inputs_for(fp, "hip"), None, set(), None)
    lines = app.face_status_lines(result)
    assert "- Güneybatı yüzü: güneş verisi alınamadı" in lines and len(lines) == 4  # every face listed


def test_face_status_lines_cover_every_face_with_its_status(pvgis_http):
    fp = footprint(-23.13)
    analysis = hesapla(inputs_for(fp, "hip"))
    inputs = {**inputs_for(fp, "hip"), "structure": "unknown"}
    result, _, _ = app.evaluate(analysis, inputs, None, set(), None)
    lines = app.face_status_lines(result)
    assert any(ln.startswith("- Tamamen düz · Düz teras: kullanıldı") for ln in lines)
    for name in ("Kuzeydoğu", "Güneybatı", "Kuzeybatı", "Güneydoğu"):
        assert sum(f"Tamamen eğimli · {name} yüzü: " in ln for ln in lines) == 1
    assert any("Kuzeybatı yüzü: elendi (düz bölüm veriminin %80'inin altında) (verim oranı %" in ln for ln in lines)


def test_zero_capacity_has_one_warning_and_no_chart_label(pvgis_http, monkeypatch):
    monkeypatch.setattr(__import__(__name__), "face_factor", lambda aspect: 0.5)  # every face below 80 %
    fp = footprint(-23.13)
    analysis = hesapla(inputs_for(fp, "gable"))
    result, _, _ = app.evaluate(analysis, inputs_for(fp, "gable"), None, set(), None)
    assert result["kurulabilir_guc_kwp_aralik"] == [0, 0] and not app.has_capacity(result)
    assert app.zero_capacity_warning(result) == app.ZERO_PITCHED_WARNING
    assert app.building_mode.chart_rows(result, app.MONTHS_TR) == []  # no "Düz teras" series


def test_flat_roof_lists_only_the_flat_section(pvgis_http):
    fp = footprint(-23.13)
    analysis = hesapla(inputs_for(fp, "hip"))
    result, _, _ = app.evaluate(analysis, {**inputs_for(fp, "hip"), "structure": "flat"}, None, set(), None)
    lines = app.face_status_lines(result)
    assert len(lines) == 1 and "Düz teras" in lines[0] and "yüzü" not in lines[0]
