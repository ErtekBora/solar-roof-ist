import json
from pathlib import Path

import pytest
import requests

from core import solar
from core.solar import SolarDataError

FIXTURES = Path(__file__).parent / "fixtures"
OPTIMAL = json.loads((FIXTURES / "pvgis_optimal.json").read_text(encoding="utf-8"))
TERRAIN = json.loads((FIXTURES / "pvgis_terrain.json").read_text(encoding="utf-8"))
LAT, LON = 41.015, 28.979


class FakeResponse:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code = status
        self._payload = payload
        self._bad_json = bad_json

    def json(self):
        if self._bad_json or self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    solar.clear_cache()
    monkeypatch.setattr(solar, "_throttle_sleep", lambda s: None)
    monkeypatch.setattr(solar, "_sleep", lambda s: None)
    yield
    solar.clear_cache()


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(solar, "_sleep", recorded.append)
    return recorded


@pytest.fixture
def http(monkeypatch):
    """Mock requests.get; the queue holds responses (or exceptions) returned in order."""
    calls = []
    queue = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(solar.requests, "get", fake_get)
    return type("Http", (), {"calls": calls, "queue": queue})


def test_parse_optimal_fixture():
    r = solar._parse_pvcalc(OPTIMAL)
    assert r["e_y"] == pytest.approx(1372.68)
    assert len(r["e_m"]) == 12
    assert r["e_m"][0] == pytest.approx(64.71)
    assert r["h_i_y"] == pytest.approx(1742.57)
    assert r["l_tg"] == pytest.approx(-6.48)
    assert r["l_total"] == pytest.approx(-21.23)
    assert r["angle"] == 33
    assert r["aspect"] == 8


def test_parse_terrain_fixture():
    r = solar._parse_pvcalc(TERRAIN)
    assert r["e_y"] == pytest.approx(1369.8)
    assert (r["angle"], r["aspect"]) == (33, 8)


@pytest.mark.parametrize("mutate", [
    lambda d: d["outputs"]["totals"]["fixed"].pop("E_y"),
    lambda d: d["outputs"]["monthly"]["fixed"].pop(),
    lambda d: d["inputs"].pop("mounting_system"),
])
def test_parse_bad_payload_raises(mutate):
    data = json.loads(json.dumps(OPTIMAL))
    mutate(data)
    with pytest.raises(SolarDataError):
        solar._parse_pvcalc(data)


def test_fetch_optimal_request_params(http):
    http.queue.append(FakeResponse(200, OPTIMAL))
    solar.fetch_pvcalc(LAT, LON, optimal=True, usehorizon=False)
    call = http.calls[0]
    assert call["timeout"] == 30
    assert call["headers"]["User-Agent"]
    p = call["params"]
    assert (p["optimalangles"], p["usehorizon"], p["peakpower"], p["loss"]) == (1, 0, 1, 14)
    assert p["outputformat"] == "json"
    assert "userhorizon" not in p and "angle" not in p


def test_fetch_fixed_angle_with_userhorizon(http):
    http.queue.append(FakeResponse(200, TERRAIN))
    solar.fetch_pvcalc(LAT, LON, angle=33, aspect=8, usehorizon=True, userhorizon=[10] * 72)
    p = http.calls[0]["params"]
    assert (p["optimalangles"], p["angle"], p["aspect"], p["usehorizon"]) == (0, 33, 8, 1)
    assert p["userhorizon"] == ",".join(["10"] * 72)


@pytest.mark.parametrize("bad", [
    [10] * 71,
    [10] * 73,
    [10] * 71 + [float("nan")],
    [10] * 71 + [float("inf")],
    [10] * 71 + [None],
    [10] * 71 + [-5],
    [10] * 71 + [95],
    [10] * 71 + [True],
    [10] * 71 + ["10"],
])
def test_userhorizon_validation(http, bad):
    with pytest.raises(ValueError):
        solar.fetch_pvcalc(LAT, LON, angle=33, aspect=8, userhorizon=bad)
    assert http.calls == []


def test_userhorizon_boundaries_accepted(http):
    http.queue.append(FakeResponse(200, TERRAIN))
    solar.fetch_pvcalc(LAT, LON, angle=33, aspect=8, userhorizon=[0] * 36 + [90] * 36)


@pytest.mark.parametrize("status", [429, 529])
def test_retry_then_success(http, sleeps, status):
    http.queue.extend([FakeResponse(status), FakeResponse(status), FakeResponse(200, OPTIMAL)])
    result = solar.fetch_pvcalc(LAT, LON, optimal=True)
    assert result["e_y"] == pytest.approx(1372.68)
    assert len(http.calls) == 3
    assert sleeps == [1, 2]


def test_retry_exhausted_raises(http, sleeps):
    http.queue.extend([FakeResponse(429)] * 4)
    with pytest.raises(SolarDataError) as exc:
        solar.fetch_pvcalc(LAT, LON, optimal=True)
    assert len(http.calls) == 4
    assert sleeps == [1, 2, 4]
    assert "yoğun" in str(exc.value)


def test_http_400_uses_message(http):
    http.queue.append(FakeResponse(400, {"message": "lat: Incorrect value.", "status": 400}))
    with pytest.raises(SolarDataError) as exc:
        solar.fetch_pvcalc(999, LON, optimal=True)
    assert "400" in str(exc.value) and "lat: Incorrect value." in str(exc.value)
    assert len(http.calls) == 1


def test_timeout_raises_solar_error(http):
    http.queue.append(requests.Timeout())
    with pytest.raises(SolarDataError):
        solar.fetch_pvcalc(LAT, LON, optimal=True)


def test_bad_json_raises_solar_error(http):
    http.queue.append(FakeResponse(200, bad_json=True))
    with pytest.raises(SolarDataError):
        solar.fetch_pvcalc(LAT, LON, optimal=True)


def test_get_solar_yield_reuses_optimal_angles(http):
    http.queue.extend([FakeResponse(200, OPTIMAL), FakeResponse(200, TERRAIN)])
    result = solar.get_solar_yield(LAT, LON)
    assert set(result) == {"ideal", "shaded", "angle", "aspect"}
    assert (result["angle"], result["aspect"]) == (33, 8)
    assert result["ideal"]["e_y"] == pytest.approx(1372.68)
    assert result["shaded"]["e_y"] == pytest.approx(1369.8)
    first, second = (c["params"] for c in http.calls)
    assert (first["optimalangles"], first["usehorizon"]) == (1, 0)
    assert (second["optimalangles"], second["usehorizon"]) == (0, 1)
    assert (second["angle"], second["aspect"]) == (33, 8)
    assert "userhorizon" not in second


def test_get_solar_yield_passes_horizon(http):
    http.queue.extend([FakeResponse(200, OPTIMAL), FakeResponse(200, TERRAIN)])
    solar.get_solar_yield(LAT, LON, horizon72=[5.0] * 72)
    assert http.calls[1]["params"]["userhorizon"] == ",".join(["5"] * 72)


def test_get_solar_yield_validates_horizon_before_network(http):
    with pytest.raises(ValueError):
        solar.get_solar_yield(LAT, LON, horizon72=[5.0] * 10)
    assert http.calls == []


def test_get_solar_yield_cache(http):
    http.queue.extend([FakeResponse(200, OPTIMAL), FakeResponse(200, TERRAIN)])
    first = solar.get_solar_yield(LAT, LON)
    second = solar.get_solar_yield(LAT + 0.00001, LON)  # same after rounding to 4 decimals
    assert len(http.calls) == 2
    assert first == second
    second["ideal"]["e_y"] = 0  # returned copies must not corrupt the cache
    assert solar.get_solar_yield(LAT, LON)["ideal"]["e_y"] == pytest.approx(1372.68)


def test_cache_key_includes_horizon(http):
    http.queue.extend([FakeResponse(200, OPTIMAL), FakeResponse(200, TERRAIN)] * 2)
    solar.get_solar_yield(LAT, LON)
    solar.get_solar_yield(LAT, LON, horizon72=[5.0] * 72)
    assert len(http.calls) == 4


def test_clear_cache(http):
    http.queue.extend([FakeResponse(200, OPTIMAL), FakeResponse(200, TERRAIN)] * 2)
    solar.get_solar_yield(LAT, LON)
    solar.clear_cache()
    solar.get_solar_yield(LAT, LON)
    assert len(http.calls) == 4


def test_failed_run_is_not_cached(http):
    http.queue.extend([FakeResponse(500), FakeResponse(200, OPTIMAL), FakeResponse(200, TERRAIN)])
    with pytest.raises(SolarDataError):
        solar.get_solar_yield(LAT, LON)
    assert solar.get_solar_yield(LAT, LON)["angle"] == 33


def test_terrain_horizon_is_stub():
    with pytest.raises(NotImplementedError):
        solar.get_terrain_horizon(LAT, LON)
