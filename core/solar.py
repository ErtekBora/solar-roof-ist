"""PVGIS PVcalc access with in-module caching and retries.

PVGIS non-interactive API docs:
https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en

Aspect convention: 0 = South, 90 = West, -90 = East. Server-side only; no Streamlit imports.
"""
import copy
import math
import time

import requests

from core.config import DEFAULT_ROOF_PITCH_DEG, PVGIS_LOSS

PVCALC_URL = "https://re.jrc.ec.europa.eu/api/v5_3/PVcalc"
USER_AGENT = "istanbul-rooftop-solar/0.1 (GreenTech COP31 student project)"
TIMEOUT_S = 30
RETRY_STATUSES = (429, 529)
BACKOFF_S = (1, 2, 4)
MAX_REQUESTS_PER_S = 30

_cache = {}
_last_request_at = 0.0
_sleep = time.sleep  # retry backoff; patched in tests
_throttle_sleep = time.sleep  # rate limiter; patched in tests


class SolarDataError(RuntimeError):
    """PVGIS data could not be obtained; the message is Turkish and UI-safe."""


def clear_cache():
    """Drop all cached yields."""
    _cache.clear()


def _validate_userhorizon(values):
    if len(values) != 72:
        raise ValueError(f"userhorizon must have exactly 72 values, got {len(values)}")
    for x in values:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(f"userhorizon values must be numbers, got {x!r}")
        if not math.isfinite(x) or not 0 <= x <= 90:
            raise ValueError(f"userhorizon values must be finite and within 0..90, got {x!r}")


def _throttle():
    """Keep at most MAX_REQUESTS_PER_S requests per second."""
    global _last_request_at
    wait = _last_request_at + 1.0 / MAX_REQUESTS_PER_S - time.monotonic()
    if wait > 0:
        _throttle_sleep(wait)
    _last_request_at = time.monotonic()


def _parse_pvcalc(data):
    """Extract the clean result dict from a PVcalc JSON response."""
    try:
        totals = data["outputs"]["totals"]["fixed"]
        monthly = sorted(data["outputs"]["monthly"]["fixed"], key=lambda r: r["month"])
        e_m = [float(r["E_m"]) for r in monthly]
        mounting = data["inputs"]["mounting_system"]["fixed"]
        result = {
            "e_y": float(totals["E_y"]),
            "e_m": e_m,
            "h_i_y": float(totals["H(i)_y"]),
            "l_tg": float(totals["l_tg"]),
            "l_total": float(totals["l_total"]),
            "angle": mounting["slope"]["value"],
            "aspect": mounting["azimuth"]["value"],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise SolarDataError("PVGIS yanıtı beklenen biçimde değil. Lütfen daha sonra tekrar deneyin.") from exc
    if len(e_m) != 12:
        raise SolarDataError("PVGIS yanıtında 12 aylık üretim verisi bulunamadı.")
    return result


def fetch_pvcalc(lat, lon, *, optimal=False, angle=None, aspect=None, usehorizon=True, userhorizon=None):
    """Call PVcalc once and return {e_y, e_m, h_i_y, l_tg, l_total, angle, aspect}.

    With optimal=True PVGIS ignores angle/aspect and returns the optimal ones.
    """
    if userhorizon is not None:
        userhorizon = list(userhorizon)
        _validate_userhorizon(userhorizon)
    params = {
        "lat": lat,
        "lon": lon,
        "peakpower": 1,
        "loss": PVGIS_LOSS,
        "outputformat": "json",
        "usehorizon": 1 if usehorizon else 0,
    }
    if optimal:
        params["optimalangles"] = 1
    else:
        params["optimalangles"] = 0
        params["angle"] = angle
        params["aspect"] = aspect
    if userhorizon is not None:
        params["userhorizon"] = ",".join(f"{x:g}" for x in userhorizon)

    resp = None
    for attempt in range(len(BACKOFF_S) + 1):
        _throttle()
        try:
            resp = requests.get(PVCALC_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
        except requests.RequestException as exc:
            raise SolarDataError("PVGIS sunucusuna ulaşılamadı. İnternet bağlantınızı kontrol edip tekrar deneyin.") from exc
        if resp.status_code not in RETRY_STATUSES:
            break
        if attempt < len(BACKOFF_S):
            _sleep(BACKOFF_S[attempt])
    if resp.status_code in RETRY_STATUSES:
        raise SolarDataError("PVGIS şu an çok yoğun. Lütfen birkaç saniye sonra tekrar deneyin.")
    if resp.status_code != 200:
        detail = ""
        try:
            detail = resp.json().get("message", "")
        except (ValueError, AttributeError):
            pass
        raise SolarDataError(f"PVGIS isteği başarısız oldu (HTTP {resp.status_code}). {detail}".strip())
    try:
        data = resp.json()
    except ValueError as exc:
        raise SolarDataError("PVGIS yanıtı okunamadı. Lütfen daha sonra tekrar deneyin.") from exc
    return _parse_pvcalc(data)


def get_solar_yield(lat, lon, horizon72=None):
    """Return {ideal, shaded, angle, aspect}.

    ideal: optimal angles, no horizon. shaded: same angles with the PVGIS terrain horizon
    (Demo: terrain only), or with userhorizon=horizon72 when given.
    """
    if horizon72 is not None:
        horizon72 = tuple(horizon72)
        _validate_userhorizon(horizon72)
    key = (round(lat, 4), round(lon, 4), horizon72)
    if key not in _cache:
        ideal = fetch_pvcalc(lat, lon, optimal=True, usehorizon=False)
        shaded = fetch_pvcalc(
            lat, lon, angle=ideal["angle"], aspect=ideal["aspect"], usehorizon=True, userhorizon=horizon72
        )
        _cache[key] = {"ideal": ideal, "shaded": shaded, "angle": ideal["angle"], "aspect": ideal["aspect"]}
    return copy.deepcopy(_cache[key])


def get_face_yield(lat, lon, aspect, angle=DEFAULT_ROOF_PITCH_DEG, horizon72=None):
    """One cached PVcalc run for a pitched roof face: fixed `angle` and `aspect`, optimalangles=0, usehorizon=1.

    aspect uses the PVGIS convention (0 = South, 90 = West, -90 = East). Raises SolarDataError after the retries.
    """
    if horizon72 is not None:
        horizon72 = tuple(horizon72)
        _validate_userhorizon(horizon72)
    key = ("face", round(lat, 4), round(lon, 4), angle, aspect, horizon72)
    if key not in _cache:
        _cache[key] = fetch_pvcalc(lat, lon, angle=angle, aspect=aspect, usehorizon=True, userhorizon=horizon72)
    return copy.deepcopy(_cache[key])


def get_terrain_horizon(lat, lon):
    """PVGIS printhorizon resampled to 72 North-based values (Prototype stage)."""
    raise NotImplementedError
