"""Central configuration: constants, model names and secret/URL accessors.

All numbers used by calculations live here; they never come from an LLM.
"""
import os
from pathlib import Path

CO2_FACTOR_T_PER_MWH = 0.6345  # ETKB TUESEF 2021, tCO2/MWh
USABLE_RATIO_NO_PHOTO = 0.276  # Ember
EDGE_FACTOR = 0.85
M2_PER_KWP_FLAT = 6.5
M2_PER_KWP_PITCHED = 5.5
LEVEL_HEIGHT_M = 3
FALLBACK_HEIGHT_M = 15.0  # only for building files with no known height at all (e.g. a tiny fixture)
SETBACK_M = 1.0  # assumption: edge strip left free around the footprint
PVGIS_LOSS = 14
MAP_CENTER = (40.9720, 29.0760)  # Prototype coverage: Erenköy / Suadiye / Göztepe
MAP_ZOOM = 15
BUILDING_ZOOM = 18  # one-shot zoom right after a building is found

# Local Overture buildings extract (GeoParquet) enriched with an ML height estimate on Kaggle
BUILDINGS_PARQUET = Path(__file__).parent.parent / "data" / "buildings" / "erenkoy_suadiye_goztepe_ml.parquet"
BUILDINGS_REPORT_JSON = Path(__file__).parent.parent / "data" / "buildings" / "height_model_report.json"
BUILDINGS_BBOX = (29.050, 40.950, 29.100, 40.990)  # (min_lon, min_lat, max_lon, max_lat), WGS84
NEAREST_MAX_M = 15  # nearest footprint accepted when the click is not inside one
NEIGHBOUR_RADIUS_M = 150

# EPDK residential tariff, two tiers (consumption above the daily limit is billed at the HIGH tier)
TARIFF_SOURCE = "EPDK, AG tek terim mesken, yürürlük 2026-04-04"
TARIFF_LOW_TL_PER_KWH_EXCL_TAX = 2.918965  # 49.4065 + 242.4900 kr/kWh
TARIFF_HIGH_TL_PER_KWH_EXCL_TAX = 4.320708  # 189.5808 + 242.4900 kr/kWh
TIER_LIMIT_KWH_PER_DAY = 8  # monthly limit = 8 x 30 = 240 kWh for bill inversion
TAX_FACTOR = 1.26  # 5% BTV then 20% KDV, approximate

# Model IDs verified with client.models.list()
VISION_MODEL = "gemini-3.8-flash"
AGENT_MODEL = "gemini-3.8-flash"
EXPLAIN_MODEL = "gemini-3.5-flash-lite"

# Single obstacle class list (SPEC section 7); Gemini may additionally return "other"
OBSTACLE_CLASSES = [
    "water tank",
    "air conditioner unit",
    "chimney",
    "solar panel",
    "satellite dish",
    "skylight",
]
MAX_IMAGE_SIDE_PX = 1280
MAX_UPLOAD_MB = 5

_MISSING_KEY_MSG = (
    "Gemini API anahtarı bulunamadı. Lütfen GEMINI_API_KEY ortam değişkenini "
    "veya .streamlit/secrets.toml dosyasındaki GEMINI_API_KEY değerini ayarlayın."
)


class MissingApiKeyError(RuntimeError):
    """Raised when no Gemini API key is configured (message is Turkish, UI-safe)."""


_PLACEHOLDER = "your-key-here"


def _clean(value):
    """Return the stripped value, or None if it is empty or the example placeholder."""
    if value is None:
        return None
    value = str(value).strip()
    if not value or value == _PLACEHOLDER:
        return None
    return value


def _from_secrets(name):
    """Read a value from st.secrets; Streamlit raises when no secrets file exists."""
    try:
        import streamlit as st

        value = st.secrets[name]
    except Exception:
        return None
    return _clean(value)


def get_gemini_api_key() -> str:
    """Return the Gemini API key (env var first, then st.secrets). Never log it."""
    key = _clean(os.environ.get("GEMINI_API_KEY")) or _from_secrets("GEMINI_API_KEY")
    if not key:
        raise MissingApiKeyError(_MISSING_KEY_MSG)
    return key


def get_yolo_url():
    """Return the YOLO service base URL, or None when unset/invalid (YOLO is optional)."""
    url = _clean(os.environ.get("YOLO_URL")) or _from_secrets("YOLO_URL")
    return url.rstrip("/") if url else None
