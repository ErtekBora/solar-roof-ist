"""Central configuration: constants, model names and secret/URL accessors.

All numbers used by calculations live here; they never come from an LLM.
"""
import os

CO2_FACTOR_T_PER_MWH = 0.6345  # ETKB TUESEF 2021, tCO2/MWh
USABLE_RATIO_NO_PHOTO = 0.276  # Ember
EDGE_FACTOR = 0.85
M2_PER_KWP_FLAT = 6.5
M2_PER_KWP_PITCHED = 5.5
DEFAULT_HEIGHT_M = 15
LEVEL_HEIGHT_M = 3
PVGIS_LOSS = 14
MAP_CENTER = (41.015, 28.979)
MAP_ZOOM = 12

# TODO: set from EPDK before demo
TARIFF_TL_PER_KWH = 0.0

# TODO: verify current model IDs in the Gemini API docs
VISION_MODEL = "TODO-flash-model-id"
AGENT_MODEL = "TODO-flash-model-id"
EXPLAIN_MODEL = "TODO-flash-lite-model-id"

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
