import importlib

import pytest

from core import config

CORE_MODULES = [
    "config", "buildings", "horizon", "solar", "vision", "yolo_client",
    "fusion", "calc", "metrics", "explain", "agent",
]


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_module_imports(name):
    assert importlib.import_module(f"core.{name}")


def test_config_constants():
    assert config.CO2_FACTOR_T_PER_MWH == 0.6345
    assert config.USABLE_RATIO_NO_PHOTO == 0.276
    assert config.EDGE_FACTOR == 0.85
    assert config.M2_PER_KWP_FLAT == 6.5
    assert config.M2_PER_KWP_PITCHED == 5.5
    assert config.FALLBACK_HEIGHT_M == 15.0
    assert config.LEVEL_HEIGHT_M == 3
    assert config.PVGIS_LOSS == 14
    assert config.MAP_CENTER == (40.9720, 29.0760)
    assert config.MAP_ZOOM == 15


class _Secrets(dict):
    """Stand-in for st.secrets; an empty instance behaves like a missing secrets file."""

    def __getitem__(self, name):
        if name not in self:
            raise KeyError(name)
        return dict.__getitem__(self, name)


@pytest.fixture
def secrets(monkeypatch):
    """Isolate st.secrets and clear env vars so local files never affect tests."""
    import streamlit as st

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("YOLO_URL", raising=False)
    fake = _Secrets()
    monkeypatch.setattr(st, "secrets", fake)
    return fake


def test_key_from_env(monkeypatch, secrets):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert config.get_gemini_api_key() == "test-key"


def test_no_key_raises_friendly_error(secrets):
    with pytest.raises(config.MissingApiKeyError) as exc:
        config.get_gemini_api_key()
    assert "GEMINI_API_KEY" in str(exc.value)
    assert "bulunamadı" in str(exc.value)


@pytest.mark.parametrize("value", ["   ", "your-key-here"])
def test_invalid_env_key_raises_friendly_error(monkeypatch, secrets, value):
    monkeypatch.setenv("GEMINI_API_KEY", value)
    with pytest.raises(config.MissingApiKeyError) as exc:
        config.get_gemini_api_key()
    assert "bulunamadı" in str(exc.value)


def test_invalid_env_falls_back_to_secrets(monkeypatch, secrets):
    monkeypatch.setenv("GEMINI_API_KEY", "your-key-here")
    secrets["GEMINI_API_KEY"] = "secret-key"
    assert config.get_gemini_api_key() == "secret-key"


def test_yolo_url_none_when_missing(secrets):
    assert config.get_yolo_url() is None


def test_app_main_is_callable():
    import app

    assert callable(app.main)
