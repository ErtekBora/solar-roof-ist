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
    assert config.DEFAULT_HEIGHT_M == 15
    assert config.LEVEL_HEIGHT_M == 3
    assert config.PVGIS_LOSS == 14
    assert config.MAP_CENTER == (41.015, 28.979)
    assert config.MAP_ZOOM == 12


def test_missing_key_raises_friendly_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # make st.secrets unavailable: any access raises, as on a machine with no secrets file
    import streamlit as st

    class NoSecrets:
        def __getitem__(self, name):
            raise FileNotFoundError("no secrets file")

    monkeypatch.setattr(st, "secrets", NoSecrets())
    with pytest.raises(config.MissingApiKeyError) as exc:
        config.get_gemini_api_key()
    assert "GEMINI_API_KEY" in str(exc.value)
    assert "bulunamadı" in str(exc.value)


def test_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    assert config.get_gemini_api_key() == "dummy"


def test_yolo_url_none_when_unset(monkeypatch):
    monkeypatch.delenv("YOLO_URL", raising=False)
    import streamlit as st

    class NoSecrets:
        def __getitem__(self, name):
            raise FileNotFoundError("no secrets file")

    monkeypatch.setattr(st, "secrets", NoSecrets())
    assert config.get_yolo_url() is None
