import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import vision

FIXTURES = Path(__file__).parent / "fixtures"


class FakeGemini:
    """Stands in for genai.Client: pops one scripted output (JSON text or exception) per call."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls += 1
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(text=out if isinstance(out, str) else json.dumps(out))


@pytest.fixture
def roof10_json():
    return json.loads((FIXTURES / "gemini_roof_10.json").read_text(encoding="utf-8"))


@pytest.fixture
def fake_gemini(monkeypatch):
    """Install a scripted fake Gemini client; returns install(outputs) -> FakeGemini. Sleeps are recorded."""
    vision.clear_cache()
    sleeps = []
    monkeypatch.setattr(vision, "_sleep", sleeps.append)

    def install(outputs):
        client = FakeGemini(outputs)
        monkeypatch.setattr(vision, "_get_client", lambda api_key=None: client)
        client.sleeps = sleeps
        return client

    yield install
    vision.clear_cache()


@pytest.fixture
def fake_explain(monkeypatch):
    """Install a scripted fake Gemini client for core.explain; returns install(outputs) -> FakeGemini."""
    from core import explain

    explain.clear_cache()

    def install(outputs):
        client = FakeGemini(outputs)
        monkeypatch.setattr(explain, "_get_client", lambda api_key=None: client)
        return client

    yield install
    explain.clear_cache()
