import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_blue_uses_liquid_model_and_credentials(monkeypatch):
    from core.config import blue_client_kwargs, get_blue_model, get_blue_provider

    monkeypatch.setenv("LIQUID_API_KEY", "test-liquid-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    assert get_blue_provider() == "liquid"
    assert get_blue_model() == "lfm3-1.7b"
    assert blue_client_kwargs() == {
        "api_key": "test-liquid-key",
        "base_url": "https://api.liquid.ai/openai/v1",
    }
