from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import patch

import pytest


@pytest.fixture
def image_tool(monkeypatch):
    captured = {}

    class FakeImage:
        def __init__(self, url=None, b64_json=None):
            self.url = url
            self.b64_json = b64_json

    class FakeImages:
        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return types.SimpleNamespace(data=[FakeImage(url="https://example.com/generated.png")])

    class FakeOpenAI:
        def __init__(self, api_key, base_url, **kwargs):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["client_kwargs"] = kwargs
            self.images = FakeImages()

        def close(self):
            captured["close_calls"] = captured.get("close_calls", 0) + 1

    fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    import tools.image_generation_tool as mod
    mod = importlib.reload(mod)
    mod._test_capture = captured
    return mod


def test_openai_compatible_backend_uses_custom_base_url_and_model(image_tool, monkeypatch):
    monkeypatch.setenv("GPT2API_IMAGE_API_KEY", "sk-test")

    with patch(
        "hermes_cli.config.load_config",
        return_value={
            "image_gen": {
                "backend": "openai-compatible",
                "model": "gpt-image-2",
                "base_url": "https://gpt2api.example.com/v1",
                "api_key_env": "GPT2API_IMAGE_API_KEY",
            }
        },
    ):
        result = image_tool.image_generate_tool("draw a black cat", aspect_ratio="square")

    data = image_tool.json.loads(result)
    assert data["success"] is True
    assert data["image"] == "https://example.com/generated.png"
    assert image_tool._test_capture["api_key"] == "sk-test"
    assert image_tool._test_capture["base_url"] == "https://gpt2api.example.com/v1"
    assert image_tool._test_capture["generate_kwargs"]["model"] == "gpt-image-2"
    assert image_tool._test_capture["generate_kwargs"]["prompt"] == "draw a black cat"
    assert image_tool._test_capture["generate_kwargs"]["size"] == "1024x1024"
    assert image_tool._test_capture["generate_kwargs"]["n"] == 1
    assert image_tool._test_capture["close_calls"] == 1


def test_openai_compatible_backend_requires_api_key(image_tool, monkeypatch):
    monkeypatch.delenv("GPT2API_IMAGE_API_KEY", raising=False)

    with patch(
        "hermes_cli.config.load_config",
        return_value={
            "image_gen": {
                "backend": "openai-compatible",
                "model": "gpt-image-2",
                "base_url": "https://gpt2api.example.com/v1",
                "api_key_env": "GPT2API_IMAGE_API_KEY",
            }
        },
    ):
        result = image_tool.image_generate_tool("draw a black cat")

    data = image_tool.json.loads(result)
    assert data["success"] is False
    assert "GPT2API_IMAGE_API_KEY" in data["error"]


def test_openai_compatible_check_requirements_without_fal(image_tool, monkeypatch):
    monkeypatch.setenv("GPT2API_IMAGE_API_KEY", "sk-test")

    with patch(
        "hermes_cli.config.load_config",
        return_value={
            "image_gen": {
                "backend": "openai-compatible",
                "model": "gpt-image-2",
                "base_url": "https://gpt2api.example.com/v1",
                "api_key_env": "GPT2API_IMAGE_API_KEY",
            }
        },
    ):
        assert image_tool.check_image_generation_requirements() is True


def test_openai_compatible_aspect_ratio_maps_to_size(image_tool, monkeypatch):
    monkeypatch.setenv("GPT2API_IMAGE_API_KEY", "sk-test")

    with patch(
        "hermes_cli.config.load_config",
        return_value={
            "image_gen": {
                "backend": "openai-compatible",
                "model": "gpt-image-2",
                "base_url": "https://gpt2api.example.com/v1",
                "api_key_env": "GPT2API_IMAGE_API_KEY",
            }
        },
    ):
        image_tool.image_generate_tool("draw a black cat", aspect_ratio="portrait")
        assert image_tool._test_capture["generate_kwargs"]["size"] == "1024x1536"
