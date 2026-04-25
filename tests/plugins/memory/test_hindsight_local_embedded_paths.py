"""Regression tests for local_embedded Hindsight daemon profile paths."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from plugins.memory.hindsight import HindsightMemoryProvider


class _InlineThread:
    def __init__(self, target=None, *args, **kwargs):
        self._target = target

    def start(self):
        if self._target:
            self._target()


class _FakeEmbedded:
    instances = []

    def __init__(self, **kwargs):
        self._manager = MagicMock()
        self._manager.is_running.return_value = False
        self._manager.stop.return_value = True
        self._ensure_started = MagicMock()
        _FakeEmbedded.instances.append(self)


def test_local_embedded_writes_profile_under_hermes_home(tmp_path, monkeypatch):
    """Running as root must not write /root/.hindsight profile env for daemon."""
    config = {
        "mode": "local_embedded",
        "profile": "hermes",
        "llm_provider": "openrouter",
        "llm_api_key": "test-key",
        "llm_model": "test-model",
        "llm_base_url": "http://llm.example/v1",
    }
    hermes_home = tmp_path / "hermes-home"
    config_path = hermes_home / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config))
    user_home = tmp_path / "home" / "hermes"
    user_home.mkdir(parents=True)

    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: hermes_home)
    monkeypatch.delenv("HINDSIGHT_HOME", raising=False)
    monkeypatch.setattr("plugins.memory.hindsight.threading.Thread", _InlineThread)
    _FakeEmbedded.instances.clear()

    with patch("plugins.memory.hindsight.os.getuid", return_value=0), \
         patch("pwd.getpwnam", return_value=SimpleNamespace(pw_dir=str(user_home))), \
         patch("plugins.memory.hindsight.HindsightEmbedded", _FakeEmbedded, create=True):
        provider = HindsightMemoryProvider()
        provider.initialize(session_id="test-session")

    profile_env = user_home / ".hindsight" / "profiles" / "hermes.env"
    assert profile_env.exists()
    content = profile_env.read_text()
    assert "HINDSIGHT_API_LLM_PROVIDER=openai" in content
    assert "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=3600" in content
    assert not (tmp_path / "root" / ".hindsight" / "profiles" / "hermes.env").exists()
    assert _FakeEmbedded.instances[-1]._ensure_started.called
