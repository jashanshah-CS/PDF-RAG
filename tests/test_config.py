from pathlib import Path

import pytest

from src.rag_project.config import load_settings


def test_settings_support_container_paths_and_services(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "persistent-data"
    model_cache = tmp_path / "models"
    monkeypatch.setenv("RAG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MODEL_CACHE", str(model_cache))
    monkeypatch.setenv("OLLAMA_CHAT_URL", "http://ollama:11434/api/chat")
    monkeypatch.setenv("OLLAMA_MODEL", "test-model:latest")
    monkeypatch.setenv("MAX_UPLOAD_MB", "75")
    monkeypatch.setenv("EMBEDDING_LOCAL_ONLY", "true")

    settings = load_settings()

    assert settings.chroma_path == data_dir / "chroma-v2"
    assert settings.history_path == data_dir / "history.db"
    assert settings.model_cache == model_cache
    assert settings.ollama_chat_url == "http://ollama:11434/api/chat"
    assert settings.ollama_model == "test-model:latest"
    assert settings.max_upload_mb == 75
    assert settings.embedding_local_only is True


def test_settings_reject_invalid_upload_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")

    with pytest.raises(ValueError, match="greater than zero"):
        load_settings()


def test_settings_reject_invalid_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_LOCAL_ONLY", "sometimes")

    with pytest.raises(ValueError, match="true or false"):
        load_settings()
