from pathlib import Path

from src.rag_project.config import AppSettings
from src.rag_project.startup import _ollama_tags_url, run_startup_checks


def settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        data_dir=tmp_path / "data",
        chroma_path=tmp_path / "data" / "chroma",
        history_path=tmp_path / "data" / "history.db",
        model_cache=tmp_path / "models",
        embedding_model="test-model",
        embedding_local_only=True,
        ollama_chat_url="http://ollama:11434/api/chat",
        ollama_model="qwen3:8b",
        max_upload_mb=200,
    )


def test_ollama_health_url_uses_configured_host() -> None:
    assert _ollama_tags_url("http://ollama:11434/api/chat") == (
        "http://ollama:11434/api/tags"
    )


def test_startup_reports_writable_storage_even_when_services_are_missing(
    tmp_path: Path,
) -> None:
    checks = run_startup_checks(settings(tmp_path))
    by_name = {check.name: check for check in checks}

    assert by_name["Storage"].ok is True
    assert by_name["Ollama"].ok is False
    assert by_name["Tesseract OCR"].required is False
