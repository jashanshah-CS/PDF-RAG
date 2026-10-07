"""Environment-backed application configuration with local-safe defaults."""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else default


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false.")


@dataclass(frozen=True)
class AppSettings:
    """Runtime settings shared by local and container deployments."""

    data_dir: Path
    chroma_path: Path
    history_path: Path
    model_cache: Path
    embedding_model: str
    embedding_local_only: bool
    ollama_chat_url: str
    ollama_model: str
    max_upload_mb: int


def load_settings() -> AppSettings:
    """Load validated settings from environment variables."""
    data_dir = _path("RAG_DATA_DIR", PROJECT_ROOT / "data")
    max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "200"))
    if max_upload_mb <= 0:
        raise ValueError("MAX_UPLOAD_MB must be greater than zero.")
    return AppSettings(
        data_dir=data_dir,
        chroma_path=_path("CHROMA_PATH", data_dir / "chroma-v2"),
        history_path=_path("HISTORY_PATH", data_dir / "history.db"),
        model_cache=_path("MODEL_CACHE", PROJECT_ROOT / ".model-cache"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
        embedding_local_only=_boolean("EMBEDDING_LOCAL_ONLY", False),
        ollama_chat_url=os.getenv(
            "OLLAMA_CHAT_URL", "http://localhost:11434/api/chat"
        ).strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b").strip(),
        max_upload_mb=max_upload_mb,
    )


SETTINGS = load_settings()
