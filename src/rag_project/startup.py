"""Friendly startup checks for local and container deployments."""

from dataclasses import dataclass
import json
import shutil
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from src.rag_project.config import AppSettings, SETTINGS
from src.rag_project.ocr import find_tesseract_command, OCRUnavailableError


@dataclass(frozen=True)
class StartupCheck:
    """One dependency check displayed by the application."""

    name: str
    ok: bool
    message: str
    required: bool = True


def _ollama_tags_url(chat_url: str) -> str:
    parts = urlsplit(chat_url)
    return urlunsplit((parts.scheme, parts.netloc, "/api/tags", "", ""))


def run_startup_checks(settings: AppSettings = SETTINGS) -> list[StartupCheck]:
    """Check storage, Ollama, OCR, and required Python tools."""
    checks: list[StartupCheck] = []
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.data_dir / ".write-check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(StartupCheck("Storage", True, str(settings.data_dir)))
    except OSError as error:
        checks.append(StartupCheck("Storage", False, f"Not writable: {error}"))

    try:
        with urlopen(_ollama_tags_url(settings.ollama_chat_url), timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        installed = {
            str(model.get("name", ""))
            for model in payload.get("models", [])
            if isinstance(model, dict)
        }
        ok = settings.ollama_model in installed
        checks.append(
            StartupCheck(
                "Ollama",
                ok,
                (
                    f"Connected; {settings.ollama_model} is installed."
                    if ok
                    else f"Connected, but {settings.ollama_model} is missing. Run: "
                    f"ollama pull {settings.ollama_model}"
                ),
            )
        )
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        checks.append(
            StartupCheck(
                "Ollama",
                False,
                "Ollama is unavailable. Start it and install the configured model: "
                f"ollama pull {settings.ollama_model}",
            )
        )

    try:
        command = find_tesseract_command()
        checks.append(StartupCheck("Tesseract OCR", True, command, required=False))
    except OCRUnavailableError:
        checks.append(
            StartupCheck(
                "Tesseract OCR",
                False,
                "Optional: install Tesseract to read scanned PDFs.",
                required=False,
            )
        )

    checks.append(
        StartupCheck(
            "Python environment",
            shutil.which("python") is not None,
            "Python runtime is available.",
        )
    )
    return checks
