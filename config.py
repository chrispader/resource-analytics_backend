"""Application settings loaded from environment and optional `.env` in the project root."""

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
_env_file = _PROJECT_ROOT / ".env"
if _env_file.is_file():
    load_dotenv(_env_file)

_DEFAULT_API_HOST = "localhost"
_DEFAULT_API_PORT = 9090


def _read_str(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw if raw else default


def _read_port(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


API_HOST = _read_str("API_HOST", _DEFAULT_API_HOST)
API_PORT = _read_port("API_PORT", _DEFAULT_API_PORT)


def _read_api_url() -> str:
    explicit = os.environ.get("API_URL", "").strip()
    if explicit:
        return explicit
    return f"http://{API_HOST}:{API_PORT}"


API_URL = _read_api_url()
