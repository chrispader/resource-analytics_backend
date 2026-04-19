"""Application settings loaded from environment and optional `.env` in the project root."""

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

_DEFAULT_API_URL = "http://localhost:9090"


def _read_api_url() -> str:
    raw = os.environ.get("API_URL", "").strip()
    if raw:
        return raw
    return _DEFAULT_API_URL


API_URL = _read_api_url()
