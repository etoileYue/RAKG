from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Web backend runtime settings loaded from environment variables."""

    app_name: str = os.getenv("RAKG_WEB_APP_NAME", "RAKG Web API")
    app_env: str = os.getenv("RAKG_WEB_ENV", "dev")
    api_prefix: str = "/api/v1"
    db_path: str = os.getenv("RAKG_WEB_DB_PATH", "src/web/backend/web_tasks.db")
    queue_poll_interval_sec: float = float(os.getenv("RAKG_WEB_QUEUE_POLL_SEC", "0.2"))
    cors_origins: tuple[str, ...] = tuple(
        item.strip()
        for item in os.getenv(
            "RAKG_WEB_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if item.strip()
    )
    cors_origin_regex: str = os.getenv(
        "RAKG_WEB_CORS_ORIGIN_REGEX",
        r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    )


SETTINGS = Settings()

# Repository root: /<repo>/src/web/backend/app/config.py -> parents[4]
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TASK_OUTPUT_ROOT = REPO_ROOT / "data" / "web"
LOG_ROOT = REPO_ROOT / "logs"
