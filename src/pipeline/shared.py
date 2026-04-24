"""Shared logging setup for split pipeline modules."""

import logging
import os

from src.logger import get_logger

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog"

LOG_FILE_ENV_KEY = "RAKG_LOGGER_FILE"
DEFAULT_LOGGER_FILE = "Agent.log"

DEBUG_FILE_ENV_KEY = "RAKG_DEBUG_FILE"
DEFAULT_DEBUG_FILE = "Debug.log"
DEBUG_CONSOLE_ENV_KEY = "RAKG_DEBUG_TO_CONSOLE"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

logger = get_logger(
    name=os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME),
    level=logging.INFO,
    log_file=os.getenv(LOG_FILE_ENV_KEY, DEFAULT_LOGGER_FILE),
)

debug_logger = get_logger(
    name=f"{os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME)}.debug",
    level=logging.DEBUG,
    log_file=os.getenv(DEBUG_FILE_ENV_KEY, DEFAULT_DEBUG_FILE),
    console_output=_env_flag(DEBUG_CONSOLE_ENV_KEY, False),
)
