"""Shared logging setup for split pipeline modules."""

import logging
import os

from src.logger import get_logger

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog"

logger = get_logger(
    name=os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME),
    level=logging.INFO,
    log_file="Agent.log",
)

debug_logger = get_logger(
    name=f"{os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME)}.debug",
    level=logging.DEBUG,
    log_file="Debug.log",
)
