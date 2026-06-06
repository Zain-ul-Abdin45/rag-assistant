"""
Central logging setup for RAG Assistant.

Call setup_logging() once at process start (top of main.py).
All other modules get a logger with:
    from logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import logging.config
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "env.local")
_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

LOG_DIR = Path(__file__).parent / "logs"

_FILE_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)-28s | %(funcName)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class _DailyRotatingHandler(logging.FileHandler):
    """
    Writes to logs/YYYYMMDD_logs.txt.
    On the first emit after midnight, closes the old file and opens a new one.
    """

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir = log_dir
        self._current_date = datetime.now().strftime("%Y%m%d")
        super().__init__(self._filepath(), encoding="utf-8", delay=False)

    def _filepath(self) -> str:
        return str(self._log_dir / f"{datetime.now().strftime('%Y%m%d')}_logs.txt")

    def emit(self, record: logging.LogRecord) -> None:
        today = datetime.now().strftime("%Y%m%d")
        if today != self._current_date:
            self._current_date = today
            if self.stream:
                self.stream.close()
                self.stream = None
            self.baseFilename = self._filepath()
            self.stream = self._open()
        super().emit(record)


def setup_logging() -> None:
    """Load log.ini (console config) then attach the daily file handler."""
    config_path = Path(__file__).parent / "log.ini"
    if config_path.exists():
        logging.config.fileConfig(str(config_path), disable_existing_loggers=False)

    file_handler = _DailyRotatingHandler(LOG_DIR)
    file_handler.setLevel(_LEVEL)
    file_handler.setFormatter(_FILE_FORMATTER)

    rag_logger = logging.getLogger("rag_assistant")
    for h in rag_logger.handlers:
        h.setLevel(_LEVEL)
    rag_logger.addHandler(file_handler)
    rag_logger.setLevel(_LEVEL)

    rag_logger.info(
        "logging initialised — file: logs/%s_logs.txt",
        datetime.now().strftime("%Y%m%d"),
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a rag_assistant child logger.
    Pass __name__ or a short label — both work.
    """
    leaf = name.split(".")[-1]
    return logging.getLogger(f"rag_assistant.{leaf}")
