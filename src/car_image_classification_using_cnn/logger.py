"""Centralized logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_file: Path | None = None, level: str = "INFO", rotation: str = "100 MB") -> None:
    """
    Configure the application logger.

    Args:
        log_file: Path to the log file. If None, only logs to stderr.
        level: Minimum log level to display.
        rotation: When to rotate the log file (e.g., "100 MB", "1 week").
    """
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation=rotation,
            retention="1 week",
            compression="zip",
        )
        logger.info(f"Logging to file: {log_file}")


def get_logger():
    """Get the configured logger instance."""
    return logger
