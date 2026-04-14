"""
app/utils/logger.py
===================
Structured JSON logging via structlog.

Calling ``configure_logging()`` once at app startup installs a process-wide
structlog pipeline that emits newline-delimited JSON to stdout.  All modules
obtain a bound logger via ``get_logger(__name__)``.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger


def _add_log_level(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Processor: inject the log level string into every event dict.
    structlog does not include it by default when wrapping stdlib logging.
    """
    event_dict["level"] = method_name.upper()
    return event_dict


def _drop_color_message_key(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Processor: remove the ``color_message`` key added by uvicorn's access logger
    so it does not pollute JSON output.
    """
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog for JSON output.

    Must be called exactly once before any logging occurs — typically in
    ``app/main.py`` during the lifespan startup hook.

    Args:
        log_level: One of DEBUG / INFO / WARNING / ERROR / CRITICAL.
                   Sourced from ``Settings.log_level``.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_log_level,
        _drop_color_message_key,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level.upper())

    # Silence noisy third-party loggers at WARNING unless DEBUG is requested
    if log_level.upper() != "DEBUG":
        for noisy in ("httpx", "httpcore", "google.auth", "urllib3", "neo4j"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a structlog BoundLogger bound to the given module name.

    Usage::

        from app.utils.logger import get_logger
        logger = get_logger(__name__)
        await logger.ainfo("event", key="value")

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A structlog BoundLogger instance.
    """
    return structlog.get_logger(name)
