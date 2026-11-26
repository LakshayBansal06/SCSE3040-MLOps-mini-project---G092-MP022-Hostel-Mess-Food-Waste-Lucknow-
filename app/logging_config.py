"""JSON logging for the prediction service.

One JSON object per line, written to stdout (never to a file), as required by
the handbook (section 9, "Logging"). No personal details are ever logged.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

LOGGER_NAME = "mp022-api"


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        # Anything passed as logger.info(..., extra={"fields": {...}}).
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class StdoutProxy:
    """Writes to whatever sys.stdout is at the moment of the write.

    logging.StreamHandler would otherwise hold the sys.stdout object that
    existed when the handler was built, which is the wrong stream as soon as
    anything replaces it (uvicorn reloads, pytest's output capture).
    """

    def write(self, message: str) -> int:
        return sys.stdout.write(message)

    def flush(self) -> None:
        sys.stdout.flush()


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach a single stdout handler with the JSON formatter."""
    handler = logging.StreamHandler(StdoutProxy())
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger
