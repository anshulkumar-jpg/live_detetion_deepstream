"""Structured JSON logging with correlation fields and secret redaction."""
from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from datetime import datetime, timezone

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

CONTEXT_FIELDS = ("camera_id", "worker_id", "model_id", "pipeline_id", "request_id", "event_id", "node_id")

# rtsp://user:pass@host  ->  rtsp://***:***@host
_CRED_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)(?P<creds>[^/@\s]+)@")


def redact(text: str) -> str:
    return _CRED_RE.sub(lambda m: f"{m.group('scheme')}***:***@", text)


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        for f in CONTEXT_FIELDS:
            v = getattr(record, f, None)
            if v is not None:
                payload[f] = v
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_logging(service: str, level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter(service))
    else:
        handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
