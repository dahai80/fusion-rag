"""O-P1-2 / O-P2-2: structured logging + rotation + request-id.

Before this, server.run_server called logging.basicConfig(level=...) only — a
single unbounded StreamHandler, and start.sh appended stdout/stderr to
logs/{stdout,stderr}.log with `>>` (no rotation, the file grew forever; a short
test run already left stderr.log at 8.9MB). There was no request-id/trace-id
correlation, and every log line was freeform text — unparseable by a log
aggregator (ELK/Loki) an enterprise deploy feeds into.

This module wires three things once, from run_server:
  1. RotatingFileHandler on logs/fusion-rag.log — 10MB x 5 files, so a long run
     no longer fills the disk. Env FUSION_RAG_LOG_DIR overrides the path.
  2. A JSON formatter (FUSION_RAG_LOG_FORMAT=json) emitting one log line as a
     JSON object {ts, level, logger, msg, request_id, ...extra} so an aggregator
     can parse fields instead of regex-ing freeform text. Plain text stays the
     default for local/single-user dev readability.
  3. A request-id: every HTTP request gets an X-Request-ID (incoming header
     honored, else generated), set into a contextvar that the log filter reads,
     and echoed back on the response header. A request's full log trail is then
     greppable by one id across app logs + the aggregator.

No new dependencies — stdlib logging + json + uuid only.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import ClassVar

logger = logging.getLogger(__name__)

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("fusion_rag_request_id", default="")


class _RequestIdFilter(logging.Filter):
    # O-P2-2: inject the current request_id onto every LogRecord so the
    # formatter can emit it. A Filter (not a Formatter) because the id is
    # per-request context, not a fixed attribute of the record; the filter
    # reads the contextvar at emit time.
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("")
        return True


class _JsonFormatter(logging.Formatter):
    # O-P2-2: one JSON object per line. Fields an aggregator indexes on
    # (ts/level/logger/msg/request_id) are top-level; the rest of the record's
    # __dict__ (exc_info flattened by formatException, args, custom attrs) is
    # folded into `extra` so structured logging from `logger.info("x", extra=...)`
    # survives. Redact nothing here — PII redaction happens at the call sites
    # (O-P1-3); the formatter only shapes what's already been logged.
    _RESERVED: ClassVar[frozenset[str]] = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "request_id", "message", "asctime",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", ""),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra = {k: v for k, v in record.__dict__.items() if k not in self._RESERVED and not k.startswith("_")}
        if extra:
            payload["extra"] = extra
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    # O-P1-2 + O-P2-2: idempotent — clear any prior basicConfig handlers so a
    # reload (uvicorn --reload) does not stack a second RotatingFileHandler and
    # double-log every line. Re-adding the request-id filter each call is safe.
    root = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)
    # drop existing handlers we own (tagged) so a re-configure does not double up.
    for h in list(root.handlers):
        if getattr(h, "_fusion_rag_owned", False):
            root.removeHandler(h)
    fmt = os.environ.get("FUSION_RAG_LOG_FORMAT", "text").strip().lower()
    log_dir = os.environ.get("FUSION_RAG_LOG_DIR", "")
    if not log_dir:
        log_dir = str(Path(__file__).resolve().parent.parent.parent / "logs")
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_path = str(Path(log_dir) / "fusion-rag.log")
        handler: logging.Handler = RotatingFileHandler(
            file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
    except OSError as e:
        # logging must never break startup — fall back to stderr if the log dir
        # is not writable (read-only root, sandbox). Surface it loudly.
        logger.warning("RotatingFileHandler unavailable (%s); falling back to stderr", e)
        handler = logging.StreamHandler()
    handler._fusion_rag_owned = True  # type: ignore[attr-defined]
    handler.setLevel(numeric_level)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
        )
    handler.addFilter(_RequestIdFilter())
    root.addHandler(handler)
    # keep a stderr handler so container deployments (docker logs reads stderr)
    # still see output even when the file handler is the primary sink.
    has_console = any(
        isinstance(h, logging.StreamHandler) and not getattr(h, "_fusion_rag_owned", False)
        for h in root.handlers
    )
    if not has_console:
        console = logging.StreamHandler()
        console._fusion_rag_owned = True  # type: ignore[attr-defined]
        console.setLevel(numeric_level)
        console.setFormatter(handler.formatter)
        console.addFilter(_RequestIdFilter())
        root.addHandler(console)
    logging.getLogger("fusion_rag").propagate = True


async def request_id_middleware(request, call_next):
    # O-P2-2: honor an inbound X-Request-ID (a gateway/front proxy may set one
    # to correlate across services); else mint a short uuid. Bind it to the
    # contextvar so the log filter tags every line this request emits, and echo
    # it on the response so a client can trace a call end-to-end.
    incoming = request.headers.get("x-request-id", "").strip()
    rid = incoming if incoming else uuid.uuid4().hex[:16]
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["x-request-id"] = rid
    return response
