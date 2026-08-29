"""R5: in-process RED metrics (request count, latency, error rate).

Before this the live search path only wrote per-request latency into an audit
log row — no aggregate counters, no histogram, no error rate. A production
slowdown (p99 200ms → 5s) had no metric to look at; the operator grepped audit
rows by hand. This module is a zero-dependency in-process collector exposed at
/metrics in Prometheus text format, labeled by endpoint + kb_id + status_class.

Design: no prometheus_client dependency (not in requirements, would need a
release). A fixed set of latency histogram buckets covers the ms range a RAG
service cares about; counters are plain ints under a lock. Best-effort
recording never raises into the request path."""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# R5: buckets in milliseconds. A RAG service's latency lives in the tens-of-ms
# to low-seconds range; these buckets let p50/p99 fall in a meaningful bucket
# without a thousand empty buckets. +Inf bucket is implicit in the render.
_HISTOGRAM_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


class _Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # counters keyed by (endpoint, kb_id, status_class) → int
        self._requests: dict[tuple[str, str, str], int] = {}
        self._errors: dict[tuple[str, str, str], int] = {}
        # histogram: (endpoint, kb_id) → {bucket_upper: count, "+Inf": count, "sum": float, "count": int}
        self._latency: dict[tuple[str, str], dict[str, Any]] = {}

    def _latency_bucket(self, endpoint: str, kb_id: str) -> dict[str, Any]:
        key = (endpoint, kb_id)
        bucket = self._latency.get(key)
        if bucket is None:
            bucket = {f"le={b}": 0 for b in _HISTOGRAM_BUCKETS_MS}
            bucket["le=+Inf"] = 0
            bucket["sum"] = 0.0
            bucket["count"] = 0
            self._latency[key] = bucket
        return bucket

    def record(self, endpoint: str, kb_id: str, status_code: int, latency_ms: float) -> None:
        # R5: best-effort — never let a metrics failure propagate into the
        # request path. status_class is the coarse bucket (2xx/4xx/5xx) so the
        # error-rate counter is cheap to alert on.
        status_class = f"{status_code // 100}xx"
        with self._lock:
            rk = (endpoint, kb_id, status_class)
            self._requests[rk] = self._requests.get(rk, 0) + 1
            if status_code >= 500:
                self._errors[rk] = self._errors.get(rk, 0) + 1
            bucket = self._latency_bucket(endpoint, kb_id)
            bucket["sum"] += latency_ms
            bucket["count"] += 1
            for b in _HISTOGRAM_BUCKETS_MS:
                if latency_ms <= b:
                    bucket[f"le={b}"] += 1
            bucket["le=+Inf"] += 1
        logger.debug("metrics record: %s kb=%s %s %.1fms", endpoint, kb_id, status_class, latency_ms)

    def render_prometheus(self) -> str:
        # R5: Prometheus text exposition format v0.0.4. No external dep —
        # counters + a histogram with cumulative bucket counts.
        lines: list[str] = []
        with self._lock:
            lines.append("# HELP fusion_rag_requests_total Total HTTP requests by endpoint/kb/status.")
            lines.append("# TYPE fusion_rag_requests_total counter")
            for (endpoint, kb_id, status_class), val in sorted(self._requests.items()):
                lines.append(
                    f'fusion_rag_requests_total{{endpoint="{endpoint}",kb_id="{kb_id}",status="{status_class}"}} {val}'
                )
            lines.append("# HELP fusion_rag_errors_total Total 5xx errors by endpoint/kb/status.")
            lines.append("# TYPE fusion_rag_errors_total counter")
            for (endpoint, kb_id, status_class), val in sorted(self._errors.items()):
                lines.append(
                    f'fusion_rag_errors_total{{endpoint="{endpoint}",kb_id="{kb_id}",status="{status_class}"}} {val}'
                )
            lines.append("# HELP fusion_rag_request_latency_ms Request latency in milliseconds.")
            lines.append("# TYPE fusion_rag_request_latency_ms histogram")
            for (endpoint, kb_id), bucket in sorted(self._latency.items()):
                base_labels = f'endpoint="{endpoint}",kb_id="{kb_id}"'
                for b in _HISTOGRAM_BUCKETS_MS:
                    lines.append(
                        f'fusion_rag_request_latency_ms_bucket{{{base_labels},le="{b}"}} {bucket[f"le={b}"]}'
                    )
                lines.append(
                    f'fusion_rag_request_latency_ms_bucket{{{base_labels},le="+Inf"}} {bucket["le=+Inf"]}'
                )
                lines.append(f'fusion_rag_request_latency_ms_sum{{{base_labels}}} {bucket["sum"]}')
                lines.append(f'fusion_rag_request_latency_ms_count{{{base_labels}}} {bucket["count"]}')
        return "\n".join(lines) + "\n"

    def snapshot(self) -> dict[str, Any]:
        # for tests: structured view without parsing the text format.
        with self._lock:
            return {
                "requests": {f"{e}|{k}|{s}": v for (e, k, s), v in self._requests.items()},
                "errors": {f"{e}|{k}|{s}": v for (e, k, s), v in self._errors.items()},
                "latency": {
                    f"{e}|{k}": {"count": b["count"], "sum": b["sum"]}
                    for (e, k), b in self._latency.items()
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
            self._errors.clear()
            self._latency.clear()


_metrics = _Metrics()


def get_metrics() -> _Metrics:
    return _metrics


def record_request(endpoint: str, kb_id: str, status_code: int, latency_ms: float) -> None:
    try:
        _metrics.record(endpoint, kb_id, status_code, latency_ms)
    except Exception as e:
        logger.warning("metrics record failed (never blocks request): %s", e)


# R5: paths that are infrastructure, not user traffic — skip them so a metrics
# scrape or liveness probe doesn't inflate its own counters.
_SKIP_PATHS = frozenset({"/metrics", "/health"})


def _extract_labels(request: Any) -> tuple[str, str]:
    # endpoint = matched route's path template (e.g. "/kb/bases/{kb_id}/search")
    # for low cardinality. Falls back to the raw path if no route matched (404).
    route = request.scope.get("route")
    endpoint = getattr(route, "path", None) or request.url.path
    kb_id = ""
    params = request.scope.get("path_params")
    if params:
        kb_id = str(params.get("kb_id", "") or "")
    return endpoint, kb_id


async def metrics_middleware(request: Any, call_next: Any) -> Any:
    # R5: RED middleware — record request count, latency histogram, error count
    # per endpoint+kb_id. Best-effort: a recording failure never breaks the
    # request (caught in record_request). Skips self-scrape / liveness paths.
    import time

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        if request.url.path not in _SKIP_PATHS:
            endpoint, kb_id = _extract_labels(request)
            record_request(endpoint, kb_id, 500, latency_ms)
        raise
    latency_ms = (time.perf_counter() - start) * 1000
    if request.url.path not in _SKIP_PATHS:
        endpoint, kb_id = _extract_labels(request)
        record_request(endpoint, kb_id, response.status_code, latency_ms)
    return response
