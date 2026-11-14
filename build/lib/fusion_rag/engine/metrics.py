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


def _escape_label(value: str) -> str:
    # S-P2-2: Prometheus label values must escape backslash, double-quote, and
    # newline. endpoint/kb_id come from request path params (user-influenced) —
    # an unescaped `"` or `}` in a kb_id would break the exposition AND let a
    # crafted value inject synthetic label sets / series into a scrape.
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class _Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # counters keyed by (endpoint, kb_id, status_class) → int
        self._requests: dict[tuple[str, str, str], int] = {}
        self._errors: dict[tuple[str, str, str], int] = {}
        # histogram: (endpoint, kb_id) → {bucket_upper: count, "+Inf": count, "sum": float, "count": int}
        self._latency: dict[tuple[str, str], dict[str, Any]] = {}
        # O-P1-6: in-flight requests gauge — incremented at request enter,
        # decremented at exit, so a concurrency spike (threadpool saturation) is
        # visible without inferring from latency. Keyed by endpoint only (kb_id
        # would fragment an already-low cardinality signal).
        self._inflight: dict[str, int] = {}
        # O-P1-6: embedding-cache hit/miss counters — the cache hit rate is the
        # single most load-relevant RAG metric (a miss = an LLM embed call).
        self._cache_hits = 0
        self._cache_misses = 0
        # O-P1-6: LLM call latency histogram (rerank / contextualize / generate).
        # Separate from request latency: a request does N LLM calls; this shows
        # upstream cost per call, not per request.
        self._llm_latency: dict[str, dict[str, Any]] = {}

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

    # O-P1-6: in-flight gauge. enter/exit wrap a request lifecycle so the
    # middleware can report how many requests are mid-flight at scrape time —
    # the clearest signal of threadpool saturation vs pure latency growth.
    def inflight_enter(self, endpoint: str) -> None:
        with self._lock:
            self._inflight[endpoint] = self._inflight.get(endpoint, 0) + 1

    def inflight_exit(self, endpoint: str) -> None:
        with self._lock:
            val = self._inflight.get(endpoint, 0)
            if val > 0:
                self._inflight[endpoint] = val - 1

    # O-P1-6: cache hit/miss. Called from the embedding client after a get.
    def record_cache(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    # O-P1-6: LLM call latency, labeled by stage (rerank/contextualize/generate).
    def record_llm_latency(self, stage: str, latency_ms: float) -> None:
        with self._lock:
            bucket = self._llm_latency.get(stage)
            if bucket is None:
                bucket = {f"le={b}": 0 for b in _HISTOGRAM_BUCKETS_MS}
                bucket["le=+Inf"] = 0
                bucket["sum"] = 0.0
                bucket["count"] = 0
                self._llm_latency[stage] = bucket
            bucket["sum"] += latency_ms
            bucket["count"] += 1
            for b in _HISTOGRAM_BUCKETS_MS:
                if latency_ms <= b:
                    bucket[f"le={b}"] += 1
            bucket["le=+Inf"] += 1

    def render_prometheus(self) -> str:
        # R5: Prometheus text exposition format v0.0.4. No external dep —
        # counters + a histogram with cumulative bucket counts.
        # S-P2-2: every label value is escaped — endpoint/kb_id come from
        # request path params (user-influenced); an unescaped `"`/`\`/newline
        # breaks the exposition and lets a crafted value inject series.
        lines: list[str] = []
        with self._lock:
            lines.append("# HELP fusion_rag_requests_total Total HTTP requests by endpoint/kb/status.")
            lines.append("# TYPE fusion_rag_requests_total counter")
            for (endpoint, kb_id, status_class), val in sorted(self._requests.items()):
                lines.append(
                    f'fusion_rag_requests_total{{endpoint="{_escape_label(endpoint)}",'
                    f'kb_id="{_escape_label(kb_id)}",status="{_escape_label(status_class)}"}} {val}'
                )
            lines.append("# HELP fusion_rag_errors_total Total 5xx errors by endpoint/kb/status.")
            lines.append("# TYPE fusion_rag_errors_total counter")
            for (endpoint, kb_id, status_class), val in sorted(self._errors.items()):
                lines.append(
                    f'fusion_rag_errors_total{{endpoint="{_escape_label(endpoint)}",'
                    f'kb_id="{_escape_label(kb_id)}",status="{_escape_label(status_class)}"}} {val}'
                )
            lines.append("# HELP fusion_rag_request_latency_ms Request latency in milliseconds.")
            lines.append("# TYPE fusion_rag_request_latency_ms histogram")
            for (endpoint, kb_id), bucket in sorted(self._latency.items()):
                base_labels = f'endpoint="{_escape_label(endpoint)}",kb_id="{_escape_label(kb_id)}"'
                for b in _HISTOGRAM_BUCKETS_MS:
                    lines.append(
                        f'fusion_rag_request_latency_ms_bucket{{{base_labels},le="{b}"}} {bucket[f"le={b}"]}'
                    )
                lines.append(
                    f'fusion_rag_request_latency_ms_bucket{{{base_labels},le="+Inf"}} {bucket["le=+Inf"]}'
                )
                lines.append(f'fusion_rag_request_latency_ms_sum{{{base_labels}}} {bucket["sum"]}')
                lines.append(f'fusion_rag_request_latency_ms_count{{{base_labels}}} {bucket["count"]}')
            # O-P1-6: in-flight gauge — current count of mid-flight requests.
            lines.append("# HELP fusion_rag_inflight_requests Currently in-flight HTTP requests.")
            lines.append("# TYPE fusion_rag_inflight_requests gauge")
            for endpoint, val in sorted(self._inflight.items()):
                lines.append(f'fusion_rag_inflight_requests{{endpoint="{_escape_label(endpoint)}"}} {val}')
            # O-P1-6: cache hit/miss counters.
            lines.append("# HELP fusion_rag_embed_cache_hits_total Embedding cache lookups that hit.")
            lines.append("# TYPE fusion_rag_embed_cache_hits_total counter")
            lines.append(f"fusion_rag_embed_cache_hits_total {self._cache_hits}")
            lines.append("# HELP fusion_rag_embed_cache_misses_total Embedding cache lookups that missed.")
            lines.append("# TYPE fusion_rag_embed_cache_misses_total counter")
            lines.append(f"fusion_rag_embed_cache_misses_total {self._cache_misses}")
            # O-P1-6: LLM call latency histogram by stage.
            lines.append("# HELP fusion_rag_llm_latency_ms Upstream LLM call latency by stage (ms).")
            lines.append("# TYPE fusion_rag_llm_latency_ms histogram")
            for stage, bucket in sorted(self._llm_latency.items()):
                labels = f'stage="{_escape_label(stage)}"'
                for b in _HISTOGRAM_BUCKETS_MS:
                    lines.append(
                        f'fusion_rag_llm_latency_ms_bucket{{{labels},le="{b}"}} {bucket[f"le={b}"]}'
                    )
                lines.append(f'fusion_rag_llm_latency_ms_bucket{{{labels},le="+Inf"}} {bucket["le=+Inf"]}')
                lines.append(f'fusion_rag_llm_latency_ms_sum{{{labels}}} {bucket["sum"]}')
                lines.append(f'fusion_rag_llm_latency_ms_count{{{labels}}} {bucket["count"]}')
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
                "inflight": dict(self._inflight),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "llm_latency": {
                    s: {"count": b["count"], "sum": b["sum"]} for s, b in self._llm_latency.items()
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
            self._errors.clear()
            self._latency.clear()
            self._inflight.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._llm_latency.clear()


_metrics = _Metrics()


def get_metrics() -> _Metrics:
    return _metrics


def record_request(endpoint: str, kb_id: str, status_code: int, latency_ms: float) -> None:
    try:
        _metrics.record(endpoint, kb_id, status_code, latency_ms)
    except Exception as e:
        logger.warning("metrics record failed (never blocks request): %s", e)


# O-P1-6: convenience pass-throughs for the new metrics so callers don't reach
# into the singleton directly.
def inflight_enter(endpoint: str) -> None:
    try:
        _metrics.inflight_enter(endpoint)
    except Exception as e:
        logger.warning("metrics inflight_enter failed: %s", e)


def inflight_exit(endpoint: str) -> None:
    try:
        _metrics.inflight_exit(endpoint)
    except Exception as e:
        logger.warning("metrics inflight_exit failed: %s", e)


def record_cache(hit: bool) -> None:
    try:
        _metrics.record_cache(hit)
    except Exception as e:
        logger.warning("metrics record_cache failed: %s", e)


def record_llm_latency(stage: str, latency_ms: float) -> None:
    try:
        _metrics.record_llm_latency(stage, latency_ms)
    except Exception as e:
        logger.warning("metrics record_llm_latency failed: %s", e)


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
    # O-P1-6: also maintain the in-flight gauge around the call.
    import time

    start = time.perf_counter()
    endpoint, kb_id = _extract_labels(request)
    skipped = request.url.path in _SKIP_PATHS
    if not skipped:
        inflight_enter(endpoint)
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        if not skipped:
            record_request(endpoint, kb_id, 500, latency_ms)
            inflight_exit(endpoint)
        raise
    latency_ms = (time.perf_counter() - start) * 1000
    if not skipped:
        record_request(endpoint, kb_id, response.status_code, latency_ms)
        inflight_exit(endpoint)
    return response
