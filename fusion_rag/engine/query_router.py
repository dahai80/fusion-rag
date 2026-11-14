"""QueryRouter — rule-based query complexity routing (issue #128, R6).

callers: routes_search.ask (opt-in via route param / FUSION_RAG_ROUTER=auto)
API: QueryRouter.route(question, history_len) -> RouteDecision
pattern: Adaptive-RAG (arXiv 2403.14403) / Self-Route (arXiv 2407.16833) —
route simple queries to cheap single-path retrieval, complex ones to the full
pipeline (+ retry). Rules first (deterministic, zero cost); a learned
classifier is deliberately NOT added until rules prove insufficient.
Route map: simple -> single-path (no hybrid/rerank/retry); standard ->
hybrid + rerank; complex -> full pipeline + bounded retry (R5).
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Cues that make a question multi-constraint / multi-hop rather than a single
# fact lookup. Deterministic substring/shape rules only — no model calls.
_COMPLEX_MARKERS = (
    " and then",
    "compare",
    "difference between",
    "step by step",
    "why does",
    "how do",
    "how does",
    "为什么",
    "如何",
    "区别",
    "对比",
    "步骤",
)
_SIMPLE_MAX_CHARS = 24
_COMPLEX_MIN_CHARS = 120

LEVELS = ("simple", "standard", "complex")


@dataclass
class RouteDecision:
    level: str
    reason: str


class QueryRouter:
    def __init__(self, simple_max_chars: int = _SIMPLE_MAX_CHARS, complex_min_chars: int = _COMPLEX_MIN_CHARS):
        self.simple_max_chars = simple_max_chars
        self.complex_min_chars = complex_min_chars

    def route(self, question: str, history_len: int = 0) -> RouteDecision:
        q = (question or "").strip()
        if not q:
            return RouteDecision("simple", "empty question")
        # Multi-turn sessions carry context the single path cannot resolve
        # (pronouns, ellipsis) — treat as complex so the full pipeline runs.
        if history_len > 0:
            return RouteDecision("complex", f"session history present ({history_len} turns)")
        lowered = q.lower()
        hits = [m for m in _COMPLEX_MARKERS if m in lowered or m in q]
        if hits:
            return RouteDecision("complex", f"multi-hop marker: {hits[0]!r}")
        if len(q) >= self.complex_min_chars:
            return RouteDecision("complex", f"length {len(q)} >= {self.complex_min_chars}")
        # Two or more question marks / interleaved questions rarely single-hop.
        fullwidth_qmark = chr(0xFF1F)
        if q.count("?") + q.count(fullwidth_qmark) >= 2:
            return RouteDecision("complex", "multiple question marks")
        if len(q) <= self.simple_max_chars:
            # Short single-clause lookups (SKU-101, Aurora Strip specs) — the
            # single dense path answers these; fusion adds latency, not recall.
            return RouteDecision("simple", f"short lookup (len {len(q)})")
        return RouteDecision("standard", "default bucket")
