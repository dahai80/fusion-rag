"""LLM failure signaling — stop swallowing LLM errors as valid results.

L1 (systemic): 9 engine modules catch broad `except Exception` and return
magic defaults (5.0 / 0.5 / "" / []), making LLM failure indistinguishable
from success. This module gives those modules one exception type to raise on
total LLM failure, so the route layer can map it to 503 instead of 200.

- LLMUnavailable: total LLM failure (network down, empty content, all retries
  exhausted). Callers that hit this MUST NOT return a fabricated valid value;
  they propagate it so the route decides the response.
- Non-fatal cases (one chunk's context failed but others ok) stay local.
"""

from __future__ import annotations


class LLMUnavailable(RuntimeError):
    """Raised when an LLM call cannot produce a usable result.

    Carries no internal detail by default — route layer maps to a generic 503
    without leaking MLX URLs / paths / stack fragments.
    """

    def __init__(self, msg: str = "LLM call failed (unavailable or empty)"):
        super().__init__(msg)
