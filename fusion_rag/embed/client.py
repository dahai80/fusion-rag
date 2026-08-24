"""Embedding client — generates text embeddings via fusion-mlx HTTP API.

All vector generation goes through fusion-mlx's /v1/embeddings endpoint.
Never imports MLX or mlx-lm directly.
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when embedding fails after all providers (F7 fail-loud).

    Distinguished from generic errors so callers can surface a clear
    "embedding unavailable" message instead of persisting zero vectors.
    """


class EmbeddingClient:
    """Generates text embeddings by calling fusion-mlx's /v1/embeddings.

    All model calls are via HTTP — no direct MLX imports.
    """

    # callers: routes.py search/ask/upload endpoints
    # API: embed/embed_batch with cache, schema: cache stores text_hash->vector
    # user instruction: "按照你的方案和计划落地所有phase阶段的需求"
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11432/v1",
        model: str = "BGE-M3",
        api_key: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size: int = 16,
        cache_enabled: bool = True,
        fallback_url: str = "",
        fallback_api_key: str = "",
    ):
        if not api_key:
            api_key = os.environ.get("FUSION_MLX_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        if not self.api_key:
            logger.warning(
                "EmbeddingClient started without FUSION_MLX_API_KEY; MLX calls will 401 if gateway auth is on"
            )
        else:
            logger.info("EmbeddingClient api_key loaded from FUSION_MLX_API_KEY (len=%d)", len(self.api_key))
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self.fallback_url = fallback_url.rstrip("/") if fallback_url else ""
        self.fallback_api_key = fallback_api_key
        self._client: httpx.AsyncClient | None = None
        self._fallback_client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(4)
        self._cache = None
        if cache_enabled:
            from ..engine.embedding_cache import EmbeddingCache

            self._cache = EmbeddingCache()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=headers,
            )
        return self._client

    @property
    def fallback_client(self) -> httpx.AsyncClient:
        if self._fallback_client is None:
            headers = {}
            if self.fallback_api_key:
                headers["Authorization"] = f"Bearer {self.fallback_api_key}"
            self._fallback_client = httpx.AsyncClient(
                base_url=self.fallback_url,
                timeout=self.timeout,
                headers=headers,
            )
        return self._fallback_client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._fallback_client:
            await self._fallback_client.aclose()
            self._fallback_client = None

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a vector of floats.

        Raises EmbeddingError on total failure (never returns a zero vector).
        """
        results = await self.embed_batch([text])
        if not results:
            raise EmbeddingError("embed_batch returned no vectors")
        vec = results[0]
        if not vec or all(v == 0.0 for v in vec):
            raise EmbeddingError("embedding produced a zero vector (all providers failed)")
        return vec

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns a list of vectors.

        Raises EmbeddingError if any provider path yields zero vectors, so
        callers never persist poison into the cache or vector store (F7).
        """
        if not texts:
            return []

        # Check cache
        if self._cache:
            cached = self._cache.get_batch(texts, self.model)
            uncached_indices = [i for i, v in enumerate(cached) if v is None]
            if not uncached_indices:
                logger.debug("EmbeddingCache: all %d texts hit cache", len(texts))
                return cached  # type: ignore
            if uncached_indices:
                uncached_texts = [texts[i] for i in uncached_indices]
            else:
                uncached_texts = texts
        else:
            cached = None
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Call API for uncached texts
        all_vectors: list[list[float]] = []
        for i in range(0, len(uncached_texts), self.batch_size):
            batch = uncached_texts[i : i + self.batch_size]
            vectors = await self._call_embed_api(batch)
            # F7: reject zero vectors from failure paths before they reach the cache/store.
            for v in vectors:
                if not v or all(x == 0.0 for x in v):
                    raise EmbeddingError("embedding produced a zero vector (provider failure)")
            all_vectors.extend(vectors)

        # Cache new results — only non-zero vectors (guard repeated in loop above)
        if self._cache and all_vectors and all(any(x != 0.0 for x in v) for v in all_vectors):
            self._cache.set_batch(uncached_texts, all_vectors, self.model)

        # Merge cached + new
        if cached:
            result = list(cached)
            for idx, vec in zip(uncached_indices, all_vectors):
                result[idx] = vec
            return result  # type: ignore

        return all_vectors

    async def _call_embed_api(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    payload = {
                        "model": self.model,
                        "input": texts,
                    }
                    resp = await self.client.post("/embeddings", json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                    vectors = []
                    for item in data.get("data", []):
                        vector = item.get("embedding", [])
                        if vector:
                            vectors.append(vector)
                    return vectors

            except httpx.TimeoutException:
                logger.warning("Embedding timeout (attempt %d/%d)", attempt + 1, self.max_retries)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
            except Exception as e:
                logger.error("Embedding failed: %s", e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1.0)
                else:
                    if self.fallback_url:
                        return await self._call_fallback_api(texts)
                    return await self._call_local_embed(texts)

        if self.fallback_url:
            return await self._call_fallback_api(texts)
        return await self._call_local_embed(texts)

    async def _call_fallback_api(self, texts: list[str]) -> list[list[float]]:
        logger.info("Falling back to cloud embedding: %s", self.fallback_url)
        try:
            payload = {"model": self.model, "input": texts}
            resp = await self.fallback_client.post("/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()
            vectors = []
            for item in data.get("data", []):
                vector = item.get("embedding", [])
                if vector:
                    vectors.append(vector)
            if len(vectors) == len(texts):
                logger.info("Fallback embedding succeeded: %d vectors", len(vectors))
                return vectors
            logger.warning("Fallback returned %d/%d vectors", len(vectors), len(texts))
        except Exception as e:
            logger.error("Fallback embedding also failed: %s", e)
        return await self._call_local_embed(texts)

    async def _call_local_embed(self, texts: list[str]) -> list[list[float]]:
        logger.info("Falling back to local sentence-transformers embedding")
        try:
            from .local import embed_local

            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(None, embed_local, texts, self.model)
            if vectors and any(any(v != 0.0 for v in vec) for vec in vectors):
                logger.info("Local embedding succeeded: %d vectors", len(vectors))
                return vectors
            logger.warning("Local embedding returned zero vectors")
        except Exception as e:
            logger.error("Local embedding failed: %s", e)
        # F7: never return zero vectors — raise so callers fail loud instead of
        # persisting poison into cache + vector store.
        raise EmbeddingError("all embedding providers failed (MLX + fallback + local)")

    async def health(self) -> bool:
        """Check if embedding is available (MLX HTTP or local sentence-transformers)."""
        try:
            resp = await self.client.get("/models", timeout=2.0)
            if resp.status_code == 200:
                return True
        except Exception:
            logger.debug("health check /models failed")
        try:
            from .local import get_local_model

            return get_local_model(self.model) is not None
        except Exception:
            return False
