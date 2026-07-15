"""Embedding client — generates text embeddings via fusion-mlx HTTP API.

All vector generation goes through fusion-mlx's /v1/embeddings endpoint.
Never imports MLX or mlx-lm directly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Generates text embeddings by calling fusion-mlx's /v1/embeddings.

    All model calls are via HTTP — no direct MLX imports.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "BGE-M3",
        api_key: str = "local",
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size: int = 16,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(4)  # Limit concurrent requests

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a vector of floats."""
        results = await self.embed_batch([text])
        return results[0] if results else []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns a list of vectors."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors = await self._call_embed_api(batch)
            all_vectors.extend(vectors)

        return all_vectors

    async def _call_embed_api(self, texts: list[str]) -> list[list[float]]:
        """Call fusion-mlx's /v1/embeddings endpoint."""
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
                    # Return zero vectors on failure
                    return [[0.0] * 1024 for _ in texts]

        return [[0.0] * 1024 for _ in texts]

    async def health(self) -> bool:
        """Check if fusion-mlx embedding endpoint is available."""
        try:
            resp = await self.client.get("/models", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False