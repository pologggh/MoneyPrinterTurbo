from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math
import re
from typing import Any, Protocol

import httpx
from loguru import logger

SENSITIVE_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{6,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key[\"']?\s*[:=]\s*[\"'])[A-Za-z0-9_\-\.]{6,}([\"'])", re.IGNORECASE),
]


def redact_secrets(text: str) -> str:
    """Redacts API keys and Bearer tokens from error strings or logs."""
    redacted = text
    for pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub(r"\1***\2" if pattern.groups > 1 else r"\1***", redacted)
    return redacted


class EmbeddingProviderError(Exception):
    """Base exception for embedding provider failures."""


class EmbeddingProvider(Protocol):
    """Protocol defining the interface for text embedding providers."""

    @property
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_text(self, text: str) -> list[float]:
        ...


class DeterministicFakeEmbeddingProvider:
    """Offline, deterministic embedding provider for tests and reproducible local execution.

    Generates normalized float vectors using deterministic token hashing so that texts with
    lexical and semantic overlap produce predictably higher cosine similarity than unrelated texts.
    Zero external network calls, zero cost, and NaN/Inf safe.
    """

    def __init__(
        self,
        dimension: int = 64,
        provider_name: str = "deterministic_fake",
        model_name: str = "fake-embedding-v1",
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        self._dimension = dimension
        self._provider_name = provider_name
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_token(self, token: str) -> tuple[int, float]:
        """Maps a token to a bucket index and a signed weight."""
        h = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(h[:4], "big") % self._dimension
        weight_raw = int.from_bytes(h[4:8], "big")
        # Map to weight between 0.5 and 1.5
        weight = 0.5 + (weight_raw % 1000) / 1000.0
        return bucket, weight

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dimension
            tokens = re.findall(r"\w+", text.lower())
            if not tokens:
                # Default unit vector along dimension 0 for empty text
                vec[0] = 1.0
                results.append(vec)
                continue

            for tok in tokens:
                bucket, weight = self._hash_token(tok)
                vec[bucket] += weight

            # Also hash character n-grams (3-grams) for subword similarity
            normalized_compact = re.sub(r"\s+", " ", text.lower().strip())
            for i in range(max(0, len(normalized_compact) - 2)):
                tri = normalized_compact[i : i + 3]
                bucket, weight = self._hash_token(f"tri:{tri}")
                vec[bucket] += weight * 0.3

            # L2 Normalize
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 1e-12:
                vec = [v / norm for v in vec]
            else:
                vec[0] = 1.0
            results.append(vec)
        return results


class OpenAICompatibleEmbeddingProvider:
    """Production embedding provider for OpenAI-compatible embedding endpoints.

    Supports OpenAI, SiliconFlow, Azure OpenAI, Ollama, vLLM, etc.
    Enforces bounded batching, timeout protection, NaN/Inf validation, and credential redaction.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        timeout: float = 30.0,
        batch_size: int = 32,
        provider_name: str = "openai_compatible",
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model
        self._dimension = dimension
        self._timeout = timeout
        self._batch_size = max(1, min(batch_size, 256))
        self._provider_name = provider_name
        self._client = http_client

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=self._timeout)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        if not self._api_key and not self._client:
            raise EmbeddingProviderError(
                f"API key missing for provider '{self._provider_name}'."
            )

        endpoint = f"{self._base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        all_embeddings: list[list[float]] = []
        client = self._get_client()
        should_close = self._client is None

        try:
            for i in range(0, len(texts), self._batch_size):
                batch = list(texts[i : i + self._batch_size])
                payload = {
                    "input": batch,
                    "model": self._model_name,
                }
                try:
                    resp = client.post(endpoint, json=payload, headers=headers)
                except httpx.RequestError as exc:
                    redacted_err = redact_secrets(str(exc))
                    raise EmbeddingProviderError(
                        f"Network error calling embedding provider '{self._provider_name}': {redacted_err}"
                    ) from None

                if resp.status_code != 200:
                    redacted_body = redact_secrets(resp.text[:500])
                    raise EmbeddingProviderError(
                        f"Embedding provider returned HTTP {resp.status_code}: {redacted_body}"
                    )

                data = resp.json()
                if "data" not in data or not isinstance(data["data"], list):
                    raise EmbeddingProviderError(
                        f"Malformed embedding response from provider: missing 'data' list."
                    )

                # Sort by index in case API returns out of order
                items = sorted(data["data"], key=lambda x: x.get("index", 0))
                for item in items:
                    vec = item.get("embedding")
                    if not isinstance(vec, list) or len(vec) == 0:
                        raise EmbeddingProviderError(
                            f"Empty or non-list embedding vector returned for item."
                        )
                    # Validate finite floats
                    for v in vec:
                        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                            raise EmbeddingProviderError(
                                "Invalid non-finite float (NaN or Inf) detected in embedding vector."
                            )
                    all_embeddings.append([float(v) for v in vec])
        finally:
            if should_close:
                client.close()

        if len(all_embeddings) != len(texts):
            raise EmbeddingProviderError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(all_embeddings)}."
            )
        return all_embeddings


def get_embedding_provider(
    config_dict: dict[str, Any] | None = None,
    prefer_fake: bool = False,
) -> EmbeddingProvider | None:
    """Factory to create an appropriate EmbeddingProvider instance.

    Returns:
        EmbeddingProvider or None if vector embeddings are disabled or unconfigured,
        allowing truthful, graceful degradation to lexical BM25-only retrieval.
    """
    if prefer_fake:
        return DeterministicFakeEmbeddingProvider()

    if config_dict is None:
        try:
            from app.config import config
            cfg = getattr(config, "embedding", {})
        except Exception:
            cfg = {}
    else:
        cfg = config_dict

    provider_type = (cfg.get("provider") or "").strip().lower()
    dimension = int(cfg.get("dimension") or 1536)

    # Explicit fake provider setting (useful for dev/test configurations)
    if provider_type == "fake":
        return DeterministicFakeEmbeddingProvider(dimension=dimension)

    enabled = bool(cfg.get("enabled", False))
    api_key = (cfg.get("api_key") or "").strip()

    # If disabled or missing API key, honestly return None (no vector provider)
    if not enabled or not api_key:
        logger.debug("[get_embedding_provider] Embedding disabled or API key empty; vector provider unavailable.")
        return None

    base_url = cfg.get("base_url") or "https://api.openai.com/v1"
    model = cfg.get("model") or "text-embedding-3-small"
    timeout = float(cfg.get("timeout") or 30.0)
    batch_size = int(cfg.get("batch_size") or 32)

    return OpenAICompatibleEmbeddingProvider(
        base_url=base_url,
        api_key=api_key,
        model=model,
        dimension=dimension,
        timeout=timeout,
        batch_size=batch_size,
        provider_name=provider_type or "openai_compatible",
    )
