from __future__ import annotations

import math
import httpx
import pytest

from app.services.knowledge.embedding_provider import (
    DeterministicFakeEmbeddingProvider,
    EmbeddingProviderError,
    OpenAICompatibleEmbeddingProvider,
    get_embedding_provider,
    redact_secrets,
)


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def test_deterministic_fake_embedding_provider_properties():
    provider = DeterministicFakeEmbeddingProvider(dimension=64)
    assert provider.dimension == 64
    assert provider.provider_name == "deterministic_fake"

    # Determinism
    vec1 = provider.embed_text("Quantum entanglement and particle physics")
    vec2 = provider.embed_text("Quantum entanglement and particle physics")
    assert vec1 == vec2
    assert len(vec1) == 64

    # L2 Normalized
    norm = math.sqrt(sum(v * v for v in vec1))
    assert abs(norm - 1.0) < 1e-6

    # Semantic similarity: related vs unrelated
    related = provider.embed_text("Quantum physics and entanglement experiments")
    unrelated = provider.embed_text("How to bake a sourdough bread in oven")

    sim_related = cosine_sim(vec1, related)
    sim_unrelated = cosine_sim(vec1, unrelated)
    assert sim_related > sim_unrelated, f"Expected {sim_related} > {sim_unrelated}"

    # Empty text handling
    empty_vec = provider.embed_text("   ")
    assert len(empty_vec) == 64
    assert abs(math.sqrt(sum(v * v for v in empty_vec)) - 1.0) < 1e-6


def test_redact_secrets():
    raw_error = "Failed to connect with Authorization: Bearer sk-1234567890abcdef and api_key='sk-secret99999'"
    redacted = redact_secrets(raw_error)
    assert "sk-1234567890abcdef" not in redacted
    assert "sk-secret99999" not in redacted
    assert "***" in redacted


def test_openai_compatible_embedding_provider_success():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("authorization") == "Bearer sk-test-key-12345"
        body = httpx.Response(200, json={
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        })
        return body

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://api.openai.com/v1",
        api_key="sk-test-key-12345",
        model="test-embed",
        dimension=2,
        batch_size=10,
        http_client=client,
    )

    results = provider.embed_texts(["first", "second"])
    assert len(results) == 2
    assert results[0] == [0.1, 0.2]  # Index 0
    assert results[1] == [0.3, 0.4]  # Index 1


def test_openai_compatible_embedding_provider_nan_inf_rejected():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='{"data": [{"index": 0, "embedding": [NaN, 0.5]}]}')

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    provider = OpenAICompatibleEmbeddingProvider(
        api_key="sk-test",
        http_client=client,
    )

    with pytest.raises(EmbeddingProviderError, match="NaN or Inf"):
        provider.embed_texts(["test text"])


def test_openai_compatible_embedding_provider_redacts_credentials_on_error():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized: api_key='sk-super-secret-key-12345' is invalid")

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    provider = OpenAICompatibleEmbeddingProvider(
        api_key="sk-super-secret-key-12345",
        http_client=client,
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        provider.embed_texts(["test text"])

    err_msg = str(exc_info.value)
    assert "sk-super-secret-key-12345" not in err_msg
    assert "***" in err_msg


def test_get_embedding_provider_factory():
    # Prefer fake
    fake = get_embedding_provider(prefer_fake=True)
    assert isinstance(fake, DeterministicFakeEmbeddingProvider)

    # Empty api_key honestly degrades to None (no vector provider)
    empty_key_p = get_embedding_provider({"api_key": ""})
    assert empty_key_p is None

    # Explicit fake provider setting
    explicit_fake = get_embedding_provider({"provider": "fake"})
    assert isinstance(explicit_fake, DeterministicFakeEmbeddingProvider)

    # With enabled and api_key creates OpenAICompatible
    prod_p = get_embedding_provider({"enabled": True, "api_key": "sk-real", "model": "text-embedding-3-small"})
    assert isinstance(prod_p, OpenAICompatibleEmbeddingProvider)
    assert prod_p.model_name == "text-embedding-3-small"
