from __future__ import annotations

import os
from typing import Any, Protocol, Sequence
from uuid import uuid4

import requests

from app.domain.evidence import SearchResult


class SearchProviderError(Exception):
    """Base exception for search provider operations."""


class SearchCredentialsMissingError(SearchProviderError):
    """Raised when required API credentials for the search provider are missing or not configured."""


class SearchProviderUnavailableError(SearchProviderError):
    """Raised when search provider encounters temporary connectivity/timeout issues (retryable)."""


class SearchProviderRateLimitError(SearchProviderError):
    """Raised when search provider encounters rate limit (HTTP 429) issues (retryable)."""


class SearchProvider(Protocol):
    """Protocol interface for search providers."""

    def search(self, query: str, limit: int = 5) -> Sequence[SearchResult]:
        """Execute search query and return discovery SearchResult items."""
        ...


class TavilySearchProvider:
    """Production search provider adapter using Tavily search API.

    Adheres to:
    - Zero credential hardcoding: key from env or config only.
    - Bounded timeout and bounded results.
    - Explicit retryable vs fatal error handling.
    - Testable via standard requests.Session mocking.
    """

    DEFAULT_ENDPOINT = "https://api.tavily.com/search"
    DEFAULT_TIMEOUT_SECONDS = 10.0

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        self._explicit_api_key = api_key
        self.endpoint = endpoint or os.getenv("TAVILY_API_ENDPOINT") or self.DEFAULT_ENDPOINT
        self.timeout = timeout
        self.session = session or requests.Session()

    def get_api_key(self) -> str:
        """Resolves API key from explicit arg, environment, or configuration."""
        if self._explicit_api_key:
            return self._explicit_api_key

        env_key = os.getenv("TAVILY_API_KEY")
        if env_key:
            return env_key

        try:
            from app.config import config

            cfg_key = config.app.get("tavily_api_key")
            if cfg_key:
                return str(cfg_key)
        except Exception:
            pass

        raise SearchCredentialsMissingError(
            "Tavily API key is missing. Set TAVILY_API_KEY environment variable or configure 'tavily_api_key' in config.toml."
        )

    def search(self, query: str, limit: int = 5) -> Sequence[SearchResult]:
        """Executes a bounded search via Tavily HTTP API."""
        api_key = self.get_api_key()
        clean_query = query.strip()
        if not clean_query:
            return ()

        bounded_limit = max(1, min(limit, 10))
        payload = {
            "api_key": api_key,
            "query": clean_query,
            "search_depth": "basic",
            "max_results": bounded_limit,
            "include_raw_content": False,
        }

        try:
            resp = self.session.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise SearchProviderUnavailableError(
                f"Tavily search provider network error: {exc}"
            ) from exc
        except Exception as exc:
            raise SearchProviderError(f"Tavily search request failed: {exc}") from exc

        if resp.status_code == 429:
            raise SearchProviderRateLimitError(
                f"Tavily search API rate limit exceeded: HTTP {resp.status_code}"
            )
        elif resp.status_code in (401, 403):
            raise SearchCredentialsMissingError(
                f"Tavily search API authentication failed: HTTP {resp.status_code} - {resp.text}"
            )
        elif resp.status_code >= 500:
            raise SearchProviderUnavailableError(
                f"Tavily search provider server error: HTTP {resp.status_code}"
            )
        elif not resp.ok:
            raise SearchProviderError(
                f"Tavily search API returned HTTP {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        raw_results = data.get("results", [])
        results: list[SearchResult] = []
        for idx, item in enumerate(raw_results[:bounded_limit]):
            url = (item.get("url") or "").strip()
            if not url:
                continue
            title = (item.get("title") or "").strip()
            snippet = (item.get("content") or "").strip()
            results.append(
                SearchResult(
                    result_id=f"tavily_{uuid4().hex[:16]}",
                    title=title,
                    url=url,
                    snippet=snippet,
                    provider_rank=idx + 1,
                    provider_metadata={
                        "score": item.get("score"),
                        "raw_data": {
                            k: v
                            for k, v in item.items()
                            if k not in ("url", "title", "content")
                        },
                    },
                )
            )

        return tuple(results)
