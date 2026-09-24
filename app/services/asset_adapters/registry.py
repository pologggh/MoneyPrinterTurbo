from __future__ import annotations

from app.services.asset_adapters.base import AssetExecutionAdapter
from app.services.asset_adapters.local_asset_adapter import LocalAssetAdapter
from app.services.asset_adapters.openai_image_adapter import OpenAIImageAdapter
from app.services.asset_adapters.seedance_adapter import SeedanceAdapter
from app.services.asset_adapters.stock_adapter import StockSearchAdapter
from app.services.asset_adapters.wavespeed_adapter import WaveSpeedAdapter


class AdapterNotFoundError(KeyError):
    """Raised when no execution adapter is registered for a requested provider."""

    def __init__(self, provider: str, available_providers: list[str]):
        self.provider = provider
        self.available_providers = available_providers
        super().__init__(
            f"No execution adapter registered for provider '{provider}'. "
            f"Available providers: {sorted(available_providers)}"
        )


class AdapterRegistry:
    """
    Registry providing the appropriate AssetExecutionAdapter for a candidate provider.
    """

    def __init__(self, custom_adapters: dict[str, AssetExecutionAdapter] | None = None):
        seedance_adapter = SeedanceAdapter()
        stock_adapter = StockSearchAdapter()
        local_adapter = LocalAssetAdapter()
        self._adapters: dict[str, AssetExecutionAdapter] = {
            "volcengine_seedance": seedance_adapter,
            "seedance": seedance_adapter,
            "wavespeed": WaveSpeedAdapter(),
            "pexels": stock_adapter,
            "pixabay": stock_adapter,
            "coverr": stock_adapter,
            "openai": OpenAIImageAdapter(),
            "system_source_asset": local_adapter,
            "system_user_asset": local_adapter,
            "system_diagram": local_adapter,
        }
        if custom_adapters:
            self._adapters.update({k.lower(): v for k, v in custom_adapters.items()})

    def has_adapter(self, provider: str) -> bool:
        """Check if an adapter is registered for the specified provider."""
        return provider.lower() in self._adapters

    def list_providers(self) -> list[str]:
        """Return list of all registered provider names."""
        return sorted(self._adapters.keys())

    def register_adapter(self, provider: str, adapter: AssetExecutionAdapter) -> None:
        self._adapters[provider.lower()] = adapter

    def get_adapter(self, provider: str) -> AssetExecutionAdapter:
        p = provider.lower()
        if p in self._adapters:
            return self._adapters[p]
        raise AdapterNotFoundError(provider, list(self._adapters.keys()))
