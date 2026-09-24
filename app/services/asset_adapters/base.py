from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from app.domain.asset_execution import AdapterExecutionResult, ProviderReceipt
from app.domain.asset_router import AssetRouteCandidate, AssetRoutingRequest


class ProviderExecutionCapabilities(BaseModel):
    """
    Explicit lifecycle capabilities of a provider adapter.
    Only features actually supported by the provider are exposed.
    """
    synchronous_completion: bool = True
    supports_async_status: bool = False
    supports_cancel: bool = False
    supports_idempotency: bool = False
    supports_safe_resubmission_with_same_key: bool = False


class AssetExecutionAdapter(ABC):
    """
    Abstract adapter boundary interfacing between domain execution and external media providers.
    Normalizes provider-specific responses and exceptions into standard AdapterExecutionResult.
    """

    @property
    def capabilities(self) -> ProviderExecutionCapabilities:
        return ProviderExecutionCapabilities(
            synchronous_completion=True,
            supports_async_status=False,
            supports_cancel=False,
            supports_idempotency=False,
            supports_safe_resubmission_with_same_key=False,
        )

    def get_capabilities(self) -> ProviderExecutionCapabilities:
        return self.capabilities

    @abstractmethod
    def execute(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        """Executes one attempt against the provider synchronously."""

    def submit(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
        idempotency_key: str | None = None,
    ) -> AdapterExecutionResult | ProviderReceipt:
        """
        Submits generation request to provider.
        For synchronous providers, returns AdapterExecutionResult directly.
        For asynchronous providers, returns a ProviderReceipt acknowledging task acceptance.
        """
        return self.execute(request, candidate, target_dir)

    def get_status(
        self,
        provider_job_id: str,
        target_dir: Path | None = None,
    ) -> AdapterExecutionResult:
        """Queries the status of an asynchronous job if supported."""
        raise NotImplementedError(
            f"Provider does not support asynchronous status queries: {self.__class__.__name__}"
        )

    def cancel(self, provider_job_id: str) -> bool:
        """Cancels an ongoing job if supported."""
        raise NotImplementedError(
            f"Provider does not support cancellation: {self.__class__.__name__}"
        )
