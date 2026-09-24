from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.domain.asset_execution import AdapterExecutionResult, ProviderOutcomeType
from app.domain.asset_router import AssetRouteCandidate, AssetRoutingRequest
from app.services.asset_adapters.base import AssetExecutionAdapter


class StockSearchAdapter(AssetExecutionAdapter):
    """
    Adapter wrapping Pexels and Pixabay stock video searching and downloading.
    """

    def __init__(self, search_func: Callable[..., Any] | None = None):
        self._search_func = search_func

    def execute(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        target_dir.mkdir(parents=True, exist_ok=True)
        query = request.visual_goal or request.generation_prompt or "video"

        try:
            if self._search_func is not None:
                video_paths = self._search_func(
                    query=query,
                    target_dir=str(target_dir),
                )
            else:
                from app.models.schema import VideoAspect
                from app.services import material

                aspect_str = getattr(request, "aspect_ratio", None) or getattr(request, "target_aspect_ratio", None)
                aspect_enum = (
                    VideoAspect.portrait
                    if aspect_str == "9:16"
                    else VideoAspect.landscape
                )
                min_dur = max(1, int(request.target_duration))
                provider_lower = candidate.provider.lower()
                if provider_lower == "pixabay":
                    items = material.search_videos_pixabay(
                        search_term=query,
                        minimum_duration=min_dur,
                        video_aspect=aspect_enum,
                    )
                elif provider_lower == "coverr":
                    items = material.search_videos_coverr(
                        search_term=query,
                        minimum_duration=min_dur,
                        video_aspect=aspect_enum,
                    )
                else:
                    items = material.search_videos_pexels(
                        search_term=query,
                        minimum_duration=min_dur,
                        video_aspect=aspect_enum,
                    )
                video_paths = [items[0].url] if items else []

            if not video_paths:
                # No stock materials match the visual query -> capability cannot fulfill, fallback!
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                    error_code="NO_STOCK_MATCHES",
                    error_message=f"No stock video found for query: {query!r}",
                )

            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUCCESS,
                file_path=str(video_paths[0]),
                raw_response={"provider": candidate.provider, "query": query},
            )

        except Exception as exc:  # noqa: BLE001
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                error_code="STOCK_SEARCH_FAILED",
                error_message=f"Failed to query/download stock video: {exc}",
            )
