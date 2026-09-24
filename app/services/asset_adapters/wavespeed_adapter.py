from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.domain.asset_execution import AdapterExecutionResult, ProviderOutcomeType
from app.domain.asset_router import AssetRouteCandidate, AssetRoutingRequest
from app.services.asset_adapters.base import AssetExecutionAdapter


class WaveSpeedAdapter(AssetExecutionAdapter):
    """
    Adapter wrapping WaveSpeed AI video generation.
    """

    def __init__(self, generator_func: Callable[..., Any] | None = None):
        self._generator_func = generator_func

    def execute(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        target_dir.mkdir(parents=True, exist_ok=True)
        prompt = request.generation_prompt or request.visual_goal

        try:
            if self._generator_func is not None:
                video_paths = self._generator_func(
                    search_term=prompt,
                    minimum_duration=max(1, int(request.target_duration)),
                    target_dir=str(target_dir),
                )
            else:
                from app.models.schema import VideoAspect
                from app.services.material import generate_videos_wavespeed

                aspect_str = getattr(request, "aspect_ratio", None) or getattr(request, "target_aspect_ratio", None)
                aspect_enum = (
                    VideoAspect.portrait
                    if aspect_str == "9:16"
                    else VideoAspect.landscape
                )
                items = generate_videos_wavespeed(
                    search_term=prompt,
                    minimum_duration=max(1, int(request.target_duration)),
                    video_aspect=aspect_enum,
                )
                video_paths = [items[0].url] if items else []

            if not video_paths:
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                    error_code="EMPTY_GENERATION_RESULT",
                    error_message="WaveSpeed returned no video items",
                )

            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUCCESS,
                file_path=str(video_paths[0]),
                raw_response={"provider": "wavespeed", "model": candidate.model},
            )

        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if any(k in msg for k in ("unconfirmed", "unknown task", "timeout awaiting task creation")):
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
                    error_code="SUBMISSION_OUTCOME_UNKNOWN",
                    error_message=f"WaveSpeed task submission outcome unknown: {exc}",
                )
            if any(k in msg for k in ("invalid", "unsupported", "content policy", "bad request", "400")):
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                    error_code="CAPABILITY_INCOMPATIBLE",
                    error_message=f"WaveSpeed rejected request parameters: {exc}",
                )
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                error_code="TECHNICAL_FAILURE",
                error_message=f"WaveSpeed technical error: {exc}",
            )
