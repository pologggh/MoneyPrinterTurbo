from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.domain.asset_execution import AdapterExecutionResult, ProviderOutcomeType
from app.domain.asset_router import AssetRouteCandidate, AssetRoutingRequest
from app.services.asset_adapters.base import AssetExecutionAdapter


class OpenAIImageAdapter(AssetExecutionAdapter):
    """
    Adapter wrapping OpenAI DALL-E image generation.
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
                image_paths = self._generator_func(
                    prompt=prompt,
                    target_dir=str(target_dir),
                )
            else:
                from app.models.schema import VideoAspect
                from app.services.material import generate_images_openai

                aspect_str = getattr(request, "aspect_ratio", None) or getattr(request, "target_aspect_ratio", None)
                aspect_enum = (
                    VideoAspect.portrait
                    if aspect_str == "9:16"
                    else VideoAspect.landscape
                )
                items = generate_images_openai(
                    search_term=prompt,
                    minimum_duration=max(1, int(request.target_duration)),
                    video_aspect=aspect_enum,
                    save_dir=str(target_dir),
                )
                image_paths = [items[0].url] if items else []

            if not image_paths:
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                    error_code="EMPTY_IMAGE_RESULT",
                    error_message="OpenAI returned no image items",
                )

            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUCCESS,
                file_path=str(image_paths[0]),
                raw_response={"provider": "openai", "model": candidate.model},
            )

        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if any(k in msg for k in ("content policy", "safety", "sensitive", "invalid_prompt", "400")):
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                    error_code="CAPABILITY_INCOMPATIBLE",
                    error_message=f"OpenAI rejected request: {exc}",
                )
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                error_code="TECHNICAL_FAILURE",
                error_message=f"OpenAI technical error: {exc}",
            )
