"""
Asset Capability Registry and Provider Adapters for Hybrid Asset Routing (Phase 4.1).

Defines the boundary between MoneyPrinterTurbo's media generation/retrieval services
and the domain-level asset capability catalog.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.config import config
from app.domain.asset_router import (
    AssetCapability,
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutingRequest,
    GenerationMode,
    RouteUnavailableError,
    RouteUnavailableReason,
    StaticCapabilityMetadata,
    TierLevel,
)
from app.domain.enums import VisualType
from app.services import aliyun_wan, volcengine_seedance
from app.services.material import is_openai_image_enabled


@runtime_checkable
class AssetCapabilityProvider(Protocol):
    """Protocol for provider capability adapters."""

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        """Return declared static capabilities without calling external services."""
        ...


class VolcEngineSeedanceAdapter:
    """Adapts VolcEngine Seedance text-to-video capabilities."""

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        model_id = str(
            config.app.get(
                "volcengine_seedance_model",
                volcengine_seedance.DEFAULT_MODEL_ID,
            )
            or volcengine_seedance.DEFAULT_MODEL_ID
        ).strip()

        min_dur, max_dur = volcengine_seedance._duration_bounds()
        enabled = volcengine_seedance.is_enabled()

        return (
            AssetCapability(
                capability_id=f"volcengine_seedance:{model_id}:TEXT_TO_VIDEO",
                provider="volcengine_seedance",
                model=model_id,
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
                supported_visual_types=(VisualType.AI_VIDEO,),
                supported_aspect_ratios=("16:9", "9:16", "1:1"),
                min_duration=float(min_dur),
                max_duration=float(max_dur),
                supports_prompt=True,
                supports_source_asset=False,
                supports_user_asset=False,
                enabled=enabled,
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.HIGH,
                    cost_tier=TierLevel.HIGH,
                    latency_tier=TierLevel.HIGH,
                ),
            ),
        )


class AliyunWanCapabilityAdapter:
    """Adapts Alibaba Cloud Model Studio Wan text-to-video capabilities."""

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        model_id = str(
            config.app.get("aliyun_wan_model", aliyun_wan.DEFAULT_MODEL)
            or aliyun_wan.DEFAULT_MODEL
        ).strip()
        return (
            AssetCapability(
                capability_id=f"aliyun_wan:{model_id}:TEXT_TO_VIDEO",
                provider="aliyun_wan",
                model=model_id,
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
                supported_visual_types=(VisualType.AI_VIDEO,),
                supported_aspect_ratios=("16:9", "9:16", "1:1", "4:3", "3:4"),
                min_duration=2.0,
                max_duration=15.0,
                supports_prompt=True,
                supports_source_asset=False,
                supports_user_asset=False,
                enabled=aliyun_wan.is_enabled(),
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.HIGH,
                    cost_tier=TierLevel.MEDIUM,
                    latency_tier=TierLevel.HIGH,
                ),
            ),
        )


class WaveSpeedAdapter:
    """Adapts WaveSpeed AI video capabilities."""

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        model_id = str(
            config.app.get(
                "wavespeed_text_to_video_model",
                "bytedance/seedance-2.0-fast/text-to-video",
            )
            or "bytedance/seedance-2.0-fast/text-to-video"
        ).strip()

        api_keys = config.app.get("wavespeed_api_keys")
        enabled = bool(api_keys)

        min_dur = float(config.app.get("wavespeed_min_duration", 4))
        max_dur = float(config.app.get("wavespeed_max_duration", 15))

        return (
            AssetCapability(
                capability_id=f"wavespeed:{model_id}:TEXT_TO_VIDEO",
                provider="wavespeed",
                model=model_id,
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
                supported_visual_types=(VisualType.AI_VIDEO,),
                supported_aspect_ratios=("16:9", "9:16", "1:1"),
                min_duration=min_dur,
                max_duration=max_dur,
                supports_prompt=True,
                supports_source_asset=False,
                supports_user_asset=False,
                enabled=enabled,
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.HIGH,
                    cost_tier=TierLevel.MEDIUM,
                    latency_tier=TierLevel.MEDIUM,
                ),
            ),
        )


class PexelsStockAdapter:
    """Adapts Pexels video search capabilities."""

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        enabled = bool(config.app.get("pexels_api_keys"))
        return (
            AssetCapability(
                capability_id="pexels:video-search:STOCK_SEARCH",
                provider="pexels",
                model="pexels-video-search",
                generation_mode=GenerationMode.STOCK_SEARCH,
                supported_visual_types=(VisualType.STOCK_VIDEO,),
                supported_aspect_ratios=("16:9", "9:16", "1:1"),
                min_duration=None,
                max_duration=None,
                supports_prompt=True,
                supports_source_asset=False,
                supports_user_asset=False,
                enabled=enabled,
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.MEDIUM,
                    cost_tier=TierLevel.LOW,
                    latency_tier=TierLevel.LOW,
                ),
            ),
        )


class PixabayStockAdapter:
    """Adapts Pixabay video search capabilities."""

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        enabled = bool(config.app.get("pixabay_api_keys"))
        return (
            AssetCapability(
                capability_id="pixabay:video-search:STOCK_SEARCH",
                provider="pixabay",
                model="pixabay-video-search",
                generation_mode=GenerationMode.STOCK_SEARCH,
                supported_visual_types=(VisualType.STOCK_VIDEO,),
                supported_aspect_ratios=("16:9", "9:16", "1:1"),
                min_duration=None,
                max_duration=None,
                supports_prompt=True,
                supports_source_asset=False,
                supports_user_asset=False,
                enabled=enabled,
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.LOW,
                    cost_tier=TierLevel.LOW,
                    latency_tier=TierLevel.LOW,
                ),
            ),
        )


class CoverrStockAdapter:
    """Adapts Coverr video search capabilities."""

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        enabled = bool(config.app.get("coverr_enabled", True))
        return (
            AssetCapability(
                capability_id="coverr:video-search:STOCK_SEARCH",
                provider="coverr",
                model="coverr-video-search",
                generation_mode=GenerationMode.STOCK_SEARCH,
                supported_visual_types=(VisualType.STOCK_VIDEO,),
                supported_aspect_ratios=("16:9", "9:16", "1:1"),
                min_duration=None,
                max_duration=None,
                supports_prompt=True,
                supports_source_asset=False,
                supports_user_asset=False,
                enabled=enabled,
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.LOW,
                    cost_tier=TierLevel.LOW,
                    latency_tier=TierLevel.LOW,
                ),
            ),
        )


class OpenAIImageAdapter:
    """Adapts OpenAI-compatible image generation capabilities."""

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        model = str(config.app.get("openai_image_model", "dall-e-3") or "dall-e-3").strip()
        enabled = is_openai_image_enabled()
        return (
            AssetCapability(
                capability_id=f"openai:{model}:TEXT_TO_IMAGE",
                provider="openai",
                model=model,
                generation_mode=GenerationMode.TEXT_TO_IMAGE,
                supported_visual_types=(VisualType.AI_IMAGE,),
                supported_aspect_ratios=("16:9", "9:16", "1:1"),
                min_duration=None,
                max_duration=None,
                supports_prompt=True,
                supports_source_asset=False,
                supports_user_asset=False,
                enabled=enabled,
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.MEDIUM,
                    cost_tier=TierLevel.MEDIUM,
                    latency_tier=TierLevel.LOW,
                ),
            ),
        )


class SourceAssetAdapter:
    """Adapts original knowledge source asset extraction/usage."""

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        return (
            AssetCapability(
                capability_id="system:source_asset_extractor:SOURCE_ASSET_USE",
                provider="system_source_asset",
                model="knowledge_figure_extractor",
                generation_mode=GenerationMode.SOURCE_ASSET_USE,
                supported_visual_types=(VisualType.SOURCE_ASSET,),
                supported_aspect_ratios=(),
                min_duration=None,
                max_duration=None,
                supports_prompt=False,
                supports_source_asset=True,
                supports_user_asset=False,
                enabled=True,
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.HIGH,
                    cost_tier=TierLevel.LOW,
                    latency_tier=TierLevel.LOW,
                ),
            ),
        )


class UserAssetAdapter:
    """Adapts user-supplied media assets."""

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        return (
            AssetCapability(
                capability_id="system:user_asset_loader:USER_ASSET_USE",
                provider="system_user_asset",
                model="user_media_loader",
                generation_mode=GenerationMode.USER_ASSET_USE,
                supported_visual_types=(VisualType.USER_ASSET,),
                supported_aspect_ratios=(),
                min_duration=None,
                max_duration=None,
                supports_prompt=False,
                supports_source_asset=False,
                supports_user_asset=True,
                enabled=True,
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.HIGH,
                    cost_tier=TierLevel.LOW,
                    latency_tier=TierLevel.LOW,
                ),
            ),
        )


class DiagramPlaceholderAdapter:
    """
    Local deterministic capability for rendering knowledge diagram cards.
    """

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        return (
            AssetCapability(
                capability_id="system:diagram_renderer:DIAGRAM_RENDER",
                provider="system_diagram",
                model="diagram_card_v1",
                generation_mode=GenerationMode.DIAGRAM_RENDER,
                supported_visual_types=(VisualType.DIAGRAM,),
                supported_aspect_ratios=("16:9", "9:16", "1:1"),
                min_duration=None,
                max_duration=None,
                supports_prompt=True,
                supports_source_asset=False,
                supports_user_asset=False,
                enabled=True,
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.MEDIUM,
                    cost_tier=TierLevel.LOW,
                    latency_tier=TierLevel.MEDIUM,
                ),
            ),
        )


class KnowledgeCardFallbackAdapter:
    """
    Local deterministic fallback capability for rendering knowledge cards
    when no real AI image generation provider is enabled.
    """

    def __init__(self, *, enabled: bool | None = None) -> None:
        self._explicit_enabled = enabled

    def _is_real_image_provider_enabled(self) -> bool:
        return is_openai_image_enabled()

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        if self._explicit_enabled is not None:
            enabled = self._explicit_enabled
        else:
            enabled = not self._is_real_image_provider_enabled()

        return (
            AssetCapability(
                capability_id="system:knowledge_card:TEXT_TO_IMAGE",
                provider="system_knowledge_card",
                model="knowledge_card_v1",
                generation_mode=GenerationMode.TEXT_TO_IMAGE,
                supported_visual_types=(VisualType.AI_IMAGE,),
                supported_aspect_ratios=("16:9", "9:16", "1:1"),
                min_duration=None,
                max_duration=None,
                supports_prompt=True,
                supports_source_asset=False,
                supports_user_asset=False,
                enabled=enabled,
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.MEDIUM,
                    cost_tier=TierLevel.LOW,
                    latency_tier=TierLevel.LOW,
                ),
            ),
        )



def evaluate_candidate(
    capability: AssetCapability,
    request: AssetRoutingRequest,
) -> AssetRouteCandidate:
    """
    Evaluates hard capability constraints against a routing request.
    Does NOT score candidates (Phase 4.2 owns candidate scoring).
    """
    rejection_reasons: list[RouteUnavailableReason] = []

    # 1. Visual Type check
    if request.requested_visual_type not in capability.supported_visual_types:
        rejection_reasons.append(RouteUnavailableReason.UNSUPPORTED_VISUAL_TYPE)

    # 2. Enabled check
    if not capability.enabled:
        rejection_reasons.append(RouteUnavailableReason.NO_ENABLED_PROVIDER)

    # 3. Duration check
    duration_too_short = (
        capability.min_duration is not None
        and request.target_duration < capability.min_duration
    )
    duration_too_long = (
        capability.max_duration is not None
        and request.target_duration > capability.max_duration
    )
    if duration_too_short or duration_too_long:
        rejection_reasons.append(RouteUnavailableReason.UNSUPPORTED_DURATION)

    # 4. Aspect Ratio check
    if (
        request.aspect_ratio
        and capability.supported_aspect_ratios
        and request.aspect_ratio not in capability.supported_aspect_ratios
    ):
        rejection_reasons.append(RouteUnavailableReason.UNSUPPORTED_ASPECT_RATIO)

    # 5. Source Asset check
    requires_source_asset = (
        request.requested_visual_type == VisualType.SOURCE_ASSET
        or capability.generation_mode == GenerationMode.SOURCE_ASSET_USE
    )
    if requires_source_asset and not request.source_asset_refs:
        rejection_reasons.append(RouteUnavailableReason.MISSING_SOURCE_ASSET)

    # 6. User Asset check
    requires_user_asset = (
        request.requested_visual_type == VisualType.USER_ASSET
        or capability.generation_mode == GenerationMode.USER_ASSET_USE
    )
    if requires_user_asset and not request.user_asset_refs:
        rejection_reasons.append(RouteUnavailableReason.MISSING_USER_ASSET)

    is_eligible = len(rejection_reasons) == 0

    return AssetRouteCandidate(
        capability_id=capability.capability_id,
        provider=capability.provider,
        model=capability.model,
        generation_mode=capability.generation_mode,
        requested_visual_type=request.requested_visual_type,
        is_eligible=is_eligible,
        rejection_reasons=tuple(rejection_reasons),
        static_metadata=capability.metadata,
    )


class AssetCapabilityRegistry:
    """
    Registry for asset capabilities, serving as the boundary between
    MoneyPrinterTurbo services and the Hybrid Asset Router.
    """

    def __init__(
        self,
        providers: Sequence[AssetCapabilityProvider] | None = None,
    ) -> None:
        self._providers: list[AssetCapabilityProvider] = list(providers or [])

    def register_provider(self, provider: AssetCapabilityProvider) -> None:
        self._providers.append(provider)

    def list_capabilities(self) -> tuple[AssetCapability, ...]:
        caps: list[AssetCapability] = []
        for p in self._providers:
            caps.extend(p.get_capabilities())
        return tuple(caps)

    def find_candidates(
        self,
        request: AssetRoutingRequest,
    ) -> tuple[AssetRouteCandidate, ...]:
        candidates: list[AssetRouteCandidate] = []
        for cap in self.list_capabilities():
            candidates.append(evaluate_candidate(cap, request))
        return tuple(candidates)

    def create_route_decision(
        self,
        request: AssetRoutingRequest,
        *,
        raise_if_unavailable: bool = False,
    ) -> AssetRouteDecision:
        all_candidates = self.find_candidates(request)
        eligible = tuple(c for c in all_candidates if c.is_eligible)
        rejected = tuple(c for c in all_candidates if not c.is_eligible)

        reason_codes: list[RouteUnavailableReason] = []
        if not eligible:
            # Group candidates that claimed this visual type to find specific reasons
            matching_type_candidates = [
                c
                for c in all_candidates
                if RouteUnavailableReason.UNSUPPORTED_VISUAL_TYPE
                not in c.rejection_reasons
            ]
            if not matching_type_candidates:
                reason_codes.append(RouteUnavailableReason.UNSUPPORTED_VISUAL_TYPE)
            else:
                for c in matching_type_candidates:
                    for r in c.rejection_reasons:
                        if r not in reason_codes:
                            reason_codes.append(r)
            if not reason_codes:
                reason_codes.append(RouteUnavailableReason.NO_ENABLED_PROVIDER)

            if raise_if_unavailable:
                reasons_str = ", ".join(r.value for r in reason_codes)
                raise RouteUnavailableError(
                    f"ROUTE_UNAVAILABLE for shot {request.shot_id} (revision {request.shot_revision_id}): {reasons_str}",
                    reasons=tuple(reason_codes),
                )

        return AssetRouteDecision(
            shot_id=request.shot_id,
            shot_revision_id=request.shot_revision_id,
            routing_strategy=request.routing_strategy,
            selected_candidate=None,  # Selection and ranking belong to Phase 4.2
            eligible_candidates=eligible,
            rejected_candidates=rejected,
            reason_codes=tuple(reason_codes),
        )


def get_default_capability_registry() -> AssetCapabilityRegistry:
    """
    Constructs the default AssetCapabilityRegistry pre-populated with
    adapters for all MoneyPrinterTurbo providers.
    """
    return AssetCapabilityRegistry(
        providers=[
            AliyunWanCapabilityAdapter(),
            VolcEngineSeedanceAdapter(),
            WaveSpeedAdapter(),
            PexelsStockAdapter(),
            PixabayStockAdapter(),
            CoverrStockAdapter(),
            OpenAIImageAdapter(),
            KnowledgeCardFallbackAdapter(),
            SourceAssetAdapter(),
            UserAssetAdapter(),
            DiagramPlaceholderAdapter(),
        ]
    )
