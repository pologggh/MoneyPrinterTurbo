from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import VisualType
from app.domain.shot import ShotRevision


class RoutingStrategy(str, Enum):
    QUALITY_FIRST = "QUALITY_FIRST"
    BALANCED = "BALANCED"
    COST_FIRST = "COST_FIRST"


class GenerationMode(str, Enum):
    TEXT_TO_VIDEO = "TEXT_TO_VIDEO"
    IMAGE_TO_VIDEO = "IMAGE_TO_VIDEO"
    TEXT_TO_IMAGE = "TEXT_TO_IMAGE"
    STOCK_SEARCH = "STOCK_SEARCH"
    SOURCE_ASSET_USE = "SOURCE_ASSET_USE"
    USER_ASSET_USE = "USER_ASSET_USE"
    DIAGRAM_RENDER = "DIAGRAM_RENDER"


class TierLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class StaticCapabilityMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    quality_tier: TierLevel = TierLevel.MEDIUM
    cost_tier: TierLevel = TierLevel.MEDIUM
    latency_tier: TierLevel = TierLevel.MEDIUM


class RouteUnavailableReason(str, Enum):
    UNSUPPORTED_VISUAL_TYPE = "UNSUPPORTED_VISUAL_TYPE"
    UNSUPPORTED_DURATION = "UNSUPPORTED_DURATION"
    UNSUPPORTED_ASPECT_RATIO = "UNSUPPORTED_ASPECT_RATIO"
    MISSING_SOURCE_ASSET = "MISSING_SOURCE_ASSET"
    MISSING_USER_ASSET = "MISSING_USER_ASSET"
    NO_ENABLED_PROVIDER = "NO_ENABLED_PROVIDER"
    DISABLED_CAPABILITY = "DISABLED_CAPABILITY"
    UNSUPPORTED_GENERATION_MODE = "UNSUPPORTED_GENERATION_MODE"
    PROMPT_NOT_SUPPORTED = "PROMPT_NOT_SUPPORTED"
    BELOW_MINIMUM_QUALITY = "BELOW_MINIMUM_QUALITY"


class CandidateScore(BaseModel):
    """
    Normalized explainable scores for an eligible candidate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    capability_id: str
    quality_score: float
    cost_score: float
    latency_score: float
    final_score: float
    ranking_position: int = 1


class AssetCapability(BaseModel):
    """
    Represents a discrete asset generation or retrieval capability at the
    granularity of (Provider + Model + GenerationMode).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    capability_id: str
    provider: str
    model: str
    generation_mode: GenerationMode
    supported_visual_types: tuple[VisualType, ...]
    supported_aspect_ratios: tuple[str, ...] = ()
    min_duration: float | None = None
    max_duration: float | None = None
    supports_prompt: bool = True
    supports_source_asset: bool = False
    supports_user_asset: bool = False
    enabled: bool = True
    metadata: StaticCapabilityMetadata = Field(default_factory=StaticCapabilityMetadata)


class AssetRoutingRequest(BaseModel):
    """
    Typed routing request derived from an approved ShotRevision.
    Contains only what is needed to determine capability eligibility.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    shot_id: str
    shot_revision_id: str
    requested_visual_type: VisualType
    target_duration: float = Field(gt=0, description="Target duration in seconds")
    visual_goal: str
    scene_description: str
    generation_prompt: str
    camera_movement: str
    aspect_ratio: str | None = None
    source_asset_refs: tuple[str, ...] = ()
    user_asset_refs: tuple[str, ...] = ()
    routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED
    generation_constraints: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_aspect_ratio(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Allow target_aspect_ratio as an alias for aspect_ratio
            if "aspect_ratio" not in data and "target_aspect_ratio" in data:
                data = dict(data)
                data["aspect_ratio"] = data.pop("target_aspect_ratio")
        return data

    @property
    def target_aspect_ratio(self) -> str | None:
        """Alias for aspect_ratio to maintain backwards and cross-module compatibility."""
        return self.aspect_ratio


def create_routing_request_from_shot_revision(
    shot_revision: ShotRevision,
    *,
    aspect_ratio: str | None = None,
    target_aspect_ratio: str | None = None,
    routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
    source_asset_refs: tuple[str, ...] | None = None,
    user_asset_refs: tuple[str, ...] | None = None,
    generation_constraints: dict[str, Any] | None = None,
) -> AssetRoutingRequest:
    """
    Factory function creating an AssetRoutingRequest directly from an immutable ShotRevision.
    """
    resolved_aspect = aspect_ratio if aspect_ratio is not None else target_aspect_ratio
    resolved_source_refs = (
        source_asset_refs
        if source_asset_refs is not None
        else tuple(shot_revision.evidence_refs)
    )
    resolved_user_refs = user_asset_refs if user_asset_refs is not None else ()

    return AssetRoutingRequest(
        shot_id=shot_revision.shot_id,
        shot_revision_id=shot_revision.shot_revision_id,
        requested_visual_type=shot_revision.visual_type,
        target_duration=shot_revision.target_duration,
        visual_goal=shot_revision.visual_goal,
        scene_description=shot_revision.scene_description,
        generation_prompt=shot_revision.generation_prompt,
        camera_movement=shot_revision.camera_movement,
        aspect_ratio=resolved_aspect,
        source_asset_refs=tuple(resolved_source_refs),
        user_asset_refs=tuple(resolved_user_refs),
        routing_strategy=routing_strategy,
        generation_constraints=dict(generation_constraints or {}),
    )


class AssetRouteCandidate(BaseModel):
    """
    Represents an evaluated capability candidate for a specific routing request.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    capability_id: str
    provider: str
    model: str
    generation_mode: GenerationMode
    requested_visual_type: VisualType
    is_eligible: bool
    rejection_reasons: tuple[RouteUnavailableReason, ...] = ()
    static_metadata: StaticCapabilityMetadata = Field(
        default_factory=StaticCapabilityMetadata
    )
    score: CandidateScore | None = None


class RoutingPolicy(BaseModel):
    """
    Versioned routing policy defining weights, constraints, and tie-break rules.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_version: str = "v1.0"
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    quality_weight: float = 0.45
    cost_weight: float = 0.35
    latency_weight: float = 0.20
    minimum_quality_tier: TierLevel | None = None
    tie_break_rule: tuple[str, ...] = (
        "final_score DESC",
        "quality_score DESC",
        "cost_attractiveness DESC",
        "latency_attractiveness DESC",
        "capability_id ASC",
    )


def get_default_policy(
    strategy: RoutingStrategy = RoutingStrategy.BALANCED,
) -> RoutingPolicy:
    """
    Returns the standard default RoutingPolicy for a given RoutingStrategy.
    """
    if strategy == RoutingStrategy.QUALITY_FIRST:
        return RoutingPolicy(
            policy_version="v1.0",
            strategy=RoutingStrategy.QUALITY_FIRST,
            quality_weight=0.70,
            cost_weight=0.20,
            latency_weight=0.10,
            minimum_quality_tier=None,
        )
    elif strategy == RoutingStrategy.COST_FIRST:
        return RoutingPolicy(
            policy_version="v1.0",
            strategy=RoutingStrategy.COST_FIRST,
            quality_weight=0.25,
            cost_weight=0.65,
            latency_weight=0.10,
            minimum_quality_tier=None,
        )
    else:  # BALANCED
        return RoutingPolicy(
            policy_version="v1.0",
            strategy=RoutingStrategy.BALANCED,
            quality_weight=0.45,
            cost_weight=0.35,
            latency_weight=0.20,
            minimum_quality_tier=None,
        )


def normalize_quality_score(tier: TierLevel) -> float:
    """Normalizes quality tier into an attractiveness score [0..1]."""
    if tier == TierLevel.HIGH:
        return 0.9
    elif tier == TierLevel.MEDIUM:
        return 0.6
    return 0.3


def normalize_cost_score(tier: TierLevel) -> float:
    """Normalizes cost tier into an attractiveness score (LOW cost = higher score)."""
    if tier == TierLevel.LOW:
        return 0.9
    elif tier == TierLevel.MEDIUM:
        return 0.6
    return 0.3


def normalize_latency_score(tier: TierLevel) -> float:
    """Normalizes latency tier into an attractiveness score (LOW latency = faster = higher score)."""
    if tier == TierLevel.LOW:
        return 0.9
    elif tier == TierLevel.MEDIUM:
        return 0.6
    return 0.3


def calculate_candidate_score(
    capability: AssetCapability,
    policy: RoutingPolicy,
    ranking_position: int = 1,
) -> CandidateScore:
    """
    Calculates normalized component scores and the final weighted score for an eligible candidate.
    """
    q_score = normalize_quality_score(capability.metadata.quality_tier)
    c_score = normalize_cost_score(capability.metadata.cost_tier)
    l_score = normalize_latency_score(capability.metadata.latency_tier)

    final = (
        policy.quality_weight * q_score
        + policy.cost_weight * c_score
        + policy.latency_weight * l_score
    )

    return CandidateScore(
        capability_id=capability.capability_id,
        quality_score=round(q_score, 4),
        cost_score=round(c_score, 4),
        latency_score=round(l_score, 4),
        final_score=round(final, 4),
        ranking_position=ranking_position,
    )


class AssetRouteDecision(BaseModel):
    """
    Represents the result of evaluating candidates for a routing request.
    Freezes decision ID, provenance, strategy, candidates, and scores.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    routing_decision_id: str = Field(default_factory=lambda: str(uuid4()))
    shot_id: str
    shot_revision_id: str
    routing_policy_version: str = "v1.0"
    routing_strategy: RoutingStrategy
    requested_visual_type: VisualType = VisualType.AI_VIDEO
    selected_candidate: AssetRouteCandidate | None = None
    eligible_candidates: tuple[AssetRouteCandidate, ...] = ()
    rejected_candidates: tuple[AssetRouteCandidate, ...] = ()
    selected_score: CandidateScore | None = None
    decision_reason_codes: tuple[RouteUnavailableReason, ...] = ()
    reason_codes: tuple[RouteUnavailableReason, ...] = ()
    model_selection_mode: str = "AUTO"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, context: Any, /) -> None:
        if not self.decision_reason_codes and self.reason_codes:
            object.__setattr__(self, "decision_reason_codes", self.reason_codes)
        elif not self.reason_codes and self.decision_reason_codes:
            object.__setattr__(self, "reason_codes", self.decision_reason_codes)


class RouteUnavailableError(RuntimeError):
    """
    Raised when no eligible capability satisfies the routing request.
    """

    def __init__(
        self,
        message: str,
        reasons: tuple[RouteUnavailableReason, ...] = (),
    ):
        super().__init__(message)
        self.reasons = tuple(reasons)


class StoryboardNotApprovedError(RuntimeError):
    """
    Raised when route planning is attempted on a non-APPROVED StoryboardSnapshot.
    """

    def __init__(
        self,
        message: str,
        code: str = "STORYBOARD_NOT_APPROVED",
    ):
        super().__init__(message)
        self.code = code


class ConfiguredModelNotFoundError(RuntimeError):
    """
    Raised when a requested PINNED or PREFERRED provider/model is not configured in MPT.
    """

    def __init__(
        self,
        message: str,
        code: str = "CONFIGURED_MODEL_NOT_FOUND",
    ):
        super().__init__(message)
        self.code = code


class AssetRoutePlanStatus(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class ModelSelectionMode(str, Enum):
    AUTO = "AUTO"
    PINNED = "PINNED"
    PREFERRED = "PREFERRED"


class ShotRoutePlanStatus(str, Enum):
    ROUTED = "ROUTED"
    BLOCKED = "BLOCKED"


class ShotRoutePlanEntry(BaseModel):
    """
    Explicit route entry for one storyboard shot revision.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    shot_id: str
    shot_revision_id: str
    beat_lineage_id: str
    requested_visual_type: VisualType
    asset_routing_request: AssetRoutingRequest
    route_decision: AssetRouteDecision
    route_status: ShotRoutePlanStatus


class AssetRoutePlan(BaseModel):
    """
    Immutable plan containing deterministic routing results for all shots
    in an APPROVED StoryboardSnapshot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_route_plan_id: str = Field(default_factory=lambda: str(uuid4()))
    storyboard_snapshot_id: str
    content_plan_revision_id: str
    routing_strategy: RoutingStrategy
    routing_policy_version: str
    model_selection_mode: ModelSelectionMode = ModelSelectionMode.AUTO
    selected_provider: str | None = None
    selected_model: str | None = None
    shot_routes: tuple[ShotRoutePlanEntry, ...] = ()
    status: AssetRoutePlanStatus = AssetRoutePlanStatus.READY
    total_shots: int = 0
    routed_shots: int = 0
    blocked_shots: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_ready(self) -> bool:
        return self.status == AssetRoutePlanStatus.READY

    @property
    def is_blocked(self) -> bool:
        return self.status == AssetRoutePlanStatus.BLOCKED

