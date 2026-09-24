"""
Hybrid Asset Router Decision Engine (Phase 4.2).

Deterministic and explainable routing engine that filters, scores, ranks,
and selects preferred asset generation/retrieval capabilities according to
a versioned RoutingPolicy.
"""

from collections.abc import Sequence
from typing import Any

from app.domain.asset_router import (
    AssetCapability,
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutingRequest,
    CandidateScore,
    GenerationMode,
    RouteUnavailableError,
    RouteUnavailableReason,
    RoutingPolicy,
    TierLevel,
    calculate_candidate_score,
    get_default_policy,
)
from app.domain.enums import VisualType
from app.services.asset_capability_registry import (
    AssetCapabilityRegistry,
    get_default_capability_registry,
)

_TIER_ORDINAL: dict[TierLevel, int] = {
    TierLevel.LOW: 1,
    TierLevel.MEDIUM: 2,
    TierLevel.HIGH: 3,
}


def filter_hard_constraints(
    capability: AssetCapability,
    request: AssetRoutingRequest,
    policy: RoutingPolicy,
) -> tuple[bool, tuple[RouteUnavailableReason, ...]]:
    """
    Evaluates all hard eligibility constraints before scoring.
    Ineligible candidates are never scored.
    """
    rejection_reasons: list[RouteUnavailableReason] = []

    # 1. Enabled check
    if not capability.enabled:
        rejection_reasons.append(RouteUnavailableReason.DISABLED_CAPABILITY)
        rejection_reasons.append(RouteUnavailableReason.NO_ENABLED_PROVIDER)

    # 2. Visual Type support
    if request.requested_visual_type not in capability.supported_visual_types:
        rejection_reasons.append(RouteUnavailableReason.UNSUPPORTED_VISUAL_TYPE)

    # 3. Target Duration bounds
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

    # 4. Aspect Ratio compatibility
    if (
        request.aspect_ratio
        and capability.supported_aspect_ratios
        and request.aspect_ratio not in capability.supported_aspect_ratios
    ):
        rejection_reasons.append(RouteUnavailableReason.UNSUPPORTED_ASPECT_RATIO)

    # 5. SOURCE_ASSET reference requirement
    requires_source_asset = (
        request.requested_visual_type == VisualType.SOURCE_ASSET
        or capability.generation_mode == GenerationMode.SOURCE_ASSET_USE
    )
    if requires_source_asset and not request.source_asset_refs:
        rejection_reasons.append(RouteUnavailableReason.MISSING_SOURCE_ASSET)

    # 6. USER_ASSET reference requirement
    requires_user_asset = (
        request.requested_visual_type == VisualType.USER_ASSET
        or capability.generation_mode == GenerationMode.USER_ASSET_USE
    )
    if requires_user_asset and not request.user_asset_refs:
        rejection_reasons.append(RouteUnavailableReason.MISSING_USER_ASSET)

    # 7. Minimum quality floor rule
    if policy.minimum_quality_tier is not None:
        cand_tier = _TIER_ORDINAL.get(capability.metadata.quality_tier, 0)
        min_tier = _TIER_ORDINAL.get(policy.minimum_quality_tier, 0)
        if cand_tier < min_tier:
            rejection_reasons.append(RouteUnavailableReason.BELOW_MINIMUM_QUALITY)

    is_eligible = len(rejection_reasons) == 0
    return is_eligible, tuple(rejection_reasons)


class HybridAssetRouter:
    """
    Deterministic Hybrid Asset Router selecting concrete (Provider, Model, GenerationMode)
    capabilities for an approved ShotRevision without executing providers.
    """

    def __init__(
        self,
        registry: AssetCapabilityRegistry | None = None,
    ) -> None:
        self._registry = registry or get_default_capability_registry()

    def route(
        self,
        request: AssetRoutingRequest,
        capabilities: Sequence[AssetCapability] | None = None,
        policy: RoutingPolicy | None = None,
        *,
        raise_if_unavailable: bool = False,
        trace_context: Any | None = None,
        trace_writer: Any | None = None,
        asset_route_plan_id: str = "",
    ) -> AssetRouteDecision:
        """
        Evaluates capabilities against the request and policy, producing a structured decision.
        """
        active_policy = policy or get_default_policy(request.routing_strategy)
        available_caps = (
            tuple(capabilities)
            if capabilities is not None
            else self._registry.list_capabilities()
        )

        eligible_list: list[tuple[AssetCapability, CandidateScore]] = []
        rejected_candidates: list[AssetRouteCandidate] = []

        # 1. Hard constraint filter
        for cap in available_caps:
            is_eligible, reasons = filter_hard_constraints(cap, request, active_policy)
            if is_eligible:
                # Calculate initial unranked score
                score = calculate_candidate_score(cap, active_policy)
                eligible_list.append((cap, score))
            else:
                rejected_candidates.append(
                    AssetRouteCandidate(
                        capability_id=cap.capability_id,
                        provider=cap.provider,
                        model=cap.model,
                        generation_mode=cap.generation_mode,
                        requested_visual_type=request.requested_visual_type,
                        is_eligible=False,
                        rejection_reasons=reasons,
                        static_metadata=cap.metadata,
                        score=None,
                    )
                )

        # 2. Deterministic sorting & tie-breaking:
        # Sort key: (-final_score, -quality_score, -cost_score, -latency_score, capability_id)
        eligible_list.sort(
            key=lambda item: (
                -item[1].final_score,
                -item[1].quality_score,
                -item[1].cost_score,
                -item[1].latency_score,
                item[0].capability_id,
            )
        )

        # 3. Assign 1-based ranking positions
        ranked_eligible_candidates: list[AssetRouteCandidate] = []
        for rank, (cap, score) in enumerate(eligible_list, start=1):
            ranked_score = CandidateScore(
                capability_id=score.capability_id,
                quality_score=score.quality_score,
                cost_score=score.cost_score,
                latency_score=score.latency_score,
                final_score=score.final_score,
                ranking_position=rank,
            )
            ranked_eligible_candidates.append(
                AssetRouteCandidate(
                    capability_id=cap.capability_id,
                    provider=cap.provider,
                    model=cap.model,
                    generation_mode=cap.generation_mode,
                    requested_visual_type=request.requested_visual_type,
                    is_eligible=True,
                    rejection_reasons=(),
                    static_metadata=cap.metadata,
                    score=ranked_score,
                )
            )

        # 4. Determine selection or unavailable reasons
        selected_candidate: AssetRouteCandidate | None = None
        selected_score: CandidateScore | None = None
        decision_reason_codes: list[RouteUnavailableReason] = []

        if ranked_eligible_candidates:
            selected_candidate = ranked_eligible_candidates[0]
            selected_score = selected_candidate.score
        else:
            # Aggregate reasons from rejected candidates
            matching_visual_type = [
                c
                for c in rejected_candidates
                if RouteUnavailableReason.UNSUPPORTED_VISUAL_TYPE
                not in c.rejection_reasons
            ]
            eval_group = matching_visual_type or rejected_candidates
            for c in eval_group:
                for r in c.rejection_reasons:
                    if r not in decision_reason_codes:
                        decision_reason_codes.append(r)

            if not decision_reason_codes:
                decision_reason_codes.append(RouteUnavailableReason.NO_ENABLED_PROVIDER)

            if raise_if_unavailable:
                reasons_str = ", ".join(r.value for r in decision_reason_codes)
                raise RouteUnavailableError(
                    f"ROUTE_UNAVAILABLE for shot {request.shot_id} (revision {request.shot_revision_id}): {reasons_str}",
                    reasons=tuple(decision_reason_codes),
                )

        decision = AssetRouteDecision(
            shot_id=request.shot_id,
            shot_revision_id=request.shot_revision_id,
            routing_policy_version=active_policy.policy_version,
            routing_strategy=active_policy.strategy,
            requested_visual_type=request.requested_visual_type,
            selected_candidate=selected_candidate,
            eligible_candidates=tuple(ranked_eligible_candidates),
            rejected_candidates=tuple(rejected_candidates),
            selected_score=selected_score,
            decision_reason_codes=tuple(decision_reason_codes),
            reason_codes=tuple(decision_reason_codes),
        )

        if trace_writer is not None and trace_context is not None:
            from app.domain.trace import (
                RouterDecisionTraceData,
                TraceEventStatus,
                TraceEventType,
            )

            ctx = trace_context.with_ids(
                shot_id=request.shot_id,
                shot_revision_id=request.shot_revision_id,
                asset_route_plan_id=asset_route_plan_id or trace_context.asset_route_plan_id,
            )
            r_data = RouterDecisionTraceData(
                asset_route_plan_id=ctx.asset_route_plan_id or "",
                shot_id=request.shot_id,
                shot_revision_id=request.shot_revision_id,
                requested_visual_type=request.requested_visual_type.value if hasattr(request.requested_visual_type, "value") else str(request.requested_visual_type),
                strategy=active_policy.strategy.value if hasattr(active_policy.strategy, "value") else str(active_policy.strategy),
                selection_mode="AUTO",
                selected_provider=selected_candidate.provider if selected_candidate else "NONE",
                selected_model=selected_candidate.model if selected_candidate else "NONE",
                generation_mode=selected_candidate.generation_mode.value if (selected_candidate and hasattr(selected_candidate.generation_mode, "value")) else "NONE",
                selected_score=selected_score.final_score if selected_score else None,
                eligible_candidate_count=len(ranked_eligible_candidates),
                rejected_candidate_count=len(rejected_candidates),
                key_reason_codes=tuple(str(r.value if hasattr(r, "value") else r) for r in decision_reason_codes),
                routing_policy_version=active_policy.policy_version,
            )
            trace_writer.record_event(
                context=ctx,
                event_type=TraceEventType.ASSET_ROUTE_DECISION_CREATED,
                status=TraceEventStatus.SUCCEEDED if selected_candidate else TraceEventStatus.FAILED,
                attributes=r_data,
            )

        return decision
