

from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRouteDecision,
    CandidateScore,
    GenerationMode,
    RoutingStrategy,
    StaticCapabilityMetadata,
    TierLevel,
)
from app.domain.enums import VisualType
from app.domain.evaluation import (
    DimensionEvaluationResult,
    DimensionEvaluationStatus,
    EvaluationDecision,
    EvaluationDimension,
    EvaluationReasonCode,
    EvaluationSnapshot,
)
from app.domain.quality_remediation import (
    QualityRemediationAction,
    QualityRemediationDecision,
    QualityRemediationReasonCode,
)
from app.domain.quality_remediation_policy import QualityRemediationPolicyEngine


def _make_candidate(
    cap_id: str,
    provider: str,
    model: str,
    vt: VisualType = VisualType.AI_VIDEO,
    gen_mode: GenerationMode = GenerationMode.TEXT_TO_VIDEO,
) -> AssetRouteCandidate:
    return AssetRouteCandidate(
        capability_id=cap_id,
        provider=provider,
        model=model,
        generation_mode=gen_mode,
        requested_visual_type=vt,
        is_eligible=True,
        rejection_reasons=(),
        static_metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.HIGH,
            cost_tier=TierLevel.MEDIUM,
            latency_tier=TierLevel.LOW,
        ),
        score=CandidateScore(
            capability_id=cap_id,
            quality_score=0.9,
            cost_score=0.6,
            latency_score=0.9,
            final_score=0.8,
            ranking_position=1,
        ),
    )


def _make_route_decision(
    mode: str = "AUTO",
    vt: VisualType = VisualType.AI_VIDEO,
    candidates: list[AssetRouteCandidate] | None = None,
) -> AssetRouteDecision:
    cands = candidates or [
        _make_candidate("cap-1", "mock_p1", "model-a", vt),
        _make_candidate("cap-2", "mock_p2", "model-b", vt),
        _make_candidate("cap-3", "mock_p3", "model-c", vt),
    ]
    return AssetRouteDecision(
        shot_id="shot-1",
        shot_revision_id="srev-1",
        routing_strategy=RoutingStrategy.BALANCED,
        requested_visual_type=vt,
        selected_candidate=cands[0],
        eligible_candidates=tuple(cands),
        rejected_candidates=(),
        model_selection_mode=mode,
    )


def _make_dimension_results(
    sem_score: float = 0.9,
    vis_score: float = 0.8,
    know_score: float = 0.95,
    comp_score: float = 0.85,
    sem_status: DimensionEvaluationStatus = DimensionEvaluationStatus.SCORED,
    vis_status: DimensionEvaluationStatus = DimensionEvaluationStatus.SCORED,
    know_status: DimensionEvaluationStatus = DimensionEvaluationStatus.SCORED,
    comp_status: DimensionEvaluationStatus = DimensionEvaluationStatus.SCORED,
    extra_reasons: dict[EvaluationDimension, list[str]] | None = None,
) -> list[DimensionEvaluationResult]:
    reasons = extra_reasons or {}
    return [
        DimensionEvaluationResult(
            evaluation_target_id="target-1",
            dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
            status=sem_status,
            score=sem_score if sem_status == DimensionEvaluationStatus.SCORED else None,
            reason_codes=tuple(reasons.get(EvaluationDimension.SEMANTIC_ALIGNMENT, ())),
        ),
        DimensionEvaluationResult(
            evaluation_target_id="target-1",
            dimension=EvaluationDimension.VISUAL_QUALITY,
            status=vis_status,
            score=vis_score if vis_status == DimensionEvaluationStatus.SCORED else None,
            reason_codes=tuple(reasons.get(EvaluationDimension.VISUAL_QUALITY, ())),
        ),
        DimensionEvaluationResult(
            evaluation_target_id="target-1",
            dimension=EvaluationDimension.KNOWLEDGE_ACCURACY,
            status=know_status,
            score=know_score if know_status == DimensionEvaluationStatus.SCORED else None,
            reason_codes=tuple(reasons.get(EvaluationDimension.KNOWLEDGE_ACCURACY, ())),
        ),
        DimensionEvaluationResult(
            evaluation_target_id="target-1",
            dimension=EvaluationDimension.COMPOSITION_SUITABILITY,
            status=comp_status,
            score=comp_score if comp_status == DimensionEvaluationStatus.SCORED else None,
            reason_codes=tuple(reasons.get(EvaluationDimension.COMPOSITION_SUITABILITY, ())),
        ),
    ]


def _make_snapshot(
    decision: EvaluationDecision,
    dim_results: list[DimensionEvaluationResult],
    summary_reasons: tuple[str, ...] = (),
) -> EvaluationSnapshot:
    return EvaluationSnapshot(
        evaluation_snapshot_id="snap-1",
        evaluation_target_id="target-1",
        dimension_result_ids=tuple(r.dimension_result_id for r in dim_results),
        decision=decision,
        policy_version="evaluation-policy-v1",
        evaluator_version="mock-v1",
        overall_score=0.88 if decision == EvaluationDecision.PASS else None,
        summary_reason_codes=summary_reasons,
    )


class TestQualityRemediationPolicyEngine:

    def test_pass_produces_accept_asset(self):
        """Invariant 1 & 2: PASS -> ACCEPT_ASSET, no regeneration."""
        dim_res = _make_dimension_results(0.9, 0.8, 0.95, 0.85)
        snap = _make_snapshot(EvaluationDecision.PASS, dim_res)
        route = _make_route_decision()

        decision = QualityRemediationPolicyEngine.evaluate(snap, dim_res, route)
        assert decision.action == QualityRemediationAction.ACCEPT_ASSET
        assert QualityRemediationReasonCode.QUALITY_PASS_ACCEPTED.value in decision.reason_codes
        assert decision.selected_route_candidate == route.selected_candidate

    def test_critical_indeterminate_does_not_regenerate_media(self):
        """Invariant 3: Critical evaluation INDETERMINATE does not regenerate media blindly."""
        dim_res = _make_dimension_results(
            know_score=0.0,
            know_status=DimensionEvaluationStatus.INDETERMINATE,
            extra_reasons={EvaluationDimension.KNOWLEDGE_ACCURACY: ["UNCERTAIN_FACTUALITY"]},
        )
        snap = _make_snapshot(EvaluationDecision.INDETERMINATE, dim_res)
        route = _make_route_decision()

        decision = QualityRemediationPolicyEngine.evaluate(snap, dim_res, route)
        assert decision.action == QualityRemediationAction.NEEDS_USER_ACTION
        assert decision.action != QualityRemediationAction.REGENERATE_SAME_ROUTE

    def test_transient_evaluator_error_produces_retry_evaluation_bounded(self):
        """Invariant 4: Temporary evaluator infrastructure uncertainty triggers bounded RETRY_EVALUATION."""
        dim_res = _make_dimension_results(
            vis_score=0.0,
            vis_status=DimensionEvaluationStatus.ERROR,
            extra_reasons={EvaluationDimension.VISUAL_QUALITY: [EvaluationReasonCode.EVALUATOR_CALL_FAILED.value]},
        )
        snap = _make_snapshot(EvaluationDecision.INDETERMINATE, dim_res, summary_reasons=(EvaluationReasonCode.EVALUATOR_CALL_FAILED.value,))
        route = _make_route_decision()

        # First evaluation error -> RETRY_EVALUATION
        decision = QualityRemediationPolicyEngine.evaluate(snap, dim_res, route, history=())
        assert decision.action == QualityRemediationAction.RETRY_EVALUATION
        assert QualityRemediationReasonCode.TRANSIENT_EVALUATION_ERROR.value in decision.reason_codes

        # Second evaluation error after retry budget (max 1) -> NEEDS_USER_ACTION
        history = [decision]
        decision2 = QualityRemediationPolicyEngine.evaluate(snap, dim_res, route, history=history)
        assert decision2.action == QualityRemediationAction.NEEDS_USER_ACTION
        assert QualityRemediationReasonCode.EVALUATION_RETRY_EXHAUSTED.value in decision2.reason_codes

    def test_missing_evidence_produces_needs_user_action(self):
        """Invariant 5: Missing required evidence eventually produces NEEDS_USER_ACTION."""
        dim_res = _make_dimension_results(
            know_score=0.0,
            know_status=DimensionEvaluationStatus.INDETERMINATE,
            extra_reasons={EvaluationDimension.KNOWLEDGE_ACCURACY: [EvaluationReasonCode.MISSING_EVIDENCE_CONTEXT.value]},
        )
        snap = _make_snapshot(EvaluationDecision.INDETERMINATE, dim_res, summary_reasons=(EvaluationReasonCode.MISSING_EVIDENCE_CONTEXT.value,))
        route = _make_route_decision()

        decision = QualityRemediationPolicyEngine.evaluate(snap, dim_res, route)
        assert decision.action == QualityRemediationAction.NEEDS_USER_ACTION
        assert QualityRemediationReasonCode.MISSING_EVIDENCE_NEEDS_USER.value in decision.reason_codes

    def test_visual_quality_only_failure_produces_regenerate_same_route(self):
        """Invariant 6 & 7: Visual-quality-only failure produces REGENERATE_SAME_ROUTE preserving candidate."""
        dim_res = _make_dimension_results(sem_score=0.92, vis_score=0.45, know_score=0.95, comp_score=0.8)
        snap = _make_snapshot(EvaluationDecision.FAIL, dim_res)
        route = _make_route_decision()

        decision = QualityRemediationPolicyEngine.evaluate(snap, dim_res, route, history=())
        assert decision.action == QualityRemediationAction.REGENERATE_SAME_ROUTE
        assert QualityRemediationReasonCode.VISUAL_QUALITY_STOCHASTIC_FAIL.value in decision.reason_codes
        assert decision.selected_route_candidate == route.selected_candidate

    def test_same_route_regeneration_budget_is_bounded(self):
        """Invariant 10: Same-route quality regeneration budget is bounded (moves to fallback next)."""
        dim_res = _make_dimension_results(sem_score=0.92, vis_score=0.45, know_score=0.95, comp_score=0.8)
        snap = _make_snapshot(EvaluationDecision.FAIL, dim_res)
        route = _make_route_decision(mode="AUTO")

        # Prior history already has 1 same-route regeneration
        prior = QualityRemediationDecision(
            quality_chain_id="chain-1",
            evaluation_snapshot_id="snap-0",
            shot_id="shot-1",
            shot_revision_id="srev-1",
            shot_asset_version_id="asset-0",
            action=QualityRemediationAction.REGENERATE_SAME_ROUTE,
            selected_route_candidate=route.selected_candidate,
        )

        decision = QualityRemediationPolicyEngine.evaluate(snap, dim_res, route, history=[prior])
        # Next candidate should be selected via fallback
        assert decision.action == QualityRemediationAction.FALLBACK_NEXT_ROUTE_CANDIDATE
        assert decision.selected_route_candidate.capability_id == "cap-2"

    def test_pinned_model_never_falls_back_to_another_provider_model(self):
        """Invariant 14: PINNED model never automatically falls back to another Provider/Model."""
        dim_res = _make_dimension_results(sem_score=0.60, vis_score=0.80, know_score=0.95, comp_score=0.8)
        snap = _make_snapshot(EvaluationDecision.FAIL, dim_res)
        route = _make_route_decision(mode="PINNED")

        prior = QualityRemediationDecision(
            quality_chain_id="chain-1",
            evaluation_snapshot_id="snap-0",
            shot_id="shot-1",
            shot_revision_id="srev-1",
            shot_asset_version_id="asset-0",
            action=QualityRemediationAction.REGENERATE_SAME_ROUTE,
            selected_route_candidate=route.selected_candidate,
        )

        decision = QualityRemediationPolicyEngine.evaluate(snap, dim_res, route, history=[prior])
        assert decision.action != QualityRemediationAction.FALLBACK_NEXT_ROUTE_CANDIDATE
        # In PINNED mode, it skips candidate fallback and moves to CONTROLLED_VISUAL_REPLAN
        assert decision.action == QualityRemediationAction.CONTROLLED_VISUAL_REPLAN
        assert QualityRemediationReasonCode.PINNED_MODEL_FALLBACK_PROHIBITED.value in decision.reason_codes

    def test_fallback_budget_bounded_and_moves_to_controlled_replan(self):
        """Invariant 11, 15, 16, 18: AUTO/PREFERRED uses fallback, then replan when fallback exhausted."""
        dim_res = _make_dimension_results(sem_score=0.60, vis_score=0.80, know_score=0.95, comp_score=0.8)
        snap = _make_snapshot(EvaluationDecision.FAIL, dim_res)
        route = _make_route_decision(mode="PREFERRED")

        # Prior: 1 same-route regen, 1 fallback already used
        prior_regen = QualityRemediationDecision(
            quality_chain_id="chain-1",
            evaluation_snapshot_id="snap-0",
            shot_id="shot-1",
            shot_revision_id="srev-1",
            shot_asset_version_id="asset-0",
            action=QualityRemediationAction.REGENERATE_SAME_ROUTE,
            selected_route_candidate=route.selected_candidate,
        )
        prior_fallback = QualityRemediationDecision(
            quality_chain_id="chain-1",
            evaluation_snapshot_id="snap-1",
            shot_id="shot-1",
            shot_revision_id="srev-1",
            shot_asset_version_id="asset-1",
            action=QualityRemediationAction.FALLBACK_NEXT_ROUTE_CANDIDATE,
            selected_route_candidate=route.eligible_candidates[1],
        )

        decision = QualityRemediationPolicyEngine.evaluate(
            snap, dim_res, route, history=[prior_regen, prior_fallback]
        )
        assert decision.action == QualityRemediationAction.CONTROLLED_VISUAL_REPLAN
        assert QualityRemediationReasonCode.FALLBACK_BUDGET_EXHAUSTED.value in decision.reason_codes

    def test_factual_narration_failure_stops_at_needs_user_action(self):
        """Invariant 17: KNOWLEDGE_ACCURACY failure caused by factual narration maps to NEEDS_USER_ACTION."""
        dim_res = _make_dimension_results(
            sem_score=0.9,
            vis_score=0.9,
            know_score=0.4,
            extra_reasons={EvaluationDimension.KNOWLEDGE_ACCURACY: [EvaluationReasonCode.FACTUAL_INCONSISTENCY.value]},
        )
        snap = _make_snapshot(
            EvaluationDecision.FAIL,
            dim_res,
            summary_reasons=(EvaluationReasonCode.FACTUAL_INCONSISTENCY.value,),
        )
        route = _make_route_decision()

        decision = QualityRemediationPolicyEngine.evaluate(snap, dim_res, route, history=())
        assert decision.action == QualityRemediationAction.NEEDS_USER_ACTION
        assert QualityRemediationReasonCode.FACTUAL_NARRATION_CONFLICT.value in decision.reason_codes

    def test_structural_replan_required_stops_at_needs_user_action(self):
        """Invariant 33: Structural replan requirement produces STRUCTURAL_REPLAN_REQUIRED -> NEEDS_USER_ACTION."""
        dim_res = _make_dimension_results(sem_score=0.5, vis_score=0.9)
        snap = _make_snapshot(
            EvaluationDecision.FAIL,
            dim_res,
            summary_reasons=("STRUCTURAL_REPLAN_REQUIRED",),
        )
        route = _make_route_decision()

        decision = QualityRemediationPolicyEngine.evaluate(snap, dim_res, route)
        assert decision.action == QualityRemediationAction.NEEDS_USER_ACTION
        assert QualityRemediationReasonCode.STRUCTURAL_REPLAN_REQUIRED.value in decision.reason_codes

    def test_replan_budget_exhausted_stops_at_needs_user_action(self):
        """Invariant 31, 32, 40: Max 1 auto-replan; failing replan candidate yields NEEDS_USER_ACTION."""
        dim_res = _make_dimension_results(sem_score=0.6, vis_score=0.6)
        snap = _make_snapshot(EvaluationDecision.FAIL, dim_res)
        route = _make_route_decision()

        # Prior history contains same-route regen, fallback, and 1 controlled visual replan
        history = [
            QualityRemediationDecision(
                quality_chain_id="chain-1",
                evaluation_snapshot_id="snap-0",
                shot_id="shot-1",
                shot_revision_id="srev-1",
                shot_asset_version_id="asset-0",
                action=QualityRemediationAction.REGENERATE_SAME_ROUTE,
                selected_route_candidate=route.selected_candidate,
            ),
            QualityRemediationDecision(
                quality_chain_id="chain-1",
                evaluation_snapshot_id="snap-1",
                shot_id="shot-1",
                shot_revision_id="srev-1",
                shot_asset_version_id="asset-1",
                action=QualityRemediationAction.FALLBACK_NEXT_ROUTE_CANDIDATE,
                selected_route_candidate=route.eligible_candidates[1],
            ),
            QualityRemediationDecision(
                quality_chain_id="chain-1",
                evaluation_snapshot_id="snap-2",
                shot_id="shot-1",
                shot_revision_id="srev-2",
                shot_asset_version_id="asset-2",
                action=QualityRemediationAction.CONTROLLED_VISUAL_REPLAN,
                selected_route_candidate=route.selected_candidate,
            ),
        ]

        decision = QualityRemediationPolicyEngine.evaluate(snap, dim_res, route, history=history)
        assert decision.action == QualityRemediationAction.NEEDS_USER_ACTION
        assert QualityRemediationReasonCode.TOTAL_BUDGET_EXHAUSTED.value in decision.reason_codes
