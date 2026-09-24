from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar
from uuid import uuid4

from app.domain.asset_router import AssetRouteCandidate, AssetRouteDecision
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
    QualityRemediationPolicy,
    QualityRemediationReasonCode,
)


class QualityRemediationPolicyEngine:
    """
    Pure deterministic policy engine for deciding quality remediation actions.
    Performs zero external I/O or LLM calls.
    Adheres strictly to bounded retry/fallback/replan policies.
    """

    DEFAULT_POLICY: ClassVar[QualityRemediationPolicy] = QualityRemediationPolicy()

    @classmethod
    def evaluate(
        cls,
        snapshot: EvaluationSnapshot,
        dimension_results: Sequence[DimensionEvaluationResult],
        route_decision: AssetRouteDecision,
        history: Sequence[QualityRemediationDecision] = (),
        policy: QualityRemediationPolicy | None = None,
        quality_chain_id: str | None = None,
        shot_id: str | None = None,
        shot_revision_id: str | None = None,
        shot_asset_version_id: str | None = None,
    ) -> QualityRemediationDecision:
        pol = policy or cls.DEFAULT_POLICY
        chain_id = quality_chain_id or (
            history[0].quality_chain_id if history else str(uuid4())
        )
        attempt_index = len(history)

        dim_map: dict[EvaluationDimension, DimensionEvaluationResult] = {
            res.dimension: res for res in dimension_results
        }

        # Resolve target identifiers
        sid = shot_id or (history[-1].shot_id if history else route_decision.shot_id)
        srev_id = shot_revision_id or (
            history[-1].shot_revision_id if history else route_decision.shot_revision_id
        )
        sasset_id = shot_asset_version_id or (
            history[-1].shot_asset_version_id if history else ""
        )

        # ----------------------------------------------------------------------
        # 1. PASS BEHAVIOR
        # ----------------------------------------------------------------------
        if snapshot.decision == EvaluationDecision.PASS:
            return QualityRemediationDecision(
                quality_chain_id=chain_id,
                evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
                shot_id=sid,
                shot_revision_id=srev_id,
                shot_asset_version_id=sasset_id,
                action=QualityRemediationAction.ACCEPT_ASSET,
                reason_codes=(QualityRemediationReasonCode.QUALITY_PASS_ACCEPTED.value,),
                remediation_policy_version=pol.policy_version,
                quality_attempt_index=attempt_index,
                selected_route_candidate=route_decision.selected_candidate,
                explanation="Evaluation passed all policy thresholds; asset accepted.",
            )

        # ----------------------------------------------------------------------
        # 2. INDETERMINATE BEHAVIOR
        # ----------------------------------------------------------------------
        if snapshot.decision == EvaluationDecision.INDETERMINATE:
            # Check for missing evidence context -> always NEEDS_USER_ACTION
            all_reasons = set(snapshot.summary_reason_codes)
            for res in dimension_results:
                all_reasons.update(res.reason_codes)

            if EvaluationReasonCode.MISSING_EVIDENCE_CONTEXT.value in all_reasons:
                return QualityRemediationDecision(
                    quality_chain_id=chain_id,
                    evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
                    shot_id=sid,
                    shot_revision_id=srev_id,
                    shot_asset_version_id=sasset_id,
                    action=QualityRemediationAction.NEEDS_USER_ACTION,
                    reason_codes=(
                        QualityRemediationReasonCode.MISSING_EVIDENCE_NEEDS_USER.value,
                        EvaluationReasonCode.MISSING_EVIDENCE_CONTEXT.value,
                    ),
                    remediation_policy_version=pol.policy_version,
                    quality_attempt_index=attempt_index,
                    selected_route_candidate=route_decision.selected_candidate,
                    explanation="Knowledge accuracy cannot be evaluated: missing required evidence context.",
                )

            # Check for structural replan or content requirement
            if "STRUCTURAL_REPLAN_REQUIRED" in all_reasons:
                return QualityRemediationDecision(
                    quality_chain_id=chain_id,
                    evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
                    shot_id=sid,
                    shot_revision_id=srev_id,
                    shot_asset_version_id=sasset_id,
                    action=QualityRemediationAction.NEEDS_USER_ACTION,
                    reason_codes=(
                        QualityRemediationReasonCode.STRUCTURAL_REPLAN_REQUIRED.value,
                    ),
                    remediation_policy_version=pol.policy_version,
                    quality_attempt_index=attempt_index,
                    selected_route_candidate=route_decision.selected_candidate,
                    explanation="Structural replan required; automatic remediation stopped.",
                )

            # Check if any dimension is ERROR (transient infrastructure/call failure)
            has_error = any(
                res.status == DimensionEvaluationStatus.ERROR
                for res in dimension_results
            )
            transient_codes = {
                EvaluationReasonCode.EVALUATOR_CALL_FAILED.value,
                EvaluationReasonCode.EVALUATOR_SCHEMA_ERROR.value,
                EvaluationReasonCode.EVALUATION_MEDIA_UNREADABLE.value,
                EvaluationReasonCode.EVALUATOR_MODEL_UNAVAILABLE.value,
            }
            is_transient = has_error or any(c in all_reasons for c in transient_codes)

            eval_retries_used = sum(
                1 for d in history if d.action == QualityRemediationAction.RETRY_EVALUATION
            )

            if is_transient and eval_retries_used < pol.max_evaluation_retries:
                return QualityRemediationDecision(
                    quality_chain_id=chain_id,
                    evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
                    shot_id=sid,
                    shot_revision_id=srev_id,
                    shot_asset_version_id=sasset_id,
                    action=QualityRemediationAction.RETRY_EVALUATION,
                    reason_codes=(
                        QualityRemediationReasonCode.TRANSIENT_EVALUATION_ERROR.value,
                    ),
                    remediation_policy_version=pol.policy_version,
                    quality_attempt_index=attempt_index,
                    selected_route_candidate=route_decision.selected_candidate,
                    explanation="Transient evaluation error occurred; retrying evaluation within budget.",
                )

            if is_transient:
                reason_code = QualityRemediationReasonCode.EVALUATION_RETRY_EXHAUSTED.value
            else:
                reason_code = QualityRemediationReasonCode.EVALUATION_UNCERTAINTY_UNRESOLVED.value

            return QualityRemediationDecision(
                quality_chain_id=chain_id,
                evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
                shot_id=sid,
                shot_revision_id=srev_id,
                shot_asset_version_id=sasset_id,
                action=QualityRemediationAction.NEEDS_USER_ACTION,
                reason_codes=(reason_code,),
                remediation_policy_version=pol.policy_version,
                quality_attempt_index=attempt_index,
                selected_route_candidate=route_decision.selected_candidate,
                explanation="Evaluation uncertainty unresolved; media generation credit spent is prevented.",
            )

        # ----------------------------------------------------------------------
        # 3. FAIL BEHAVIOR
        # ----------------------------------------------------------------------
        all_reasons = set(snapshot.summary_reason_codes)
        for res in dimension_results:
            all_reasons.update(res.reason_codes)

        # Hard Rule: Structural replan needed -> NEEDS_USER_ACTION
        if "STRUCTURAL_REPLAN_REQUIRED" in all_reasons:
            return QualityRemediationDecision(
                quality_chain_id=chain_id,
                evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
                shot_id=sid,
                shot_revision_id=srev_id,
                shot_asset_version_id=sasset_id,
                action=QualityRemediationAction.NEEDS_USER_ACTION,
                reason_codes=(
                    QualityRemediationReasonCode.STRUCTURAL_REPLAN_REQUIRED.value,
                ),
                remediation_policy_version=pol.policy_version,
                quality_attempt_index=attempt_index,
                selected_route_candidate=route_decision.selected_candidate,
                explanation="Structural replan required; automatic remediation stopped.",
            )

        # Hard Rule: Factual narration/knowledge conflict -> NEVER auto-replan narration
        if (
            EvaluationReasonCode.FACTUAL_INCONSISTENCY.value in all_reasons
            or "FACTUAL_NARRATION_CONFLICT" in all_reasons
        ):
            return QualityRemediationDecision(
                quality_chain_id=chain_id,
                evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
                shot_id=sid,
                shot_revision_id=srev_id,
                shot_asset_version_id=sasset_id,
                action=QualityRemediationAction.NEEDS_USER_ACTION,
                reason_codes=(
                    QualityRemediationReasonCode.FACTUAL_NARRATION_CONFLICT.value,
                    EvaluationReasonCode.FACTUAL_INCONSISTENCY.value,
                ),
                remediation_policy_version=pol.policy_version,
                quality_attempt_index=attempt_index,
                selected_route_candidate=route_decision.selected_candidate,
                explanation="Factual inconsistency with narration/evidence detected; user intervention required.",
            )

        # Calculate budgets consumed in this quality chain
        # Note: Same-route quality regenerations count for the current route candidate
        current_candidate_id = (
            route_decision.selected_candidate.capability_id
            if route_decision.selected_candidate
            else None
        )
        same_route_regens_used = sum(
            1
            for d in history
            if d.action == QualityRemediationAction.REGENERATE_SAME_ROUTE
        )
        fallbacks_used = sum(
            1
            for d in history
            if d.action == QualityRemediationAction.FALLBACK_NEXT_ROUTE_CANDIDATE
        )
        replans_used = sum(
            1
            for d in history
            if d.action == QualityRemediationAction.CONTROLLED_VISUAL_REPLAN
        )

        # Determine failure category
        sem_res = dim_map.get(EvaluationDimension.SEMANTIC_ALIGNMENT)
        vis_res = dim_map.get(EvaluationDimension.VISUAL_QUALITY)
        comp_res = dim_map.get(EvaluationDimension.COMPOSITION_SUITABILITY)

        sem_failed = (
            sem_res is not None
            and sem_res.status == DimensionEvaluationStatus.SCORED
            and sem_res.score is not None
            and sem_res.score < 0.80
        )
        vis_failed = (
            vis_res is not None
            and vis_res.status == DimensionEvaluationStatus.SCORED
            and vis_res.score is not None
            and vis_res.score < 0.65
        )
        comp_failed = (
            comp_res is not None
            and comp_res.status == DimensionEvaluationStatus.SCORED
            and comp_res.score is not None
            and comp_res.score < 0.65
        )

        # 3.1 Initial stochastic visual quality or semantic failure -> REGENERATE_SAME_ROUTE
        can_regenerate_same = same_route_regens_used < pol.max_same_route_quality_regenerations

        if can_regenerate_same:
            if vis_failed and not sem_failed:
                return QualityRemediationDecision(
                    quality_chain_id=chain_id,
                    evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
                    shot_id=sid,
                    shot_revision_id=srev_id,
                    shot_asset_version_id=sasset_id,
                    action=QualityRemediationAction.REGENERATE_SAME_ROUTE,
                    reason_codes=(
                        QualityRemediationReasonCode.VISUAL_QUALITY_STOCHASTIC_FAIL.value,
                    ),
                    remediation_policy_version=pol.policy_version,
                    quality_attempt_index=attempt_index,
                    selected_route_candidate=route_decision.selected_candidate,
                    explanation="Visual quality failed on stochastic generation; regenerating same route candidate.",
                )
            if sem_failed or comp_failed or snapshot.overall_score is not None:
                return QualityRemediationDecision(
                    quality_chain_id=chain_id,
                    evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
                    shot_id=sid,
                    shot_revision_id=srev_id,
                    shot_asset_version_id=sasset_id,
                    action=QualityRemediationAction.REGENERATE_SAME_ROUTE,
                    reason_codes=(
                        QualityRemediationReasonCode.SEMANTIC_ALIGNMENT_INITIAL_FAIL.value,
                    ),
                    remediation_policy_version=pol.policy_version,
                    quality_attempt_index=attempt_index,
                    selected_route_candidate=route_decision.selected_candidate,
                    explanation="First attempt failed quality thresholds; attempting same-route stochastic regeneration.",
                )

        # 3.2 Candidate mismatch / Repeated failure on current candidate -> FALLBACK_NEXT_ROUTE_CANDIDATE
        # Model selection mode check: PINNED model prohibits provider/model fallback
        is_pinned = (
            getattr(route_decision, "model_selection_mode", "AUTO").upper() == "PINNED"
        )

        if not is_pinned and fallbacks_used < pol.max_route_fallbacks:
            # Find next eligible candidate with matching VisualType from frozen candidate list
            expected_vt = route_decision.requested_visual_type
            used_capability_ids = {
                current_candidate_id,
                *(
                    d.selected_route_candidate.capability_id
                    for d in history
                    if d.selected_route_candidate is not None
                ),
            }

            next_candidate: AssetRouteCandidate | None = None
            for cand in route_decision.eligible_candidates:
                if (
                    cand.requested_visual_type == expected_vt
                    and cand.capability_id not in used_capability_ids
                    and cand.is_eligible
                ):
                    next_candidate = cand
                    break

            if next_candidate is not None:
                return QualityRemediationDecision(
                    quality_chain_id=chain_id,
                    evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
                    shot_id=sid,
                    shot_revision_id=srev_id,
                    shot_asset_version_id=sasset_id,
                    action=QualityRemediationAction.FALLBACK_NEXT_ROUTE_CANDIDATE,
                    reason_codes=(QualityRemediationReasonCode.ROUTE_REPEATED_FAIL.value,),
                    remediation_policy_version=pol.policy_version,
                    quality_attempt_index=attempt_index,
                    selected_route_candidate=next_candidate,
                    explanation=f"Route candidate {current_candidate_id} failed repeatedly; falling back to next frozen candidate {next_candidate.capability_id}.",
                )

        # If pinned, record that fallback was prohibited
        extra_reason_codes: list[str] = []
        if is_pinned:
            extra_reason_codes.append(
                QualityRemediationReasonCode.PINNED_MODEL_FALLBACK_PROHIBITED.value
            )
        elif fallbacks_used >= pol.max_route_fallbacks:
            extra_reason_codes.append(
                QualityRemediationReasonCode.FALLBACK_BUDGET_EXHAUSTED.value
            )
        else:
            extra_reason_codes.append(
                QualityRemediationReasonCode.NO_ELIGIBLE_FALLBACK_CANDIDATES.value
            )

        # 3.3 Visual Planning Issue -> CONTROLLED_VISUAL_REPLAN
        if replans_used < pol.max_controlled_auto_replans:
            return QualityRemediationDecision(
                quality_chain_id=chain_id,
                evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
                shot_id=sid,
                shot_revision_id=srev_id,
                shot_asset_version_id=sasset_id,
                action=QualityRemediationAction.CONTROLLED_VISUAL_REPLAN,
                reason_codes=(
                    QualityRemediationReasonCode.VISUAL_PLANNING_ISSUE.value,
                    *extra_reason_codes,
                ),
                remediation_policy_version=pol.policy_version,
                quality_attempt_index=attempt_index,
                selected_route_candidate=route_decision.selected_candidate,
                explanation="Quality failure persists across available routes; triggering bounded controlled visual replan.",
            )

        # 3.4 All remediation budgets exhausted -> NEEDS_USER_ACTION
        return QualityRemediationDecision(
            quality_chain_id=chain_id,
            evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
            shot_id=sid,
            shot_revision_id=srev_id,
            shot_asset_version_id=sasset_id,
            action=QualityRemediationAction.NEEDS_USER_ACTION,
            reason_codes=(
                QualityRemediationReasonCode.TOTAL_BUDGET_EXHAUSTED.value,
                QualityRemediationReasonCode.REPLAN_BUDGET_EXHAUSTED.value,
                *extra_reason_codes,
            ),
            remediation_policy_version=pol.policy_version,
            quality_attempt_index=attempt_index,
            selected_route_candidate=route_decision.selected_candidate,
            explanation="All automatic quality remediation budgets exhausted; user intervention required.",
        )
