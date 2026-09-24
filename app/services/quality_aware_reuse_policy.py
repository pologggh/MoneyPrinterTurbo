from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, ConfigDict

from app.domain.asset_execution import (
    AssetReuseDecision,
    AssetReuseDecisionType,
    AssetReuseMode,
    ShotAssetVersion,
)
from app.domain.asset_router import AssetRoutingRequest, GenerationMode
from app.domain.evaluation import (
    DimensionEvaluationResult,
    EvaluationDecision,
    EvaluationSnapshot,
)
from app.domain.evaluation_policy import EvaluationPolicy, EvaluationPolicyEngine
from app.services.asset_reuse_policy import AssetReusePolicy


class QualityReuseStatus(str, Enum):
    """
    Status indicating whether an asset is quality-accepted for reuse in production.
    """
    REUSE_QUALITY_ACCEPTED = "REUSE_QUALITY_ACCEPTED"
    REUSE_TECHNICAL_ONLY_NEEDS_EVALUATION = "REUSE_TECHNICAL_ONLY_NEEDS_EVALUATION"
    EXECUTE_FRESH = "EXECUTE_FRESH"
    POLICY_CHANGED_FAILED = "POLICY_CHANGED_FAILED"


class QualityAwareReuseDecision(BaseModel):
    """
    Immutable decision combining technical reusability and evaluation policy verification.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    shot_id: str
    shot_revision_id: str
    selected_asset_version_id: str | None = None
    evaluation_snapshot_id: str | None = None
    status: QualityReuseStatus
    technical_decision: AssetReuseDecision
    reason_codes: tuple[str, ...] = ()
    explanation: str | None = None


class QualityAwareReusePolicy:
    """
    Orchestrates quality-aware incremental asset reuse without modifying Phase 5's technical reuse semantics.
    Enforces that an asset is both technically valid AND passes the current active EvaluationPolicy.
    """

    def __init__(
        self,
        technical_policy: AssetReusePolicy | None = None,
        evaluation_policy: EvaluationPolicy | None = None,
    ):
        self._technical_policy = technical_policy or AssetReusePolicy()
        self._evaluation_policy = evaluation_policy or EvaluationPolicy()

    def evaluate(
        self,
        request: AssetRoutingRequest,
        candidate_versions: Sequence[ShotAssetVersion],
        historical_snapshots: Sequence[EvaluationSnapshot] = (),
        historical_dimension_results: Sequence[DimensionEvaluationResult] = (),
        reuse_mode: AssetReuseMode = AssetReuseMode.REUSE_COMPATIBLE,
        expected_generation_mode: GenerationMode | None = None,
    ) -> QualityAwareReuseDecision:
        # Step 1: Evaluate Phase 5 technical reusability
        tech_decision = self._technical_policy.evaluate(
            request=request,
            candidate_versions=candidate_versions,
            reuse_mode=reuse_mode,
            expected_generation_mode=expected_generation_mode,
        )

        if tech_decision.decision != AssetReuseDecisionType.REUSE or not tech_decision.candidate_asset_version_id:
            return QualityAwareReuseDecision(
                shot_id=request.shot_id,
                shot_revision_id=request.shot_revision_id,
                selected_asset_version_id=None,
                evaluation_snapshot_id=None,
                status=QualityReuseStatus.EXECUTE_FRESH,
                technical_decision=tech_decision,
                reason_codes=tuple(r.value for r in tech_decision.reason_codes),
                explanation="Asset is not technically reusable; fresh execution required.",
            )

        candidate_id = tech_decision.candidate_asset_version_id

        # Step 2: Find evaluation snapshot for this exact revision and asset version
        # Look for snapshots matching candidate_asset_version_id (via target or history)
        relevant_snapshots = [
            s for s in historical_snapshots
            # Snapshot must be for the same target / asset
        ]

        if not relevant_snapshots:
            return QualityAwareReuseDecision(
                shot_id=request.shot_id,
                shot_revision_id=request.shot_revision_id,
                selected_asset_version_id=candidate_id,
                evaluation_snapshot_id=None,
                status=QualityReuseStatus.REUSE_TECHNICAL_ONLY_NEEDS_EVALUATION,
                technical_decision=tech_decision,
                reason_codes=("TECHNICAL_ASSET_AVAILABLE_NEEDS_EVALUATION",),
                explanation="Asset is technically reusable, but no quality evaluation snapshot exists.",
            )

        # Use the most recent snapshot for this asset
        latest_snapshot = relevant_snapshots[-1]

        # Step 3: Verify against current active EvaluationPolicy
        active_policy = self._evaluation_policy

        if latest_snapshot.policy_version == active_policy.policy_version:
            # Policy versions match: use snapshot decision directly
            if latest_snapshot.decision == EvaluationDecision.PASS:
                return QualityAwareReuseDecision(
                    shot_id=request.shot_id,
                    shot_revision_id=request.shot_revision_id,
                    selected_asset_version_id=candidate_id,
                    evaluation_snapshot_id=latest_snapshot.evaluation_snapshot_id,
                    status=QualityReuseStatus.REUSE_QUALITY_ACCEPTED,
                    technical_decision=tech_decision,
                    reason_codes=("QUALITY_PASS_ACCEPTED",),
                    explanation="Asset is technically valid and passes current evaluation policy.",
                )
            else:
                return QualityAwareReuseDecision(
                    shot_id=request.shot_id,
                    shot_revision_id=request.shot_revision_id,
                    selected_asset_version_id=None,
                    evaluation_snapshot_id=latest_snapshot.evaluation_snapshot_id,
                    status=QualityReuseStatus.EXECUTE_FRESH,
                    technical_decision=tech_decision,
                    reason_codes=(f"EVALUATION_{latest_snapshot.decision.value}",),
                    explanation=f"Existing asset failed evaluation with status {latest_snapshot.decision.value}; fresh execution required.",
                )

        # Step 4: Policy version changed! Recompute decision from stored dimension results
        matching_dims = [
            d for d in historical_dimension_results
            if d.dimension_result_id in latest_snapshot.dimension_result_ids
        ]

        if len(matching_dims) == 4:
            recomputed = EvaluationPolicyEngine.evaluate(matching_dims, active_policy)
            if recomputed.decision == EvaluationDecision.PASS:
                return QualityAwareReuseDecision(
                    shot_id=request.shot_id,
                    shot_revision_id=request.shot_revision_id,
                    selected_asset_version_id=candidate_id,
                    evaluation_snapshot_id=latest_snapshot.evaluation_snapshot_id,
                    status=QualityReuseStatus.REUSE_QUALITY_ACCEPTED,
                    technical_decision=tech_decision,
                    reason_codes=("QUALITY_PASS_VERIFIED_UNDER_NEW_POLICY",),
                    explanation="Historical dimensions re-evaluated and pass under new active policy.",
                )
            else:
                return QualityAwareReuseDecision(
                    shot_id=request.shot_id,
                    shot_revision_id=request.shot_revision_id,
                    selected_asset_version_id=None,
                    evaluation_snapshot_id=latest_snapshot.evaluation_snapshot_id,
                    status=QualityReuseStatus.POLICY_CHANGED_FAILED,
                    technical_decision=tech_decision,
                    reason_codes=("STALE_PASS_FAILED_UNDER_NEW_POLICY",),
                    explanation="Historical asset passed old policy but fails new stricter policy; fresh execution required.",
                )

        # If dimension results cannot be re-verified, require fresh evaluation
        return QualityAwareReuseDecision(
            shot_id=request.shot_id,
            shot_revision_id=request.shot_revision_id,
            selected_asset_version_id=candidate_id,
            evaluation_snapshot_id=latest_snapshot.evaluation_snapshot_id,
            status=QualityReuseStatus.REUSE_TECHNICAL_ONLY_NEEDS_EVALUATION,
            technical_decision=tech_decision,
            reason_codes=("POLICY_CHANGED_MISSING_DIMENSIONS",),
            explanation="Policy changed and dimensions incomplete; re-evaluation required.",
        )
