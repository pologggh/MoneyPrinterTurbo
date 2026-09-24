from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.asset_router import AssetRouteCandidate


class QualityRemediationAction(str, Enum):
    """
    Deterministic remediation actions decided strictly after an EvaluationSnapshot.
    """
    ACCEPT_ASSET = "ACCEPT_ASSET"
    RETRY_EVALUATION = "RETRY_EVALUATION"
    REGENERATE_SAME_ROUTE = "REGENERATE_SAME_ROUTE"
    FALLBACK_NEXT_ROUTE_CANDIDATE = "FALLBACK_NEXT_ROUTE_CANDIDATE"
    CONTROLLED_VISUAL_REPLAN = "CONTROLLED_VISUAL_REPLAN"
    NEEDS_USER_ACTION = "NEEDS_USER_ACTION"


class QualityRemediationReasonCode(str, Enum):
    """
    Standard structured reason codes for remediation decisions.
    """
    QUALITY_PASS_ACCEPTED = "QUALITY_PASS_ACCEPTED"
    TRANSIENT_EVALUATION_ERROR = "TRANSIENT_EVALUATION_ERROR"
    EVALUATION_RETRY_EXHAUSTED = "EVALUATION_RETRY_EXHAUSTED"
    MISSING_EVIDENCE_NEEDS_USER = "MISSING_EVIDENCE_NEEDS_USER"
    EVALUATION_UNCERTAINTY_UNRESOLVED = "EVALUATION_UNCERTAINTY_UNRESOLVED"
    VISUAL_QUALITY_STOCHASTIC_FAIL = "VISUAL_QUALITY_STOCHASTIC_FAIL"
    SEMANTIC_ALIGNMENT_INITIAL_FAIL = "SEMANTIC_ALIGNMENT_INITIAL_FAIL"
    ROUTE_REPEATED_FAIL = "ROUTE_REPEATED_FAIL"
    PINNED_MODEL_FALLBACK_PROHIBITED = "PINNED_MODEL_FALLBACK_PROHIBITED"
    FALLBACK_BUDGET_EXHAUSTED = "FALLBACK_BUDGET_EXHAUSTED"
    NO_ELIGIBLE_FALLBACK_CANDIDATES = "NO_ELIGIBLE_FALLBACK_CANDIDATES"
    VISUAL_PLANNING_ISSUE = "VISUAL_PLANNING_ISSUE"
    REPLAN_BUDGET_EXHAUSTED = "REPLAN_BUDGET_EXHAUSTED"
    FACTUAL_NARRATION_CONFLICT = "FACTUAL_NARRATION_CONFLICT"
    STRUCTURAL_REPLAN_REQUIRED = "STRUCTURAL_REPLAN_REQUIRED"
    TOTAL_BUDGET_EXHAUSTED = "TOTAL_BUDGET_EXHAUSTED"


DEFAULT_REMEDIATION_POLICY_VERSION = "remediation-policy-v1"


class QualityRemediationPolicy(BaseModel):
    """
    Immutable versioned quality remediation policy governing retry and fallback budgets.
    Completely decoupled from Phase 5 technical execution retry limits.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_version: str = DEFAULT_REMEDIATION_POLICY_VERSION
    max_same_route_quality_regenerations: int = 1
    max_route_fallbacks: int = 1
    max_controlled_auto_replans: int = 1
    max_evaluation_retries: int = 1


class QualityRemediationDecision(BaseModel):
    """
    Immutable typed decision capturing the chosen quality remediation action and provenance.
    Append-only: old remediation decisions are never mutated.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    remediation_decision_id: str = Field(default_factory=lambda: str(uuid4()))
    quality_chain_id: str
    evaluation_snapshot_id: str
    shot_id: str
    shot_revision_id: str
    shot_asset_version_id: str
    action: QualityRemediationAction
    reason_codes: tuple[str, ...] = ()
    remediation_policy_version: str = DEFAULT_REMEDIATION_POLICY_VERSION
    quality_attempt_index: int = 0
    selected_route_candidate: AssetRouteCandidate | None = None
    explanation: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ShotQualitySelection(BaseModel):
    """
    Immutable formal asset quality selection identifying the exact ShotAssetVersion
    and exact EvaluationSnapshot accepted for production.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    quality_selection_id: str = Field(default_factory=lambda: str(uuid4()))
    quality_chain_id: str
    shot_id: str
    shot_revision_id: str
    shot_asset_version_id: str
    evaluation_snapshot_id: str
    selection_source: str = "EVALUATION_PASS"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
