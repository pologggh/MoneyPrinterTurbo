from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.asset_execution import ShotAssetVersion
from app.domain.shot import ShotRevision

# ==============================================================================
# EVALUATION ENUMS & CONSTANTS
# ==============================================================================


class EvaluationDimension(str, Enum):
    """
    Exactly four V1 evaluation dimensions for multimodal asset evaluation.
    """
    SEMANTIC_ALIGNMENT = "SEMANTIC_ALIGNMENT"
    VISUAL_QUALITY = "VISUAL_QUALITY"
    KNOWLEDGE_ACCURACY = "KNOWLEDGE_ACCURACY"
    COMPOSITION_SUITABILITY = "COMPOSITION_SUITABILITY"


# Locked Critical Dimensions: must be SCORED to allow a PASS decision
CRITICAL_EVALUATION_DIMENSIONS: frozenset[EvaluationDimension] = frozenset({
    EvaluationDimension.SEMANTIC_ALIGNMENT,
    EvaluationDimension.KNOWLEDGE_ACCURACY,
})


class DimensionEvaluationStatus(str, Enum):
    """
    Explicit status of a single dimension evaluation.
    SCORED: meaningful assessment with numeric score.
    INDETERMINATE: evaluator cannot confidently determine the result (score must be None).
    ERROR: evaluation process itself failed (score must be None).
    """
    SCORED = "SCORED"
    INDETERMINATE = "INDETERMINATE"
    ERROR = "ERROR"


class EvaluationDecision(str, Enum):
    """
    Final snapshot-level decision contract.
    """
    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"


class EvaluationReasonCode(str, Enum):
    """
    Standard reason codes for evaluation dimension results and policy decisions.
    """
    EVALUATION_MEDIA_UNREADABLE = "EVALUATION_MEDIA_UNREADABLE"
    EVALUATOR_MODEL_UNAVAILABLE = "EVALUATOR_MODEL_UNAVAILABLE"
    MISSING_EVIDENCE_CONTEXT = "MISSING_EVIDENCE_CONTEXT"
    LOW_SEMANTIC_ALIGNMENT = "LOW_SEMANTIC_ALIGNMENT"
    LOW_VISUAL_QUALITY = "LOW_VISUAL_QUALITY"
    FACTUAL_INCONSISTENCY = "FACTUAL_INCONSISTENCY"
    UNSUITABLE_COMPOSITION = "UNSUITABLE_COMPOSITION"
    EVALUATOR_SCHEMA_ERROR = "EVALUATOR_SCHEMA_ERROR"
    EVALUATOR_CALL_FAILED = "EVALUATOR_CALL_FAILED"
    OVERALL_SCORE_BELOW_MINIMUM = "OVERALL_SCORE_BELOW_MINIMUM"
    CRITICAL_DIMENSION_BELOW_THRESHOLD = "CRITICAL_DIMENSION_BELOW_THRESHOLD"
    NON_CRITICAL_DIMENSION_BELOW_THRESHOLD = "NON_CRITICAL_DIMENSION_BELOW_THRESHOLD"
    DIMENSION_INDETERMINATE = "DIMENSION_INDETERMINATE"
    DIMENSION_ERROR = "DIMENSION_ERROR"


# ==============================================================================
# DOMAIN EXCEPTIONS
# ==============================================================================


class EvaluationError(Exception):
    """Base exception for evaluation domain invariant violations."""


class EvaluationMediaUnreadableError(EvaluationError):
    """Raised when asset media file cannot be read, decoded, or sampled."""


class EvaluationDimensionMissingError(EvaluationError, ValueError):
    """Raised when an EvaluationSnapshot does not contain all four required dimensions."""


class DuplicateEvaluationDimensionError(EvaluationError, ValueError):
    """Raised when an EvaluationSnapshot contains duplicate results for the same dimension."""


class EvaluationTargetMismatchError(EvaluationError, ValueError):
    """Raised when dimension results belong to a different EvaluationTarget than the snapshot."""


class InvalidEvaluationDecisionError(EvaluationError, ValueError):
    """Raised when decision violates structural invariants (e.g. PASS with critical INDETERMINATE/ERROR)."""


class InvalidEvaluationScoreError(EvaluationError, ValueError):
    """Raised when DimensionEvaluationResult score violates status-based numeric invariants."""


# ==============================================================================
# IMMUTABLE DOMAIN MODELS
# ==============================================================================


class EvaluationContextRef(BaseModel):
    """
    Extensible boundary for future evaluation context references (e.g., composition preview).
    Avoids future schema redesign while keeping Phase 6.1 boundary clean.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str  # e.g., "COMPOSITION_PREVIEW", "NEIGHBORING_SHOT", "SOURCE_MEDIA"
    reference_id: str


class EvaluationTarget(BaseModel):
    """
    Immutable evaluation target freezing exact content and exact media identities.
    Answers: 'What exact content and exact media are being evaluated together?'
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_target_id: str = Field(default_factory=lambda: str(uuid4()))
    shot_id: str
    shot_revision_id: str
    shot_asset_version_id: str
    evidence_refs_snapshot: tuple[str, ...] = Field(default=())
    context_refs: tuple[EvaluationContextRef, ...] = Field(default=())
    target_version: str = "v1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("shot_id", "shot_revision_id", "shot_asset_version_id")
    @classmethod
    def validate_non_empty_ids(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Target IDs (shot_id, shot_revision_id, shot_asset_version_id) must not be empty")
        return v.strip()


class DimensionEvaluationResult(BaseModel):
    """
    Immutable evaluation outcome for a single dimension on an exact EvaluationTarget.
    Enforces status-based score invariants and avoids model chain-of-thought.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension_result_id: str = Field(default_factory=lambda: str(uuid4()))
    evaluation_target_id: str
    dimension: EvaluationDimension
    status: DimensionEvaluationStatus
    score: float | None = None
    reason_codes: tuple[str, ...] = Field(default=())
    concise_summary: str | None = None
    evaluator_version: str = "v1.0"
    dimension_semantics_version: str = "v1.0"
    dependency_fingerprint: str | None = None
    error_code: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_score_status_invariant(self) -> DimensionEvaluationResult:
        if self.status == DimensionEvaluationStatus.SCORED:
            if self.score is None:
                raise InvalidEvaluationScoreError(
                    f"DimensionEvaluationResult with status SCORED requires a numeric score, got None for {self.dimension.value}"
                )
            if not (0.0 <= self.score <= 1.0):
                raise InvalidEvaluationScoreError(
                    f"Score must be within [0.0, 1.0], got {self.score} for {self.dimension.value}"
                )
        else:
            if self.score is not None:
                raise InvalidEvaluationScoreError(
                    f"DimensionEvaluationResult with status {self.status.value} must not contain a numeric score, got {self.score}"
                )
        return self


class EvaluationSnapshot(BaseModel):
    """
    Immutable snapshot freezing a complete evaluation view for an exact target.
    Contains exactly one result for each of the four dimensions.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    evaluation_target_id: str
    dimension_result_ids: tuple[str, ...] = Field(
        description="Frozen tuple of exact dimension_result_ids"
    )
    decision: EvaluationDecision
    policy_version: str = "v1.0"
    evaluator_version: str = "v1.0"
    overall_score: float | None = Field(default=None, ge=0.0, le=1.0)
    summary_reason_codes: tuple[str, ...] = Field(default=())
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==============================================================================
# FACTORY & VALIDATION FUNCTIONS
# ==============================================================================


def create_evaluation_target_from_shot(
    shot_revision: ShotRevision,
    shot_asset_version: ShotAssetVersion,
    context_refs: Sequence[EvaluationContextRef] = (),
    target_id: str | None = None,
    target_version: str = "v1.0",
) -> EvaluationTarget:
    """
    Factory creating an immutable EvaluationTarget from a ShotRevision and ShotAssetVersion.
    Validates that both belong to the same Shot identity.
    """
    if shot_revision.shot_id != shot_asset_version.shot_id:
        raise ValueError(
            f"ShotRevision shot_id '{shot_revision.shot_id}' does not match "
            f"ShotAssetVersion shot_id '{shot_asset_version.shot_id}'"
        )

    return EvaluationTarget(
        evaluation_target_id=target_id or str(uuid4()),
        shot_id=shot_revision.shot_id,
        shot_revision_id=shot_revision.shot_revision_id,
        shot_asset_version_id=shot_asset_version.shot_asset_version_id,
        evidence_refs_snapshot=tuple(shot_revision.evidence_refs),
        context_refs=tuple(context_refs),
        target_version=target_version,
    )


def create_evaluation_snapshot(
    target: EvaluationTarget,
    dimension_results: Sequence[DimensionEvaluationResult],
    decision: EvaluationDecision,
    policy_version: str = "v1.0",
    evaluator_version: str = "v1.0",
    overall_score: float | None = None,
    summary_reason_codes: Sequence[str] = (),
    snapshot_id: str | None = None,
) -> EvaluationSnapshot:
    """
    Factory creating an immutable EvaluationSnapshot enforcing all Phase 6.1 invariants:
    1. All dimension results must belong to the exact target.
    2. Exactly four dimensions must be present (no missing, no duplicate, no unknown).
    3. Structural decision safety: PASS is rejected if critical dimensions are INDETERMINATE or ERROR.
    4. Deterministic dimension result ordering by dimension enum value.
    """
    # 1. Target consistency check
    for r in dimension_results:
        if r.evaluation_target_id != target.evaluation_target_id:
            raise EvaluationTargetMismatchError(
                f"DimensionEvaluationResult {r.dimension_result_id} belongs to target "
                f"'{r.evaluation_target_id}', expected '{target.evaluation_target_id}'"
            )

    # 2. Dimensions completeness and uniqueness check
    results_by_dim: dict[EvaluationDimension, DimensionEvaluationResult] = {}
    for r in dimension_results:
        if r.dimension in results_by_dim:
            raise DuplicateEvaluationDimensionError(
                f"Duplicate dimension result provided for dimension '{r.dimension.value}'"
            )
        results_by_dim[r.dimension] = r

    all_required_dims = set(EvaluationDimension)
    missing_dims = all_required_dims - set(results_by_dim.keys())
    if missing_dims:
        missing_names = ", ".join(sorted(d.value for d in missing_dims))
        raise EvaluationDimensionMissingError(
            f"EvaluationSnapshot requires exactly four dimensions. Missing: {missing_names}"
        )

    # 3. Decision structural safety check
    if decision == EvaluationDecision.PASS:
        for crit_dim in CRITICAL_EVALUATION_DIMENSIONS:
            crit_res = results_by_dim[crit_dim]
            if crit_res.status != DimensionEvaluationStatus.SCORED:
                raise InvalidEvaluationDecisionError(
                    f"Cannot claim PASS decision when critical dimension {crit_dim.value} "
                    f"has status {crit_res.status.value}. Critical dimensions must be SCORED."
                )

    # 4. Deterministic dimension_result_ids ordering by EvaluationDimension definition order
    ordered_ids = tuple(
        results_by_dim[dim].dimension_result_id
        for dim in EvaluationDimension
    )

    return EvaluationSnapshot(
        evaluation_snapshot_id=snapshot_id or str(uuid4()),
        evaluation_target_id=target.evaluation_target_id,
        dimension_result_ids=ordered_ids,
        decision=decision,
        policy_version=policy_version,
        evaluator_version=evaluator_version,
        overall_score=overall_score,
        summary_reason_codes=tuple(summary_reason_codes),
    )
