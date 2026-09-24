from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.domain.evaluation import (
    CRITICAL_EVALUATION_DIMENSIONS,
    DimensionEvaluationResult,
    DimensionEvaluationStatus,
    EvaluationDecision,
    EvaluationDimension,
    EvaluationDimensionMissingError,
)

DEFAULT_POLICY_VERSION = "evaluation-policy-v1"

DEFAULT_DIMENSION_THRESHOLDS: dict[EvaluationDimension, float] = {
    EvaluationDimension.SEMANTIC_ALIGNMENT: 0.80,
    EvaluationDimension.VISUAL_QUALITY: 0.65,
    EvaluationDimension.KNOWLEDGE_ACCURACY: 0.80,
    EvaluationDimension.COMPOSITION_SUITABILITY: 0.65,
}

DEFAULT_DIMENSION_WEIGHTS: dict[EvaluationDimension, float] = {
    EvaluationDimension.SEMANTIC_ALIGNMENT: 0.30,
    EvaluationDimension.VISUAL_QUALITY: 0.20,
    EvaluationDimension.KNOWLEDGE_ACCURACY: 0.35,
    EvaluationDimension.COMPOSITION_SUITABILITY: 0.15,
}

DEFAULT_MINIMUM_OVERALL_SCORE = 0.75


class EvaluationPolicy(BaseModel):
    """
    Immutable versioned evaluation policy defining thresholds, weights, and minimum overall score.
    Centralizes all engineering defaults without scattering threshold literals across application code.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_version: str = DEFAULT_POLICY_VERSION
    thresholds: Mapping[EvaluationDimension, float] = Field(
        default_factory=lambda: dict(DEFAULT_DIMENSION_THRESHOLDS)
    )
    weights: Mapping[EvaluationDimension, float] = Field(
        default_factory=lambda: dict(DEFAULT_DIMENSION_WEIGHTS)
    )
    minimum_overall_score: float = DEFAULT_MINIMUM_OVERALL_SCORE
    critical_dimensions: frozenset[EvaluationDimension] = Field(
        default=CRITICAL_EVALUATION_DIMENSIONS
    )


class EvaluationPolicyDecision(BaseModel):
    """
    Deterministic result produced by the EvaluationPolicyEngine.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: EvaluationDecision
    overall_score: float | None = None
    failed_dimensions: tuple[EvaluationDimension, ...] = ()
    indeterminate_dimensions: tuple[EvaluationDimension, ...] = ()
    error_dimensions: tuple[EvaluationDimension, ...] = ()
    reason_codes: tuple[str, ...] = ()
    policy_version: str = DEFAULT_POLICY_VERSION


class EvaluationPolicyEngine:
    """
    Pure deterministic policy evaluation engine.
    Performs zero external I/O or LLM/VLM calls.
    Adheres strictly to the 5-step deterministic decision order.
    """

    DEFAULT_POLICY: ClassVar[EvaluationPolicy] = EvaluationPolicy()

    @classmethod
    def evaluate(
        cls,
        dimension_results: Sequence[DimensionEvaluationResult],
        policy: EvaluationPolicy | None = None,
    ) -> EvaluationPolicyDecision:
        pol = policy or cls.DEFAULT_POLICY

        # 1. Completeness validation: exactly 4 dimensions required
        dim_map: dict[EvaluationDimension, DimensionEvaluationResult] = {}
        for res in dimension_results:
            dim_map[res.dimension] = res

        missing = set(EvaluationDimension) - set(dim_map.keys())
        if missing:
            missing_str = ", ".join(sorted(d.value for d in missing))
            raise EvaluationDimensionMissingError(
                f"PolicyEngine requires all four dimensions. Missing: {missing_str}"
            )

        failed_dims: list[EvaluationDimension] = []
        indet_dims: list[EvaluationDimension] = []
        error_dims: list[EvaluationDimension] = []
        reason_codes: list[str] = []

        # Classify dimension statuses
        for dim in (
            EvaluationDimension.SEMANTIC_ALIGNMENT,
            EvaluationDimension.VISUAL_QUALITY,
            EvaluationDimension.KNOWLEDGE_ACCURACY,
            EvaluationDimension.COMPOSITION_SUITABILITY,
        ):
            res = dim_map[dim]
            if res.status == DimensionEvaluationStatus.ERROR:
                error_dims.append(dim)
            elif res.status == DimensionEvaluationStatus.INDETERMINATE:
                indet_dims.append(dim)
            elif res.status == DimensionEvaluationStatus.SCORED:
                thresh = pol.thresholds.get(dim, 0.0)
                if res.score is not None and res.score < thresh:
                    failed_dims.append(dim)

        critical_dims = pol.critical_dimensions

        # Rule 1: Critical dimension ERROR or INDETERMINATE -> INDETERMINATE
        critical_errors = [d for d in error_dims if d in critical_dims]
        critical_indets = [d for d in indet_dims if d in critical_dims]
        if critical_errors or critical_indets:
            for d in critical_errors:
                reason_codes.append(f"{d.value}_ERROR")
            for d in critical_indets:
                reason_codes.append(f"{d.value}_INDETERMINATE")
            return EvaluationPolicyDecision(
                decision=EvaluationDecision.INDETERMINATE,
                overall_score=None,
                failed_dimensions=tuple(failed_dims),
                indeterminate_dimensions=tuple(indet_dims),
                error_dimensions=tuple(error_dims),
                reason_codes=tuple(reason_codes),
                policy_version=pol.policy_version,
            )

        # Rule 2: Critical dimension SCORED below critical threshold -> FAIL (Hard Gate)
        critical_fails = [d for d in failed_dims if d in critical_dims]
        if critical_fails:
            for d in critical_fails:
                reason_codes.append(f"{d.value}_BELOW_THRESHOLD")
            return EvaluationPolicyDecision(
                decision=EvaluationDecision.FAIL,
                overall_score=None,  # Do not invent overall score on threshold failure
                failed_dimensions=tuple(failed_dims),
                indeterminate_dimensions=tuple(indet_dims),
                error_dimensions=tuple(error_dims),
                reason_codes=tuple(reason_codes),
                policy_version=pol.policy_version,
            )

        # Rule 3: Non-critical dimension ERROR or INDETERMINATE -> INDETERMINATE
        non_critical_errors = [d for d in error_dims if d not in critical_dims]
        non_critical_indets = [d for d in indet_dims if d not in critical_dims]
        if non_critical_errors or non_critical_indets:
            for d in non_critical_errors:
                reason_codes.append(f"{d.value}_ERROR")
            for d in non_critical_indets:
                reason_codes.append(f"{d.value}_INDETERMINATE")
            return EvaluationPolicyDecision(
                decision=EvaluationDecision.INDETERMINATE,
                overall_score=None,
                failed_dimensions=tuple(failed_dims),
                indeterminate_dimensions=tuple(indet_dims),
                error_dimensions=tuple(error_dims),
                reason_codes=tuple(reason_codes),
                policy_version=pol.policy_version,
            )

        # Rule 4: Non-critical dimension SCORED below threshold -> FAIL
        non_critical_fails = [d for d in failed_dims if d not in critical_dims]
        if non_critical_fails:
            for d in non_critical_fails:
                reason_codes.append(f"{d.value}_BELOW_THRESHOLD")
            return EvaluationPolicyDecision(
                decision=EvaluationDecision.FAIL,
                overall_score=None,
                failed_dimensions=tuple(failed_dims),
                indeterminate_dimensions=tuple(indet_dims),
                error_dimensions=tuple(error_dims),
                reason_codes=tuple(reason_codes),
                policy_version=pol.policy_version,
            )

        # Rule 5: All four dimensions SCORED and above dimension thresholds
        # Calculate weighted overall score
        total_weight = sum(pol.weights.get(dim, 0.0) for dim in EvaluationDimension)
        if total_weight <= 0:
            total_weight = 1.0

        weighted_score = sum(
            (dim_map[dim].score or 0.0) * pol.weights.get(dim, 0.0)
            for dim in EvaluationDimension
        ) / total_weight

        overall_score = round(weighted_score, 4)

        if overall_score < pol.minimum_overall_score:
            reason_codes.append("OVERALL_SCORE_BELOW_THRESHOLD")
            return EvaluationPolicyDecision(
                decision=EvaluationDecision.FAIL,
                overall_score=overall_score,
                failed_dimensions=tuple(failed_dims),
                indeterminate_dimensions=(),
                error_dimensions=(),
                reason_codes=tuple(reason_codes),
                policy_version=pol.policy_version,
            )

        reason_codes.append("ALL_DIMENSIONS_PASSED")
        return EvaluationPolicyDecision(
            decision=EvaluationDecision.PASS,
            overall_score=overall_score,
            failed_dimensions=(),
            indeterminate_dimensions=(),
            error_dimensions=(),
            reason_codes=tuple(reason_codes),
            policy_version=pol.policy_version,
        )
