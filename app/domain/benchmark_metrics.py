from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MetricValueStatus(str, Enum):
    """Explicit status of a calculated metric value."""
    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"


class MetricDirection(str, Enum):
    """Optimization direction for a metric."""
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class MetricDefinition:
    """Formal definition of a benchmark metric."""
    metric_key: str
    metric_version: str
    description: str
    unit: str  # "ratio", "count", "score", "ms", "currency"
    direction: MetricDirection
    numerator_semantics: str
    denominator_semantics: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_key": self.metric_key,
            "metric_version": self.metric_version,
            "description": self.description,
            "unit": self.unit,
            "direction": self.direction.value,
            "numerator_semantics": self.numerator_semantics,
            "denominator_semantics": self.denominator_semantics,
        }


@dataclass(frozen=True)
class MetricValue:
    """
    Immutable value instance for a metric, adhering to strict status invariants:
    - If status is UNAVAILABLE or NOT_APPLICABLE, value MUST be None.
    - If status is AVAILABLE or PARTIAL, value MUST be a float.
    - Coverage ratio is explicitly tracked when eligible_count > 0.
    """
    metric_key: str
    metric_version: str
    status: MetricValueStatus
    value: float | None
    sample_count: int = 0
    eligible_count: int = 0
    coverage_ratio: float | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.status in (MetricValueStatus.UNAVAILABLE, MetricValueStatus.NOT_APPLICABLE):
            if self.value is not None:
                raise ValueError(
                    f"MetricValue with status {self.status.value} must have value=None, got {self.value}"
                )
        elif self.status in (MetricValueStatus.AVAILABLE, MetricValueStatus.PARTIAL) and self.value is None:
            raise ValueError(
                f"MetricValue with status {self.status.value} must have a numeric float value, got None"
            )

        # Compute coverage ratio if not explicitly supplied
        if self.coverage_ratio is None and self.eligible_count > 0:
            ratio = round(min(1.0, max(0.0, self.sample_count / self.eligible_count)), 4)
            object.__setattr__(self, "coverage_ratio", ratio)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_key": self.metric_key,
            "metric_version": self.metric_version,
            "status": self.status.value,
            "value": round(self.value, 4) if self.value is not None else None,
            "sample_count": self.sample_count,
            "eligible_count": self.eligible_count,
            "coverage_ratio": self.coverage_ratio,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CostSummary:
    """
    Summary of costs observed vs estimated during benchmark execution.
    Maintains status distinction and never fabricates UNKNOWN costs into 0.0.
    """
    observed_by_currency: dict[str, float] = field(default_factory=dict)
    estimated_by_currency: dict[str, float] = field(default_factory=dict)
    observed_ops: int = 0
    estimated_ops: int = 0
    unknown_ops: int = 0
    completeness: str = "UNKNOWN"  # "COMPLETE", "PARTIAL", "UNKNOWN"
    is_synthetic: bool = False  # True for OFFLINE mode synthetic fake costs

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_by_currency": {k: round(v, 4) for k, v in self.observed_by_currency.items()},
            "estimated_by_currency": {k: round(v, 4) for k, v in self.estimated_by_currency.items()},
            "observed_ops": self.observed_ops,
            "estimated_ops": self.estimated_ops,
            "unknown_ops": self.unknown_ops,
            "completeness": self.completeness,
            "is_synthetic": self.is_synthetic,
        }


@dataclass(frozen=True)
class StageLatencySummary:
    """Latency distribution for a specific pipeline stage."""
    stage: str
    mean_ms: float | None = None
    p50_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "mean_ms": round(self.mean_ms, 2) if self.mean_ms is not None else None,
            "p50_ms": round(self.p50_ms, 2) if self.p50_ms is not None else None,
            "min_ms": round(self.min_ms, 2) if self.min_ms is not None else None,
            "max_ms": round(self.max_ms, 2) if self.max_ms is not None else None,
            "sample_count": self.sample_count,
        }


# =============================================================================
# Standard V1 Metric Definitions Catalog
# =============================================================================

METRIC_SET_VERSION_V1 = "v1"

V1_METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    # -------------------------------------------------------------------------
    # Planning & Scripting
    # -------------------------------------------------------------------------
    "plan_validation_success_rate": MetricDefinition(
        metric_key="plan_validation_success_rate",
        metric_version=METRIC_SET_VERSION_V1,
        description="Ratio of benchmark cases passing structural constraint validation",
        unit="ratio",
        direction=MetricDirection.HIGHER_IS_BETTER,
        numerator_semantics="Cases passing structural constraints",
        denominator_semantics="Total benchmark cases evaluated",
    ),
    "storyboard_generation_success_rate": MetricDefinition(
        metric_key="storyboard_generation_success_rate",
        metric_version=METRIC_SET_VERSION_V1,
        description="Ratio of benchmark cases where storyboard snapshot was successfully generated",
        unit="ratio",
        direction=MetricDirection.HIGHER_IS_BETTER,
        numerator_semantics="Cases with valid storyboard snapshot",
        denominator_semantics="Total benchmark cases evaluated",
    ),
    "evidence_support_rate": MetricDefinition(
        metric_key="evidence_support_rate",
        metric_version=METRIC_SET_VERSION_V1,
        description="Ratio of shots backed by non-empty evidence references in their source beat",
        unit="ratio",
        direction=MetricDirection.HIGHER_IS_BETTER,
        numerator_semantics="Shots whose beat contains evidence references",
        denominator_semantics="Total shots across all cases",
    ),

    # -------------------------------------------------------------------------
    # Execution & Generation
    # -------------------------------------------------------------------------
    "first_asset_quality_pass_rate": MetricDefinition(
        metric_key="first_asset_quality_pass_rate",
        metric_version=METRIC_SET_VERSION_V1,
        description="Ratio of shots whose FIRST generated asset evaluation snapshot resulted in PASS",
        unit="ratio",
        direction=MetricDirection.HIGHER_IS_BETTER,
        numerator_semantics="Shots whose initial evaluation snapshot was PASS",
        denominator_semantics="Total shots evaluated at least once",
    ),
    "final_shot_quality_pass_rate": MetricDefinition(
        metric_key="final_shot_quality_pass_rate",
        metric_version=METRIC_SET_VERSION_V1,
        description="Ratio of shots that ultimately obtained an accepted ShotAssetVersion",
        unit="ratio",
        direction=MetricDirection.HIGHER_IS_BETTER,
        numerator_semantics="Shots with accepted ShotAssetVersion",
        denominator_semantics="Total shots across all cases",
    ),
    "mean_generation_attempts_per_shot": MetricDefinition(
        metric_key="mean_generation_attempts_per_shot",
        metric_version=METRIC_SET_VERSION_V1,
        description="Average number of asset execution attempts executed per shot",
        unit="count",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Total execution attempts across all shots",
        denominator_semantics="Total shots across all cases",
    ),
    "fallback_rate": MetricDefinition(
        metric_key="fallback_rate",
        metric_version=METRIC_SET_VERSION_V1,
        description="Ratio of execution attempts that used a fallback provider/route candidate",
        unit="ratio",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Execution attempts with fallback triggered",
        denominator_semantics="Total execution attempts",
    ),
    "recovery_required_rate": MetricDefinition(
        metric_key="recovery_required_rate",
        metric_version=METRIC_SET_VERSION_V1,
        description="Ratio of execution attempts requiring recovery from unknown submission outcomes",
        unit="ratio",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Execution attempts requiring recovery",
        denominator_semantics="Total execution attempts",
    ),

    # -------------------------------------------------------------------------
    # Quality & Remediation
    # -------------------------------------------------------------------------
    "quality_remediation_rate": MetricDefinition(
        metric_key="quality_remediation_rate",
        metric_version=METRIC_SET_VERSION_V1,
        description="Ratio of evaluated shots that required at least one quality remediation decision",
        unit="ratio",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Shots triggering quality remediation",
        denominator_semantics="Total shots evaluated",
    ),
    "mean_quality_remediations_per_shot": MetricDefinition(
        metric_key="mean_quality_remediations_per_shot",
        metric_version=METRIC_SET_VERSION_V1,
        description="Average number of quality remediation decisions per shot",
        unit="count",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Total quality remediation decisions",
        denominator_semantics="Total shots across all cases",
    ),
    "controlled_visual_replan_rate": MetricDefinition(
        metric_key="controlled_visual_replan_rate",
        metric_version=METRIC_SET_VERSION_V1,
        description="Ratio of evaluated shots that triggered controlled visual replanning",
        unit="ratio",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Shots with CONTROLLED_VISUAL_REPLAN decision",
        denominator_semantics="Total shots evaluated",
    ),
    "needs_user_action_rate": MetricDefinition(
        metric_key="needs_user_action_rate",
        metric_version=METRIC_SET_VERSION_V1,
        description="Ratio of benchmark cases resulting in NEEDS_USER_ACTION status",
        unit="ratio",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Cases with status NEEDS_USER_ACTION",
        denominator_semantics="Total benchmark cases evaluated",
    ),

    # -------------------------------------------------------------------------
    # Multimodal Quality Dimensions
    # -------------------------------------------------------------------------
    "dimension_semantic_alignment_score": MetricDefinition(
        metric_key="dimension_semantic_alignment_score",
        metric_version=METRIC_SET_VERSION_V1,
        description="Mean score across SCORED evaluations for SEMANTIC_ALIGNMENT (excluding INDETERMINATE/ERROR)",
        unit="score",
        direction=MetricDirection.HIGHER_IS_BETTER,
        numerator_semantics="Sum of SEMANTIC_ALIGNMENT scores with status SCORED",
        denominator_semantics="Count of SEMANTIC_ALIGNMENT evaluations with status SCORED",
    ),
    "dimension_visual_quality_score": MetricDefinition(
        metric_key="dimension_visual_quality_score",
        metric_version=METRIC_SET_VERSION_V1,
        description="Mean score across SCORED evaluations for VISUAL_QUALITY (excluding INDETERMINATE/ERROR)",
        unit="score",
        direction=MetricDirection.HIGHER_IS_BETTER,
        numerator_semantics="Sum of VISUAL_QUALITY scores with status SCORED",
        denominator_semantics="Count of VISUAL_QUALITY evaluations with status SCORED",
    ),
    "dimension_knowledge_accuracy_score": MetricDefinition(
        metric_key="dimension_knowledge_accuracy_score",
        metric_version=METRIC_SET_VERSION_V1,
        description="Mean score across SCORED evaluations for KNOWLEDGE_ACCURACY (excluding INDETERMINATE/ERROR)",
        unit="score",
        direction=MetricDirection.HIGHER_IS_BETTER,
        numerator_semantics="Sum of KNOWLEDGE_ACCURACY scores with status SCORED",
        denominator_semantics="Count of KNOWLEDGE_ACCURACY evaluations with status SCORED",
    ),
    "dimension_composition_suitability_score": MetricDefinition(
        metric_key="dimension_composition_suitability_score",
        metric_version=METRIC_SET_VERSION_V1,
        description="Mean score across SCORED evaluations for COMPOSITION_SUITABILITY (excluding INDETERMINATE/ERROR)",
        unit="score",
        direction=MetricDirection.HIGHER_IS_BETTER,
        numerator_semantics="Sum of COMPOSITION_SUITABILITY scores with status SCORED",
        denominator_semantics="Count of COMPOSITION_SUITABILITY evaluations with status SCORED",
    ),

    # -------------------------------------------------------------------------
    # Human-in-the-Loop (NOT_APPLICABLE for unattended benchmarks)
    # -------------------------------------------------------------------------
    "human_edit_rate": MetricDefinition(
        metric_key="human_edit_rate",
        metric_version=METRIC_SET_VERSION_V1,
        description="Ratio of shots edited by human in workbench. NOT_APPLICABLE for unattended benchmarks",
        unit="ratio",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Shots edited by human user",
        denominator_semantics="Total shots",
    ),
    "beat_replan_rate": MetricDefinition(
        metric_key="beat_replan_rate",
        metric_version=METRIC_SET_VERSION_V1,
        description="Ratio of beats manually replanned. NOT_APPLICABLE for unattended benchmarks",
        unit="ratio",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Beats manually replanned",
        denominator_semantics="Total beats",
    ),

    # -------------------------------------------------------------------------
    # Latency Metrics
    # -------------------------------------------------------------------------
    "latency_end_to_end_mean_ms": MetricDefinition(
        metric_key="latency_end_to_end_mean_ms",
        metric_version=METRIC_SET_VERSION_V1,
        description="Mean end-to-end case duration in milliseconds",
        unit="ms",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Sum of end-to-end durations across completed cases",
        denominator_semantics="Count of cases with observed duration",
    ),
    "latency_end_to_end_p50_ms": MetricDefinition(
        metric_key="latency_end_to_end_p50_ms",
        metric_version=METRIC_SET_VERSION_V1,
        description="Median (P50) end-to-end case duration in milliseconds",
        unit="ms",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="P50 duration across completed cases",
        denominator_semantics="Count of cases with observed duration",
    ),
    "latency_planning_mean_ms": MetricDefinition(
        metric_key="latency_planning_mean_ms",
        metric_version=METRIC_SET_VERSION_V1,
        description="Mean content planning duration in milliseconds",
        unit="ms",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Sum of content planning durations",
        denominator_semantics="Count of planning executions",
    ),
    "latency_storyboard_mean_ms": MetricDefinition(
        metric_key="latency_storyboard_mean_ms",
        metric_version=METRIC_SET_VERSION_V1,
        description="Mean storyboard generation duration in milliseconds",
        unit="ms",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Sum of storyboard generation durations",
        denominator_semantics="Count of storyboard executions",
    ),
    "latency_routing_mean_ms": MetricDefinition(
        metric_key="latency_routing_mean_ms",
        metric_version=METRIC_SET_VERSION_V1,
        description="Mean asset routing planning duration in milliseconds",
        unit="ms",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Sum of asset route planning durations",
        denominator_semantics="Count of routing executions",
    ),
    "latency_generation_mean_ms": MetricDefinition(
        metric_key="latency_generation_mean_ms",
        metric_version=METRIC_SET_VERSION_V1,
        description="Mean asset generation attempt duration in milliseconds",
        unit="ms",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Sum of generation attempt durations",
        denominator_semantics="Count of generation attempts",
    ),
    "latency_evaluation_mean_ms": MetricDefinition(
        metric_key="latency_evaluation_mean_ms",
        metric_version=METRIC_SET_VERSION_V1,
        description="Mean multimodal evaluation duration in milliseconds",
        unit="ms",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Sum of evaluation durations",
        denominator_semantics="Count of evaluations",
    ),
    "latency_remediation_mean_ms": MetricDefinition(
        metric_key="latency_remediation_mean_ms",
        metric_version=METRIC_SET_VERSION_V1,
        description="Mean quality remediation cycle duration in milliseconds",
        unit="ms",
        direction=MetricDirection.LOWER_IS_BETTER,
        numerator_semantics="Sum of quality remediation cycle durations",
        denominator_semantics="Count of remediation decisions",
    ),
}
