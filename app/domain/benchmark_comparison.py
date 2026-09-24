from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.domain.benchmark import BenchmarkExecutionMode
from app.domain.benchmark_metrics import (
    METRIC_SET_VERSION_V1,
    MetricDirection,
    MetricValue,
    MetricValueStatus,
)


@dataclass(frozen=True)
class MetricDelta:
    """Deterministic delta between baseline and candidate values for a specific metric."""
    metric_key: str
    metric_version: str
    baseline_value: float | None
    candidate_value: float | None
    absolute_delta: float | None
    relative_delta: float | None
    direction: MetricDirection
    is_improvement: bool | None
    status: str  # "COMPARED", "PARTIAL", "UNAVAILABLE", "NOT_APPLICABLE"
    coverage_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_key": self.metric_key,
            "metric_version": self.metric_version,
            "baseline_value": round(self.baseline_value, 4) if self.baseline_value is not None else None,
            "candidate_value": round(self.candidate_value, 4) if self.candidate_value is not None else None,
            "absolute_delta": round(self.absolute_delta, 4) if self.absolute_delta is not None else None,
            "relative_delta": round(self.relative_delta, 4) if self.relative_delta is not None else None,
            "direction": self.direction.value,
            "is_improvement": self.is_improvement,
            "status": self.status,
            "coverage_warning": self.coverage_warning,
        }


@dataclass(frozen=True)
class LatencyDelta:
    """Latency comparison for a specific pipeline stage."""
    stage: str
    baseline_mean_ms: float | None
    candidate_mean_ms: float | None
    absolute_delta_ms: float | None
    relative_delta: float | None
    is_improvement: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "baseline_mean_ms": round(self.baseline_mean_ms, 2) if self.baseline_mean_ms is not None else None,
            "candidate_mean_ms": round(self.candidate_mean_ms, 2) if self.candidate_mean_ms is not None else None,
            "absolute_delta_ms": round(self.absolute_delta_ms, 2) if self.absolute_delta_ms is not None else None,
            "relative_delta": round(self.relative_delta, 4) if self.relative_delta is not None else None,
            "is_improvement": self.is_improvement,
        }


@dataclass(frozen=True)
class CaseDrilldownDelta:
    """Case-level drilldown preserving exact trace_id for both runs."""
    case_key: str
    baseline_trace_id: str
    candidate_trace_id: str
    baseline_status: str
    candidate_status: str
    baseline_duration_ms: float | None = None
    candidate_duration_ms: float | None = None
    duration_delta_ms: float | None = None
    baseline_failure_stage: str | None = None
    candidate_failure_stage: str | None = None
    baseline_failure_code: str | None = None
    candidate_failure_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_key": self.case_key,
            "baseline_trace_id": self.baseline_trace_id,
            "candidate_trace_id": self.candidate_trace_id,
            "baseline_status": self.baseline_status,
            "candidate_status": self.candidate_status,
            "baseline_duration_ms": round(self.baseline_duration_ms, 2) if self.baseline_duration_ms is not None else None,
            "candidate_duration_ms": round(self.candidate_duration_ms, 2) if self.candidate_duration_ms is not None else None,
            "duration_delta_ms": round(self.duration_delta_ms, 2) if self.duration_delta_ms is not None else None,
            "baseline_failure_stage": self.baseline_failure_stage,
            "candidate_failure_stage": self.candidate_failure_stage,
            "baseline_failure_code": self.baseline_failure_code,
            "candidate_failure_code": self.candidate_failure_code,
        }


def compute_metric_delta(
    base_val: MetricValue | None,
    cand_val: MetricValue | None,
    direction: MetricDirection,
    metric_key: str,
    metric_version: str = METRIC_SET_VERSION_V1,
) -> MetricDelta:
    """Computes a deterministic MetricDelta with direction-aware improvement calculation."""
    if base_val is None and cand_val is None:
        return MetricDelta(
            metric_key=metric_key,
            metric_version=metric_version,
            baseline_value=None,
            candidate_value=None,
            absolute_delta=None,
            relative_delta=None,
            direction=direction,
            is_improvement=None,
            status="UNAVAILABLE",
        )

    if (base_val and base_val.status == MetricValueStatus.NOT_APPLICABLE) and (
        cand_val and cand_val.status == MetricValueStatus.NOT_APPLICABLE
    ):
        return MetricDelta(
            metric_key=metric_key,
            metric_version=metric_version,
            baseline_value=None,
            candidate_value=None,
            absolute_delta=None,
            relative_delta=None,
            direction=direction,
            is_improvement=None,
            status="NOT_APPLICABLE",
        )

    b_val = base_val.value if base_val and base_val.status in (MetricValueStatus.AVAILABLE, MetricValueStatus.PARTIAL) else None
    c_val = cand_val.value if cand_val and cand_val.status in (MetricValueStatus.AVAILABLE, MetricValueStatus.PARTIAL) else None

    coverage_warning = None
    if base_val and cand_val and (base_val.status == MetricValueStatus.PARTIAL or cand_val.status == MetricValueStatus.PARTIAL):
        coverage_warning = f"Partial coverage: baseline={base_val.coverage_ratio}, candidate={cand_val.coverage_ratio}"

    if b_val is not None and c_val is not None:
        abs_delta = c_val - b_val
        rel_delta = (abs_delta / b_val) if b_val != 0.0 else None

        if direction == MetricDirection.HIGHER_IS_BETTER:
            is_imp = abs_delta > 1e-6
        elif direction == MetricDirection.LOWER_IS_BETTER:
            is_imp = abs_delta < -1e-6
        else:
            is_imp = None

        return MetricDelta(
            metric_key=metric_key,
            metric_version=metric_version,
            baseline_value=b_val,
            candidate_value=c_val,
            absolute_delta=abs_delta,
            relative_delta=rel_delta,
            direction=direction,
            is_improvement=is_imp,
            status="COMPARED",
            coverage_warning=coverage_warning,
        )

    # One or both unavailable
    return MetricDelta(
        metric_key=metric_key,
        metric_version=metric_version,
        baseline_value=b_val,
        candidate_value=c_val,
        absolute_delta=None,
        relative_delta=None,
        direction=direction,
        is_improvement=None,
        status="PARTIAL" if (b_val is not None or c_val is not None) else "UNAVAILABLE",
        coverage_warning=coverage_warning,
    )


def compute_comparison_fingerprint(
    baseline_report_id: str,
    candidate_report_id: str,
    baseline_fingerprint: str,
    candidate_fingerprint: str,
    metric_deltas: dict[str, MetricDelta],
) -> str:
    """Deterministic SHA-256 fingerprint for a BenchmarkComparisonReport."""
    payload = {
        "baseline_report_id": baseline_report_id,
        "candidate_report_id": candidate_report_id,
        "baseline_fingerprint": baseline_fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
        "metric_deltas": {k: metric_deltas[k].to_dict() for k in sorted(metric_deltas.keys())},
    }
    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


@dataclass(frozen=True)
class BenchmarkComparisonReport:
    """
    Immutable comparison between two compatible BenchmarkReports.
    Presents objective deltas and trade-offs without fabricating an arbitrary single score.
    """
    baseline_report_id: str
    candidate_report_id: str
    baseline_variant_key: str
    candidate_variant_key: str
    suite_key: str
    suite_version: str
    suite_fingerprint: str
    execution_mode: BenchmarkExecutionMode
    metric_deltas: dict[str, MetricDelta]
    latency_deltas: dict[str, LatencyDelta]
    cost_deltas: dict[str, Any]
    tradeoff_summary: dict[str, Any]
    case_drilldowns: list[CaseDrilldownDelta]
    failure_breakdown: dict[str, Any]
    comparison_id: str = field(default_factory=lambda: f"bcomp_{uuid4().hex[:16]}")
    metric_definition_set_version: str = METRIC_SET_VERSION_V1
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.source_fingerprint:
            fp = compute_comparison_fingerprint(
                baseline_report_id=self.baseline_report_id,
                candidate_report_id=self.candidate_report_id,
                baseline_fingerprint=self.baseline_report_id,
                candidate_fingerprint=self.candidate_report_id,
                metric_deltas=self.metric_deltas,
            )
            object.__setattr__(self, "source_fingerprint", fp)

    def to_dict(self) -> dict[str, Any]:
        """Deterministic export of comparison report."""
        return {
            "comparison_id": self.comparison_id,
            "baseline_report_id": self.baseline_report_id,
            "candidate_report_id": self.candidate_report_id,
            "baseline_variant_key": self.baseline_variant_key,
            "candidate_variant_key": self.candidate_variant_key,
            "suite_key": self.suite_key,
            "suite_version": self.suite_version,
            "suite_fingerprint": self.suite_fingerprint,
            "execution_mode": self.execution_mode.value,
            "metric_definition_set_version": self.metric_definition_set_version,
            "source_fingerprint": self.source_fingerprint,
            "created_at": self.created_at.isoformat(),
            "metric_deltas": {k: v.to_dict() for k, v in sorted(self.metric_deltas.items())},
            "latency_deltas": {k: v.to_dict() for k, v in sorted(self.latency_deltas.items())},
            "cost_deltas": self.cost_deltas,
            "tradeoff_summary": self.tradeoff_summary,
            "case_drilldowns": [c.to_dict() for c in self.case_drilldowns],
            "failure_breakdown": self.failure_breakdown,
        }

    def to_json(self, indent: int = 2) -> str:
        """Deterministic JSON string export."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
