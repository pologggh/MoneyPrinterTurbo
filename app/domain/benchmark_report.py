from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.domain.benchmark import BenchmarkExecutionMode, BenchmarkVariant
from app.domain.benchmark_metrics import (
    METRIC_SET_VERSION_V1,
    CostSummary,
    MetricValue,
    StageLatencySummary,
)


def compute_benchmark_report_fingerprint(
    benchmark_run_id: str,
    suite_key: str,
    suite_version: str,
    suite_fingerprint: str,
    variant_key: str,
    execution_mode: str,
    metric_definition_set_version: str,
    case_count: int,
    shot_count: int,
    metrics: dict[str, MetricValue],
) -> str:
    """
    Deterministic SHA-256 fingerprint for a BenchmarkReport based on canonical JSON.
    Guarantees tamper-evident traceability back to source runs and metrics.
    """
    payload = {
        "benchmark_run_id": benchmark_run_id,
        "suite_key": suite_key,
        "suite_version": suite_version,
        "suite_fingerprint": suite_fingerprint,
        "variant_key": variant_key,
        "execution_mode": execution_mode,
        "metric_definition_set_version": metric_definition_set_version,
        "case_count": case_count,
        "shot_count": shot_count,
        "metrics": {k: metrics[k].to_dict() for k in sorted(metrics.keys())},
    }
    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


@dataclass(frozen=True)
class BenchmarkReport:
    """
    Immutable benchmark report containing aggregated versioned metrics,
    stage latencies, cost summaries, and provider usage derived from a BenchmarkRun.
    """
    benchmark_run_id: str
    suite_key: str
    suite_version: str
    suite_fingerprint: str
    variant: BenchmarkVariant
    execution_mode: BenchmarkExecutionMode
    metrics: dict[str, MetricValue]
    stage_latencies: dict[str, StageLatencySummary]
    cost_summary: CostSummary
    provider_model_usage: dict[str, Any]
    case_count: int
    shot_count: int
    report_id: str = field(default_factory=lambda: f"brep_{uuid4().hex[:16]}")
    metric_definition_set_version: str = METRIC_SET_VERSION_V1
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.source_fingerprint:
            fp = compute_benchmark_report_fingerprint(
                benchmark_run_id=self.benchmark_run_id,
                suite_key=self.suite_key,
                suite_version=self.suite_version,
                suite_fingerprint=self.suite_fingerprint,
                variant_key=self.variant.variant_key,
                execution_mode=self.execution_mode.value,
                metric_definition_set_version=self.metric_definition_set_version,
                case_count=self.case_count,
                shot_count=self.shot_count,
                metrics=self.metrics,
            )
            object.__setattr__(self, "source_fingerprint", fp)

    def to_dict(self) -> dict[str, Any]:
        """Deterministic, complete dictionary representation for JSON export and audit."""
        return {
            "report_id": self.report_id,
            "benchmark_run_id": self.benchmark_run_id,
            "suite_key": self.suite_key,
            "suite_version": self.suite_version,
            "suite_fingerprint": self.suite_fingerprint,
            "variant": self.variant.to_dict(),
            "execution_mode": self.execution_mode.value,
            "metric_definition_set_version": self.metric_definition_set_version,
            "source_fingerprint": self.source_fingerprint,
            "case_count": self.case_count,
            "shot_count": self.shot_count,
            "generated_at": self.generated_at.isoformat(),
            "metrics": {k: v.to_dict() for k, v in sorted(self.metrics.items())},
            "stage_latencies": {k: v.to_dict() for k, v in sorted(self.stage_latencies.items())},
            "cost_summary": self.cost_summary.to_dict(),
            "provider_model_usage": self.provider_model_usage,
        }

    def to_json(self, indent: int = 2) -> str:
        """Deterministic JSON string export."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
