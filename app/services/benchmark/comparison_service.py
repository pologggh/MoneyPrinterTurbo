from __future__ import annotations

from typing import Any

from app.domain.benchmark_comparison import (
    BenchmarkComparisonReport,
    CaseDrilldownDelta,
    LatencyDelta,
    compute_metric_delta,
)
from app.domain.benchmark_metrics import V1_METRIC_DEFINITIONS, MetricDirection
from app.domain.benchmark_report import BenchmarkReport
from app.persistence.repositories import BenchmarkRepository
from app.services.benchmark.comparison_gate import assert_benchmark_comparability


class BenchmarkComparisonService:
    """
    Constructs a BenchmarkComparisonReport between two compatible BenchmarkReports.
    Enforces compatibility hard gates and calculates deterministic metric deltas.
    """

    def __init__(self, bench_repo: BenchmarkRepository | None = None) -> None:
        self._bench_repo = bench_repo

    def compare_reports(
        self,
        baseline_report: BenchmarkReport,
        candidate_report: BenchmarkReport,
    ) -> BenchmarkComparisonReport:
        """
        Validates compatibility and calculates complete comparative deltas,
        case-level drilldowns, and trade-off summary.
        """
        # 1. Enforce strict compatibility hard gates
        assert_benchmark_comparability(baseline_report, candidate_report)

        # 2. Compute Metric Deltas across all standard V1 metrics
        metric_deltas: dict[str, Any] = {}
        all_keys = set(baseline_report.metrics.keys()).union(candidate_report.metrics.keys())
        for mkey in sorted(all_keys):
            b_val = baseline_report.metrics.get(mkey)
            c_val = candidate_report.metrics.get(mkey)
            defn = V1_METRIC_DEFINITIONS.get(mkey)
            direction = defn.direction if defn else MetricDirection.NEUTRAL

            delta = compute_metric_delta(
                base_val=b_val,
                cand_val=c_val,
                direction=direction,
                metric_key=mkey,
            )
            metric_deltas[mkey] = delta

        # 3. Compute Latency Deltas
        latency_deltas: dict[str, LatencyDelta] = {}
        all_stages = set(baseline_report.stage_latencies.keys()).union(candidate_report.stage_latencies.keys())
        for stage in sorted(all_stages):
            b_lat = baseline_report.stage_latencies.get(stage)
            c_lat = candidate_report.stage_latencies.get(stage)

            b_mean = b_lat.mean_ms if b_lat and b_lat.mean_ms is not None else None
            c_mean = c_lat.mean_ms if c_lat and c_lat.mean_ms is not None else None

            if b_mean is not None and c_mean is not None:
                abs_delta = c_mean - b_mean
                rel_delta = (abs_delta / b_mean) if b_mean != 0.0 else None
                is_imp = abs_delta < -1e-6  # Lower latency is better
            else:
                abs_delta = None
                rel_delta = None
                is_imp = None

            latency_deltas[stage] = LatencyDelta(
                stage=stage,
                baseline_mean_ms=b_mean,
                candidate_mean_ms=c_mean,
                absolute_delta_ms=abs_delta,
                relative_delta=rel_delta,
                is_improvement=is_imp,
            )

        # 4. Compute Cost Deltas
        cost_deltas: dict[str, Any] = {
            "is_synthetic": baseline_report.cost_summary.is_synthetic,
            "currency_deltas": {},
        }
        all_currencies = set(baseline_report.cost_summary.observed_by_currency.keys()).union(
            candidate_report.cost_summary.observed_by_currency.keys()
        )
        for curr in sorted(all_currencies):
            b_cost = baseline_report.cost_summary.observed_by_currency.get(curr, 0.0)
            c_cost = candidate_report.cost_summary.observed_by_currency.get(curr, 0.0)
            diff = c_cost - b_cost
            cost_deltas["currency_deltas"][curr] = {
                "baseline": round(b_cost, 4),
                "candidate": round(c_cost, 4),
                "delta": round(diff, 4),
            }

        # 5. Extract Case Drilldowns and Failure Breakdown
        case_drilldowns: list[CaseDrilldownDelta] = []
        failure_breakdown: dict[str, Any] = {
            "baseline_failures": {},
            "candidate_failures": {},
        }

        if self._bench_repo:
            b_cases = {
                c.case_key: c
                for c in self._bench_repo.list_case_results_for_run(baseline_report.benchmark_run_id)
            }
            c_cases = {
                c.case_key: c
                for c in self._bench_repo.list_case_results_for_run(candidate_report.benchmark_run_id)
            }

            for ckey in sorted(b_cases.keys()):
                b_c = b_cases[ckey]
                c_c = c_cases.get(ckey)

                c_trace_id = c_c.trace_id if c_c else ""
                c_status = c_c.status.value if c_c else "MISSING"
                c_dur = c_c.duration_ms if c_c else None
                b_dur = b_c.duration_ms
                dur_delta = (c_dur - b_dur) if (c_dur is not None and b_dur is not None) else None

                cdelta = CaseDrilldownDelta(
                    case_key=ckey,
                    baseline_trace_id=b_c.trace_id,
                    candidate_trace_id=c_trace_id,
                    baseline_status=b_c.status.value,
                    candidate_status=c_status,
                    baseline_duration_ms=b_dur,
                    candidate_duration_ms=c_dur,
                    duration_delta_ms=dur_delta,
                    baseline_failure_stage=b_c.failure_stage.value if b_c.failure_stage else None,
                    candidate_failure_stage=c_c.failure_stage.value if (c_c and c_c.failure_stage) else None,
                    baseline_failure_code=b_c.failure_code,
                    candidate_failure_code=c_c.failure_code if c_c else None,
                )
                case_drilldowns.append(cdelta)

                # Update failure breakdown
                if b_c.failure_stage:
                    stage_str = b_c.failure_stage.value
                    failure_breakdown["baseline_failures"][stage_str] = (
                        failure_breakdown["baseline_failures"].get(stage_str, 0) + 1
                    )
                if c_c and c_c.failure_stage:
                    stage_str = c_c.failure_stage.value
                    failure_breakdown["candidate_failures"][stage_str] = (
                        failure_breakdown["candidate_failures"].get(stage_str, 0) + 1
                    )

        # 6. Objective Trade-Off Summary (No Arbitrary Score)
        tradeoff_summary: dict[str, Any] = {
            "quality": {
                "first_pass_delta": metric_deltas.get("first_asset_quality_pass_rate", None).to_dict() if "first_asset_quality_pass_rate" in metric_deltas else None,
                "final_pass_delta": metric_deltas.get("final_shot_quality_pass_rate", None).to_dict() if "final_shot_quality_pass_rate" in metric_deltas else None,
            },
            "reliability": {
                "fallback_delta": metric_deltas.get("fallback_rate", None).to_dict() if "fallback_rate" in metric_deltas else None,
                "recovery_delta": metric_deltas.get("recovery_required_rate", None).to_dict() if "recovery_required_rate" in metric_deltas else None,
                "attempts_per_shot_delta": metric_deltas.get("mean_generation_attempts_per_shot", None).to_dict() if "mean_generation_attempts_per_shot" in metric_deltas else None,
            },
            "efficiency": {
                "e2e_latency_delta_ms": latency_deltas.get("end_to_end", None).to_dict() if "end_to_end" in latency_deltas else None,
                "cost_deltas": cost_deltas,
            },
            "findings": self._generate_tradeoff_findings(metric_deltas, latency_deltas),
        }

        return BenchmarkComparisonReport(
            baseline_report_id=baseline_report.report_id,
            candidate_report_id=candidate_report.report_id,
            baseline_variant_key=baseline_report.variant.variant_key,
            candidate_variant_key=candidate_report.variant.variant_key,
            suite_key=baseline_report.suite_key,
            suite_version=baseline_report.suite_version,
            suite_fingerprint=baseline_report.suite_fingerprint,
            execution_mode=baseline_report.execution_mode,
            metric_deltas=metric_deltas,
            latency_deltas=latency_deltas,
            cost_deltas=cost_deltas,
            tradeoff_summary=tradeoff_summary,
            case_drilldowns=case_drilldowns,
            failure_breakdown=failure_breakdown,
        )

    def _generate_tradeoff_findings(
        self,
        metric_deltas: dict[str, Any],
        latency_deltas: dict[str, LatencyDelta],
    ) -> list[str]:
        """Generates clear, factual bullet points highlighting observed trade-offs."""
        findings: list[str] = []

        # Quality finding
        fp_delta = metric_deltas.get("first_asset_quality_pass_rate")
        if fp_delta and fp_delta.absolute_delta is not None:
            if fp_delta.absolute_delta > 0:
                findings.append(
                    f"Candidate improved first-asset quality pass rate by {fp_delta.absolute_delta * 100:.1f}%."
                )
            elif fp_delta.absolute_delta < 0:
                findings.append(
                    f"Candidate reduced first-asset quality pass rate by {abs(fp_delta.absolute_delta) * 100:.1f}%."
                )
            else:
                findings.append("First-asset quality pass rate remained unchanged.")

        # Latency finding
        e2e_lat = latency_deltas.get("end_to_end")
        if e2e_lat and e2e_lat.absolute_delta_ms is not None:
            if e2e_lat.absolute_delta_ms > 0:
                findings.append(
                    f"Candidate mean end-to-end latency increased by {e2e_lat.absolute_delta_ms:.0f} ms."
                )
            elif e2e_lat.absolute_delta_ms < 0:
                findings.append(
                    f"Candidate mean end-to-end latency decreased by {abs(e2e_lat.absolute_delta_ms):.0f} ms."
                )

        # Reliability finding
        fb_delta = metric_deltas.get("fallback_rate")
        if fb_delta and fb_delta.absolute_delta is not None:
            if fb_delta.absolute_delta < 0:
                findings.append(
                    f"Candidate reduced fallback rate by {abs(fb_delta.absolute_delta) * 100:.1f}%."
                )
            elif fb_delta.absolute_delta > 0:
                findings.append(
                    f"Candidate had {fb_delta.absolute_delta * 100:.1f}% higher fallback rate."
                )

        return findings
