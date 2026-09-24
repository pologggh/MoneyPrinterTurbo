import unittest

from app.domain.asset_router import ModelSelectionMode, RoutingStrategy
from app.domain.benchmark import BenchmarkExecutionMode, BenchmarkVariant
from app.domain.benchmark_comparison import (
    compute_metric_delta,
)
from app.domain.benchmark_metrics import (
    CostSummary,
    MetricDirection,
    MetricValue,
    MetricValueStatus,
    StageLatencySummary,
)
from app.domain.benchmark_report import BenchmarkReport
from app.services.benchmark.comparison_service import BenchmarkComparisonService


class TestBenchmarkComparisonReport(unittest.TestCase):
    def test_compute_metric_delta_higher_is_better(self):
        """Improvement when candidate > baseline for HIGHER_IS_BETTER."""
        base = MetricValue(
            metric_key="first_asset_quality_pass_rate",
            metric_version="v1",
            status=MetricValueStatus.AVAILABLE,
            value=0.70,
            sample_count=7,
            eligible_count=10,
        )
        cand = MetricValue(
            metric_key="first_asset_quality_pass_rate",
            metric_version="v1",
            status=MetricValueStatus.AVAILABLE,
            value=0.85,
            sample_count=8,
            eligible_count=10,
        )
        delta = compute_metric_delta(
            base_val=base,
            cand_val=cand,
            direction=MetricDirection.HIGHER_IS_BETTER,
            metric_key="first_asset_quality_pass_rate",
        )
        self.assertEqual(delta.status, "COMPARED")
        self.assertAlmostEqual(delta.absolute_delta, 0.15, places=4)
        self.assertAlmostEqual(delta.relative_delta, 0.15 / 0.70, places=4)
        self.assertTrue(delta.is_improvement)

    def test_compute_metric_delta_lower_is_better(self):
        """Improvement when candidate < baseline for LOWER_IS_BETTER."""
        base = MetricValue(
            metric_key="fallback_rate",
            metric_version="v1",
            status=MetricValueStatus.AVAILABLE,
            value=0.20,
            sample_count=2,
            eligible_count=10,
        )
        cand = MetricValue(
            metric_key="fallback_rate",
            metric_version="v1",
            status=MetricValueStatus.AVAILABLE,
            value=0.05,
            sample_count=1,
            eligible_count=20,
        )
        delta = compute_metric_delta(
            base_val=base,
            cand_val=cand,
            direction=MetricDirection.LOWER_IS_BETTER,
            metric_key="fallback_rate",
        )
        self.assertEqual(delta.status, "COMPARED")
        self.assertAlmostEqual(delta.absolute_delta, -0.15, places=4)
        self.assertTrue(delta.is_improvement)  # negative delta is improvement for LOWER_IS_BETTER

    def test_compute_metric_delta_partial_and_unavailable(self):
        """Handles UNAVAILABLE and NOT_APPLICABLE properly."""
        base = MetricValue(
            metric_key="dimension_visual_quality_score",
            metric_version="v1",
            status=MetricValueStatus.UNAVAILABLE,
            value=None,
        )
        cand = MetricValue(
            metric_key="dimension_visual_quality_score",
            metric_version="v1",
            status=MetricValueStatus.AVAILABLE,
            value=0.9,
            sample_count=5,
            eligible_count=5,
        )
        delta = compute_metric_delta(
            base_val=base,
            cand_val=cand,
            direction=MetricDirection.HIGHER_IS_BETTER,
            metric_key="dimension_visual_quality_score",
        )
        self.assertEqual(delta.status, "PARTIAL")
        self.assertIsNone(delta.absolute_delta)
        self.assertIsNone(delta.is_improvement)

    def test_comparison_service_e2e_deltas_and_export(self):
        """Verifies full comparative report generation between two BenchmarkReports."""
        metrics_a = {
            "first_asset_quality_pass_rate": MetricValue(
                metric_key="first_asset_quality_pass_rate",
                metric_version="v1",
                status=MetricValueStatus.AVAILABLE,
                value=0.60,
                sample_count=6,
                eligible_count=10,
            ),
            "fallback_rate": MetricValue(
                metric_key="fallback_rate",
                metric_version="v1",
                status=MetricValueStatus.AVAILABLE,
                value=0.25,
                sample_count=5,
                eligible_count=20,
            ),
        }
        lat_a = {
            "end_to_end": StageLatencySummary(stage="end_to_end", mean_ms=5000.0, p50_ms=4800.0, sample_count=10)
        }
        rep_a = BenchmarkReport(
            report_id="rep-base",
            benchmark_run_id="brun-base",
            suite_key="suite-v1",
            suite_version="v1",
            suite_fingerprint="fp-common",
            variant=BenchmarkVariant(
                variant_key="balanced-auto",
                routing_strategy=RoutingStrategy.BALANCED,
                model_selection_mode=ModelSelectionMode.AUTO,
            ),
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            metrics=metrics_a,
            stage_latencies=lat_a,
            cost_summary=CostSummary(observed_by_currency={"USD": 1.0}, is_synthetic=True),
            provider_model_usage={},
            case_count=10,
            shot_count=20,
        )

        metrics_b = {
            "first_asset_quality_pass_rate": MetricValue(
                metric_key="first_asset_quality_pass_rate",
                metric_version="v1",
                status=MetricValueStatus.AVAILABLE,
                value=0.85,
                sample_count=17,
                eligible_count=20,
            ),
            "fallback_rate": MetricValue(
                metric_key="fallback_rate",
                metric_version="v1",
                status=MetricValueStatus.AVAILABLE,
                value=0.05,
                sample_count=1,
                eligible_count=20,
            ),
        }
        lat_b = {
            "end_to_end": StageLatencySummary(stage="end_to_end", mean_ms=7500.0, p50_ms=7200.0, sample_count=10)
        }
        rep_b = BenchmarkReport(
            report_id="rep-cand",
            benchmark_run_id="brun-cand",
            suite_key="suite-v1",
            suite_version="v1",
            suite_fingerprint="fp-common",
            variant=BenchmarkVariant(
                variant_key="quality-auto",
                routing_strategy=RoutingStrategy.QUALITY_FIRST,
                model_selection_mode=ModelSelectionMode.AUTO,
            ),
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            metrics=metrics_b,
            stage_latencies=lat_b,
            cost_summary=CostSummary(observed_by_currency={"USD": 2.5}, is_synthetic=True),
            provider_model_usage={},
            case_count=10,
            shot_count=20,
        )

        svc = BenchmarkComparisonService()
        comp = svc.compare_reports(rep_a, rep_b)

        self.assertEqual(comp.baseline_variant_key, "balanced-auto")
        self.assertEqual(comp.candidate_variant_key, "quality-auto")

        # First pass improved +25%
        fp_delta = comp.metric_deltas["first_asset_quality_pass_rate"]
        self.assertAlmostEqual(fp_delta.absolute_delta, 0.25, places=4)
        self.assertTrue(fp_delta.is_improvement)

        # Fallback rate reduced -20%
        fb_delta = comp.metric_deltas["fallback_rate"]
        self.assertAlmostEqual(fb_delta.absolute_delta, -0.20, places=4)
        self.assertTrue(fb_delta.is_improvement)

        # Latency increased +2500 ms (so is_improvement is False)
        lat_delta = comp.latency_deltas["end_to_end"]
        self.assertEqual(lat_delta.absolute_delta_ms, 2500.0)
        self.assertFalse(lat_delta.is_improvement)

        # Cost delta USD 1.50
        self.assertEqual(comp.cost_deltas["currency_deltas"]["USD"]["delta"], 1.5)

        # Trade-off summary findings
        self.assertTrue(len(comp.tradeoff_summary["findings"]) >= 2)

        # Serialization
        d = comp.to_dict()
        self.assertEqual(d["baseline_report_id"], "rep-base")
        self.assertEqual(d["candidate_report_id"], "rep-cand")
        self.assertTrue(len(comp.source_fingerprint) == 64)

        json_str = comp.to_json()
        self.assertIn("quality-auto", json_str)
        self.assertIn("fp-common", json_str)


if __name__ == "__main__":
    unittest.main()
