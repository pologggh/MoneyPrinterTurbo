import unittest

from app.domain.asset_router import ModelSelectionMode, RoutingStrategy
from app.domain.benchmark import BenchmarkExecutionMode, BenchmarkVariant
from app.domain.benchmark_metrics import CostSummary, MetricValue, MetricValueStatus
from app.domain.benchmark_report import BenchmarkReport
from app.services.benchmark.comparison_gate import (
    IncompatibleBenchmarkComparisonError,
    assert_benchmark_comparability,
    validate_benchmark_comparability,
)


def make_test_report(
    report_id: str,
    suite_key: str = "knowledge-video-v1",
    suite_version: str = "v1",
    suite_fingerprint: str = "fp-suite-1",
    variant_key: str = "balanced-auto",
    execution_mode: BenchmarkExecutionMode = BenchmarkExecutionMode.OFFLINE,
    metric_version: str = "v1",
    case_count: int = 12,
) -> BenchmarkReport:
    metrics = {
        "first_asset_quality_pass_rate": MetricValue(
            metric_key="first_asset_quality_pass_rate",
            metric_version=metric_version,
            status=MetricValueStatus.AVAILABLE,
            value=0.8,
            sample_count=8,
            eligible_count=10,
        )
    }
    return BenchmarkReport(
        report_id=report_id,
        benchmark_run_id=f"brun-{report_id}",
        suite_key=suite_key,
        suite_version=suite_version,
        suite_fingerprint=suite_fingerprint,
        variant=BenchmarkVariant(
            variant_key=variant_key,
            routing_strategy=RoutingStrategy.BALANCED,
            model_selection_mode=ModelSelectionMode.AUTO,
        ),
        execution_mode=execution_mode,
        metric_definition_set_version=metric_version,
        metrics=metrics,
        stage_latencies={},
        cost_summary=CostSummary(),
        provider_model_usage={},
        case_count=case_count,
        shot_count=24,
    )


class TestBenchmarkComparisonGate(unittest.TestCase):
    def test_identical_reports_pass_gate(self):
        """Compatible reports with different variants pass the hard gate."""
        rep_a = make_test_report("rep-1", variant_key="balanced-auto")
        rep_b = make_test_report("rep-2", variant_key="quality-auto")

        is_valid, reason = validate_benchmark_comparability(rep_a, rep_b)
        self.assertTrue(is_valid)
        self.assertIsNone(reason)
        # Should not raise
        assert_benchmark_comparability(rep_a, rep_b)

    def test_suite_key_mismatch_rejected(self):
        """Mismatched suite_key is rejected."""
        rep_a = make_test_report("rep-1", suite_key="suite-a")
        rep_b = make_test_report("rep-2", suite_key="suite-b")

        is_valid, reason = validate_benchmark_comparability(rep_a, rep_b)
        self.assertFalse(is_valid)
        self.assertIn("Suite key mismatch", reason)

        with self.assertRaises(IncompatibleBenchmarkComparisonError):
            assert_benchmark_comparability(rep_a, rep_b)

    def test_suite_version_mismatch_rejected(self):
        """Mismatched suite_version is rejected."""
        rep_a = make_test_report("rep-1", suite_version="v1")
        rep_b = make_test_report("rep-2", suite_version="v2")

        is_valid, reason = validate_benchmark_comparability(rep_a, rep_b)
        self.assertFalse(is_valid)
        self.assertIn("Suite version mismatch", reason)

    def test_suite_fingerprint_mismatch_rejected(self):
        """Mismatched suite content fingerprint is rejected."""
        rep_a = make_test_report("rep-1", suite_fingerprint="fp-alpha")
        rep_b = make_test_report("rep-2", suite_fingerprint="fp-beta")

        is_valid, reason = validate_benchmark_comparability(rep_a, rep_b)
        self.assertFalse(is_valid)
        self.assertIn("fingerprint mismatch", reason)

    def test_execution_mode_mismatch_rejected(self):
        """Comparing OFFLINE against REAL is strictly prohibited."""
        rep_a = make_test_report("rep-1", execution_mode=BenchmarkExecutionMode.OFFLINE)
        rep_b = make_test_report("rep-2", execution_mode=BenchmarkExecutionMode.REAL)

        is_valid, reason = validate_benchmark_comparability(rep_a, rep_b)
        self.assertFalse(is_valid)
        self.assertIn("Execution mode mismatch", reason)

    def test_case_count_mismatch_rejected(self):
        """Comparing reports evaluated on different subsets of cases is rejected."""
        rep_a = make_test_report("rep-1", case_count=12)
        rep_b = make_test_report("rep-2", case_count=6)

        is_valid, reason = validate_benchmark_comparability(rep_a, rep_b)
        self.assertFalse(is_valid)
        self.assertIn("Case count mismatch", reason)


if __name__ == "__main__":
    unittest.main()
