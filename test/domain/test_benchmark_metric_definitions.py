import unittest

from app.domain.benchmark_metrics import (
    METRIC_SET_VERSION_V1,
    V1_METRIC_DEFINITIONS,
    CostSummary,
    MetricDirection,
    MetricValue,
    MetricValueStatus,
    StageLatencySummary,
)


class TestBenchmarkMetricDefinitions(unittest.TestCase):
    def test_catalog_integrity(self):
        """Verifies that the standard V1 metrics catalog is complete and valid."""
        self.assertGreaterEqual(len(V1_METRIC_DEFINITIONS), 20)
        for key, defn in V1_METRIC_DEFINITIONS.items():
            self.assertEqual(defn.metric_key, key)
            self.assertEqual(defn.metric_version, METRIC_SET_VERSION_V1)
            self.assertIn(defn.direction, [MetricDirection.HIGHER_IS_BETTER, MetricDirection.LOWER_IS_BETTER, MetricDirection.NEUTRAL])
            self.assertTrue(len(defn.description) > 0)
            self.assertTrue(len(defn.numerator_semantics) > 0)
            self.assertTrue(len(defn.denominator_semantics) > 0)
            self.assertIn(defn.unit, ["ratio", "count", "score", "ms", "currency"])

    def test_metric_value_available_status(self):
        """AVAILABLE metric value must have a numeric float and computes coverage."""
        mv = MetricValue(
            metric_key="plan_validation_success_rate",
            metric_version="v1",
            status=MetricValueStatus.AVAILABLE,
            value=0.9167,
            sample_count=11,
            eligible_count=12,
        )
        self.assertEqual(mv.value, 0.9167)
        self.assertEqual(mv.coverage_ratio, 0.9167)
        d = mv.to_dict()
        self.assertEqual(d["status"], "AVAILABLE")
        self.assertEqual(d["value"], 0.9167)

    def test_metric_value_partial_status(self):
        """PARTIAL metric value must have a numeric float and explicitly indicates partial coverage."""
        mv = MetricValue(
            metric_key="dimension_semantic_alignment_score",
            metric_version="v1",
            status=MetricValueStatus.PARTIAL,
            value=0.85,
            sample_count=6,
            eligible_count=10,
        )
        self.assertEqual(mv.value, 0.85)
        self.assertEqual(mv.coverage_ratio, 0.6)
        self.assertEqual(mv.status, MetricValueStatus.PARTIAL)

    def test_metric_value_unavailable_invariant(self):
        """UNAVAILABLE metric value must have value=None and rejects float values."""
        mv = MetricValue(
            metric_key="first_asset_quality_pass_rate",
            metric_version="v1",
            status=MetricValueStatus.UNAVAILABLE,
            value=None,
            eligible_count=0,
        )
        self.assertIsNone(mv.value)
        self.assertIsNone(mv.coverage_ratio)

        with self.assertRaises(ValueError):
            MetricValue(
                metric_key="first_asset_quality_pass_rate",
                metric_version="v1",
                status=MetricValueStatus.UNAVAILABLE,
                value=0.0,  # Zero-fabrication rejected!
            )

    def test_metric_value_not_applicable_invariant(self):
        """NOT_APPLICABLE metric value must have value=None and rejects float values."""
        mv = MetricValue(
            metric_key="human_edit_rate",
            metric_version="v1",
            status=MetricValueStatus.NOT_APPLICABLE,
            value=None,
            notes="Unattended benchmark run",
        )
        self.assertIsNone(mv.value)
        self.assertEqual(mv.status, MetricValueStatus.NOT_APPLICABLE)

        with self.assertRaises(ValueError):
            MetricValue(
                metric_key="human_edit_rate",
                metric_version="v1",
                status=MetricValueStatus.NOT_APPLICABLE,
                value=0.0,
            )

    def test_cost_summary_serialization(self):
        """Verifies CostSummary maintains currency grouping and synthetic flag."""
        cs = CostSummary(
            observed_by_currency={"USD": 1.25},
            estimated_by_currency={"USD": 0.50},
            observed_ops=5,
            estimated_ops=2,
            unknown_ops=1,
            completeness="PARTIAL",
            is_synthetic=True,
        )
        d = cs.to_dict()
        self.assertEqual(d["observed_by_currency"]["USD"], 1.25)
        self.assertTrue(d["is_synthetic"])
        self.assertEqual(d["unknown_ops"], 1)

    def test_stage_latency_summary(self):
        """Verifies StageLatencySummary serialization."""
        lat = StageLatencySummary(
            stage="planning",
            mean_ms=1234.56,
            p50_ms=1200.0,
            min_ms=1000.0,
            max_ms=1500.0,
            sample_count=5,
        )
        d = lat.to_dict()
        self.assertEqual(d["mean_ms"], 1234.56)
        self.assertEqual(d["p50_ms"], 1200.0)
        self.assertEqual(d["sample_count"], 5)


if __name__ == "__main__":
    unittest.main()
