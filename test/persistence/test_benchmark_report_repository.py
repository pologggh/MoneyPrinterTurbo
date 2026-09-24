import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.asset_router import ModelSelectionMode, RoutingStrategy
from app.domain.benchmark import (
    BenchmarkExecutionMode,
    BenchmarkRun,
    BenchmarkRunStatus,
    BenchmarkVariant,
)
from app.domain.benchmark_comparison import BenchmarkComparisonReport, MetricDelta
from app.domain.benchmark_metrics import (
    CostSummary,
    MetricDirection,
    MetricValue,
    MetricValueStatus,
    StageLatencySummary,
)
from app.domain.benchmark_report import BenchmarkReport
from app.persistence.models import Base
from app.persistence.repositories import BenchmarkRepository


class TestBenchmarkReportRepository(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.session = self.session_factory()
        self.repo = BenchmarkRepository(self.session)

        # Setup 2 runs
        self.variant_a = BenchmarkVariant(
            variant_key="balanced-auto",
            routing_strategy=RoutingStrategy.BALANCED,
            model_selection_mode=ModelSelectionMode.AUTO,
        )
        self.variant_b = BenchmarkVariant(
            variant_key="quality-auto",
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
            model_selection_mode=ModelSelectionMode.AUTO,
        )

        self.run_a = BenchmarkRun(
            benchmark_run_id="brun-repo-a",
            benchmark_suite_id="bsuite-1",
            suite_key="test-suite",
            suite_version="v1",
            suite_fingerprint="fp1",
            variant=self.variant_a,
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            status=BenchmarkRunStatus.COMPLETED,
        )
        self.run_b = BenchmarkRun(
            benchmark_run_id="brun-repo-b",
            benchmark_suite_id="bsuite-1",
            suite_key="test-suite",
            suite_version="v1",
            suite_fingerprint="fp1",
            variant=self.variant_b,
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            status=BenchmarkRunStatus.COMPLETED,
        )
        self.repo.add_run(self.run_a)
        self.repo.add_run(self.run_b)
        self.session.commit()

    def tearDown(self):
        self.session.close()

    def test_add_and_get_report(self):
        """Verifies saving and retrieving a BenchmarkReport."""
        metrics = {
            "first_asset_quality_pass_rate": MetricValue(
                metric_key="first_asset_quality_pass_rate",
                metric_version="v1",
                status=MetricValueStatus.AVAILABLE,
                value=0.85,
                sample_count=17,
                eligible_count=20,
            )
        }
        latencies = {
            "end_to_end": StageLatencySummary(stage="end_to_end", mean_ms=3500.0, p50_ms=3200.0, sample_count=5)
        }
        cost = CostSummary(observed_by_currency={"USD": 0.45}, is_synthetic=True)

        report = BenchmarkReport(
            report_id="brep-test-1",
            benchmark_run_id="brun-repo-a",
            suite_key="test-suite",
            suite_version="v1",
            suite_fingerprint="fp1",
            variant=self.variant_a,
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            metrics=metrics,
            stage_latencies=latencies,
            cost_summary=cost,
            provider_model_usage={"mock/model-1": {"attempts": 10}},
            case_count=5,
            shot_count=10,
        )

        saved = self.repo.add_report(report)
        self.session.commit()

        self.assertEqual(saved.report_id, "brep-test-1")
        self.assertEqual(saved.suite_key, "test-suite")
        self.assertTrue(len(saved.source_fingerprint) == 64)

        # Retrieve by report_id
        fetched = self.repo.get_report("brep-test-1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.report_id, "brep-test-1")
        self.assertEqual(fetched.metrics["first_asset_quality_pass_rate"].value, 0.85)
        self.assertEqual(fetched.stage_latencies["end_to_end"].mean_ms, 3500.0)
        self.assertTrue(fetched.cost_summary.is_synthetic)

        # Retrieve by run_id
        by_run = self.repo.get_report_by_run("brun-repo-a", "v1")
        self.assertIsNotNone(by_run)
        self.assertEqual(by_run.report_id, "brep-test-1")

        # List for suite
        suite_reports = self.repo.list_reports_for_suite("test-suite")
        self.assertEqual(len(suite_reports), 1)

    def test_add_and_get_comparison_report(self):
        """Verifies saving and retrieving a BenchmarkComparisonReport."""
        rep_a = BenchmarkReport(
            report_id="brep-a",
            benchmark_run_id="brun-repo-a",
            suite_key="test-suite",
            suite_version="v1",
            suite_fingerprint="fp1",
            variant=self.variant_a,
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            metrics={},
            stage_latencies={},
            cost_summary=CostSummary(),
            provider_model_usage={},
            case_count=5,
            shot_count=10,
        )
        rep_b = BenchmarkReport(
            report_id="brep-b",
            benchmark_run_id="brun-repo-b",
            suite_key="test-suite",
            suite_version="v1",
            suite_fingerprint="fp1",
            variant=self.variant_b,
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            metrics={},
            stage_latencies={},
            cost_summary=CostSummary(),
            provider_model_usage={},
            case_count=5,
            shot_count=10,
        )
        self.repo.add_report(rep_a)
        self.repo.add_report(rep_b)
        self.session.commit()

        comp = BenchmarkComparisonReport(
            comparison_id="bcomp-test-1",
            baseline_report_id="brep-a",
            candidate_report_id="brep-b",
            baseline_variant_key="balanced-auto",
            candidate_variant_key="quality-auto",
            suite_key="test-suite",
            suite_version="v1",
            suite_fingerprint="fp1",
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            metric_deltas={
                "first_asset_quality_pass_rate": MetricDelta(
                    metric_key="first_asset_quality_pass_rate",
                    metric_version="v1",
                    baseline_value=0.6,
                    candidate_value=0.8,
                    absolute_delta=0.2,
                    relative_delta=0.3333,
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    is_improvement=True,
                    status="COMPARED",
                )
            },
            latency_deltas={},
            cost_deltas={"is_synthetic": True},
            tradeoff_summary={"findings": ["Quality improved +20%."]},
            case_drilldowns=[],
            failure_breakdown={},
        )

        saved_comp = self.repo.add_comparison_report(comp)
        self.session.commit()

        self.assertEqual(saved_comp.comparison_id, "bcomp-test-1")

        fetched_comp = self.repo.get_comparison_report("bcomp-test-1")
        self.assertIsNotNone(fetched_comp)
        self.assertEqual(fetched_comp.baseline_variant_key, "balanced-auto")
        self.assertEqual(fetched_comp.candidate_variant_key, "quality-auto")
        delta = fetched_comp.metric_deltas["first_asset_quality_pass_rate"]
        self.assertEqual(delta.absolute_delta, 0.2)
        self.assertTrue(delta.is_improvement)


if __name__ == "__main__":
    unittest.main()
