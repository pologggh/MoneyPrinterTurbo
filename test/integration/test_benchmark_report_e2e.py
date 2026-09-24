import json
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.controllers.v1 import benchmark as benchmark_controller
from app.controllers.v1.benchmark import (
    ApiCompareReportsRequest,
    ApiGenerateReportRequest,
)
from app.domain.asset_router import RoutingStrategy
from app.domain.benchmark import (
    BenchmarkExecutionMode,
    BenchmarkRunStatus,
    BenchmarkVariant,
)
from app.domain.benchmark_metrics import MetricValueStatus
from app.domain.benchmark_report import BenchmarkReport
from app.persistence.models import Base
from app.persistence.repositories import (
    TraceRepository,
)
from app.services.benchmark.benchmark_runner import (
    BenchmarkRunInput,
    BenchmarkRunner,
)
from app.services.benchmark.comparison_gate import (
    validate_benchmark_comparability,
)
from app.services.benchmark.report_service import BenchmarkReportService
from app.services.trace_service import TraceService


class TestBenchmarkReportE2E(unittest.TestCase):
    """
    End-to-End Integration tests verifying Phase 7.3:
    - Benchmark Report generation from immutable production runs & traces
    - Versioned metrics aggregation and strict status invariants
    - Compatibility hard gates
    - Strategy comparison and deterministic deltas
    - Case-level drilldown preserving trace_id
    - JSON exports and API controller endpoints
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name)

        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)

        self.trace_session = self.session_factory()
        self.trace_repo = TraceRepository(self.trace_session)
        self.trace_service = TraceService(repository=self.trace_repo)
        self.runner = BenchmarkRunner(
            session_factory=self.session_factory,
            trace_service=self.trace_service,
        )
        self.report_service = BenchmarkReportService(session_factory=self.session_factory)

    def tearDown(self):
        self.trace_session.close()
        self.temp_dir.cleanup()

    def test_benchmark_report_and_strategy_comparison_e2e(self):
        """
        Executes two benchmark variants offline on the same cases:
        - Variant A: BALANCED
        - Variant B: QUALITY_FIRST
        Extracts BenchmarkReports, validates compatibility gate,
        generates BenchmarkComparisonReport, and checks JSON export.
        """
        # 1. Run Variant A (BALANCED)
        run_input_a = BenchmarkRunInput(
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            selected_case_keys=("transformer-attention", "tcp-three-way-handshake"),
            storage_dir=self.storage_dir / "run_a",
            variant=BenchmarkVariant(
                variant_key="balanced-auto",
                routing_strategy=RoutingStrategy.BALANCED,
            ),
        )
        run_a = self.runner.run_benchmark(run_input_a)
        self.assertEqual(run_a.status, BenchmarkRunStatus.COMPLETED)
        self.assertEqual(run_a.completed_cases, 2)

        # 2. Run Variant B (QUALITY_FIRST)
        run_input_b = BenchmarkRunInput(
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            selected_case_keys=("transformer-attention", "tcp-three-way-handshake"),
            storage_dir=self.storage_dir / "run_b",
            variant=BenchmarkVariant(
                variant_key="quality-auto",
                routing_strategy=RoutingStrategy.QUALITY_FIRST,
            ),
        )
        run_b = self.runner.run_benchmark(run_input_b)
        self.assertEqual(run_b.status, BenchmarkRunStatus.COMPLETED)
        self.assertEqual(run_b.completed_cases, 2)

        # 3. Generate BenchmarkReport for Run A
        report_a = self.report_service.generate_report(run_a.benchmark_run_id)
        self.assertIsNotNone(report_a)
        self.assertEqual(report_a.benchmark_run_id, run_a.benchmark_run_id)
        self.assertEqual(report_a.case_count, 2)
        self.assertGreater(report_a.shot_count, 0)
        self.assertEqual(len(report_a.source_fingerprint), 64)

        # Check key metrics in Report A
        metrics_a = report_a.metrics
        self.assertIn("first_asset_quality_pass_rate", metrics_a)
        self.assertIn("final_shot_quality_pass_rate", metrics_a)
        self.assertIn("plan_validation_success_rate", metrics_a)
        self.assertIn("storyboard_generation_success_rate", metrics_a)
        self.assertIn("dimension_semantic_alignment_score", metrics_a)
        self.assertIn("human_edit_rate", metrics_a)

        # Planning validation must be 1.0 (both succeeded)
        self.assertEqual(metrics_a["plan_validation_success_rate"].value, 1.0)
        # Human metrics must be NOT_APPLICABLE
        self.assertEqual(metrics_a["human_edit_rate"].status, MetricValueStatus.NOT_APPLICABLE)
        self.assertIsNone(metrics_a["human_edit_rate"].value)

        # Cost summary synthetic flag
        self.assertTrue(report_a.cost_summary.is_synthetic)

        # Verify idempotency: generating report again returns existing without duplicate insertion
        report_a_again = self.report_service.generate_report(run_a.benchmark_run_id)
        self.assertEqual(report_a.report_id, report_a_again.report_id)
        self.assertEqual(report_a.source_fingerprint, report_a_again.source_fingerprint)

        # 4. Generate BenchmarkReport for Run B
        report_b = self.report_service.generate_report(run_b.benchmark_run_id)
        self.assertIsNotNone(report_b)
        self.assertEqual(report_b.variant.variant_key, "quality-auto")
        self.assertEqual(len(report_b.source_fingerprint), 64)

        # 5. Validate Compatibility Hard Gates
        is_valid, reason = validate_benchmark_comparability(report_a, report_b)
        self.assertTrue(is_valid, f"Expected reports to be comparable, but got: {reason}")

        # Test gate failure on mismatched execution mode
        report_b_fake_real = self.report_service.get_report(report_b.report_id)
        report_b_fake_real = BenchmarkReport(
            report_id="brep-fake-real",
            benchmark_run_id=run_b.benchmark_run_id,
            suite_key=report_b.suite_key,
            suite_version=report_b.suite_version,
            suite_fingerprint=report_b.suite_fingerprint,
            variant=report_b.variant,
            execution_mode=BenchmarkExecutionMode.REAL,  # Mismatched!
            metrics=report_b.metrics,
            stage_latencies=report_b.stage_latencies,
            cost_summary=report_b.cost_summary,
            provider_model_usage=report_b.provider_model_usage,
            case_count=report_b.case_count,
            shot_count=report_b.shot_count,
        )
        gate_ok, gate_err = validate_benchmark_comparability(report_a, report_b_fake_real)
        self.assertFalse(gate_ok)
        self.assertIn("Execution mode mismatch", gate_err)

        # 6. Execute Strategy Comparison
        comparison = self.report_service.compare_reports(report_a.report_id, report_b.report_id)
        self.assertIsNotNone(comparison)
        self.assertEqual(comparison.baseline_variant_key, "balanced-auto")
        self.assertEqual(comparison.candidate_variant_key, "quality-auto")
        self.assertEqual(len(comparison.source_fingerprint), 64)

        # Check metric deltas
        deltas = comparison.metric_deltas
        self.assertIn("first_asset_quality_pass_rate", deltas)
        self.assertIn("final_shot_quality_pass_rate", deltas)
        self.assertIn("fallback_rate", deltas)
        self.assertIn("mean_generation_attempts_per_shot", deltas)

        # Check case drilldowns with exact trace_id
        self.assertEqual(len(comparison.case_drilldowns), 2)
        for cd in comparison.case_drilldowns:
            self.assertTrue(cd.baseline_trace_id.startswith("tr_bcase_"))
            self.assertTrue(cd.candidate_trace_id.startswith("tr_bcase_"))
            self.assertNotEqual(cd.baseline_trace_id, cd.candidate_trace_id)
            self.assertEqual(cd.baseline_status, "SUCCEEDED")
            self.assertEqual(cd.candidate_status, "SUCCEEDED")

        # Check trade-off summary
        self.assertIn("quality", comparison.tradeoff_summary)
        self.assertIn("reliability", comparison.tradeoff_summary)
        self.assertIn("efficiency", comparison.tradeoff_summary)

        # 7. Check Deterministic JSON Exports
        rep_json = report_a.to_json()
        parsed_rep = json.loads(rep_json)
        self.assertEqual(parsed_rep["report_id"], report_a.report_id)
        self.assertEqual(parsed_rep["source_fingerprint"], report_a.source_fingerprint)

        comp_json = comparison.to_json()
        parsed_comp = json.loads(comp_json)
        self.assertEqual(parsed_comp["comparison_id"], comparison.comparison_id)
        self.assertEqual(parsed_comp["source_fingerprint"], comparison.source_fingerprint)
        self.assertEqual(len(parsed_comp["case_drilldowns"]), 2)

    def test_benchmark_api_controller_endpoints(self):
        """
        Tests FastAPI controller endpoint logic directly:
        - POST /benchmark/runs/{run_id}/report
        - GET /benchmark/reports/{report_id}
        - POST /benchmark/compare
        - GET /benchmark/comparisons/{comparison_id}
        """
        # Run a 1-case benchmark
        run_input = BenchmarkRunInput(
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            selected_case_keys=("transformer-attention",),
            storage_dir=self.storage_dir / "api_run",
            variant=BenchmarkVariant(variant_key="balanced-auto"),
        )
        run = self.runner.run_benchmark(run_input)

        # Patch session in controller to use our static test engine
        import app.controllers.v1.benchmark as bc_module
        old_service_cls = bc_module.BenchmarkReportService

        class TestableReportService(BenchmarkReportService):
            def __init__(self, _session_getter=None):
                super().__init__(session_factory=self_test.session_factory)

        self_test = self
        bc_module.BenchmarkReportService = TestableReportService
        try:
            # 1. Generate report via controller
            gen_resp = benchmark_controller.generate_benchmark_report(
                run_id=run.benchmark_run_id,
                body=ApiGenerateReportRequest(metric_version="v1"),
            )
            self.assertEqual(gen_resp["status"], 200)
            rep_data = gen_resp["data"]
            report_id = rep_data["report_id"]
            self.assertEqual(rep_data["benchmark_run_id"], run.benchmark_run_id)

            # 2. Retrieve report via controller
            get_resp = benchmark_controller.get_benchmark_report(report_id=report_id)
            self.assertEqual(get_resp["status"], 200)
            self.assertEqual(get_resp["data"]["report_id"], report_id)

            # 3. Compare reports via controller (comparing with self for contract test)
            comp_resp = benchmark_controller.compare_benchmarks(
                body=ApiCompareReportsRequest(
                    baseline_report_id=report_id,
                    candidate_report_id=report_id,
                )
            )
            self.assertEqual(comp_resp["status"], 200)
            comp_data = comp_resp["data"]
            comp_id = comp_data["comparison_id"]
            self.assertEqual(comp_data["baseline_report_id"], report_id)

            # 4. Retrieve comparison via controller
            get_comp = benchmark_controller.get_benchmark_comparison(comparison_id=comp_id)
            self.assertEqual(get_comp["status"], 200)
            self.assertEqual(get_comp["data"]["comparison_id"], comp_id)

        finally:
            bc_module.BenchmarkReportService = old_service_cls


if __name__ == "__main__":
    unittest.main()
