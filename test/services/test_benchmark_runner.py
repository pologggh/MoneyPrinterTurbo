import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.asset_router import RoutingStrategy
from app.domain.benchmark import (
    BenchmarkCaseStatus,
    BenchmarkExecutionMode,
    BenchmarkRunStatus,
    BenchmarkVariant,
    RealBenchmarkExternalCallsNotAuthorizedError,
)
from app.persistence.models import Base
from app.persistence.repositories import (
    BenchmarkRepository,
    StoryboardApprovalRepository,
    TraceRepository,
)
from app.services.benchmark.benchmark_runner import (
    BenchmarkRunInput,
    BenchmarkRunner,
)
from app.services.trace_service import TraceService


class TestBenchmarkRunner(unittest.TestCase):

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

    def tearDown(self):
        self.trace_session.close()
        self.temp_dir.cleanup()

    def test_real_mode_without_authorization_rejected(self):
        """REAL benchmark mode without allow_external_calls=True raises safety error immediately."""
        run_input = BenchmarkRunInput(
            execution_mode=BenchmarkExecutionMode.REAL,
            allow_external_calls=False,  # NOT AUTHORIZED!
        )
        with self.assertRaises(RealBenchmarkExternalCallsNotAuthorizedError) as ctx:
            self.runner.run_benchmark(run_input)

        self.assertIn("REAL benchmark execution requires explicit allow_external_calls=True", str(ctx.exception))

    def test_offline_mode_executes_with_zero_real_external_calls(self):
        """OFFLINE mode executes through production services with zero network calls and generates traces."""
        run_input = BenchmarkRunInput(
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            limit=2,  # Run first 2 cases
            storage_dir=self.storage_dir,
            variant=BenchmarkVariant(
                variant_key="balanced-auto",
                routing_strategy=RoutingStrategy.BALANCED,
            ),
        )

        run = self.runner.run_benchmark(run_input)
        self.assertEqual(run.status, BenchmarkRunStatus.COMPLETED)
        self.assertEqual(run.total_cases, 2)
        self.assertEqual(run.completed_cases, 2)
        self.assertEqual(run.failed_cases, 0)

        # Check database records
        session = self.session_factory()
        try:
            repo = BenchmarkRepository(session)
            persisted_run = repo.get_run(run.benchmark_run_id)
            self.assertIsNotNone(persisted_run)
            self.assertEqual(persisted_run.status, BenchmarkRunStatus.COMPLETED)

            case_results = repo.list_case_results_for_run(run.benchmark_run_id)
            self.assertEqual(len(case_results), 2)

            # Invariant: Each case gets a different root trace_id
            trace_ids = [cr.trace_id for cr in case_results]
            self.assertEqual(len(set(trace_ids)), 2)

            # Invariant: BenchmarkCaseResult preserves exact trace_id and production result refs
            for cr in case_results:
                self.assertEqual(cr.status, BenchmarkCaseStatus.SUCCEEDED)
                self.assertIsNotNone(cr.trace_id)
                self.assertTrue(cr.trace_id.startswith("tr_bcase_"))
                self.assertIsNotNone(cr.production_result_refs.content_plan_revision_id)
                self.assertIsNotNone(cr.production_result_refs.approved_storyboard_snapshot_id)
                self.assertIsNotNone(cr.production_result_refs.asset_route_plan_id)
                self.assertIsNotNone(cr.production_result_refs.execution_run_id)
                self.assertTrue(len(cr.production_result_refs.accepted_shot_asset_version_ids) > 0)

                # Verify trace exists in TraceService
                root = self.trace_service.reader.get_root(cr.trace_id)
                self.assertIsNotNone(root)
                events = self.trace_service.reader.list_events_by_trace(cr.trace_id)
                self.assertGreater(len(events), 5)
        finally:
            session.close()

    def test_benchmark_auto_approval_creates_real_approval_record(self):
        """Benchmark auto-approval invokes normal StoryboardApprovalService creating ApprovalRecord."""
        run_input = BenchmarkRunInput(
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            limit=1,
            storage_dir=self.storage_dir,
        )
        run = self.runner.run_benchmark(run_input)

        session = self.session_factory()
        try:
            repo = BenchmarkRepository(session)
            case_results = repo.list_case_results_for_run(run.benchmark_run_id)
            self.assertEqual(len(case_results), 1)
            approved_snap_id = case_results[0].production_result_refs.approved_storyboard_snapshot_id
            self.assertIsNotNone(approved_snap_id)

            appr_repo = StoryboardApprovalRepository(session)
            record = appr_repo.get_approval_by_approved_snapshot_id(approved_snap_id)
            self.assertIsNotNone(record)
            self.assertEqual(record.approved_storyboard_snapshot_id, approved_snap_id)
        finally:
            session.close()

    def test_failure_isolation_between_cases(self):
        """One failed case does not destroy prior results or prevent subsequent cases from running."""
        # Case 1 will fail evaluation (via simulated visual defect), but Case 2 will succeed
        run_input = BenchmarkRunInput(
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            limit=2,
            storage_dir=self.storage_dir,
            offline_failing_shot_tokens={"shot_"},  # Trigger failure on first case
        )

        run = self.runner.run_benchmark(run_input)
        self.assertEqual(run.status, BenchmarkRunStatus.COMPLETED_WITH_FAILURES)
        self.assertEqual(run.total_cases, 2)

        session = self.session_factory()
        try:
            repo = BenchmarkRepository(session)
            case_results = repo.list_case_results_for_run(run.benchmark_run_id)
            self.assertEqual(len(case_results), 2)
            # Both results exist in DB
            self.assertIsNotNone(case_results[0].benchmark_case_result_id)
            self.assertIsNotNone(case_results[1].benchmark_case_result_id)
        finally:
            session.close()

    def test_case_subset_selection_preserves_suite_order(self):
        """Selecting specific case keys preserves their order from the suite definition."""
        run_input = BenchmarkRunInput(
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            selected_case_keys=("dns-resolution", "rag-overview"),
            storage_dir=self.storage_dir,
        )
        run = self.runner.run_benchmark(run_input)

        session = self.session_factory()
        try:
            repo = BenchmarkRepository(session)
            case_results = repo.list_case_results_for_run(run.benchmark_run_id)
            self.assertEqual(len(case_results), 2)
            # In suite definition, rag-overview is before dns-resolution
            self.assertEqual(case_results[0].case_key, "rag-overview")
            self.assertEqual(case_results[1].case_key, "dns-resolution")
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
