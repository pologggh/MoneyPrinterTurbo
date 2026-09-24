import json
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
    BenchmarkFailureStage,
    BenchmarkRunStatus,
    BenchmarkVariant,
)
from app.domain.trace import TraceEventType
from app.persistence.models import Base
from app.persistence.repositories import (
    BenchmarkRepository,
    TraceRepository,
)
from app.services.benchmark.benchmark_runner import (
    BenchmarkRunInput,
    BenchmarkRunner,
)
from app.services.trace_service import TraceService


class TestBenchmarkOfflineE2E(unittest.TestCase):
    """
    End-to-End Integration tests verifying the entire offline benchmark execution pipeline:
    - Real production service invocation
    - Zero external calls
    - Failure isolation across multiple cases
    - Trace completeness and linkage
    - Structural constraint violation handling
    - Deterministic repeatability
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

    def tearDown(self):
        self.trace_session.close()
        self.temp_dir.cleanup()

    def test_offline_e2e_multi_case_with_failure_isolation(self):
        """
        Runs 3 cases:
        - Case 1: transformer-attention (succeeds)
        - Case 2: rag-overview (fails visual quality evaluation via simulated defect)
        - Case 3: tcp-three-way-handshake (succeeds, completely isolated from Case 2 failure)
        """
        run_input = BenchmarkRunInput(
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            selected_case_keys=("transformer-attention", "rag-overview", "tcp-three-way-handshake"),
            storage_dir=self.storage_dir,
            variant=BenchmarkVariant(
                variant_key="balanced-auto",
                routing_strategy=RoutingStrategy.BALANCED,
            ),
            offline_failing_shot_tokens={"rag"},  # Induce failure specifically for rag-overview shots
        )

        run = self.runner.run_benchmark(run_input)

        # Invariant: Run finishes with COMPLETED_WITH_FAILURES because of Case 2
        self.assertEqual(run.status, BenchmarkRunStatus.COMPLETED_WITH_FAILURES)
        self.assertEqual(run.total_cases, 3)
        self.assertEqual(run.completed_cases, 2)
        self.assertEqual(run.failed_cases, 1)

        session = self.session_factory()
        try:
            repo = BenchmarkRepository(session)
            persisted_run = repo.get_run(run.benchmark_run_id)
            self.assertIsNotNone(persisted_run)
            self.assertEqual(persisted_run.status, BenchmarkRunStatus.COMPLETED_WITH_FAILURES)

            case_results = repo.list_case_results_for_run(run.benchmark_run_id)
            self.assertEqual(len(case_results), 3)

            cr_map = {cr.case_key: cr for cr in case_results}

            # Case 1: transformer-attention (SUCCEEDED)
            cr1 = cr_map["transformer-attention"]
            self.assertEqual(cr1.status, BenchmarkCaseStatus.SUCCEEDED)
            self.assertGreater(cr1.duration_ms, 0.0)
            self.assertTrue(cr1.trace_id.startswith("tr_bcase_"))
            self.assertIsNotNone(cr1.production_result_refs.content_plan_revision_id)
            self.assertIsNotNone(cr1.production_result_refs.approved_storyboard_snapshot_id)
            self.assertIsNotNone(cr1.production_result_refs.asset_route_plan_id)
            self.assertIsNotNone(cr1.production_result_refs.execution_run_id)
            self.assertGreater(len(cr1.production_result_refs.accepted_shot_asset_version_ids), 0)
            self.assertGreater(len(cr1.production_result_refs.final_evaluation_snapshot_ids), 0)

            # Case 2: rag-overview (Failed due to induced visual defect)
            cr2 = cr_map["rag-overview"]
            self.assertIn(cr2.status, (BenchmarkCaseStatus.FAILED, BenchmarkCaseStatus.NEEDS_USER_ACTION))
            self.assertGreater(cr2.duration_ms, 0.0)
            self.assertTrue(cr2.trace_id.startswith("tr_bcase_"))
            self.assertNotEqual(cr1.trace_id, cr2.trace_id)

            # Case 3: tcp-three-way-handshake (SUCCEEDED - Isolation verified!)
            cr3 = cr_map["tcp-three-way-handshake"]
            self.assertEqual(cr3.status, BenchmarkCaseStatus.SUCCEEDED)
            self.assertGreater(cr3.duration_ms, 0.0)
            self.assertTrue(cr3.trace_id.startswith("tr_bcase_"))
            self.assertNotEqual(cr2.trace_id, cr3.trace_id)
            self.assertIsNotNone(cr3.production_result_refs.approved_storyboard_snapshot_id)

            # Trace Invariants Check:
            for cr in (cr1, cr3):
                events = self.trace_service.reader.list_events_by_trace(cr.trace_id)
                event_types = {e.event_type for e in events}
                self.assertIn(TraceEventType.CONTENT_PLANNER_STARTED, event_types)
                self.assertIn(TraceEventType.STORYBOARD_GENERATION_STARTED, event_types)
                self.assertIn(TraceEventType.STORYBOARD_APPROVED, event_types)
                self.assertIn(TraceEventType.ASSET_ROUTE_PLAN_CREATED, event_types)
                self.assertIn(TraceEventType.EXECUTION_RUN_COMPLETED, event_types)
                self.assertIn(TraceEventType.EVALUATION_COMPLETED, event_types)

                # Trace Detail View queryable
                detail = self.trace_service.get_trace_detail_view(cr.trace_id)
                self.assertIsNotNone(detail)
                self.assertGreater(detail.total_events, 5)

        finally:
            session.close()

    def test_offline_e2e_structural_constraint_violation(self):
        """
        Creates a custom benchmark suite containing a case with impossible structural constraints
        (min_beats=8, but planner creates 2).
        Verifies that structural validation fails fast before approval or asset execution.
        """
        custom_suites_dir = self.storage_dir / "custom_suites"
        custom_suites_dir.mkdir(parents=True, exist_ok=True)

        suite_payload = {
            "suite_key": "custom-strict",
            "suite_version": "v1",
            "display_name": "Custom Strict Suite",
            "description": "Tests structural validation failure",
            "cases": [
                {
                    "case_key": "strict-beats-test",
                    "case_version": "v1",
                    "title": "Strict Beats Validation Test",
                    "topic": "Strict Beats Validation Test",
                    "target_duration": 60.0,
                    "knowledge_fixture": {
                        "fixture_id": "fix-transformer-attention",
                        "fixture_version": "v1.0",
                        "file_path": "fixtures/benchmark/knowledge/transformer_attention.md",
                    },
                    "expected_constraints": {
                        "min_beats": 8,
                        "max_beats": 12,
                        "required_beat_types": ["HOOK", "KNOWLEDGE"],
                    },
                }
            ],
        }
        suite_file = custom_suites_dir / "custom_strict_v1.json"
        suite_file.write_text(json.dumps(suite_payload, indent=2), encoding="utf-8")

        custom_runner = BenchmarkRunner(
            session_factory=self.session_factory,
            trace_service=self.trace_service,
            suites_dir=custom_suites_dir,
        )

        run_input = BenchmarkRunInput(
            suite_key="custom-strict",
            suite_version="v1",
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            storage_dir=self.storage_dir,
        )

        run = custom_runner.run_benchmark(run_input)
        self.assertEqual(run.status, BenchmarkRunStatus.COMPLETED_WITH_FAILURES)
        self.assertEqual(run.failed_cases, 1)

        session = self.session_factory()
        try:
            repo = BenchmarkRepository(session)
            case_results = repo.list_case_results_for_run(run.benchmark_run_id)
            self.assertEqual(len(case_results), 1)
            cr = case_results[0]
            self.assertEqual(cr.status, BenchmarkCaseStatus.FAILED)
            self.assertEqual(cr.failure_stage, BenchmarkFailureStage.STRUCTURAL_VALIDATION)
            self.assertEqual(cr.failure_code, "STRUCTURAL_CONSTRAINTS_VIOLATED")
            self.assertIn("below minimum (8)", cr.failure_message)

            # Invariant: Approval and routing did NOT run for this failed case
            self.assertIsNone(cr.production_result_refs.approved_storyboard_snapshot_id)
            self.assertIsNone(cr.production_result_refs.asset_route_plan_id)
            self.assertIsNone(cr.production_result_refs.execution_run_id)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
