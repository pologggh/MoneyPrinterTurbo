import unittest
from datetime import UTC, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.domain.asset_router import ModelSelectionMode, RoutingStrategy
from app.domain.benchmark import (
    BenchmarkCaseResult,
    BenchmarkCaseStatus,
    BenchmarkExecutionMode,
    BenchmarkFailureStage,
    BenchmarkProductionResultRefs,
    BenchmarkRun,
    BenchmarkRunStatus,
    BenchmarkVariant,
)
from app.persistence.models import Base
from app.persistence.repositories import BenchmarkRepository


class TestBenchmarkRepository(unittest.TestCase):

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.session = self.session_factory()
        self.repo = BenchmarkRepository(self.session)

    def tearDown(self):
        self.session.close()

    def test_add_and_get_benchmark_run(self):
        """Verify adding and retrieving a BenchmarkRun."""
        variant = BenchmarkVariant(
            variant_key="balanced-auto",
            routing_strategy=RoutingStrategy.BALANCED,
            model_selection_mode=ModelSelectionMode.AUTO,
        )
        run = BenchmarkRun(
            benchmark_run_id="brun-test-1",
            benchmark_suite_id="bsuite-1",
            suite_key="knowledge-video-v1",
            suite_version="v1",
            suite_fingerprint="fp12345",
            variant=variant,
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            status=BenchmarkRunStatus.RUNNING,
            started_at=datetime(2026, 9, 11, 12, 0, 0, tzinfo=UTC),
            total_cases=12,
        )
        saved = self.repo.add_run(run)
        self.session.commit()

        self.assertEqual(saved.benchmark_run_id, "brun-test-1")
        self.assertEqual(saved.suite_key, "knowledge-video-v1")
        self.assertEqual(saved.status, BenchmarkRunStatus.RUNNING)

        fetched = self.repo.get_run("brun-test-1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.benchmark_run_id, "brun-test-1")
        self.assertEqual(fetched.variant.variant_key, "balanced-auto")
        self.assertEqual(fetched.total_cases, 12)

    def test_update_benchmark_run(self):
        """Verify updating status and counters on a BenchmarkRun."""
        variant = BenchmarkVariant(variant_key="quality-auto")
        run = BenchmarkRun(
            benchmark_run_id="brun-test-2",
            benchmark_suite_id="bsuite-1",
            suite_key="knowledge-video-v1",
            suite_version="v1",
            suite_fingerprint="fp12345",
            variant=variant,
            status=BenchmarkRunStatus.RUNNING,
            total_cases=12,
        )
        self.repo.add_run(run)
        self.session.commit()

        # Update run
        updated_run = BenchmarkRun(
            benchmark_run_id="brun-test-2",
            benchmark_suite_id="bsuite-1",
            suite_key="knowledge-video-v1",
            suite_version="v1",
            suite_fingerprint="fp12345",
            variant=variant,
            status=BenchmarkRunStatus.COMPLETED,
            finished_at=datetime.now(UTC),
            total_cases=12,
            completed_cases=12,
            failed_cases=0,
        )
        saved = self.repo.update_run(updated_run)
        self.session.commit()

        self.assertEqual(saved.status, BenchmarkRunStatus.COMPLETED)
        self.assertEqual(saved.completed_cases, 12)
        self.assertEqual(saved.failed_cases, 0)

    def test_add_and_list_case_results(self):
        """Verify saving and listing BenchmarkCaseResults linked to BenchmarkRun."""
        variant = BenchmarkVariant(variant_key="balanced-auto")
        run = BenchmarkRun(
            benchmark_run_id="brun-test-3",
            benchmark_suite_id="bsuite-1",
            suite_key="knowledge-video-v1",
            suite_version="v1",
            suite_fingerprint="fp12345",
            variant=variant,
            status=BenchmarkRunStatus.RUNNING,
            total_cases=2,
        )
        self.repo.add_run(run)
        self.session.commit()

        res1 = BenchmarkCaseResult(
            benchmark_case_result_id="bcres-1",
            benchmark_run_id="brun-test-3",
            benchmark_case_id="bcase-1",
            case_key="transformer-attention",
            case_version="v1",
            case_fingerprint="fp-c1",
            trace_id="tr-111",
            status=BenchmarkCaseStatus.SUCCEEDED,
            started_at=datetime(2026, 9, 11, 12, 1, 0, tzinfo=UTC),
            finished_at=datetime(2026, 9, 11, 12, 1, 10, tzinfo=UTC),
            duration_ms=10000.0,
            production_result_refs=BenchmarkProductionResultRefs(
                content_plan_revision_id="cprev-1",
                approved_storyboard_snapshot_id="snap-1",
                execution_run_id="run-1",
            ),
        )
        res2 = BenchmarkCaseResult(
            benchmark_case_result_id="bcres-2",
            benchmark_run_id="brun-test-3",
            benchmark_case_id="bcase-2",
            case_key="rag-overview",
            case_version="v1",
            case_fingerprint="fp-c2",
            trace_id="tr-222",
            status=BenchmarkCaseStatus.FAILED,
            started_at=datetime(2026, 9, 11, 12, 1, 15, tzinfo=UTC),
            finished_at=datetime(2026, 9, 11, 12, 1, 20, tzinfo=UTC),
            duration_ms=5000.0,
            failure_stage=BenchmarkFailureStage.EVALUATION,
            failure_code="VISUAL_QUALITY_FAIL",
            failure_message="Blurry artifacts in visual observation",
        )
        self.repo.add_case_result(res1)
        self.repo.add_case_result(res2)
        self.session.commit()

        results = self.repo.list_case_results_for_run("brun-test-3")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].case_key, "transformer-attention")
        self.assertEqual(results[0].status, BenchmarkCaseStatus.SUCCEEDED)
        self.assertEqual(results[0].trace_id, "tr-111")
        self.assertEqual(results[0].production_result_refs.content_plan_revision_id, "cprev-1")

        self.assertEqual(results[1].case_key, "rag-overview")
        self.assertEqual(results[1].status, BenchmarkCaseStatus.FAILED)
        self.assertEqual(results[1].failure_stage, BenchmarkFailureStage.EVALUATION)
        self.assertEqual(results[1].failure_code, "VISUAL_QUALITY_FAIL")


if __name__ == "__main__":
    unittest.main()
