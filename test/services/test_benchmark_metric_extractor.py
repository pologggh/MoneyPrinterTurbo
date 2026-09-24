import unittest
from datetime import UTC, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.benchmark import (
    BenchmarkCaseResult,
    BenchmarkCaseStatus,
    BenchmarkExecutionMode,
    BenchmarkProductionResultRefs,
    BenchmarkRun,
    BenchmarkRunStatus,
    BenchmarkVariant,
)
from app.domain.benchmark_metrics import MetricValueStatus
from app.domain.trace import (
    CostEstimateStatus,
    CostUsageTraceData,
    EvaluationCompletedTraceData,
    ExecutionAttemptTraceData,
    TraceContext,
    TraceEvent,
    TraceEventType,
    TraceRoot,
)
from app.persistence.models import Base
from app.persistence.repositories import (
    BenchmarkRepository,
    ExecutionRepository,
    TraceRepository,
)
from app.services.benchmark.metric_extractor import BenchmarkMetricExtractor


class TestBenchmarkMetricExtractor(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.session = self.session_factory()
        self.extractor = BenchmarkMetricExtractor(self.session)
        self.trace_repo = TraceRepository(self.session)
        self.bench_repo = BenchmarkRepository(self.session)
        self.exec_repo = ExecutionRepository(self.session)

    def tearDown(self):
        self.session.close()

    def test_first_pass_vs_final_pass_isolation(self):
        """
        Shot 1 passes on first evaluation.
        Shot 2 fails on first evaluation, then passes on remediation.
        First-pass rate MUST be 0.5 (1/2), and final-pass rate MUST be 1.0 (2/2).
        Remediated later pass must NOT alter first-pass rate.
        """
        trace_id_1 = "tr-case-1"
        self.trace_repo.add_root(TraceRoot(trace_id=trace_id_1))

        # Shot 1: attempt -> eval pass
        ctx_s1 = TraceContext(trace_id=trace_id_1, shot_id="shot-1", execution_attempt_id="att-1")
        self.trace_repo.add_event(TraceEvent(
            trace_id=trace_id_1,
            event_type=TraceEventType.EXECUTION_ATTEMPT_COMPLETED,
            context=ctx_s1,
            duration_ms=1000.0,
            attributes=ExecutionAttemptTraceData(
                execution_run_id="run-1",
                shot_execution_id="sexec-1",
                execution_attempt_id="att-1",
                shot_revision_id="srev-1",
                provider="mock",
                model="mock-v1",
                generation_mode="VIDEO",
                attempt_number=1,
                resulting_shot_asset_version_id="asset-1",
            ).model_dump(mode="json"),
        ))
        self.trace_repo.add_event(TraceEvent(
            trace_id=trace_id_1,
            event_type=TraceEventType.EVALUATION_COMPLETED,
            context=ctx_s1,
            duration_ms=500.0,
            started_at=datetime(2026, 9, 11, 12, 0, 1, tzinfo=UTC),
            attributes=EvaluationCompletedTraceData(
                evaluation_target_id="target-1",
                evaluation_snapshot_id="snap-1",
                shot_asset_version_id="asset-1",
                evaluation_policy_version="v1.0",
                dimension_statuses={"SEMANTIC_ALIGNMENT": "SCORED", "VISUAL_QUALITY": "SCORED"},
                dimension_scores={"SEMANTIC_ALIGNMENT": 0.9, "VISUAL_QUALITY": 0.85},
                decision="PASS",
            ).model_dump(mode="json"),
        ))

        # Shot 2: attempt -> eval fail -> remediation -> eval pass
        ctx_s2 = TraceContext(trace_id=trace_id_1, shot_id="shot-2", execution_attempt_id="att-2")
        self.trace_repo.add_event(TraceEvent(
            trace_id=trace_id_1,
            event_type=TraceEventType.EXECUTION_ATTEMPT_COMPLETED,
            context=ctx_s2,
            duration_ms=1000.0,
            attributes=ExecutionAttemptTraceData(
                execution_run_id="run-1",
                shot_execution_id="sexec-2",
                execution_attempt_id="att-2",
                shot_revision_id="srev-2",
                provider="mock",
                model="mock-v1",
                generation_mode="VIDEO",
                attempt_number=1,
                resulting_shot_asset_version_id="asset-2",
            ).model_dump(mode="json"),
        ))
        self.trace_repo.add_event(TraceEvent(
            trace_id=trace_id_1,
            event_type=TraceEventType.EVALUATION_COMPLETED,
            context=ctx_s2,
            duration_ms=500.0,
            started_at=datetime(2026, 9, 11, 12, 0, 2, tzinfo=UTC),
            attributes=EvaluationCompletedTraceData(
                evaluation_target_id="target-2",
                evaluation_snapshot_id="snap-2",
                shot_asset_version_id="asset-2",
                evaluation_policy_version="v1.0",
                dimension_statuses={"SEMANTIC_ALIGNMENT": "SCORED", "VISUAL_QUALITY": "SCORED"},
                dimension_scores={"SEMANTIC_ALIGNMENT": 0.4, "VISUAL_QUALITY": 0.5},
                decision="FAIL",
            ).model_dump(mode="json"),
        ))
        self.trace_repo.add_event(TraceEvent(
            trace_id=trace_id_1,
            event_type=TraceEventType.QUALITY_REMEDIATION_DECIDED,
            context=ctx_s2,
            duration_ms=200.0,
            attributes={"action": "REGENERATE_SAME_ROUTE"},
        ))
        # Remediated evaluation for shot-2
        self.trace_repo.add_event(TraceEvent(
            trace_id=trace_id_1,
            event_type=TraceEventType.EVALUATION_COMPLETED,
            context=ctx_s2,
            duration_ms=500.0,
            started_at=datetime(2026, 9, 11, 12, 0, 5, tzinfo=UTC),
            attributes=EvaluationCompletedTraceData(
                evaluation_target_id="target-2-remed",
                evaluation_snapshot_id="snap-2-remed",
                shot_asset_version_id="asset-2-new",
                evaluation_policy_version="v1.0",
                dimension_statuses={"SEMANTIC_ALIGNMENT": "SCORED", "VISUAL_QUALITY": "SCORED"},
                dimension_scores={"SEMANTIC_ALIGNMENT": 0.88, "VISUAL_QUALITY": 0.82},
                decision="PASS",
            ).model_dump(mode="json"),
        ))

        # Emit QUALITY_ASSET_ACCEPTED events
        self.trace_repo.add_event(TraceEvent(
            trace_id=trace_id_1,
            event_type=TraceEventType.QUALITY_ASSET_ACCEPTED,
            context=ctx_s1,
            attributes={"accepted_shot_asset_version_id": "asset-1"},
        ))
        self.trace_repo.add_event(TraceEvent(
            trace_id=trace_id_1,
            event_type=TraceEventType.QUALITY_ASSET_ACCEPTED,
            context=ctx_s2,
            attributes={"accepted_shot_asset_version_id": "asset-2-new"},
        ))

        # Benchmark run and case result
        brun = BenchmarkRun(
            benchmark_run_id="brun-eval-1",
            benchmark_suite_id="bsuite-1",
            suite_key="test-suite",
            suite_version="v1",
            suite_fingerprint="fp1",
            variant=BenchmarkVariant(variant_key="balanced-auto"),
            execution_mode=BenchmarkExecutionMode.OFFLINE,
            status=BenchmarkRunStatus.COMPLETED,
        )
        cres = BenchmarkCaseResult(
            benchmark_case_result_id="bcres-1",
            benchmark_run_id="brun-eval-1",
            benchmark_case_id="bcase-1",
            case_key="test-case-1",
            case_version="v1",
            case_fingerprint="fp-c1",
            trace_id=trace_id_1,
            status=BenchmarkCaseStatus.SUCCEEDED,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            duration_ms=3000.0,
            production_result_refs=BenchmarkProductionResultRefs(
                accepted_shot_asset_version_ids=("asset-1", "asset-2-new"),
                remediation_decision_ids=("remed-1",),
            ),
        )
        self.bench_repo.add_run(brun)
        self.bench_repo.add_case_result(cres)
        self.session.commit()

        metrics, _latencies, _cost, _usage, total_cases, total_shots = self.extractor.extract_metrics_for_run(
            brun, [cres]
        )

        self.assertEqual(total_cases, 1)
        self.assertEqual(total_shots, 2)

        # First asset quality pass rate: 1 of 2 passes initially = 0.5
        first_pass = metrics["first_asset_quality_pass_rate"]
        self.assertEqual(first_pass.status, MetricValueStatus.AVAILABLE)
        self.assertEqual(first_pass.value, 0.5)
        self.assertEqual(first_pass.sample_count, 1)
        self.assertEqual(first_pass.eligible_count, 2)

        # Final shot quality pass rate: 2 of 2 eventually accepted = 1.0
        final_pass = metrics["final_shot_quality_pass_rate"]
        self.assertEqual(final_pass.status, MetricValueStatus.AVAILABLE)
        self.assertEqual(final_pass.value, 1.0)
        self.assertEqual(final_pass.sample_count, 2)
        self.assertEqual(final_pass.eligible_count, 2)

        # Quality remediation rate: 1 of 2 shots required remediation = 0.5
        remed_rate = metrics["quality_remediation_rate"]
        self.assertEqual(remed_rate.value, 0.5)

    def test_dimension_mean_excludes_indeterminate_and_error(self):
        """
        Multimodal dimension evaluation:
        Sample 1: SCORED (0.9)
        Sample 2: INDETERMINATE (None)
        Sample 3: SCORED (0.8)
        Mean MUST be calculated strictly from the two SCORED samples: (0.9 + 0.8) / 2 = 0.85.
        INDETERMINATE must NEVER be coerced to 0.0 (which would have yielded 0.5667).
        Status must be PARTIAL with coverage 2/3 (0.6667).
        """
        trace_id = "tr-dim-test"
        self.trace_repo.add_root(TraceRoot(trace_id=trace_id))

        # Sample 1: SCORED
        self.trace_repo.add_event(TraceEvent(
            trace_id=trace_id,
            event_type=TraceEventType.DIMENSION_EVALUATION_COMPLETED,
            context=TraceContext(trace_id=trace_id, shot_id="shot-1"),
            attributes={"dimension": "SEMANTIC_ALIGNMENT", "status": "SCORED", "score": 0.9},
        ))
        # Sample 2: INDETERMINATE
        self.trace_repo.add_event(TraceEvent(
            trace_id=trace_id,
            event_type=TraceEventType.DIMENSION_EVALUATION_COMPLETED,
            context=TraceContext(trace_id=trace_id, shot_id="shot-2"),
            attributes={"dimension": "SEMANTIC_ALIGNMENT", "status": "INDETERMINATE", "score": None},
        ))
        # Sample 3: SCORED
        self.trace_repo.add_event(TraceEvent(
            trace_id=trace_id,
            event_type=TraceEventType.DIMENSION_EVALUATION_COMPLETED,
            context=TraceContext(trace_id=trace_id, shot_id="shot-3"),
            attributes={"dimension": "SEMANTIC_ALIGNMENT", "status": "SCORED", "score": 0.8},
        ))

        brun = BenchmarkRun(
            benchmark_run_id="brun-dim-1",
            benchmark_suite_id="bsuite-1",
            suite_key="test-suite",
            suite_version="v1",
            suite_fingerprint="fp1",
            variant=BenchmarkVariant(variant_key="balanced-auto"),
        )
        cres = BenchmarkCaseResult(
            benchmark_case_result_id="bcres-dim",
            benchmark_run_id="brun-dim-1",
            benchmark_case_id="bcase-1",
            case_key="test-case-dim",
            case_version="v1",
            case_fingerprint="fp-c1",
            trace_id=trace_id,
            status=BenchmarkCaseStatus.SUCCEEDED,
            started_at=datetime.now(UTC),
        )

        metrics, _, _, _, _, _ = self.extractor.extract_metrics_for_run(brun, [cres])

        dim_val = metrics["dimension_semantic_alignment_score"]
        self.assertEqual(dim_val.status, MetricValueStatus.PARTIAL)
        self.assertEqual(dim_val.value, 0.85)  # Strictly (0.9 + 0.8) / 2
        self.assertEqual(dim_val.sample_count, 2)
        self.assertEqual(dim_val.eligible_count, 3)
        self.assertEqual(dim_val.coverage_ratio, 0.6667)

    def test_cost_summary_and_human_metrics(self):
        """
        Verifies:
        1. OFFLINE run marks cost as synthetic.
        2. OBSERVED cost aggregated into currency map.
        3. UNKNOWN cost not added to totals.
        4. Human edit and beat replan rates are NOT_APPLICABLE.
        """
        trace_id = "tr-cost-test"
        self.trace_repo.add_root(TraceRoot(trace_id=trace_id))

        # Event with observed cost
        self.trace_repo.add_event(TraceEvent(
            trace_id=trace_id,
            event_type=TraceEventType.CONTENT_PLANNER_COMPLETED,
            context=TraceContext(trace_id=trace_id),
            duration_ms=450.0,
            attributes={
                "content_plan_revision_id": "cprev-1",
                "usage": CostUsageTraceData(
                    cost_status=CostEstimateStatus.OBSERVED,
                    cost_amount=0.015,
                    currency="USD",
                ).model_dump(mode="json"),
            },
        ))
        # Event with unknown cost
        self.trace_repo.add_event(TraceEvent(
            trace_id=trace_id,
            event_type=TraceEventType.EXECUTION_ATTEMPT_COMPLETED,
            context=TraceContext(trace_id=trace_id),
            duration_ms=1200.0,
            attributes={
                "execution_run_id": "run-1",
                "usage": CostUsageTraceData(
                    cost_status=CostEstimateStatus.UNKNOWN,
                    cost_amount=None,
                ).model_dump(mode="json"),
            },
        ))

        brun = BenchmarkRun(
            benchmark_run_id="brun-cost-1",
            benchmark_suite_id="bsuite-1",
            suite_key="test-suite",
            suite_version="v1",
            suite_fingerprint="fp1",
            variant=BenchmarkVariant(variant_key="balanced-auto"),
            execution_mode=BenchmarkExecutionMode.OFFLINE,
        )
        cres = BenchmarkCaseResult(
            benchmark_case_result_id="bcres-cost",
            benchmark_run_id="brun-cost-1",
            benchmark_case_id="bcase-1",
            case_key="test-case-cost",
            case_version="v1",
            case_fingerprint="fp-c1",
            trace_id=trace_id,
            status=BenchmarkCaseStatus.SUCCEEDED,
            started_at=datetime.now(UTC),
            duration_ms=1650.0,
        )

        metrics, latencies, cost, _, _, _ = self.extractor.extract_metrics_for_run(brun, [cres])

        # Cost assertions
        self.assertTrue(cost.is_synthetic)
        self.assertEqual(cost.observed_ops, 1)
        self.assertEqual(cost.unknown_ops, 1)
        self.assertEqual(cost.observed_by_currency.get("USD"), 0.015)
        self.assertEqual(cost.completeness, "PARTIAL")

        # Human metrics
        self.assertEqual(metrics["human_edit_rate"].status, MetricValueStatus.NOT_APPLICABLE)
        self.assertIsNone(metrics["human_edit_rate"].value)
        self.assertEqual(metrics["beat_replan_rate"].status, MetricValueStatus.NOT_APPLICABLE)
        self.assertIsNone(metrics["beat_replan_rate"].value)

        # Latencies
        self.assertEqual(latencies["end_to_end"].mean_ms, 1650.0)
        self.assertEqual(latencies["planning"].mean_ms, 450.0)
        self.assertEqual(latencies["generation"].mean_ms, 1200.0)


if __name__ == "__main__":
    unittest.main()
