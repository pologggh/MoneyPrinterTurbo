from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import sessionmaker

from app.domain.benchmark import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkCaseStatus,
    BenchmarkExecutionMode,
    BenchmarkFailureStage,
    BenchmarkProductionResultRefs,
    BenchmarkRun,
    BenchmarkRunStatus,
    BenchmarkVariant,
    RealBenchmarkExternalCallsNotAuthorizedError,
)
from app.domain.trace import TraceContext, TraceEventType
from app.persistence.repositories import BenchmarkRepository
from app.services.benchmark.dataset_loader import load_benchmark_suite
from app.services.benchmark.offline_adapters import (
    build_offline_benchmark_environment,
    create_deterministic_offline_planner_llm,
)
from app.services.benchmark.production_pipeline_runner import (
    ProductionPipelineResult,
    ProductionPipelineRunner,
)
from app.services.trace_service import TraceService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkRunInput:
    """Input parameters for starting a benchmark run."""
    suite_key: str = "knowledge-video-v1"
    suite_version: str = "v1"
    variant: BenchmarkVariant = field(default_factory=lambda: BenchmarkVariant(variant_key="balanced-auto"))
    execution_mode: BenchmarkExecutionMode = BenchmarkExecutionMode.OFFLINE
    allow_external_calls: bool = False
    selected_case_keys: tuple[str, ...] | None = None
    limit: int | None = None
    storage_dir: Path | None = None
    offline_failing_shot_tokens: set[str] | None = None


class BenchmarkRunner:
    """
    Orchestrates repeatable, isolated benchmark runs over the real production pipeline.
    """

    def __init__(
        self,
        session_factory: sessionmaker,
        trace_service: TraceService,
        pipeline_runner: ProductionPipelineRunner | None = None,
        suites_dir: Path | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.trace_service = trace_service
        self.pipeline_runner = pipeline_runner or ProductionPipelineRunner(session_factory)
        self.suites_dir = suites_dir

    def run_benchmark(self, run_input: BenchmarkRunInput) -> BenchmarkRun:
        """
        Executes a benchmark suite with explicit failure isolation and Trace linkage.
        """
        # 1. Safety Gate: REAL mode authorization check
        if run_input.execution_mode == BenchmarkExecutionMode.REAL and not run_input.allow_external_calls:
            raise RealBenchmarkExternalCallsNotAuthorizedError(
                "REAL benchmark execution requires explicit allow_external_calls=True. "
                "Configured API credentials alone do not grant permission."
            )

        # 2. Load suite and exact frozen case versions
        suite, all_cases = load_benchmark_suite(
            suite_key=run_input.suite_key,
            suite_version=run_input.suite_version,
            suites_dir=self.suites_dir,
        )

        # Filter cases while preserving explicit suite order
        selected_cases: list[BenchmarkCase] = []
        for cref in suite.benchmark_case_refs:
            if run_input.selected_case_keys is not None and cref.case_key not in run_input.selected_case_keys:
                continue
            case = all_cases.get(cref.case_key)
            if case:
                selected_cases.append(case)
            if run_input.limit is not None and len(selected_cases) >= run_input.limit:
                break

        # 3. Create BenchmarkRun record
        run = BenchmarkRun(
            benchmark_run_id=f"brun_{uuid4().hex[:16]}",
            benchmark_suite_id=suite.benchmark_suite_id,
            suite_key=suite.suite_key,
            suite_version=suite.suite_version,
            suite_fingerprint=suite.content_fingerprint,
            variant=run_input.variant,
            execution_mode=run_input.execution_mode,
            status=BenchmarkRunStatus.RUNNING,
            started_at=datetime.now(UTC),
            total_cases=len(selected_cases),
        )

        session = self.session_factory()
        try:
            repo = BenchmarkRepository(session)
            repo.add_run(run)
            session.commit()
        finally:
            session.close()

        # 4. Prepare execution environment
        storage_dir = run_input.storage_dir or Path(".benchmark_storage")
        storage_dir.mkdir(parents=True, exist_ok=True)

        if run_input.execution_mode == BenchmarkExecutionMode.OFFLINE:
            env = build_offline_benchmark_environment(
                storage_dir=storage_dir,
                failing_shot_tokens=run_input.offline_failing_shot_tokens,
            )
            adapter_registry = env["adapter_registry"]
            evaluator_adapter = env["evaluator_adapter"]
            capabilities = env["capabilities"]
            storyboard_agent = env["storyboard_agent"]
        else:
            # In REAL mode, resolve from MoneyPrinterTurbo configuration
            # (only entered when allow_external_calls=True)
            from app.services.asset_adapters.registry import default_adapter_registry
            adapter_registry = default_adapter_registry
            evaluator_adapter = None  # Uses default configured MPT evaluator adapter
            capabilities = ()

        completed_count = 0
        failed_count = 0

        # 5. Execute each case sequentially with failure isolation
        for case in selected_cases:
            t0 = time.perf_counter()
            case_start_time = datetime.now(UTC)
            trace_id = f"tr_bcase_{uuid4().hex[:12]}"

            # Initialize root trace for this benchmark case
            self.trace_service.create_root(
                trace_id=trace_id,
                root_reference_id=case.benchmark_case_id,
            )
            ctx = TraceContext(trace_id=trace_id)

            pipeline_res: ProductionPipelineResult
            try:
                # Build planner LLM for this case's topic & evidence
                planner_llm = create_deterministic_offline_planner_llm(
                    topic=case.topic,
                    target_duration=case.target_duration,
                    evidence_id=case.knowledge_fixture_ref.fixture_id,
                )

                pipeline_res = self.pipeline_runner.run(
                    case=case,
                    variant=run_input.variant,
                    storage_base_dir=storage_dir,
                    trace_context=ctx,
                    trace_writer=self.trace_service.writer,
                    planner_llm_caller=planner_llm,
                    storyboard_agent=storyboard_agent,
                    adapter_registry=adapter_registry,
                    evaluator_adapter=evaluator_adapter,
                    capabilities=capabilities,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Uncaught infrastructure exception on case {case.case_key}: {exc}")
                pipeline_res = ProductionPipelineResult(
                    status=BenchmarkCaseStatus.FAILED,
                    production_result_refs=BenchmarkProductionResultRefs(),
                    failure_stage=BenchmarkFailureStage.INFRASTRUCTURE,
                    failure_code="UNCAUGHT_INFRASTRUCTURE_ERROR",
                    failure_message=str(exc),
                )

            t1 = time.perf_counter()
            duration_ms = round((t1 - t0) * 1000.0, 2)
            case_end_time = datetime.now(UTC)

            # Post-case Trace Completeness Check
            trace_events = self.trace_service.reader.list_events_by_trace(trace_id)
            trace_attributes: dict[str, Any] = {}
            if pipeline_res.status == BenchmarkCaseStatus.SUCCEEDED:
                event_types = {e.event_type for e in trace_events}
                if TraceEventType.EXECUTION_RUN_COMPLETED not in event_types or TraceEventType.EVALUATION_COMPLETED not in event_types:
                    trace_attributes["trace_completeness_warning"] = "BENCHMARK_TRACE_INCOMPLETE"

            # Persist BenchmarkCaseResult in short transaction
            case_res = BenchmarkCaseResult(
                benchmark_case_result_id=f"bcres_{uuid4().hex[:16]}",
                benchmark_run_id=run.benchmark_run_id,
                benchmark_case_id=case.benchmark_case_id,
                case_key=case.case_key,
                case_version=case.case_version,
                case_fingerprint=case.content_fingerprint,
                trace_id=trace_id,
                status=pipeline_res.status,
                started_at=case_start_time,
                finished_at=case_end_time,
                duration_ms=duration_ms,
                failure_stage=pipeline_res.failure_stage,
                failure_code=pipeline_res.failure_code,
                failure_message=pipeline_res.failure_message,
                production_result_refs=pipeline_res.production_result_refs,
                attributes=trace_attributes,
            )

            session = self.session_factory()
            try:
                repo = BenchmarkRepository(session)
                repo.add_case_result(case_res)
                session.commit()
            finally:
                session.close()

            if pipeline_res.status == BenchmarkCaseStatus.SUCCEEDED:
                completed_count += 1
            else:
                failed_count += 1

        # 6. Finalize BenchmarkRun state
        final_status = (
            BenchmarkRunStatus.COMPLETED
            if failed_count == 0
            else BenchmarkRunStatus.COMPLETED_WITH_FAILURES
        )
        final_run = BenchmarkRun(
            benchmark_run_id=run.benchmark_run_id,
            benchmark_suite_id=run.benchmark_suite_id,
            suite_key=run.suite_key,
            suite_version=run.suite_version,
            suite_fingerprint=run.suite_fingerprint,
            variant=run.variant,
            execution_mode=run.execution_mode,
            status=final_status,
            started_at=run.started_at,
            finished_at=datetime.now(UTC),
            total_cases=len(selected_cases),
            completed_cases=completed_count,
            failed_cases=failed_count,
        )

        session = self.session_factory()
        try:
            repo = BenchmarkRepository(session)
            repo.update_run(final_run)
            session.commit()
        finally:
            session.close()

        return final_run
