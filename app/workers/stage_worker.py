from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from loguru import logger
from sqlalchemy.orm import Session, sessionmaker

from app.application.knowledge_video_workflow import KnowledgeVideoWorkflow
from app.application.stage_executor_protocol import (
    StageExecutionResult,
    StageExecutorProtocol,
)
from app.application.stage_executor_registry import (
    StageExecutorRegistry,
    get_default_executor_registry,
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.stage_execution import StageExecution
from app.domain.task_artifact import TaskArtifactRef
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import JobStatus, Stage, TaskStatus
from app.persistence.database_lifecycle import run_database_migrations, wait_for_database
from app.persistence.repositories import (
    KnowledgeVideoTaskRepository,
    StageExecutionRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)
from app.persistence.session import create_db_engine


class _HeartbeatRunner:
    """Background runner for renewing active job lease during stage execution."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        job_id: str,
        worker_id: str,
        interval_seconds: float,
        extend_seconds: int,
    ) -> None:
        self.session_factory = session_factory
        self.job_id = job_id
        self.worker_id = worker_id
        self.interval_seconds = interval_seconds
        self.extend_seconds = extend_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=self.interval_seconds):
            try:
                with self.session_factory() as session:
                    repo = WorkflowJobRepository(session)
                    repo.renew_lease(
                        job_id=self.job_id,
                        owner=self.worker_id,
                        extend_seconds=self.extend_seconds,
                    )
                    session.commit()
                    logger.debug(
                        f"Heartbeat renewed lease on job '{self.job_id}' for worker '{self.worker_id}'."
                    )
            except Exception as exc:
                logger.warning(f"Heartbeat failed on job '{self.job_id}': {exc}")


class StageWorker:
    """
    Durable, database-backed background worker executing workflow stage jobs.

    Adheres to:
    - Polling-based job leasing with supported_stages filtering.
    - Periodic recovery of expired leases.
    - Short database transactions: DB transaction is NEVER held open during stage execution.
    - Active background lease renewal (heartbeat) during long execution runs.
    - Recording audit trails in StageExecution and TaskArtifactRef.
    - Advancing or pausing workflow state via KnowledgeVideoWorkflow.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        registry: StageExecutorRegistry | None = None,
        worker_id: str | None = None,
        poll_interval_seconds: float = 2.0,
        lease_duration_seconds: int = 300,
        heartbeat_interval_seconds: float = 30.0,
        recovery_interval_seconds: float = 60.0,
    ) -> None:
        self.session_factory = session_factory
        self.registry = (
            registry
            if registry is not None
            else get_default_executor_registry(session_factory=session_factory)
        )
        self.worker_id = worker_id or f"stage-worker-{uuid4().hex[:8]}"
        self.poll_interval_seconds = poll_interval_seconds
        self.lease_duration_seconds = lease_duration_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.recovery_interval_seconds = recovery_interval_seconds
        self._stop_event = threading.Event()
        self._last_recovery_time = 0.0

    def stop(self) -> None:
        """Signals worker to stop gracefully."""
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def recover_leases_if_needed(self, now: datetime | None = None, force: bool = False) -> int:
        """Recovers expired job leases back to QUEUED status."""
        current_mono = time.monotonic()
        if not force and (current_mono - self._last_recovery_time < self.recovery_interval_seconds):
            return 0
        self._last_recovery_time = current_mono
        with self.session_factory() as session:
            repo = WorkflowJobRepository(session)
            recovered = repo.recover_expired_leases(now=now)
            if recovered > 0:
                session.commit()
                logger.info(f"Worker '{self.worker_id}' recovered {recovered} expired job leases.")
            return recovered

    def run_once(self, now: datetime | None = None, force_recovery: bool = False) -> bool:
        """
        Executes a single polling iteration:
        1. Recovers expired leases if needed.
        2. In Short TX 1: Acquires next available job supported by registry.
        3. Executes the stage via registered executor (outside DB transaction).
        4. In Short TX 2: Records artifacts, audit execution, and advances workflow.
        Returns True if a job was acquired and processed, False otherwise.
        """
        self.recover_leases_if_needed(now=now, force=force_recovery)

        supported_stages = self.registry.list_supported_stages()
        if not supported_stages:
            return False

        # Short TX 1: Job acquisition & initial task transition
        acquired_job: WorkflowJob | None = None
        task: KnowledgeVideoTask | None = None

        with self.session_factory() as session:
            job_repo = WorkflowJobRepository(session)
            task_repo = KnowledgeVideoTaskRepository(session)

            job = job_repo.acquire_next_available_job(
                owner=self.worker_id,
                supported_stages=supported_stages,
                lease_duration_seconds=self.lease_duration_seconds,
                now=now,
            )
            if job is None:
                return False

            job_task = task_repo.get_task(job.task_id)
            if job_task is None:
                logger.error(f"Job '{job.job_id}' references non-existent task '{job.task_id}'.")
                job.mark_failed("TASK_NOT_FOUND", f"Task '{job.task_id}' not found", is_retryable=False, now=now)
                job_repo.update_job(job)
                session.commit()
                return True

            job.start(now=now)
            job_repo.update_job(job)

            if job_task.task_status == TaskStatus.CREATED:
                job_task.transition_to(TaskStatus.RUNNING, now=now)
                task_repo.save_task(job_task)

            session.commit()
            acquired_job = job
            task = job_task

        # Execution outside DB transaction
        executor = self.registry.get_executor(acquired_job.stage)
        if executor is None:
            logger.warning(
                f"No executor registered for acquired stage '{acquired_job.stage}'."
            )
            with self.session_factory() as session:
                job_repo = WorkflowJobRepository(session)
                acquired_job.mark_failed("NO_EXECUTOR", f"No executor for stage {acquired_job.stage}", is_retryable=True, now=now)
                job_repo.update_job(acquired_job)
                session.commit()
            return True

        started_at = now or datetime.now(UTC)
        heartbeat = _HeartbeatRunner(
            session_factory=self.session_factory,
            job_id=acquired_job.job_id,
            worker_id=self.worker_id,
            interval_seconds=self.heartbeat_interval_seconds,
            extend_seconds=self.lease_duration_seconds,
        )
        heartbeat.start()

        result: StageExecutionResult
        try:
            result = executor.execute(task, acquired_job)
        except Exception as exc:
            logger.exception(
                f"Unhandled exception in executor for stage '{acquired_job.stage}' on task '{task.task_id}': {exc}"
            )
            result = StageExecutionResult(
                success=False,
                error_type="FATAL",
                error_message=f"Unhandled exception: {exc}",
                is_retryable=False,
            )
        finally:
            heartbeat.stop()

        finished_at = now or datetime.now(UTC)

        # Short TX 2: Result persistence, execution audit, and workflow state progression
        with self.session_factory() as session:
            job_repo = WorkflowJobRepository(session)
            task_repo = KnowledgeVideoTaskRepository(session)
            artifact_repo = TaskArtifactRepository(session)
            exec_repo = StageExecutionRepository(session)
            workflow = KnowledgeVideoWorkflow(session)

            current_task = task_repo.get_task(task.task_id)
            current_job = job_repo.get_job(acquired_job.job_id)
            if current_task is None or current_job is None:
                logger.critical(
                    f"Task '{task.task_id}' or job '{acquired_job.job_id}' missing during result persistence."
                )
                return True

            output_ref_id: str | None = None
            if result.output_artifact_ref is not None:
                artifact_repo.save_artifact_ref(result.output_artifact_ref)
                output_ref_id = result.output_artifact_ref.task_artifact_ref_id
            else:
                output_ref_id = result.output_task_artifact_ref_id or result.output_artifact_revision_id

            stage_exec = StageExecution.record(
                task_id=current_task.task_id,
                job_id=current_job.job_id,
                stage=current_job.stage,
                attempt_number=current_job.attempt_number,
                status="SUCCEEDED" if result.success else "FAILED",
                started_at=started_at,
                finished_at=finished_at,
                input_task_artifact_ref_id=current_job.input_task_artifact_ref_id,
                output_task_artifact_ref_id=output_ref_id,
                error_type=result.error_type if not result.success else None,
                error_message=result.error_message if not result.success else None,
            )
            exec_repo.record_execution(stage_exec)

            if result.success:
                current_job.mark_succeeded(output_task_artifact_ref_id=output_ref_id, now=finished_at)
                job_repo.update_job(current_job)
                workflow.on_stage_completed(current_task, current_job, now=finished_at)
            else:
                current_job.mark_failed(
                    error_type=result.error_type or "FATAL",
                    error_message=result.error_message or "",
                    is_retryable=result.is_retryable,
                    now=finished_at,
                )
                job_repo.update_job(current_job)
                workflow.on_stage_failed(
                    current_task,
                    current_job,
                    error_type=result.error_type or "FATAL",
                    error_message=result.error_message or "",
                    now=finished_at,
                )

            session.commit()

        return True

    def run_forever(self) -> None:
        """Continuous polling execution loop with graceful shutdown."""
        logger.info(
            f"StageWorker '{self.worker_id}' started. Supported stages: {self.registry.list_supported_stages()}"
        )
        while not self._stop_event.is_set():
            try:
                processed = self.run_once()
                if not processed:
                    self._stop_event.wait(timeout=self.poll_interval_seconds)
            except Exception as exc:
                logger.exception(f"Unexpected exception in StageWorker loop: {exc}")
                self._stop_event.wait(timeout=self.poll_interval_seconds)
        logger.info(f"StageWorker '{self.worker_id}' stopped cleanly.")


def main() -> None:
    """Standalone entry point for launching a background StageWorker."""
    parser = argparse.ArgumentParser(description="MoneyPrinterTurbo StageWorker")
    parser.add_argument("--worker-id", type=str, default=None, help="Unique worker identifier")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds")
    parser.add_argument("--lease-duration", type=int, default=300, help="Lease duration in seconds")
    parser.add_argument("--heartbeat-interval", type=float, default=30.0, help="Heartbeat interval in seconds")
    parser.add_argument("--recovery-interval", type=float, default=60.0, help="Lease recovery interval in seconds")
    args = parser.parse_args()

    logger.info("Initializing StageWorker standalone process...")
    engine = create_db_engine()
    if not wait_for_database(engine, timeout=10.0):
        logger.error("Database connection check timed out.")
        sys.exit(1)

    run_database_migrations()
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    registry = get_default_executor_registry()
    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id=args.worker_id,
        poll_interval_seconds=args.poll_interval,
        lease_duration_seconds=args.lease_duration,
        heartbeat_interval_seconds=args.heartbeat_interval,
        recovery_interval_seconds=args.recovery_interval,
    )

    def _signal_handler(sig, frame):
        logger.info(f"Received exit signal {sig}, stopping StageWorker gracefully...")
        worker.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    worker.run_forever()


if __name__ == "__main__":
    main()
