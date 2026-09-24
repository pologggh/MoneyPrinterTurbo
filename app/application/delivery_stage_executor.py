"""
Delivery Stage Executor.

Implements StageExecutorProtocol for Stage.DELIVERY.
Final stage of the Knowledge Video Agent workflow:
1. Validates accepted QUALITY_REVIEW artifact (EvaluationSnapshot.decision == PASS).
2. Authoritatively reuses accepted CompositionOutput without rerendering or invoking FFmpeg.
3. Reuses AudioOutput subtitles (if available).
4. Generates authoritative source_report.md and execution_report.md via DeliveryReportService.
5. Produces immutable DeliveryManifest domain record & TaskArtifactRef(DELIVERY_MANIFEST).
6. Emits trace events:
   - DELIVERY_STAGE_STARTED
   - DELIVERY_MANIFEST_CREATED
   - DELIVERY_STAGE_COMPLETED
   - DELIVERY_STAGE_FAILED (on error)
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger
from sqlalchemy.orm import Session

from app.application.stage_executor_protocol import (
    StageExecutionResult,
    StageExecutorProtocol,
)
from app.domain.delivery import DeliveryManifest
from app.domain.evaluation import EvaluationDecision
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.task_artifact import TaskArtifactRef
from app.domain.trace import TraceEventType
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    ArtifactType,
    JobErrorType,
    Stage,
)
from app.persistence.repositories import (
    AudioOutputRepository,
    CompositionOutputRepository,
    DeliveryManifestRepository,
    EvaluationRepository,
    TaskArtifactRepository,
)
from app.persistence.session import get_session
from app.services.delivery_report_service import DeliveryReportService
from app.utils import utils


@contextmanager
def _managed_session(
    session_factory: Callable[[], Session] | None,
) -> Generator[Session, None, None]:
    if session_factory is None:
        with get_session() as session:
            yield session
    else:
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def compute_file_sha256(file_path: str) -> str:
    """Computes SHA-256 checksum for a local file in 64KB blocks."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class DeliveryStageExecutor(StageExecutorProtocol):
    """
    Production executor for Stage.DELIVERY.
    Produces immutable DeliveryManifest and signals workflow completion.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
        storage_base_dir: str | Path | None = None,
        report_service: DeliveryReportService | None = None,
        trace_writer: Any | None = None,
        trace_emitter: Any | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._storage_base_dir = Path(storage_base_dir) if storage_base_dir else None
        self._report_service = report_service
        self.trace_writer = trace_writer or trace_emitter
        self._trace_emitter = self.trace_writer

    @property
    def stage(self) -> Stage:
        return Stage.DELIVERY

    def _emit_trace(
        self,
        task_id: str,
        event_type: TraceEventType,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        emitter = self.trace_writer or self._trace_emitter
        if emitter is None:
            return
        try:
            if hasattr(emitter, "emit_trace_event"):
                emitter.emit_trace_event(
                    task_id=task_id,
                    event_type=event_type,
                    attributes=attributes or {},
                )
            elif hasattr(emitter, "write_event"):
                emitter.write_event(
                    task_id=task_id,
                    event_type=event_type,
                    attributes=attributes or {},
                )
        except Exception as exc:
            logger.warning(
                f"[DeliveryStageExecutor] Failed to emit trace event {event_type}: {exc}"
            )

    def execute(
        self,
        task: KnowledgeVideoTask,
        job: WorkflowJob,
    ) -> StageExecutionResult:
        task_id = task.task_id
        logger.info(
            f"[DeliveryStageExecutor] Starting delivery for task '{task_id}', "
            f"job '{job.job_id}', attempt {job.attempt_number}."
        )

        self._emit_trace(
            task_id=task_id,
            event_type=TraceEventType.DELIVERY_STAGE_STARTED,
            attributes={
                "task_id": task_id,
                "job_id": job.job_id,
                "attempt_number": job.attempt_number,
            },
        )

        with _managed_session(self._session_factory) as session:
            art_repo = TaskArtifactRepository(session)
            eval_repo = EvaluationRepository(session)
            comp_repo = CompositionOutputRepository(session)
            audio_repo = AudioOutputRepository(session)
            deliv_repo = DeliveryManifestRepository(session)

            # -----------------------------------------------------------------
            # 1. Resolve & Validate Quality Review Input (EvaluationSnapshot)
            # -----------------------------------------------------------------
            eval_snap_id = None
            if job.input_task_artifact_ref_id:
                ref = art_repo.get_artifact_ref(job.input_task_artifact_ref_id)
                if ref is not None:
                    if ref.task_id != task_id:
                        err = f"Task ID mismatch: job task '{task_id}' != input artifact task '{ref.task_id}'."
                        logger.error(f"[DeliveryStageExecutor] {err}")
                        self._emit_trace(task_id, TraceEventType.DELIVERY_STAGE_FAILED, {"error": err})
                        return StageExecutionResult(
                            success=False,
                            error_type=JobErrorType.FATAL.value,
                            error_message=err,
                            is_retryable=False,
                        )
                    if ref.is_stale:
                        err = f"Input evaluation snapshot artifact ref '{ref.task_artifact_ref_id}' is marked stale."
                        logger.error(f"[DeliveryStageExecutor] {err}")
                        self._emit_trace(task_id, TraceEventType.DELIVERY_STAGE_FAILED, {"error": err})
                        return StageExecutionResult(
                            success=False,
                            error_type=JobErrorType.FATAL.value,
                            error_message=err,
                            is_retryable=False,
                        )
                    eval_snap_id = ref.artifact_id

            if eval_snap_id is None:
                latest_eval_ref = art_repo.get_current_artifact_ref(
                    task_id=task_id,
                    stage=Stage.QUALITY_REVIEW,
                    artifact_type=ArtifactType.EVALUATION_SNAPSHOT,
                )
                if latest_eval_ref is not None and not latest_eval_ref.is_stale:
                    eval_snap_id = latest_eval_ref.artifact_id

            if eval_snap_id is None:
                err = f"EvaluationSnapshot missing for task '{task_id}', job '{job.job_id}'."
                logger.error(f"[DeliveryStageExecutor] {err}")
                self._emit_trace(task_id, TraceEventType.DELIVERY_STAGE_FAILED, {"error": err})
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            eval_snap = eval_repo.get_snapshot(eval_snap_id)
            if eval_snap is None:
                err = f"EvaluationSnapshot '{eval_snap_id}' record not found in repository."
                logger.error(f"[DeliveryStageExecutor] {err}")
                self._emit_trace(task_id, TraceEventType.DELIVERY_STAGE_FAILED, {"error": err})
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            decision_str = eval_snap.decision.value if hasattr(eval_snap.decision, "value") else str(eval_snap.decision)
            if decision_str != EvaluationDecision.PASS.value:
                err = f"EvaluationSnapshot '{eval_snap_id}' decision is '{decision_str}', expected 'PASS'. Delivery cannot proceed."
                logger.error(f"[DeliveryStageExecutor] {err}")
                self._emit_trace(task_id, TraceEventType.DELIVERY_STAGE_FAILED, {"error": err, "decision": decision_str})
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            # -----------------------------------------------------------------
            # 2. Resolve & Validate Accepted CompositionOutput (No Rerender/FFmpeg)
            # -----------------------------------------------------------------
            comp_output_id = None
            latest_comp_ref = art_repo.get_current_artifact_ref(
                task_id=task_id,
                stage=Stage.COMPOSITION,
                artifact_type=ArtifactType.COMPOSITION_OUTPUT,
            )
            if latest_comp_ref is not None and not latest_comp_ref.is_stale:
                comp_output_id = latest_comp_ref.artifact_id

            if comp_output_id is None:
                latest_comp = comp_repo.get_latest_composition_output_for_task(task_id)
                if latest_comp is not None:
                    comp_output_id = latest_comp.composition_output_id

            if comp_output_id is None:
                err = f"CompositionOutput missing for task '{task_id}', job '{job.job_id}'."
                logger.error(f"[DeliveryStageExecutor] {err}")
                self._emit_trace(task_id, TraceEventType.DELIVERY_STAGE_FAILED, {"error": err})
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            comp_output = comp_repo.get_composition_output(comp_output_id)
            if comp_output is None:
                err = f"CompositionOutput '{comp_output_id}' record not found."
                logger.error(f"[DeliveryStageExecutor] {err}")
                self._emit_trace(task_id, TraceEventType.DELIVERY_STAGE_FAILED, {"error": err})
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if not os.path.exists(comp_output.video_path) or os.path.getsize(comp_output.video_path) == 0:
                err = f"Final video file missing or empty at '{comp_output.video_path}'."
                logger.error(f"[DeliveryStageExecutor] {err}")
                self._emit_trace(task_id, TraceEventType.DELIVERY_STAGE_FAILED, {"error": err})
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            final_video_size_bytes = os.path.getsize(comp_output.video_path)
            final_video_hash = compute_file_sha256(comp_output.video_path)

            # -----------------------------------------------------------------
            # 3. Resolve Subtitles & Audio Output
            # -----------------------------------------------------------------
            subtitle_path = None
            subtitle_hash = None
            if comp_output.audio_output_id:
                audio_output = audio_repo.get_audio_output(comp_output.audio_output_id)
                if (
                    audio_output is not None
                    and audio_output.subtitle_path
                    and os.path.exists(audio_output.subtitle_path)
                ):
                    subtitle_path = audio_output.subtitle_path
                    subtitle_hash = compute_file_sha256(subtitle_path)

            # -----------------------------------------------------------------
            # 4. Generate Authoritative Delivery Reports (Frozen MD + SHA-256)
            # -----------------------------------------------------------------
            delivery_manifest_id = f"dm_{uuid4().hex[:24]}"
            if self._storage_base_dir is not None:
                task_dir = self._storage_base_dir / task_id
            else:
                task_dir = Path(utils.task_dir(task_id))
            task_dir.mkdir(parents=True, exist_ok=True)

            source_report_target = str(task_dir / f"source_report_{delivery_manifest_id}.md")
            exec_report_target = str(task_dir / f"execution_report_{delivery_manifest_id}.md")

            report_svc = self._report_service or DeliveryReportService(session)
            source_report_path, source_report_hash = report_svc.generate_source_report(
                task_id=task_id,
                output_file_path=source_report_target,
            )
            exec_report_path, exec_report_hash = report_svc.generate_execution_report(
                task_id=task_id,
                output_file_path=exec_report_target,
                evaluation_snapshot_id=eval_snap.evaluation_snapshot_id,
                composition_output_id=comp_output.composition_output_id,
            )

            # -----------------------------------------------------------------
            # 5. Construct & Persist Immutable DeliveryManifest
            # -----------------------------------------------------------------
            delivery_params = {
                "duration": comp_output.duration,
                "width": comp_output.width,
                "height": comp_output.height,
                "fps": comp_output.fps,
                "video_codec": comp_output.video_codec,
                "audio_codec": comp_output.audio_codec,
            }

            manifest = DeliveryManifest.create(
                delivery_manifest_id=delivery_manifest_id,
                task_id=task_id,
                composition_output_id=comp_output.composition_output_id,
                evaluation_snapshot_id=eval_snap.evaluation_snapshot_id,
                final_video_path=comp_output.video_path,
                final_video_hash=final_video_hash,
                final_video_size_bytes=final_video_size_bytes,
                subtitle_path=subtitle_path,
                subtitle_hash=subtitle_hash,
                source_report_path=source_report_path,
                source_report_hash=source_report_hash,
                execution_report_path=exec_report_path,
                execution_report_hash=exec_report_hash,
                delivery_params_snapshot=delivery_params,
            )

            deliv_repo.save_delivery_manifest(manifest)

            # -----------------------------------------------------------------
            # 6. Create TaskArtifactRef & Emit Completion Traces
            # -----------------------------------------------------------------
            output_artifact_ref = TaskArtifactRef.create(
                task_id=task_id,
                stage=Stage.DELIVERY,
                artifact_type=ArtifactType.DELIVERY_MANIFEST,
                artifact_id=manifest.delivery_manifest_id,
                metadata_json={
                    "final_video_path": manifest.final_video_path,
                    "final_video_hash": manifest.final_video_hash,
                    "final_video_size_bytes": manifest.final_video_size_bytes,
                    "source_report_path": manifest.source_report_path,
                    "source_report_hash": manifest.source_report_hash,
                    "execution_report_path": manifest.execution_report_path,
                    "execution_report_hash": manifest.execution_report_hash,
                },
            )
            saved_ref = art_repo.save_artifact_ref(output_artifact_ref)

            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.DELIVERY_MANIFEST_CREATED,
                attributes={
                    "task_id": task_id,
                    "delivery_manifest_id": manifest.delivery_manifest_id,
                    "final_video_path": manifest.final_video_path,
                    "final_video_hash": manifest.final_video_hash,
                    "final_video_size_bytes": manifest.final_video_size_bytes,
                },
            )

            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.DELIVERY_STAGE_COMPLETED,
                attributes={
                    "task_id": task_id,
                    "delivery_manifest_id": manifest.delivery_manifest_id,
                },
            )

            logger.info(
                f"[DeliveryStageExecutor] Successfully generated DeliveryManifest '{manifest.delivery_manifest_id}' "
                f"for task '{task_id}'. Video: '{manifest.final_video_path}'."
            )

            return StageExecutionResult(
                success=True,
                output_artifact_ref=saved_ref,
                output_task_artifact_ref_id=saved_ref.task_artifact_ref_id,
                metadata_json={
                    "delivery_manifest_id": manifest.delivery_manifest_id,
                    "final_video_path": manifest.final_video_path,
                    "final_video_size_bytes": manifest.final_video_size_bytes,
                },
            )
