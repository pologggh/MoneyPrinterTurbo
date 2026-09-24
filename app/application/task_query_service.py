from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.persistence.repositories import (
    KnowledgeVideoTaskRepository,
    StageExecutionRepository,
    WorkflowJobRepository,
)


class TaskQueryService:
    """
    Application query service providing read models and status inspection for KnowledgeVideoTasks.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self.task_repo = KnowledgeVideoTaskRepository(session)
        self.job_repo = WorkflowJobRepository(session)
        self.exec_repo = StageExecutionRepository(session)

    def get_task_detail(self, task_id: str) -> dict[str, Any] | None:
        """
        Retrieves detailed task state including current stage job and execution history.
        """
        task = self.task_repo.get_task(task_id)
        if task is None:
            return None

        current_job = self.job_repo.get_current_job_for_task(task_id)
        executions = self.exec_repo.list_executions_for_task(task_id)

        job_dict = None
        if current_job is not None:
            job_dict = {
                "job_id": current_job.job_id,
                "task_id": current_job.task_id,
                "stage": current_job.stage.value if hasattr(current_job.stage, "value") else str(current_job.stage),
                "status": current_job.status.value if hasattr(current_job.status, "value") else str(current_job.status),
                "attempt_number": current_job.attempt_number,
                "max_attempts": current_job.max_attempts,
                "available_at": current_job.available_at.isoformat() if current_job.available_at else None,
                "lease_owner": current_job.lease_owner,
                "lease_expires_at": current_job.lease_expires_at.isoformat() if current_job.lease_expires_at else None,
                "heartbeat_at": current_job.heartbeat_at.isoformat() if current_job.heartbeat_at else None,
                "input_artifact_revision_id": current_job.input_artifact_revision_id,
                "output_artifact_revision_id": current_job.output_artifact_revision_id,
                "error_type": current_job.error_type,
                "error_message": current_job.error_message,
                "created_at": current_job.created_at.isoformat() if current_job.created_at else None,
                "started_at": current_job.started_at.isoformat() if current_job.started_at else None,
                "finished_at": current_job.finished_at.isoformat() if current_job.finished_at else None,
            }

        exec_list = []
        for exc in executions:
            exec_list.append({
                "stage_execution_id": exc.stage_execution_id,
                "task_id": exc.task_id,
                "job_id": exc.job_id,
                "stage": exc.stage.value if hasattr(exc.stage, "value") else str(exc.stage),
                "attempt_number": exc.attempt_number,
                "status": exc.status,
                "duration_ms": exc.duration_ms,
                "started_at": exc.started_at.isoformat() if exc.started_at else None,
                "finished_at": exc.finished_at.isoformat() if exc.finished_at else None,
                "error_type": exc.error_type,
                "error_message": exc.error_message,
                "input_artifact_revision_id": exc.input_artifact_revision_id,
                "output_artifact_revision_id": exc.output_artifact_revision_id,
            })

        return {
            "task_id": task.task_id,
            "topic": task.topic,
            "task_status": task.task_status.value if hasattr(task.task_status, "value") else str(task.task_status),
            "current_stage": task.current_stage.value if hasattr(task.current_stage, "value") else str(task.current_stage),
            "workflow_policy": task.workflow_policy.value if hasattr(task.workflow_policy, "value") else str(task.workflow_policy),
            "target_duration": task.target_duration,
            "aspect_ratio": task.aspect_ratio,
            "language": task.language,
            "waiting_reason": task.waiting_reason,
            "error_type": task.error_type.value if task.error_type and hasattr(task.error_type, "value") else (str(task.error_type) if task.error_type else None),
            "error_message": task.error_message,
            "task_metadata": task.task_metadata,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "finished_at": task.finished_at.isoformat() if task.finished_at else None,
            "current_job": job_dict,
            "stage_executions": exec_list,
        }

    def list_tasks(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """
        Lists summaries of tasks ordered by creation time descending.
        """
        tasks = self.task_repo.list_tasks(limit=limit, offset=offset)
        result = []
        for t in tasks:
            result.append({
                "task_id": t.task_id,
                "topic": t.topic,
                "task_status": t.task_status.value if hasattr(t.task_status, "value") else str(t.task_status),
                "current_stage": t.current_stage.value if hasattr(t.current_stage, "value") else str(t.current_stage),
                "workflow_policy": t.workflow_policy.value if hasattr(t.workflow_policy, "value") else str(t.workflow_policy),
                "target_duration": t.target_duration,
                "aspect_ratio": t.aspect_ratio,
                "language": t.language,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "updated_at": t.updated_at.isoformat() if t.updated_at else None,
                "finished_at": t.finished_at.isoformat() if t.finished_at else None,
            })
        return result

    def get_task_artifacts(self, task_id: str) -> list[dict[str, Any]]:
        """
        Lists all task artifact references associated with the task.
        """
        from app.persistence.repositories import TaskArtifactRepository
        art_repo = TaskArtifactRepository(self._session)
        refs = art_repo.list_artifact_refs_for_task(task_id)
        result = []
        for r in refs:
            result.append({
                "task_artifact_ref_id": r.task_artifact_ref_id,
                "task_id": r.task_id,
                "stage": r.stage.value if hasattr(r.stage, "value") else str(r.stage),
                "artifact_type": r.artifact_type.value if hasattr(r.artifact_type, "value") else str(r.artifact_type),
                "artifact_id": r.artifact_id,
                "artifact_version": r.artifact_version,
                "is_current": r.is_current,
                "is_stale": r.is_stale,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "metadata": r.metadata_json or {},
            })
        return result

    def get_task_events(self, task_id: str) -> list[dict[str, Any]]:
        """
        Lists all trace events associated with the task in chronological order.
        """
        from sqlalchemy import select
        from app.persistence.models import TraceEventORM, TraceRootORM

        stmt = (
            select(TraceEventORM)
            .where(
                (TraceEventORM.trace_id == task_id)
                | (
                    TraceEventORM.trace_id.in_(
                        select(TraceRootORM.trace_id).where(TraceRootORM.root_reference_id == task_id)
                    )
                )
            )
            .order_by(TraceEventORM.started_at.asc(), TraceEventORM.created_at.asc())
        )
        events = self._session.scalars(stmt).all()
        result = []
        for ev in events:
            result.append({
                "trace_event_id": ev.trace_event_id,
                "trace_id": ev.trace_id,
                "event_type": ev.event_type,
                "status": ev.status,
                "started_at": ev.started_at.isoformat() if ev.started_at else None,
                "completed_at": ev.completed_at.isoformat() if ev.completed_at else None,
                "duration_ms": ev.duration_ms,
                "attributes": ev.attributes_json or {},
                "created_at": ev.created_at.isoformat() if ev.created_at else None,
            })
        return result

    def get_delivery_manifest(self, task_id: str) -> dict[str, Any] | None:
        """
        Retrieves final delivery manifest details for a task.
        """
        from app.persistence.repositories import DeliveryManifestRepository

        deliv_repo = DeliveryManifestRepository(self._session)
        manifest = deliv_repo.get_delivery_manifest_for_task(task_id)
        if manifest is None:
            return None
        return {
            "delivery_manifest_id": manifest.delivery_manifest_id,
            "task_id": manifest.task_id,
            "composition_output_id": manifest.composition_output_id,
            "evaluation_snapshot_id": manifest.evaluation_snapshot_id,
            "final_video_path": manifest.final_video_path,
            "final_video_hash": manifest.final_video_hash,
            "final_video_size_bytes": manifest.final_video_size_bytes,
            "subtitle_path": manifest.subtitle_path,
            "subtitle_hash": manifest.subtitle_hash,
            "source_report_path": manifest.source_report_path,
            "source_report_hash": manifest.source_report_hash,
            "execution_report_path": manifest.execution_report_path,
            "execution_report_hash": manifest.execution_report_hash,
            "delivery_params_snapshot": manifest.delivery_params_snapshot,
            "created_at": manifest.created_at.isoformat() if manifest.created_at else None,
        }
