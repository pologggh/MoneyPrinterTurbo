"""
Composition Stage Executor.

Implements StageExecutorProtocol for Stage.COMPOSITION.
Consumes:
  1. AudioOutput (from Stage.AUDIO)
  2. COMPLETED ExecutionRun & selected ShotAssetVersions (from Stage.ASSET)
  3. APPROVED StoryboardSnapshot
Produces:
  - Final rendered MP4 video
  - Immutable CompositionOutput domain record & TaskArtifactRef(COMPOSITION_OUTPUT)
Advances workflow to Stage.QUALITY_REVIEW (which remains safely parked in QUEUED).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger
from sqlalchemy.orm import Session

from app.application.stage_executor_protocol import (
    StageExecutionResult,
    StageExecutorProtocol,
)
from app.domain.asset_execution import ExecutionStatus
from app.domain.composition import CompositionOutput
from app.domain.enums import StoryboardSnapshotState
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.task_artifact import TaskArtifactRef
from app.domain.trace import TraceEventType
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    ArtifactType,
    JobErrorType,
    Stage,
)
from app.models.schema import VideoAspect, VideoParams
from app.persistence.repositories import (
    AudioOutputRepository,
    CompositionOutputRepository,
    ContentPlanRepository,
    ExecutionRepository,
    StoryboardRepository,
    TaskArtifactRepository,
)
from app.persistence.session import get_session
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


def _compute_sha256(file_path: str) -> str:
    """Computes SHA-256 checksum for a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def default_media_validator(
    video_path: str,
    require_audio: bool = True,
) -> dict[str, Any]:
    """
    Validates technical media properties of a rendered MP4 file.

    Verifies:
    - File exists on disk and is non-empty
    - Readable video container
    - Positive duration
    - Positive width and height
    - Positive fps
    - Audio stream presence when require_audio is True
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Rendered video file not found at '{video_path}'")
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        raise ValueError(f"Rendered video file at '{video_path}' is empty (0 bytes)")

    from app.services.video import _open_video_clip_quietly, close_clip

    clip = _open_video_clip_quietly(video_path, audio=True)
    try:
        duration = float(clip.duration or 0)
        if duration <= 0:
            raise ValueError(
                f"Rendered video file '{video_path}' has invalid duration: {duration}"
            )
        width, height = clip.size
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Rendered video file '{video_path}' has invalid dimensions: ({width}, {height})"
            )
        fps = float(getattr(clip, "fps", 0) or 0)
        has_audio = clip.audio is not None
        if require_audio and not has_audio:
            raise ValueError(
                f"Rendered video file '{video_path}' lacks required audio stream"
            )
        return {
            "duration": duration,
            "width": int(width),
            "height": int(height),
            "file_size": file_size,
            "fps": fps,
            "has_audio": has_audio,
            "video_codec": "h264",
            "audio_codec": "aac" if has_audio else None,
        }
    finally:
        close_clip(clip)


class CompositionStageExecutor(StageExecutorProtocol):
    """
    Executes the COMPOSITION workflow stage for a KnowledgeVideoTask.

    Workflow contract:
    1. Resolves and validates the input TaskArtifactRef (AUDIO_OUTPUT).
    2. Verifies AudioOutput integrity, ExecutionRun status (COMPLETED), and StoryboardSnapshot status (APPROVED).
    3. Preserves deterministic shot ordering (Beat.order, then Shot.local_order).
    4. Reuses existing media composition engine (video.combine_videos & video.generate_video).
    5. Renders video to a temporary path, executes technical media validation, and atomically promotes to final MP4.
    6. Persists immutable CompositionOutput record and returns TaskArtifactRef(COMPOSITION_OUTPUT).
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
        storage_base_dir: Path | str | None = None,
        assembly_service: Any | None = None,
        media_validator: Callable[[str, bool], dict[str, Any]] | None = None,
        trace_writer: Any | None = None,
    ) -> None:
        self._session_factory = session_factory
        self.storage_base_dir = storage_base_dir
        self._assembly_service = assembly_service
        self._media_validator = media_validator
        self.trace_writer = trace_writer

    @property
    def assembly_service(self) -> Any:
        if self._assembly_service is not None:
            return self._assembly_service
        from app.services.storyboard_video_assembly_service import (
            StoryboardVideoAssemblyService,
        )
        return StoryboardVideoAssemblyService(session_factory=self._session_factory)

    @assembly_service.setter
    def assembly_service(self, val: Any | None) -> None:
        self._assembly_service = val

    @property
    def media_validator(self) -> Callable[[str, bool], dict[str, Any]]:
        if self._media_validator is not None:
            return self._media_validator
        return default_media_validator

    @media_validator.setter
    def media_validator(
        self, val: Callable[[str, bool], dict[str, Any]] | None
    ) -> None:
        self._media_validator = val

    def _emit_trace(
        self,
        task_id: str,
        event_type: TraceEventType,
        attributes: dict[str, Any],
    ) -> None:
        if self.trace_writer is None:
            return
        try:
            self.trace_writer.write_event(
                task_id=task_id,
                event_type=event_type,
                attributes=attributes,
            )
        except Exception as exc:
            logger.warning(
                f"[CompositionStageExecutor] Failed to emit trace event {event_type}: {exc}"
            )

    def execute(
        self,
        task: KnowledgeVideoTask,
        job: WorkflowJob,
    ) -> StageExecutionResult:
        task_id = task.task_id
        logger.info(
            f"[CompositionStageExecutor] Starting execution for task '{task_id}', "
            f"job '{job.job_id}', attempt {job.attempt_number}."
        )

        self._emit_trace(
            task_id=task_id,
            event_type=TraceEventType.COMPOSITION_STAGE_STARTED,
            attributes={"job_id": job.job_id, "attempt": job.attempt_number},
        )

        # ---------------------------------------------------------------------
        # 1. Resolve and Validate Input Artifact (AudioOutput)
        # ---------------------------------------------------------------------
        if job.input_task_artifact_ref_id is None:
            err = f"Input artifact ref missing for task '{task_id}', job '{job.job_id}'."
            logger.error(f"[CompositionStageExecutor] {err}")
            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                attributes={"error": err},
            )
            return StageExecutionResult(
                success=False,
                error_type=JobErrorType.FATAL.value,
                error_message=err,
                is_retryable=False,
            )

        with _managed_session(self._session_factory) as session:
            art_repo = TaskArtifactRepository(session)
            audio_repo = AudioOutputRepository(session)
            exec_repo = ExecutionRepository(session)
            sb_repo = StoryboardRepository(session)
            plan_repo = ContentPlanRepository(session)

            input_ref = art_repo.get_artifact_ref(job.input_task_artifact_ref_id)
            if input_ref is None or input_ref.artifact_type != ArtifactType.AUDIO_OUTPUT:
                err = (
                    f"Input artifact ref '{job.input_task_artifact_ref_id}' for task '{task_id}' "
                    f"is not of type AUDIO_OUTPUT."
                )
                logger.error(f"[CompositionStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if input_ref.task_id != task_id:
                err = (
                    f"Task ID mismatch: input ref belongs to '{input_ref.task_id}', "
                    f"current task is '{task_id}'."
                )
                logger.error(f"[CompositionStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            audio_output = audio_repo.get_audio_output(input_ref.artifact_id)
            if audio_output is None:
                err = f"AudioOutput artifact '{input_ref.artifact_id}' not found in database."
                logger.error(f"[CompositionStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if audio_output.task_id != task_id:
                err = (
                    f"AudioOutput task_id mismatch: audio belongs to '{audio_output.task_id}', "
                    f"current task is '{task_id}'."
                )
                logger.error(f"[CompositionStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            # Incomplete AUDIO output check
            if not os.path.exists(audio_output.narration_audio_path) or os.path.getsize(audio_output.narration_audio_path) == 0:
                err = f"Narration audio file is missing or empty at '{audio_output.narration_audio_path}'."
                logger.error(f"[CompositionStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if audio_output.actual_narration_duration <= 0:
                err = f"AudioOutput actual_narration_duration is invalid ({audio_output.actual_narration_duration}s)."
                logger.error(f"[CompositionStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            # -----------------------------------------------------------------
            # 2. Resolve & Validate Upstream ExecutionRun & StoryboardSnapshot
            # -----------------------------------------------------------------
            run = exec_repo.get_execution_run(audio_output.execution_run_id)
            if run is None:
                err = f"ExecutionRun '{audio_output.execution_run_id}' not found in database."
                logger.error(f"[CompositionStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if run.status != ExecutionStatus.COMPLETED:
                err = (
                    f"ExecutionRun '{run.execution_run_id}' is not in COMPLETED state "
                    f"(current: {run.status.value}). Incomplete assets cannot be composed."
                )
                logger.error(f"[CompositionStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            snapshot = sb_repo.get_snapshot(run.storyboard_snapshot_id)
            if snapshot is None:
                err = f"StoryboardSnapshot '{run.storyboard_snapshot_id}' not found in database."
                logger.error(f"[CompositionStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if snapshot.snapshot_state != StoryboardSnapshotState.APPROVED:
                err = (
                    f"StoryboardSnapshot '{snapshot.storyboard_snapshot_id}' is not APPROVED "
                    f"(current state: '{snapshot.snapshot_state.value}'). Cannot compose unapproved storyboard."
                )
                logger.error(f"[CompositionStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            # Incomplete ASSET output check
            frozen_revisions = sb_repo.get_snapshot_shot_revisions(snapshot.storyboard_snapshot_id)
            if not frozen_revisions:
                err = f"StoryboardSnapshot '{snapshot.storyboard_snapshot_id}' contains no shot revisions."
                logger.error(f"[CompositionStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            for rev in frozen_revisions:
                asset_v = exec_repo.get_asset_version_for_run_and_shot(
                    execution_run_id=run.execution_run_id,
                    shot_revision_id=rev.shot_revision_id,
                )
                if not asset_v or not asset_v.file_path or not os.path.exists(asset_v.file_path):
                    err = (
                        f"Shot '{rev.shot_id}' has missing or unresolved asset in execution run "
                        f"'{run.execution_run_id}'."
                    )
                    logger.error(f"[CompositionStageExecutor] {err}")
                    self._emit_trace(
                        task_id=task_id,
                        event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                        attributes={"error": err},
                    )
                    return StageExecutionResult(
                        success=False,
                        error_type=JobErrorType.FATAL.value,
                        error_message=err,
                        is_retryable=False,
                    )

            plan = plan_repo.get_revision(snapshot.content_plan_revision_id)
            plan_topic = plan.topic if plan else "知识视频"

        # ---------------------------------------------------------------------
        # 3. Prepare Paths and Configuration
        # ---------------------------------------------------------------------
        meta = task.task_metadata or {}
        aspect_val = meta.get("video_aspect") or "16:9"
        # Normalize aspect ratio string if needed
        try:
            aspect_enum = VideoAspect(aspect_val)
        except ValueError:
            aspect_enum = VideoAspect.landscape if "16:9" in aspect_val or "landscape" in aspect_val else VideoAspect.portrait

        video_params = VideoParams(
            video_subject=plan_topic,
            video_aspect=aspect_enum.value,
            bgm_type=meta.get("bgm_type") or "none",
            bgm_volume=float(meta.get("bgm_volume", 0.2)) if meta.get("bgm_volume") is not None else 0.2,
            subtitle_enabled=bool(audio_output.subtitle_path and os.path.exists(audio_output.subtitle_path)),
            font_name=meta.get("font_name", "STHeitiMedium.ttc"),
            text_fore_color=meta.get("text_fore_color", "#FFFFFF"),
            stroke_color=meta.get("stroke_color", "#000000"),
            stroke_width=int(meta.get("stroke_width", 2)),
            font_size=int(meta.get("font_size", 40)),
            match_materials_to_script=True,
        )

        composition_output_id = f"co_{uuid4().hex[:24]}"
        if self.storage_base_dir is not None:
            task_dir = Path(self.storage_base_dir) / task_id
        else:
            task_dir = Path(utils.task_dir(task_id))
        task_dir.mkdir(parents=True, exist_ok=True)

        temp_video_path = str(task_dir / f"temp_{composition_output_id}.mp4")
        final_video_path = str(task_dir / f"final_{composition_output_id}.mp4")

        # ---------------------------------------------------------------------
        # 4. Render Video via Media Composition Engine
        # ---------------------------------------------------------------------
        try:
            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.SHOT_TIMELINE_BUILT,
                attributes={"shots_count": len(frozen_revisions)},
            )
            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.MEDIA_RENDER_STARTED,
                attributes={"temp_output_path": temp_video_path},
            )

            logger.info(
                f"[CompositionStageExecutor] Assembling video for task '{task_id}', "
                f"snapshot='{snapshot.storyboard_snapshot_id}', run='{run.execution_run_id}', "
                f"output='{temp_video_path}'"
            )

            assembly_res = self.assembly_service.assemble_video(
                storyboard_snapshot_id=snapshot.storyboard_snapshot_id,
                execution_run_id=run.execution_run_id,
                video_params=video_params,
                task_id=task_id,
                audio_path=audio_output.narration_audio_path,
                subtitle_path=audio_output.subtitle_path,
                audio_duration=audio_output.actual_narration_duration,
                bgm_path=audio_output.bgm_path,
                bgm_volume=audio_output.bgm_volume,
                output_file_path=temp_video_path,
            )

            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.MEDIA_RENDER_COMPLETED,
                attributes={"temp_output_path": temp_video_path},
            )

            # -----------------------------------------------------------------
            # 5. Technical Output Validation
            # -----------------------------------------------------------------
            validation_info = self.media_validator(temp_video_path, True)
            logger.info(
                f"[CompositionStageExecutor] Video technical validation passed: {validation_info}"
            )

            # Atomically promote temp work file to final output artifact
            os.replace(temp_video_path, final_video_path)
            video_hash = _compute_sha256(final_video_path)

        except Exception as exc:
            # Clean up temp file on failure
            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except Exception:
                    pass
            if os.path.exists(final_video_path):
                try:
                    os.remove(final_video_path)
                except Exception:
                    pass

            err_msg = str(exc)[:500]
            logger.error(
                f"[CompositionStageExecutor] Composition render/validation failed for task '{task_id}': {err_msg}"
            )
            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.COMPOSITION_VALIDATION_FAILED,
                attributes={"error": err_msg},
            )
            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.COMPOSITION_STAGE_FAILED,
                attributes={"error": err_msg},
            )
            return StageExecutionResult(
                success=False,
                error_type=JobErrorType.RETRYABLE.value,
                error_message=f"Composition rendering failed: {err_msg}",
                is_retryable=True,
            )

        # ---------------------------------------------------------------------
        # 6. Persist Immutable CompositionOutput & Construct ArtifactRef
        # ---------------------------------------------------------------------
        composition_params_snapshot = {
            "video_aspect": video_params.video_aspect,
            "bgm_type": video_params.bgm_type,
            "bgm_volume": video_params.bgm_volume,
            "subtitle_enabled": video_params.subtitle_enabled,
            "font_name": video_params.font_name,
        }

        comp_output = CompositionOutput.create(
            composition_output_id=composition_output_id,
            task_id=task_id,
            storyboard_snapshot_id=snapshot.storyboard_snapshot_id,
            execution_run_id=run.execution_run_id,
            audio_output_id=audio_output.audio_output_id,
            video_path=final_video_path,
            video_hash=video_hash,
            duration=validation_info["duration"],
            width=validation_info["width"],
            height=validation_info["height"],
            file_size=validation_info["file_size"],
            video_codec=validation_info.get("video_codec"),
            audio_codec=validation_info.get("audio_codec"),
            fps=validation_info.get("fps"),
            composition_params_snapshot=composition_params_snapshot,
        )

        with _managed_session(self._session_factory) as session:
            comp_repo = CompositionOutputRepository(session)
            comp_repo.save_composition_output(comp_output)

        self._emit_trace(
            task_id=task_id,
            event_type=TraceEventType.COMPOSITION_STAGE_COMPLETED,
            attributes={
                "composition_output_id": comp_output.composition_output_id,
                "video_path": final_video_path,
                "duration": comp_output.duration,
                "file_size": comp_output.file_size,
                "video_hash": video_hash,
            },
        )

        output_artifact_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.COMPOSITION,
            artifact_type=ArtifactType.COMPOSITION_OUTPUT,
            artifact_id=comp_output.composition_output_id,
        )

        logger.success(
            f"[CompositionStageExecutor] Completed composition stage for task '{task_id}'. "
            f"Artifact: {output_artifact_ref.task_artifact_ref_id}, output_id: {comp_output.composition_output_id}"
        )

        return StageExecutionResult(
            success=True,
            output_artifact_ref=output_artifact_ref,
        )
