"""
Audio Stage Executor.

Implements StageExecutorProtocol for Stage.AUDIO.
Consumes:
  1. COMPLETED ExecutionRun (from Stage.ASSET)
  2. APPROVED StoryboardSnapshot
  3. ScriptRevision (authoritative source of spoken text)
Produces:
  - Immutable AudioOutput domain record & TaskArtifactRef(AUDIO_OUTPUT)
  - Narration audio file & SHA-256 hash
  - Synchronized subtitle (.srt) & SHA-256 hash
  - Optional prepared BGM reference
Strictly obeys the bounded context: A1 does NOT perform final video composition.
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
from app.config import config
from app.domain.asset_execution import ExecutionStatus
from app.domain.audio import AudioOutput
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
from app.persistence.repositories import (
    AudioOutputRepository,
    ExecutionRepository,
    ScriptRepository,
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


class AudioStageExecutor(StageExecutorProtocol):
    """
    Executes the AUDIO workflow stage for a KnowledgeVideoTask.

    Workflow contract:
    1. Resolves and validates the input TaskArtifactRef (EXECUTION_RUN).
    2. Verifies ExecutionRun is COMPLETED and derives StoryboardSnapshot & ScriptRevision.
    3. Assembles authoritative narration text from ordered ScriptSegments (no LLM rewriting).
    4. Invokes TTS synthesis and generates synchronized subtitles.
    5. Measures actual narration duration and enforces duration tolerance.
    6. Resolves optional background music reference.
    7. Persists immutable AudioOutput and produces TaskArtifactRef(AUDIO_OUTPUT).
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
        storage_base_dir: Path | str | None = None,
        tts_runner: Callable[..., Any] | None = None,
        duration_getter: Callable[[Any], float] | None = None,
        subtitle_creator: Callable[..., Any] | None = None,
        trace_writer: Any | None = None,
    ) -> None:
        self._session_factory = session_factory
        self.storage_base_dir = storage_base_dir
        self._tts_runner = tts_runner
        self._duration_getter = duration_getter
        self._subtitle_creator = subtitle_creator
        self.trace_writer = trace_writer

    @property
    def tts_runner(self) -> Callable[..., Any]:
        if self._tts_runner is not None:
            return self._tts_runner
        from app.services import voice
        return voice.tts

    @tts_runner.setter
    def tts_runner(self, val: Callable[..., Any] | None) -> None:
        self._tts_runner = val

    @property
    def duration_getter(self) -> Callable[[Any], float]:
        if self._duration_getter is not None:
            return self._duration_getter
        from app.services import voice
        return voice.get_audio_duration

    @duration_getter.setter
    def duration_getter(self, val: Callable[[Any], float] | None) -> None:
        self._duration_getter = val

    @property
    def subtitle_creator(self) -> Callable[..., Any]:
        if self._subtitle_creator is not None:
            return self._subtitle_creator
        from app.services import voice
        return voice.create_subtitle

    @subtitle_creator.setter
    def subtitle_creator(self, val: Callable[..., Any] | None) -> None:
        self._subtitle_creator = val

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
                f"[AudioStageExecutor] Failed to emit trace event {event_type}: {exc}"
            )

    def execute(
        self,
        task: KnowledgeVideoTask,
        job: WorkflowJob,
    ) -> StageExecutionResult:
        task_id = task.task_id
        logger.info(
            f"[AudioStageExecutor] Starting execution for task '{task_id}', "
            f"job '{job.job_id}', attempt {job.attempt_number}."
        )

        self._emit_trace(
            task_id=task_id,
            event_type=TraceEventType.AUDIO_STAGE_STARTED,
            attributes={
                "task_id": task_id,
                "job_id": job.job_id,
                "attempt_number": job.attempt_number,
            },
        )

        # ---------------------------------------------------------------------
        # 1. Input Resolution & Lineage Verification
        # ---------------------------------------------------------------------
        input_ref_id = job.input_task_artifact_ref_id

        with _managed_session(self._session_factory) as session:
            art_repo = TaskArtifactRepository(session)
            exec_repo = ExecutionRepository(session)
            sb_repo = StoryboardRepository(session)
            script_repo = ScriptRepository(session)

            input_ref: TaskArtifactRef | None = None
            if input_ref_id:
                input_ref = art_repo.get_artifact_ref(input_ref_id)
            else:
                input_ref = art_repo.get_latest_artifact_ref(
                    task_id=task_id,
                    stage=Stage.ASSET,
                    artifact_type=ArtifactType.EXECUTION_RUN,
                )

            if input_ref is None:
                err = f"Input artifact ref missing for task '{task_id}', job '{job.job_id}'."
                logger.error(f"[AudioStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.AUDIO_STAGE_FAILED,
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
                    f"Task ID mismatch: Job task '{task_id}' != "
                    f"input artifact task '{input_ref.task_id}'."
                )
                logger.error(f"[AudioStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.AUDIO_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if input_ref.artifact_type != ArtifactType.EXECUTION_RUN:
                err = (
                    f"Input artifact type '{input_ref.artifact_type}' is invalid, "
                    f"expected '{ArtifactType.EXECUTION_RUN}'."
                )
                logger.error(f"[AudioStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.AUDIO_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            run = exec_repo.get_execution_run(input_ref.artifact_id)
            if run is None:
                err = f"ExecutionRun '{input_ref.artifact_id}' not found in repository."
                logger.error(f"[AudioStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.AUDIO_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if run.status != ExecutionStatus.COMPLETED:
                err = f"ExecutionRun '{run.execution_run_id}' is not in COMPLETED state (status='{run.status.value}')."
                logger.error(f"[AudioStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.AUDIO_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            # Validate StoryboardSnapshot
            sb_snapshot = sb_repo.get_snapshot(run.storyboard_snapshot_id)
            if sb_snapshot is None:
                err = f"StoryboardSnapshot '{run.storyboard_snapshot_id}' referenced by ExecutionRun not found."
                logger.error(f"[AudioStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.AUDIO_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if sb_snapshot.snapshot_state != StoryboardSnapshotState.APPROVED:
                err = f"StoryboardSnapshot '{sb_snapshot.storyboard_snapshot_id}' is not in APPROVED state (state='{sb_snapshot.snapshot_state}')."
                logger.error(f"[AudioStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.AUDIO_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            # Validate ScriptRevision
            script_rev = script_repo.get_latest_revision_for_task(task_id)
            if script_rev is None or not script_rev.segments:
                err = f"Authoritative ScriptRevision with segments missing for task '{task_id}'."
                logger.error(f"[AudioStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.AUDIO_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            # Extract authoritative narration text in strict sequence order
            ordered_segments = sorted(script_rev.segments, key=lambda s: s.order)
            narration_texts = [
                seg.narration_text.strip()
                for seg in ordered_segments
                if seg.narration_text and seg.narration_text.strip()
            ]
            full_narration_text = "\n".join(narration_texts)
            if not full_narration_text:
                err = f"Narration text is empty across all segments of ScriptRevision '{script_rev.script_revision_id}'."
                logger.error(f"[AudioStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.AUDIO_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

        # ---------------------------------------------------------------------
        # 2. Voice Configuration & Output Paths
        # ---------------------------------------------------------------------
        meta = task.task_metadata or {}
        voice_name = meta.get("voice_name") or config.ui.get("voice_name", "zh-CN-XiaoxiaoNeural")
        voice_rate = float(meta.get("voice_rate", 1.0))
        voice_volume = float(meta.get("voice_volume", 1.0))
        bgm_type = meta.get("bgm_type")
        bgm_file = meta.get("bgm_file")
        bgm_volume = float(meta.get("bgm_volume", 0.2)) if meta.get("bgm_volume") is not None else 0.2
        subtitle_word_level = bool(meta.get("subtitle_word_level", False))

        voice_config_snapshot = {
            "voice_name": voice_name,
            "voice_rate": voice_rate,
            "voice_volume": voice_volume,
            "bgm_type": bgm_type,
            "bgm_file": bgm_file,
            "bgm_volume": bgm_volume,
            "subtitle_word_level": subtitle_word_level,
        }

        audio_output_id = f"ao_{uuid4().hex[:24]}"
        if self.storage_base_dir is not None:
            task_dir = Path(self.storage_base_dir) / task_id
        else:
            task_dir = Path(utils.task_dir(task_id))
        task_dir.mkdir(parents=True, exist_ok=True)

        audio_filename = f"narration_{audio_output_id[:12]}.mp3"
        subtitle_filename = f"subtitle_{audio_output_id[:12]}.srt"
        audio_file_path = str(task_dir / audio_filename)
        subtitle_file_path = str(task_dir / subtitle_filename)

        # ---------------------------------------------------------------------
        # 3. TTS Narration Audio & Subtitle Generation
        # ---------------------------------------------------------------------
        try:
            logger.info(
                f"[AudioStageExecutor] Invoking TTS synthesis for task '{task_id}', "
                f"voice='{voice_name}', rate={voice_rate}, target_file='{audio_file_path}'"
            )
            sub_maker = self.tts_runner(
                text=full_narration_text,
                voice_name=voice_name,
                voice_rate=voice_rate,
                voice_file=audio_file_path,
                voice_volume=voice_volume,
            )

            if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
                err = f"TTS generation failed to produce non-empty audio file at '{audio_file_path}'."
                logger.error(f"[AudioStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.AUDIO_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.RETRYABLE.value,
                    error_message=err,
                    is_retryable=True,
                )

            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.TTS_COMPLETED,
                attributes={
                    "task_id": task_id,
                    "audio_file_path": audio_file_path,
                    "voice_name": voice_name,
                },
            )

            # Generate subtitles if sub_maker is available
            subtitle_created = False
            if sub_maker is not None:
                try:
                    self.subtitle_creator(
                        sub_maker=sub_maker,
                        text=full_narration_text,
                        subtitle_file=subtitle_file_path,
                        word_level=subtitle_word_level,
                    )
                    if os.path.exists(subtitle_file_path) and os.path.getsize(subtitle_file_path) > 0:
                        subtitle_created = True
                        self._emit_trace(
                            task_id=task_id,
                            event_type=TraceEventType.SUBTITLE_COMPLETED,
                            attributes={
                                "task_id": task_id,
                                "subtitle_file_path": subtitle_file_path,
                            },
                        )
                except Exception as sub_err:
                    logger.warning(
                        f"[AudioStageExecutor] Subtitle generation encountered error: {sub_err}"
                    )

            # Measure actual duration
            actual_duration = float(self.duration_getter(audio_file_path))
            if actual_duration <= 0.0:
                # Fallback to sub_maker duration measurement if provided
                if sub_maker is not None:
                    actual_duration = float(self.duration_getter(sub_maker))

            if actual_duration <= 0.0:
                err = f"Measured narration audio duration is invalid ({actual_duration}s)."
                logger.error(f"[AudioStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.AUDIO_STAGE_FAILED,
                    attributes={"error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.RETRYABLE.value,
                    error_message=err,
                    is_retryable=True,
                )

            # -----------------------------------------------------------------
            # 4. Audio Duration Tolerance Validation
            # -----------------------------------------------------------------
            target_duration = script_rev.overall_target_duration
            max_allowed_deviation = max(10.0, target_duration * 0.6)
            deviation = abs(actual_duration - target_duration)

            if deviation > max_allowed_deviation:
                err = (
                    f"Narration duration ({actual_duration:.1f}s) deviates excessively from "
                    f"target duration ({target_duration:.1f}s, tolerance ±{max_allowed_deviation:.1f}s). "
                    f"Audio generation failed duration tolerance."
                )
                logger.error(f"[AudioStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.AUDIO_STAGE_FAILED,
                    attributes={
                        "error": err,
                        "actual_duration": actual_duration,
                        "target_duration": target_duration,
                    },
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.RETRYABLE.value,
                    error_message=err,
                    is_retryable=True,
                )

            # Compute content hashes
            narration_audio_hash = _compute_sha256(audio_file_path)
            subtitle_hash = _compute_sha256(subtitle_file_path) if subtitle_created else None
            final_subtitle_path = subtitle_file_path if subtitle_created else None

            # -----------------------------------------------------------------
            # 5. Optional Background Music Preparation
            # -----------------------------------------------------------------
            from app.services import bgm

            resolved_bgm_path: str | None = None
            if bgm.should_use_bgm(bgm_type, bgm_volume):
                if bgm_file:
                    try:
                        resolved_bgm_path = bgm.resolve_bgm_file(bgm_file)
                    except Exception as bgm_err:
                        logger.warning(
                            f"[AudioStageExecutor] Failed to resolve requested BGM file '{bgm_file}': {bgm_err}"
                        )
                elif bgm_type == "random":
                    available_songs = bgm.list_bgm_files()
                    if available_songs:
                        resolved_bgm_path = available_songs[0]

                if resolved_bgm_path:
                    self._emit_trace(
                        task_id=task_id,
                        event_type=TraceEventType.BGM_PREPARED,
                        attributes={
                            "task_id": task_id,
                            "bgm_path": resolved_bgm_path,
                            "bgm_volume": bgm_volume,
                        },
                    )

            # -----------------------------------------------------------------
            # 6. Output Persistence & TaskArtifactRef Production
            # -----------------------------------------------------------------
            domain_audio_output = AudioOutput.create(
                audio_output_id=audio_output_id,
                task_id=task_id,
                script_revision_id=script_rev.script_revision_id,
                execution_run_id=run.execution_run_id,
                narration_audio_path=audio_file_path,
                narration_audio_hash=narration_audio_hash,
                actual_narration_duration=actual_duration,
                subtitle_path=final_subtitle_path,
                subtitle_hash=subtitle_hash,
                bgm_path=resolved_bgm_path,
                bgm_volume=bgm_volume if resolved_bgm_path else None,
                voice_config_snapshot=voice_config_snapshot,
            )

            with _managed_session(self._session_factory) as session:
                audio_repo = AudioOutputRepository(session)
                audio_repo.save_audio_output(domain_audio_output)

            out_ref = TaskArtifactRef.create(
                task_id=task_id,
                stage=Stage.AUDIO,
                artifact_type=ArtifactType.AUDIO_OUTPUT,
                artifact_id=domain_audio_output.audio_output_id,
                metadata_json={
                    "narration_audio_path": audio_file_path,
                    "narration_audio_hash": narration_audio_hash,
                    "actual_narration_duration": actual_duration,
                    "subtitle_path": final_subtitle_path,
                    "subtitle_hash": subtitle_hash,
                    "bgm_path": resolved_bgm_path,
                    "bgm_volume": bgm_volume if resolved_bgm_path else None,
                },
            )

            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.AUDIO_STAGE_COMPLETED,
                attributes={
                    "task_id": task_id,
                    "audio_output_id": audio_output_id,
                    "actual_narration_duration": actual_duration,
                    "narration_audio_hash": narration_audio_hash,
                    "has_subtitle": bool(final_subtitle_path),
                    "has_bgm": bool(resolved_bgm_path),
                },
            )

            logger.info(
                f"[AudioStageExecutor] Completed audio stage for task '{task_id}'. "
                f"AudioOutput ID: '{audio_output_id}', duration: {actual_duration:.2f}s."
            )

            return StageExecutionResult(
                success=True,
                output_artifact_revision_id=domain_audio_output.audio_output_id,
                output_artifact_ref=out_ref,
                output_task_artifact_ref_id=out_ref.task_artifact_ref_id,
                metadata_json={
                    "audio_output_id": domain_audio_output.audio_output_id,
                    "actual_narration_duration": actual_duration,
                    "narration_audio_hash": narration_audio_hash,
                    "subtitle_path": final_subtitle_path,
                    "bgm_path": resolved_bgm_path,
                },
            )

        except Exception as exc:
            logger.exception(f"[AudioStageExecutor] Unexpected failure: {exc}")
            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.AUDIO_STAGE_FAILED,
                attributes={"error": str(exc)},
            )
            return StageExecutionResult(
                success=False,
                error_type=JobErrorType.FATAL.value,
                error_message=str(exc),
                is_retryable=False,
            )
