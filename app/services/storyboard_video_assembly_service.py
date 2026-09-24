from __future__ import annotations

import os
from collections import defaultdict
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.domain.asset_execution import ExecutionStatus
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.models import const
from app.models.schema import VideoAspect, VideoConcatMode, VideoParams
from app.persistence.repositories import (
    ContentPlanRepository,
    ExecutionRepository,
    ShotRepository,
    StoryboardRepository,
)
from app.persistence.session import get_session
from app.services import state as sm
from app.utils import utils


class StoryboardAssemblyError(Exception):
    """Base exception for storyboard video assembly errors."""

    def __init__(self, message: str, code: str = "STORYBOARD_ASSEMBLY_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message


class StoryboardVideoAssemblyResult(BaseModel):
    """Result of assembling a full video from a StoryboardSnapshot and its executed assets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    storyboard_snapshot_id: str
    execution_run_id: str
    final_video_paths: list[str]
    combined_video_paths: list[str]
    audio_path: str
    subtitle_path: str
    audio_duration: float
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "COMPLETED"


class StoryboardVideoAssemblyService:
    """
    Connects executed storyboard shot assets with MoneyPrinterTurbo's media pipeline:
    - Verifies all shots have valid produced media files
    - Preprocesses static images into motion video clips via render_image_zoom_video
    - Synthesizes narration audio via TTS (voice.tts / task.generate_audio)
    - Generates synchronized subtitles (task.generate_subtitle)
    - Composes ordered shot clips with BGM mixing (task.generate_final_videos)
    - Updates MPT task state and returns final MP4 paths
    """

    def __init__(self, session_factory: Any | None = None):
        self._session_factory = session_factory

    def assemble_video(
        self,
        storyboard_snapshot_id: str,
        execution_run_id: str | None = None,
        video_params: VideoParams | dict[str, Any] | None = None,
        task_id: str | None = None,
        session: Session | None = None,
        audio_path: str | None = None,
        subtitle_path: str | None = None,
        audio_duration: float | None = None,
        bgm_path: str | None = None,
        bgm_volume: float | None = None,
        output_file_path: str | None = None,
    ) -> StoryboardVideoAssemblyResult:
        """
        Executes end-to-end video assembly for an approved or completed storyboard.
        """
        active_task_id = task_id or str(uuid4())
        task_dir = utils.task_dir(active_task_id)
        os.makedirs(task_dir, exist_ok=True)

        sm.state.update_task(
            active_task_id,
            state=const.TASK_STATE_PROCESSING,
            progress=5,
        )

        try:
            if session is not None:
                return self._execute_assembly(
                    session=session,
                    storyboard_snapshot_id=storyboard_snapshot_id,
                    execution_run_id=execution_run_id,
                    video_params=video_params,
                    task_id=active_task_id,
                    audio_path=audio_path,
                    subtitle_path=subtitle_path,
                    audio_duration=audio_duration,
                    bgm_path=bgm_path,
                    bgm_volume=bgm_volume,
                    output_file_path=output_file_path,
                )

            with get_session(self._session_factory) as db_session:
                return self._execute_assembly(
                    session=db_session,
                    storyboard_snapshot_id=storyboard_snapshot_id,
                    execution_run_id=execution_run_id,
                    video_params=video_params,
                    task_id=active_task_id,
                    audio_path=audio_path,
                    subtitle_path=subtitle_path,
                    audio_duration=audio_duration,
                    bgm_path=bgm_path,
                    bgm_volume=bgm_volume,
                    output_file_path=output_file_path,
                )
        except Exception:
            sm.state.patch_task(
                active_task_id,
                state=const.TASK_STATE_FAILED,
                progress=100,
                error_code="STORYBOARD_ASSEMBLY_FAILED",
                error="Storyboard video assembly failed. Check server logs for details.",
            )
            raise

    def _execute_assembly(
        self,
        session: Session,
        storyboard_snapshot_id: str,
        execution_run_id: str | None,
        video_params: VideoParams | dict[str, Any] | None,
        task_id: str,
        audio_path: str | None = None,
        subtitle_path: str | None = None,
        audio_duration: float | None = None,
        bgm_path: str | None = None,
        bgm_volume: float | None = None,
        output_file_path: str | None = None,
    ) -> StoryboardVideoAssemblyResult:
        from app.services import task, video

        plan_repo = ContentPlanRepository(session)
        shot_repo = ShotRepository(session)
        storyboard_repo = StoryboardRepository(session)
        exec_repo = ExecutionRepository(session)

        # 1. Load Storyboard Snapshot and Plan
        snapshot: StoryboardSnapshot | None = storyboard_repo.get_snapshot(
            storyboard_snapshot_id
        )
        if snapshot is None:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            raise StoryboardAssemblyError(
                f"StoryboardSnapshot '{storyboard_snapshot_id}' not found."
            )

        plan = plan_repo.get_revision(snapshot.content_plan_revision_id)
        if plan is None:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            raise StoryboardAssemblyError(
                f"ContentPlanRevision '{snapshot.content_plan_revision_id}' not found."
            )

        frozen_revisions = storyboard_repo.get_snapshot_shot_revisions(
            snapshot.storyboard_snapshot_id
        )
        if not frozen_revisions:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            raise StoryboardAssemblyError(
                f"StoryboardSnapshot '{storyboard_snapshot_id}' contains no shots."
            )

        # Resolve and validate execution_run_id
        active_run_id: str
        if execution_run_id:
            run = exec_repo.get_execution_run(execution_run_id)
            if run is None:
                sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
                raise StoryboardAssemblyError(
                    f"ExecutionRun '{execution_run_id}' not found."
                )
            if run.storyboard_snapshot_id != snapshot.storyboard_snapshot_id:
                sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
                raise StoryboardAssemblyError(
                    f"ExecutionRun '{execution_run_id}' belongs to storyboard_snapshot_id "
                    f"'{run.storyboard_snapshot_id}', not '{snapshot.storyboard_snapshot_id}'."
                )
            if run.status != ExecutionStatus.COMPLETED:
                sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
                raise StoryboardAssemblyError(
                    f"ExecutionRun '{execution_run_id}' is not completed; "
                    f"current status is '{run.status.value}'."
                )
            active_run_id = execution_run_id
        else:
            latest_run = exec_repo.get_latest_terminal_execution_run(
                snapshot.storyboard_snapshot_id
            )
            if latest_run is not None:
                active_run_id = latest_run.execution_run_id
            else:
                sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
                raise StoryboardAssemblyError(
                    f"No completed or successful execution run found for StoryboardSnapshot "
                    f"'{snapshot.storyboard_snapshot_id}'. Shot '{frozen_revisions[0].shot_id}' "
                    f"(revision '{frozen_revisions[0].shot_revision_id}') has no produced asset."
                )

        # Order shots: first by Beat.order, then by Shot.local_order
        revisions_by_lineage: dict[str, list[ShotRevision]] = defaultdict(list)
        for rev in frozen_revisions:
            revisions_by_lineage[rev.beat_lineage_id].append(rev)

        ordered_shot_revisions: list[ShotRevision] = []
        for beat in sorted(plan.beats, key=lambda b: b.order):
            shots_in_beat = revisions_by_lineage.get(beat.beat_lineage_id, [])
            # Sort shots by local_order
            shots_with_order: list[tuple[int, ShotRevision]] = []
            for r in shots_in_beat:
                shot_entity: Shot | None = shot_repo.get_shot(r.shot_id)
                local_order = shot_entity.local_order if shot_entity else 1
                shots_with_order.append((local_order, r))
            shots_with_order.sort(key=lambda x: x[0])
            ordered_shot_revisions.extend([r for _, r in shots_with_order])

        # 2. Retrieve & Verify Produced Assets for Each Shot in execution run
        ordered_video_materials: list[str] = []
        image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

        for rev in ordered_shot_revisions:
            selected_asset = exec_repo.get_asset_version_for_run_and_shot(
                execution_run_id=active_run_id,
                shot_revision_id=rev.shot_revision_id,
            )
            if not selected_asset:
                err_msg = (
                    f"Shot '{rev.shot_id}' (revision '{rev.shot_revision_id}') has no "
                    f"produced asset in execution run '{active_run_id}'. Please run asset execution before assembling video."
                )
                sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
                raise StoryboardAssemblyError(err_msg)

            file_path = selected_asset.file_path

            if not file_path or not os.path.isfile(file_path):
                err_msg = (
                    f"Asset file '{file_path}' for shot '{rev.shot_id}' (revision '{rev.shot_revision_id}') "
                    f"in execution run '{active_run_id}' does not exist on disk."
                )
                sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
                raise StoryboardAssemblyError(err_msg)

            # Preprocess static image into video clip if necessary
            lower_path = file_path.lower()
            if lower_path.endswith(image_extensions) or selected_asset.media_type.name == "IMAGE":
                clip_duration = max(3, round(rev.target_duration))
                logger.info(
                    f"Rendering static image asset to video clip ({clip_duration}s): {file_path}"
                )
                rendered_clip = video.render_image_zoom_video(
                    image_path=file_path,
                    clip_duration=clip_duration,
                )
                ordered_video_materials.append(rendered_clip)
            else:
                ordered_video_materials.append(file_path)

        sm.state.update_task(task_id, progress=20)

        # 3. Build VideoParams and Full Narration Script
        full_script = "\n".join(
            rev.narration.strip()
            for rev in ordered_shot_revisions
            if rev.narration and rev.narration.strip()
        )
        if not full_script:
            full_script = plan.topic or "知识视频"

        params = self._resolve_video_params(
            video_params=video_params,
            plan_topic=plan.topic,
            full_script=full_script,
        )

        # 4. Generate or Use Prepared Narration Audio
        if audio_path:
            audio_file = audio_path
            resolved_audio_duration = audio_duration
            if resolved_audio_duration is None:
                from app.services import voice
                resolved_audio_duration = voice.get_audio_duration(audio_file)
            sub_maker = None
        else:
            logger.info(f"Generating TTS audio for storyboard: task_id={task_id}")
            audio_file, resolved_audio_duration, sub_maker = task.generate_audio(
                task_id=task_id,
                params=params,
                video_script=full_script,
            )
            if not audio_file or not os.path.isfile(audio_file):
                sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
                raise StoryboardAssemblyError(
                    f"Failed to generate TTS audio for storyboard task '{task_id}'"
                )

        sm.state.update_task(task_id, progress=40)

        # 5. Subtitles
        if subtitle_path is not None:
            resolved_subtitle_path = subtitle_path
        else:
            logger.info(f"Generating subtitles for storyboard: task_id={task_id}")
            resolved_subtitle_path = ""
            if params.subtitle_enabled:
                resolved_subtitle_path = task.generate_subtitle(
                    task_id=task_id,
                    params=params,
                    video_script=full_script,
                    sub_maker=sub_maker,
                    audio_file=audio_file,
                )

        sm.state.update_task(task_id, progress=55)

        # 6. Video Composition & BGM Mixing
        logger.info(
            f"Composing final video for storyboard: task_id={task_id}, "
            f"clips={len(ordered_video_materials)}, duration={resolved_audio_duration:.1f}s"
        )
        # Ensure sequential stitching matching storyboard shot order
        params.match_materials_to_script = True
        params.video_concat_mode = VideoConcatMode.sequential
        if bgm_volume is not None:
            params.bgm_volume = bgm_volume

        if audio_path or output_file_path or bgm_path is not None:
            # Unified workflow composition path (or custom output):
            # Directly combine clips and generate final video without legacy multi-video/AI-music re-selection
            final_video_path = output_file_path or os.path.join(
                utils.task_dir(task_id), "final-1.mp4"
            )
            combined_video_path = os.path.join(
                utils.task_dir(task_id), f"combined-{storyboard_snapshot_id[:8]}.mp4"
            )
            video.combine_videos(
                combined_video_path=combined_video_path,
                video_paths=ordered_video_materials,
                audio_file=audio_file,
                video_aspect=params.video_aspect,
                video_fit_mode=params.video_fit_mode,
                video_concat_mode=params.video_concat_mode,
                video_transition_mode=params.video_transition_mode,
                max_clip_duration=params.video_clip_duration,
                threads=params.n_threads,
                clip_speed=params.video_clip_speed,
            )

            bgm_override = bgm_path if bgm_path is not None else ("" if params.bgm_type else None)
            video.generate_video(
                video_path=combined_video_path,
                audio_path=audio_file,
                subtitle_path=resolved_subtitle_path,
                output_file=final_video_path,
                params=params,
                bgm_file_override=bgm_override,
            )
            final_video_paths = [final_video_path]
            combined_video_paths = [combined_video_path]
            warnings = []
        else:
            final_video_paths, combined_video_paths, warnings = task.generate_final_videos(
                task_id=task_id,
                params=params,
                downloaded_videos=ordered_video_materials,
                audio_file=audio_file,
                subtitle_path=resolved_subtitle_path,
                audio_duration=resolved_audio_duration,
            )

        if not final_video_paths:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            raise StoryboardAssemblyError(
                f"Video generation produced no output files for task '{task_id}'"
            )

        # 7. Finalize Task State
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            videos=final_video_paths,
            combined_videos=combined_video_paths,
        )

        logger.success(
            f"Storyboard video assembly completed: task_id={task_id}, "
            f"output={final_video_paths}"
        )

        return StoryboardVideoAssemblyResult(
            task_id=task_id,
            storyboard_snapshot_id=storyboard_snapshot_id,
            execution_run_id=active_run_id,
            final_video_paths=final_video_paths,
            combined_video_paths=combined_video_paths,
            audio_path=audio_file,
            subtitle_path=resolved_subtitle_path,
            audio_duration=float(resolved_audio_duration),
            warnings=warnings,
            status="COMPLETED",
        )

    def _resolve_video_params(
        self,
        video_params: VideoParams | dict[str, Any] | None,
        plan_topic: str,
        full_script: str,
    ) -> VideoParams:
        if isinstance(video_params, VideoParams):
            params = video_params.model_copy()
            params.video_subject = params.video_subject or plan_topic
            params.video_script = full_script
            return params

        data: dict[str, Any] = dict(video_params) if isinstance(video_params, dict) else {}
        data.setdefault("video_subject", plan_topic or "知识视频")
        data.setdefault("video_script", full_script)
        data.setdefault("video_aspect", VideoAspect.landscape.value)
        data.setdefault("voice_name", "zh-CN-XiaoxiaoNeural")
        data.setdefault("voice_rate", 1.0)
        data.setdefault("voice_volume", 1.0)
        data.setdefault("bgm_type", "random")
        data.setdefault("bgm_volume", 0.2)
        data.setdefault("subtitle_enabled", True)
        data.setdefault("match_materials_to_script", True)
        data.setdefault("video_concat_mode", VideoConcatMode.sequential.value)

        return VideoParams(**data)
