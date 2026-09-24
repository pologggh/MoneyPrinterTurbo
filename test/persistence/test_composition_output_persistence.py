from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.composition import CompositionOutput
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_state import WorkflowPolicyType
from app.persistence.models import (
    AudioOutputORM,
    Base,
    CompositionOutputORM,
    ContentPlanRevisionORM,
    ExecutionRunORM,
    ScriptRevisionORM,
    StoryboardSnapshotORM,
)
from app.persistence.repositories import (
    CompositionOutputRepository,
    KnowledgeVideoTaskRepository,
)


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal


def test_composition_output_repository_crud_and_queries(session_factory):
    """Verify CompositionOutputRepository save, get, get_latest, and list queries."""
    task_id = f"task_{uuid4().hex[:8]}"
    plan_id = f"cpr_{uuid4().hex[:8]}"
    script_id = f"scr_{uuid4().hex[:8]}"
    storyboard_id = f"sb_{uuid4().hex[:8]}"
    run_id = f"run_{uuid4().hex[:8]}"
    audio_id = f"ao_{uuid4().hex[:8]}"

    with session_factory() as session:
        # 1. Setup prerequisite records
        task_repo = KnowledgeVideoTaskRepository(session)
        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="History of Relativity",
            target_duration=45.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_repo.save_task(task)

        plan_orm = ContentPlanRevisionORM(
            content_plan_revision_id=plan_id,
            revision_number=1,
            topic="History of Relativity",
            overall_target_duration=45.0,
            created_at=datetime.now(UTC),
        )
        session.add(plan_orm)

        script_orm = ScriptRevisionORM(
            script_revision_id=script_id,
            task_id=task_id,
            content_plan_revision_id=plan_id,
            revision_number=1,
            overall_target_duration=45.0,
            content_fingerprint="fp123",
            created_at=datetime.now(UTC),
        )
        session.add(script_orm)

        sb_orm = StoryboardSnapshotORM(
            storyboard_snapshot_id=storyboard_id,
            content_plan_revision_id=plan_id,
            snapshot_state="APPROVED",
            created_at=datetime.now(UTC),
        )
        session.add(sb_orm)

        run_orm = ExecutionRunORM(
            execution_run_id=run_id,
            asset_route_plan_id=f"arp_{uuid4().hex[:8]}",
            storyboard_snapshot_id=storyboard_id,
            status="COMPLETED",
            total_shots=2,
            succeeded_shots=2,
            failed_shots=0,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )
        session.add(run_orm)

        audio_orm = AudioOutputORM(
            audio_output_id=audio_id,
            task_id=task_id,
            script_revision_id=script_id,
            execution_run_id=run_id,
            narration_audio_path="/tmp/audio.mp3",
            narration_audio_hash="hash_audio_1",
            actual_narration_duration=44.5,
            subtitle_path="/tmp/subtitle.srt",
            subtitle_hash="hash_sub_1",
            bgm_path="/tmp/bgm.mp3",
            bgm_volume=0.2,
            voice_config_snapshot_json={"voice": "test"},
            created_at=datetime.now(UTC),
        )
        session.add(audio_orm)
        session.commit()

        # 2. Persist two composition outputs
        comp_repo = CompositionOutputRepository(session)
        out1 = CompositionOutput.create(
            task_id=task_id,
            storyboard_snapshot_id=storyboard_id,
            execution_run_id=run_id,
            audio_output_id=audio_id,
            video_path="/tmp/final_1.mp4",
            video_hash="hash_v1",
            duration=44.5,
            width=1920,
            height=1080,
            file_size=10485760,
            video_codec="h264",
            audio_codec="aac",
            fps=30.0,
            composition_params_snapshot={"resolution": "1080p"},
            created_at=datetime(2026, 9, 12, 10, 0, 0, tzinfo=UTC),
        )
        saved1 = comp_repo.save_composition_output(out1)
        assert saved1.composition_output_id == out1.composition_output_id
        assert saved1.video_path == "/tmp/final_1.mp4"
        assert saved1.width == 1920
        assert saved1.height == 1080
        assert saved1.file_size == 10485760

        out2 = CompositionOutput.create(
            task_id=task_id,
            storyboard_snapshot_id=storyboard_id,
            execution_run_id=run_id,
            audio_output_id=audio_id,
            video_path="/tmp/final_2.mp4",
            video_hash="hash_v2",
            duration=44.5,
            width=1080,
            height=1920,
            file_size=12000000,
            video_codec="h264",
            audio_codec="aac",
            fps=30.0,
            composition_params_snapshot={"resolution": "portrait"},
            created_at=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
        )
        saved2 = comp_repo.save_composition_output(out2)
        session.commit()

        # 3. Test get_composition_output
        retrieved = comp_repo.get_composition_output(out1.composition_output_id)
        assert retrieved is not None
        assert retrieved.composition_output_id == out1.composition_output_id
        assert retrieved.video_hash == "hash_v1"

        # 4. Test get_latest_composition_output_for_task
        latest = comp_repo.get_latest_composition_output_for_task(task_id)
        assert latest is not None
        assert latest.composition_output_id == out2.composition_output_id
        assert latest.width == 1080

        # 5. Test list_composition_outputs_for_task
        all_outs = comp_repo.list_composition_outputs_for_task(task_id)
        assert len(all_outs) == 2
        assert all_outs[0].composition_output_id == out1.composition_output_id
        assert all_outs[1].composition_output_id == out2.composition_output_id
