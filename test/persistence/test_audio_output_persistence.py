from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.audio import AudioOutput
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.script import ScriptRevision, ScriptSegment
from app.domain.workflow_state import WorkflowPolicyType
from app.persistence.models import (
    Base,
    ContentPlanRevisionORM,
    ExecutionRunORM,
    StoryboardSnapshotORM,
)
from app.persistence.repositories import (
    AudioOutputRepository,
    KnowledgeVideoTaskRepository,
    ScriptRepository,
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


def test_audio_output_repository_crud_and_queries(session_factory):
    """Verify AudioOutputRepository save, get, get_latest, and list queries."""
    task_id = f"task_{uuid4().hex[:8]}"
    plan_id = f"cpr_{uuid4().hex[:8]}"
    script_id = f"scr_{uuid4().hex[:8]}"
    storyboard_id = f"sb_{uuid4().hex[:8]}"
    run_id = f"run_{uuid4().hex[:8]}"

    with session_factory() as session:
        # 1. Setup prerequisite records
        task_repo = KnowledgeVideoTaskRepository(session)
        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="History of Audio",
            target_duration=45.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_repo.save_task(task)

        plan_orm = ContentPlanRevisionORM(
            content_plan_revision_id=plan_id,
            revision_number=1,
            topic="History of Audio",
            overall_target_duration=45.0,
            created_at=datetime.now(UTC),
        )
        session.add(plan_orm)

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

        script_repo = ScriptRepository(session)
        seg = ScriptSegment(
            script_revision_id=script_id,
            content_beat_id=f"beat_{uuid4().hex[:8]}",
            order=1,
            narration_text="Sound has a rich history.",
            target_duration=45.0,
        )
        script_rev = ScriptRevision.create(
            script_revision_id=script_id,
            task_id=task_id,
            content_plan_revision_id=plan_id,
            revision_number=1,
            overall_target_duration=45.0,
            segments=[seg],
        )
        script_repo.save_revision(script_rev)
        session.commit()

        # 2. Test AudioOutput persistence
        audio_repo = AudioOutputRepository(session)
        audio_out = AudioOutput.create(
            task_id=task_id,
            script_revision_id=script_id,
            execution_run_id=run_id,
            narration_audio_path="/data/audio/narration_1.mp3",
            narration_audio_hash="abc123hash",
            actual_narration_duration=43.5,
            subtitle_path="/data/audio/narration_1.srt",
            subtitle_hash="def456hash",
            bgm_path="/data/audio/bgm.mp3",
            bgm_volume=0.15,
            voice_config_snapshot={"voice_name": "zh-CN-XiaoxiaoNeural", "rate": 1.0},
        )
        saved = audio_repo.save_audio_output(audio_out)
        session.commit()

        assert saved.audio_output_id == audio_out.audio_output_id
        assert saved.actual_narration_duration == 43.5
        assert saved.voice_config_snapshot["voice_name"] == "zh-CN-XiaoxiaoNeural"

        # 3. Retrieve by ID
        fetched = audio_repo.get_audio_output(audio_out.audio_output_id)
        assert fetched is not None
        assert fetched.audio_output_id == audio_out.audio_output_id
        assert fetched.narration_audio_path == "/data/audio/narration_1.mp3"
        assert fetched.narration_audio_hash == "abc123hash"
        assert fetched.subtitle_path == "/data/audio/narration_1.srt"
        assert fetched.subtitle_hash == "def456hash"
        assert fetched.bgm_path == "/data/audio/bgm.mp3"
        assert fetched.bgm_volume == 0.15

        # 4. Query latest and list
        latest = audio_repo.get_latest_audio_output_for_task(task_id)
        assert latest is not None
        assert latest.audio_output_id == audio_out.audio_output_id

        all_outputs = audio_repo.list_audio_outputs_for_task(task_id)
        assert len(all_outputs) == 1
        assert all_outputs[0].audio_output_id == audio_out.audio_output_id

        # Non-existent query
        assert audio_repo.get_audio_output("non_existent") is None
