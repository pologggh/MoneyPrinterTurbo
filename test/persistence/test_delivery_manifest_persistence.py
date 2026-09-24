from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.composition import CompositionOutput
from app.domain.delivery import DeliveryManifest
from app.domain.evaluation import EvaluationDecision, EvaluationSnapshot
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_state import WorkflowPolicyType
from app.persistence.models import (
    AudioOutputORM,
    Base,
    CompositionOutputORM,
    ContentPlanRevisionORM,
    DeliveryManifestORM,
    EvaluationSnapshotORM,
    ExecutionRunORM,
    ScriptRevisionORM,
    StoryboardSnapshotORM,
)
from app.persistence.repositories import (
    CompositionOutputRepository,
    DeliveryManifestRepository,
    EvaluationRepository,
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


def test_delivery_manifest_repository_crud_and_queries(session_factory):
    """Verify DeliveryManifestRepository save, get, and get_for_task queries."""
    task_id = f"task_{uuid4().hex[:8]}"
    plan_id = f"cpr_{uuid4().hex[:8]}"
    script_id = f"scr_{uuid4().hex[:8]}"
    storyboard_id = f"sb_{uuid4().hex[:8]}"
    run_id = f"run_{uuid4().hex[:8]}"
    audio_id = f"ao_{uuid4().hex[:8]}"
    comp_id = f"co_{uuid4().hex[:8]}"
    snap_id = f"es_{uuid4().hex[:8]}"
    delivery_id = f"dm_{uuid4().hex[:8]}"

    with session_factory() as session:
        # 1. Setup prerequisite records
        task_repo = KnowledgeVideoTaskRepository(session)
        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="Quantum Computing",
            target_duration=60.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_repo.save_task(task)

        # Plan, Script, Storyboard, ExecutionRun, AudioOutput
        session.add(
            ContentPlanRevisionORM(
                content_plan_revision_id=plan_id,
                revision_number=1,
                topic="Quantum Computing",
                overall_target_duration=60.0,
                created_at=datetime.now(UTC),
            )
        )
        session.add(
            ScriptRevisionORM(
                script_revision_id=script_id,
                task_id=task_id,
                content_plan_revision_id=plan_id,
                revision_number=1,
                overall_target_duration=60.0,
                content_fingerprint="fp_script_1",
                created_at=datetime.now(UTC),
            )
        )
        session.add(
            StoryboardSnapshotORM(
                storyboard_snapshot_id=storyboard_id,
                content_plan_revision_id=plan_id,
                snapshot_state="APPROVED",
                created_at=datetime.now(UTC),
            )
        )
        session.add(
            ExecutionRunORM(
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
        )
        session.add(
            AudioOutputORM(
                audio_output_id=audio_id,
                task_id=task_id,
                script_revision_id=script_id,
                execution_run_id=run_id,
                narration_audio_path="/fake/path/audio.mp3",
                narration_audio_hash="hash_audio_123",
                actual_narration_duration=60.0,
                subtitle_path="/fake/path/subtitles.srt",
                subtitle_hash="hash_sub_123",
                created_at=datetime.now(UTC),
            )
        )
        session.add(
            CompositionOutputORM(
                composition_output_id=comp_id,
                task_id=task_id,
                storyboard_snapshot_id=storyboard_id,
                execution_run_id=run_id,
                audio_output_id=audio_id,
                video_path="/fake/path/video.mp4",
                video_hash="hash_video_123",
                duration=60.0,
                fps=30.0,
                width=1920,
                height=1080,
                file_size=20000000,
                created_at=datetime.now(UTC),
            )
        )
        session.add(
            EvaluationSnapshotORM(
                evaluation_snapshot_id=snap_id,
                evaluation_target_id="target_1",
                decision="PASS",
                overall_score=0.95,
                policy_version="v1.0",
                evaluator_version="v1.0",
                created_at=datetime.now(UTC),
            )
        )
        session.commit()

    with session_factory() as session:
        deliv_repo = DeliveryManifestRepository(session)

        # 2. Save delivery manifest
        manifest = DeliveryManifest.create(
            delivery_manifest_id=delivery_id,
            task_id=task_id,
            composition_output_id=comp_id,
            evaluation_snapshot_id=snap_id,
            final_video_path="/fake/path/video.mp4",
            final_video_hash="hash_video_123",
            final_video_size_bytes=20000000,
            subtitle_path="/fake/path/subtitles.srt",
            subtitle_hash="hash_sub_123",
            source_report_path="/fake/path/source_report.md",
            source_report_hash="hash_source_rep_123",
            execution_report_path="/fake/path/execution_report.md",
            execution_report_hash="hash_exec_rep_123",
            delivery_params_snapshot={"format": "mp4"},
        )

        deliv_repo.save_delivery_manifest(manifest)

        # 3. Retrieve and verify
        loaded = deliv_repo.get_delivery_manifest(delivery_id)
        assert loaded is not None
        assert loaded.delivery_manifest_id == delivery_id
        assert loaded.task_id == task_id
        assert loaded.composition_output_id == comp_id
        assert loaded.evaluation_snapshot_id == snap_id
        assert loaded.final_video_path == "/fake/path/video.mp4"
        assert loaded.final_video_hash == "hash_video_123"
        assert loaded.final_video_size_bytes == 20000000
        assert loaded.subtitle_path == "/fake/path/subtitles.srt"
        assert loaded.subtitle_hash == "hash_sub_123"
        assert loaded.source_report_hash == "hash_source_rep_123"
        assert loaded.execution_report_hash == "hash_exec_rep_123"
        assert loaded.delivery_params_snapshot == {"format": "mp4"}

        # 4. Retrieve for task
        task_manifest = deliv_repo.get_delivery_manifest_for_task(task_id)
        assert task_manifest is not None
        assert task_manifest.delivery_manifest_id == delivery_id

        # 5. Non-existent returns None
        assert deliv_repo.get_delivery_manifest("non_existent_id") is None
        assert deliv_repo.get_delivery_manifest_for_task("non_existent_task") is None
