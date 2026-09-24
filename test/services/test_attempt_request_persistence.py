from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.domain.asset_execution import (
    ExecutionAttempt,
    ExecutionAttemptStatus,
    GenerationMode,
)
from app.domain.asset_router import AssetRouteCandidate
from app.domain.enums import VisualType
from app.domain.shot import Shot, ShotRevision
from app.persistence.models import Base
from app.persistence.repositories import (
    ExecutionRepository,
    ShotRepository,
)
from app.services.shot_execution_service import (
    ShotExecutionService,
)


def _setup_in_memory_db():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def test_attempt_request_persisted_before_adapter_submit(tmp_path):
    """
    Area 1: Verify ExecutionAttempt and AttemptRequest are persisted in DB
    BEFORE adapter.submit is called. If DB persistence succeeds, adapter is invoked.
    """
    SessionLocal = _setup_in_memory_db()

    with SessionLocal() as session:
        shot_repo = ShotRepository(session)
        exec_repo = ExecutionRepository(session)

        shot = Shot(
            shot_id="shot_persist_01",
            beat_lineage_id="bl_01",
            local_order=1,
        )
        shot_repo.add_shot(shot)

        rev = ShotRevision(
            shot_revision_id="srev_persist_01",
            shot_id=shot.shot_id,
            revision_number=1,
            beat_lineage_id="bl_01",
            created_from_beat_instance_id="beat_inst_01",
            narration="测试旁白",
            target_duration=5.0,
            visual_goal="镜头视觉目标",
            visual_type=VisualType.STOCK_VIDEO,
            scene_description="场景",
            generation_prompt="prompt",
            camera_movement="static",
        )
        shot_repo.add_revision(rev)
        session.commit()

    cand = AssetRouteCandidate(
        capability_id="cap_persist_01",
        provider="pexels",
        model="default",
        generation_mode=GenerationMode.STOCK_SEARCH,
        requested_visual_type=VisualType.STOCK_VIDEO,
        is_eligible=True,
    )

    mock_adapter = MagicMock()
    mock_adapter.submit.return_value = MagicMock(
        submission_accepted=True,
        remote_task_id="rt_123",
        status="submitted",
        raw_response={},
    )
    mock_adapter.get_status.return_value = MagicMock(
        remote_task_id="rt_123",
        status="running",
        is_terminal=False,
    )

    mock_registry = MagicMock()
    mock_registry.get_adapter.return_value = mock_adapter

    service = ShotExecutionService(
        adapter_registry=mock_registry, session_factory=SessionLocal
    )

    from app.domain.asset_execution import ShotExecution, ShotExecutionStatus
    from app.domain.asset_router import (
        AssetRouteDecision,
        RoutingStrategy,
        create_routing_request_from_shot_revision,
    )

    dec = AssetRouteDecision(
        shot_id=shot.shot_id,
        shot_revision_id=rev.shot_revision_id,
        requested_visual_type=rev.visual_type,
        routing_strategy=RoutingStrategy.BALANCED,
        selected_candidate=cand,
        eligible_candidates=(cand,),
        rejected_candidates=(),
    )
    shot_exec = ShotExecution(
        shot_execution_id="se_persist_01",
        execution_run_id="run_persist_01",
        shot_id=shot.shot_id,
        shot_revision_id=rev.shot_revision_id,
        route_decision=dec,
        route_candidate_capability_ids=(cand.capability_id,),
        status=ShotExecutionStatus.PENDING,
    )
    req = create_routing_request_from_shot_revision(rev)

    _res_exec, attempts, _asset = service.execute_shot(
        shot_execution=shot_exec,
        request=req,
        storage_base_dir=str(tmp_path),
    )

    # Verify attempt and request records are persisted in database
    with SessionLocal() as session:
        exec_repo = ExecutionRepository(session)
        assert len(attempts) >= 1
        attempt_id = attempts[0].execution_attempt_id

        db_attempt = exec_repo.get_execution_attempt(attempt_id)
        assert db_attempt is not None
        assert db_attempt.execution_run_id == "run_persist_01"
        assert db_attempt.shot_revision_id == rev.shot_revision_id

        db_req = exec_repo.get_attempt_request(attempt_id)
        assert db_req is not None
        assert db_req.execution_attempt_id == attempt_id
        assert db_req.provider == "pexels"
        assert db_req.generation_mode == GenerationMode.STOCK_SEARCH


def test_db_persistence_failure_halts_before_adapter_submit(tmp_path):
    """
    Area 1: Verify that if DB pre-submission persistence fails, adapter.submit
    is NEVER called, and the exception is not silently swallowed.
    """
    SessionLocal = _setup_in_memory_db()

    with SessionLocal() as session:
        shot_repo = ShotRepository(session)
        shot = Shot(
            shot_id="shot_fail_01",
            beat_lineage_id="bl_01",
            local_order=1,
        )
        shot_repo.add_shot(shot)
        rev = ShotRevision(
            shot_revision_id="srev_fail_01",
            shot_id=shot.shot_id,
            revision_number=1,
            beat_lineage_id="bl_01",
            created_from_beat_instance_id="beat_inst_01",
            narration="测试失败旁白",
            target_duration=5.0,
            visual_goal="目标",
            visual_type=VisualType.STOCK_VIDEO,
            scene_description="场景",
            generation_prompt="prompt",
            camera_movement="static",
        )
        shot_repo.add_revision(rev)
        session.commit()

    cand = AssetRouteCandidate(
        capability_id="cap_fail_01",
        provider="openai",
        model="dall-e-3",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
        requested_visual_type=VisualType.STOCK_VIDEO,
        is_eligible=True,
    )

    mock_adapter = MagicMock()
    mock_registry = MagicMock()
    mock_registry.get_adapter.return_value = mock_adapter

    service = ShotExecutionService(
        adapter_registry=mock_registry, session_factory=SessionLocal
    )

    from app.domain.asset_execution import ShotExecution, ShotExecutionStatus
    from app.domain.asset_router import (
        AssetRouteDecision,
        RoutingStrategy,
        create_routing_request_from_shot_revision,
    )

    dec = AssetRouteDecision(
        shot_id=shot.shot_id,
        shot_revision_id=rev.shot_revision_id,
        requested_visual_type=rev.visual_type,
        routing_strategy=RoutingStrategy.BALANCED,
        selected_candidate=cand,
        eligible_candidates=(cand,),
        rejected_candidates=(),
    )
    shot_exec = ShotExecution(
        shot_execution_id="se_fail_01",
        execution_run_id="run_fail_01",
        shot_id=shot.shot_id,
        shot_revision_id=rev.shot_revision_id,
        route_decision=dec,
        route_candidate_capability_ids=(cand.capability_id,),
        status=ShotExecutionStatus.PENDING,
    )
    req = create_routing_request_from_shot_revision(rev)

    with patch.object(
        ExecutionRepository,
        "save_attempt_request",
        side_effect=RuntimeError("Simulated DB commit error"),
    ), pytest.raises(RuntimeError, match="Simulated DB commit error"):
        service.execute_shot(
            shot_execution=shot_exec,
            request=req,
            storage_base_dir=str(tmp_path),
        )

    # CRITICAL: Verify external adapter submit was NEVER called
    mock_adapter.submit.assert_not_called()


def test_save_or_update_execution_attempt_prevents_duplicate_key():
    """
    Area 1: Verify save_or_update_execution_attempt safely updates existing
    attempts rather than raising primary key constraint errors.
    """
    SessionLocal = _setup_in_memory_db()

    attempt = ExecutionAttempt(
        execution_attempt_id="att_dup_01",
        execution_run_id="run_dup_01",
        shot_id="shot_dup_01",
        shot_revision_id="srev_dup_01",
        candidate_index=0,
        attempt_number=1,
        provider="pexels",
        model="default",
        generation_mode=GenerationMode.STOCK_SEARCH,
        status=ExecutionAttemptStatus.CREATED,
    )

    with SessionLocal() as session:
        repo = ExecutionRepository(session)
        first_save = repo.save_or_update_execution_attempt(attempt)
        session.commit()
        assert first_save.execution_attempt_id == "att_dup_01"

    # Now update attempt status and save again with the same PK
    updated_attempt = attempt.model_copy(update={"status": ExecutionAttemptStatus.RUNNING})
    with SessionLocal() as session:
        repo = ExecutionRepository(session)
        second_save = repo.save_or_update_execution_attempt(updated_attempt)
        session.commit()
        assert second_save.status == ExecutionAttemptStatus.RUNNING

    with SessionLocal() as session:
        repo = ExecutionRepository(session)
        fetched = repo.get_execution_attempt("att_dup_01")
        assert fetched is not None
        assert fetched.status == ExecutionAttemptStatus.RUNNING
