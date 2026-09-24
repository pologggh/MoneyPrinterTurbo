from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.asset_execution import (
    AttemptRequest,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRun,
    ExecutionStatus,
    GenerationMode,
    ProviderOutcomeType,
    ProviderReceipt,
    ShotExecution,
    ShotExecutionStatus,
)
from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutePlan,
    AssetRoutePlanStatus,
    ModelSelectionMode,
    RoutingStrategy,
    ShotRoutePlanEntry,
    ShotRoutePlanStatus,
    create_routing_request_from_shot_revision,
)
from app.domain.enums import VisualType
from app.domain.shot import Shot, ShotRevision
from app.persistence.models import Base
from app.persistence.repositories import (
    ExecutionRepository,
    ShotRepository,
)
from app.services.asset_adapters.base import AdapterExecutionResult
from app.services.asset_route_plan_execution_service import (
    AssetRoutePlanExecutionService,
)
from app.services.attempt_recovery_service import AttemptRecoveryService
from app.services.shot_execution_service import ShotExecutionService


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _create_test_shot_and_revision(session, shot_id: str, rev_id: str):
    shot_repo = ShotRepository(session)
    shot = Shot(
        shot_id=shot_id,
        beat_lineage_id="bl_01",
        local_order=1,
    )
    shot_repo.add_shot(shot)
    rev = ShotRevision(
        shot_revision_id=rev_id,
        shot_id=shot.shot_id,
        revision_number=1,
        beat_lineage_id="bl_01",
        created_from_beat_instance_id="beat_inst_01",
        narration="测试旁白内容",
        target_duration=5.0,
        visual_goal="测试镜头目标",
        visual_type=VisualType.STOCK_VIDEO,
        scene_description="场景描述",
        generation_prompt="测试提示词",
        camera_movement="static",
    )
    shot_repo.add_revision(rev)
    session.commit()
    return shot, rev


def test_receipt_persisted_before_status_polling(session_factory, tmp_path):
    """
    P0-4 requirement: Immediately persist ProviderReceipt, running_attempt, and current_exec
    to DB before polling, downloading, or probing.
    """
    with session_factory() as session:
        shot, rev = _create_test_shot_and_revision(session, "shot_rec_01", "rev_rec_01")
        exec_repo = ExecutionRepository(session)

        cand = AssetRouteCandidate(
            capability_id="cap_rec_01",
            provider="mock_provider",
            model="default",
            generation_mode=GenerationMode.STOCK_SEARCH,
            requested_visual_type=VisualType.STOCK_VIDEO,
            is_eligible=True,
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
            shot_execution_id="se_rec_01",
            execution_run_id="run_rec_01",
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            route_decision=dec,
            route_candidate_capability_ids=(cand.capability_id,),
            status=ShotExecutionStatus.PENDING,
        )
        exec_repo.add_shot_execution(shot_exec)
        session.commit()

    req = create_routing_request_from_shot_revision(rev)

    mock_adapter = MagicMock()

    def mock_submit(req, cand, storage_path, idempotency_key=None):
        return ProviderReceipt(
            execution_attempt_id="provider-placeholder-must-be-normalized",
            provider="mock_provider",
            provider_job_id="remote_task_999",
            provider_status="submitted",
            sanitized_metadata={"initial": "ok"},
        )

    mock_adapter.submit.side_effect = mock_submit

    # When get_status is called, we verify that the receipt is ALREADY in the database!
    receipt_in_db_during_poll = []

    def mock_get_status(remote_task_id, target_dir):
        with session_factory() as verify_session:
            exec_repo = ExecutionRepository(verify_session)
            attempts = exec_repo.list_attempts_for_run("run_rec_01")
            assert len(attempts) >= 1
            att_id = attempts[0].execution_attempt_id
            rec = exec_repo.get_provider_receipt(att_id)
            receipt_in_db_during_poll.append(
                rec is not None
                and rec.execution_attempt_id == att_id
                and rec.provider_job_id == "remote_task_999"
            )
            # Also verify attempt is RUNNING in DB
            db_att = exec_repo.get_execution_attempt(att_id)
            assert db_att.status == ExecutionAttemptStatus.RUNNING

        return AdapterExecutionResult(
            remote_task_id=remote_task_id,
            status="RUNNING",
            outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
            file_path=None,
            error_code="POLL_TIMEOUT",
            error_message="Remote task is still running after the polling deadline",
        )

    mock_adapter.get_status.side_effect = mock_get_status

    mock_registry = MagicMock()
    mock_registry.get_adapter.return_value = mock_adapter

    service = ShotExecutionService(
        adapter_registry=mock_registry, session_factory=session_factory
    )

    res_exec, attempts, _asset = service.execute_shot(
        shot_execution=shot_exec,
        request=req,
        storage_base_dir=str(tmp_path),
    )

    # Verify that get_status saw the receipt persisted before polling
    assert len(receipt_in_db_during_poll) >= 1
    assert all(receipt_in_db_during_poll)
    assert res_exec.status == ShotExecutionStatus.NEEDS_RECOVERY
    assert attempts[-1].status == ExecutionAttemptStatus.SUBMISSION_OUTCOME_UNKNOWN


def test_receipt_persistence_failure_stops_before_status_polling(
    session_factory, tmp_path, monkeypatch
):
    """A known provider receipt must be durable before any further provider work."""
    with session_factory() as session:
        shot, rev = _create_test_shot_and_revision(
            session, "shot_receipt_fail", "rev_receipt_fail"
        )
        candidate = AssetRouteCandidate(
            capability_id="cap_receipt_fail",
            provider="mock_provider",
            model="default",
            generation_mode=GenerationMode.STOCK_SEARCH,
            requested_visual_type=VisualType.STOCK_VIDEO,
            is_eligible=True,
        )
        decision = AssetRouteDecision(
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            requested_visual_type=rev.visual_type,
            routing_strategy=RoutingStrategy.BALANCED,
            selected_candidate=candidate,
            eligible_candidates=(candidate,),
            rejected_candidates=(),
        )
        shot_execution = ShotExecution(
            shot_execution_id="se_receipt_fail",
            execution_run_id="run_receipt_fail",
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            route_decision=decision,
            route_candidate_capability_ids=(candidate.capability_id,),
            status=ShotExecutionStatus.PENDING,
        )
        ExecutionRepository(session).add_shot_execution(shot_execution)
        session.commit()

    request = create_routing_request_from_shot_revision(rev)
    adapter = MagicMock()
    adapter.submit.side_effect = lambda *_args, **_kwargs: ProviderReceipt(
        execution_attempt_id="provider-supplied-id-is-normalized-by-service",
        provider="mock_provider",
        provider_job_id="remote-receipt-fail",
        provider_status="submitted",
        sanitized_metadata={},
    )
    registry = MagicMock()
    registry.has_adapter.return_value = True
    registry.get_adapter.return_value = adapter

    def fail_receipt_persistence(_self, _receipt):
        raise RuntimeError("database unavailable while saving receipt")

    monkeypatch.setattr(
        ExecutionRepository, "save_provider_receipt", fail_receipt_persistence
    )

    service = ShotExecutionService(
        adapter_registry=registry, session_factory=session_factory
    )
    with pytest.raises(RuntimeError, match="database unavailable"):
        service.execute_shot(
            shot_execution=shot_execution,
            request=request,
            storage_base_dir=str(tmp_path),
        )

    adapter.get_status.assert_not_called()


def test_submission_transport_error_becomes_needs_recovery(session_factory, tmp_path):
    """
    P0-4 requirement: Transport/network exception during adapter.submit is classified
    as SUBMISSION_OUTCOME_UNKNOWN (NEEDS_RECOVERY), not definitive technical failure.
    """
    with session_factory() as session:
        shot, rev = _create_test_shot_and_revision(session, "shot_trans_01", "rev_trans_01")
        exec_repo = ExecutionRepository(session)

        cand = AssetRouteCandidate(
            capability_id="cap_trans_01",
            provider="remote_provider",
            model="default",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            requested_visual_type=VisualType.STOCK_VIDEO,
            is_eligible=True,
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
            shot_execution_id="se_trans_01",
            execution_run_id="run_trans_01",
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            route_decision=dec,
            route_candidate_capability_ids=(cand.capability_id,),
            status=ShotExecutionStatus.PENDING,
        )
        exec_repo.add_shot_execution(shot_exec)
        session.commit()

    req = create_routing_request_from_shot_revision(rev)

    mock_adapter = MagicMock()
    # Adapter raises network/transport error during submit
    mock_adapter.submit.side_effect = ConnectionResetError("Connection lost during provider HTTP POST")

    mock_registry = MagicMock()
    mock_registry.get_adapter.return_value = mock_adapter

    service = ShotExecutionService(
        adapter_registry=mock_registry, session_factory=session_factory
    )

    res_exec, attempts, asset = service.execute_shot(
        shot_execution=shot_exec,
        request=req,
        storage_base_dir=str(tmp_path),
    )

    # Status must be NEEDS_RECOVERY, attempt must be SUBMISSION_OUTCOME_UNKNOWN, NO asset created
    assert res_exec.status == ShotExecutionStatus.NEEDS_RECOVERY
    assert asset is None
    assert len(attempts) == 1
    assert attempts[0].status == ExecutionAttemptStatus.SUBMISSION_OUTCOME_UNKNOWN
    assert attempts[0].error_code == "SUBMISSION_OUTCOME_UNKNOWN"
    assert "Connection lost" in (attempts[0].error_message or "")

    # Also verify persisted in DB
    with session_factory() as session:
        exec_repo = ExecutionRepository(session)
        db_exec = exec_repo.get_shot_execution("se_trans_01")
        assert db_exec is not None
        assert db_exec.status == ShotExecutionStatus.NEEDS_RECOVERY


def test_recovery_without_registered_adapter_resolves_failed(session_factory):
    """A missing recovery adapter is a typed failure, not an AttributeError."""
    with session_factory() as session:
        shot, rev = _create_test_shot_and_revision(
            session, "shot_missing_adapter", "rev_missing_adapter"
        )
    candidate = AssetRouteCandidate(
        capability_id="cap_missing_adapter",
        provider="missing_provider",
        model="default",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
        requested_visual_type=VisualType.STOCK_VIDEO,
        is_eligible=True,
    )
    decision = AssetRouteDecision(
        shot_id=shot.shot_id,
        shot_revision_id=rev.shot_revision_id,
        requested_visual_type=rev.visual_type,
        routing_strategy=RoutingStrategy.BALANCED,
        selected_candidate=candidate,
        eligible_candidates=(candidate,),
        rejected_candidates=(),
    )
    shot_execution = ShotExecution(
        execution_run_id="run_missing_adapter",
        shot_id=shot.shot_id,
        shot_revision_id=rev.shot_revision_id,
        route_decision=decision,
        status=ShotExecutionStatus.NEEDS_RECOVERY,
        attempt_ids=("attempt_missing_adapter",),
    )
    attempt = ExecutionAttempt(
        execution_attempt_id="attempt_missing_adapter",
        execution_run_id="run_missing_adapter",
        attempt_number=1,
        shot_id=shot.shot_id,
        shot_revision_id=rev.shot_revision_id,
        provider="missing_provider",
        model="default",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
        status=ExecutionAttemptStatus.SUBMITTING,
    )
    request = AttemptRequest(
        execution_attempt_id=attempt.execution_attempt_id,
        idempotency_key="missing-adapter-key",
        request_hash="missing-adapter-hash",
        provider="missing_provider",
        model="default",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
        payload={},
    )
    registry = MagicMock()
    registry.has_adapter.return_value = False
    registry.list_providers.return_value = []

    decision_result, recovered_shot, recovered_attempt, asset = (
        AttemptRecoveryService(adapter_registry=registry).recover(
            shot_execution=shot_execution,
            attempt=attempt,
            attempt_request=request,
        )
    )

    assert decision_result.reason_code == "ADAPTER_NOT_FOUND"
    assert recovered_shot.status == ShotExecutionStatus.FAILED
    assert recovered_attempt.status == ExecutionAttemptStatus.FAILED
    assert asset is None


@pytest.mark.parametrize(
    ("shot_status", "attempt_status"),
    [
        (ShotExecutionStatus.PENDING, ExecutionAttemptStatus.SUBMITTING),
        (ShotExecutionStatus.RUNNING, ExecutionAttemptStatus.RUNNING),
    ],
)
def test_recover_run_detects_abandoned_remote_attempt(
    session_factory,
    tmp_path,
    shot_status,
    attempt_status,
):
    """
    P0-4 requirement: Crash/restart recovery (recover_run) detects abandoned PENDING/RUNNING
    ShotExecution with a submitted remote attempt and durable receipt, transitions it
    to NEEDS_RECOVERY, and resolves the existing provider job without resubmitting.
    """
    with session_factory() as session:
        shot, rev = _create_test_shot_and_revision(session, "shot_abn_01", "rev_abn_01")
        exec_repo = ExecutionRepository(session)

        run = ExecutionRun(
            execution_run_id="run_abn_01",
            asset_route_plan_id="arp_01",
            storyboard_snapshot_id="sb_snap_01",
            total_shots=1,
            status=ExecutionStatus.RUNNING,
        )
        exec_repo.add_execution_run(run)

        cand = AssetRouteCandidate(
            capability_id="cap_abn_01",
            provider="async_provider",
            model="default",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            requested_visual_type=VisualType.STOCK_VIDEO,
            is_eligible=True,
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
        # Abandoned state may occur before acceptance persistence or during polling.
        shot_exec = ShotExecution(
            shot_execution_id="se_abn_01",
            execution_run_id="run_abn_01",
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            route_decision=dec,
            route_candidate_capability_ids=(cand.capability_id,),
            status=shot_status,
            attempt_ids=("att_abn_01",),
        )
        exec_repo.add_shot_execution(shot_exec)

        attempt = ExecutionAttempt(
            execution_attempt_id="att_abn_01",
            execution_run_id="run_abn_01",
            attempt_number=1,
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            provider="async_provider",
            model="default",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            status=attempt_status,
        )
        exec_repo.add_execution_attempt(attempt)

        att_req = AttemptRequest(
            execution_attempt_id="att_abn_01",
            idempotency_key="idemp_abn_123",
            request_hash="test_hash_abn",
            provider="async_provider",
            model="default",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            payload={"prompt": "test"},
        )
        exec_repo.save_attempt_request(att_req)

        receipt = ProviderReceipt(
            execution_attempt_id="att_abn_01",
            provider="async_provider",
            provider_job_id="job_remote_resolved_456",
            provider_status="submitted",
            sanitized_metadata={"id": "job_remote_resolved_456"},
        )
        exec_repo.save_provider_receipt(receipt)
        session.commit()

    # Mock adapter that resolves this remote job successfully
    mock_adapter = MagicMock()
    mock_adapter.get_capabilities.return_value = MagicMock(
        supports_async_status=True,
        supports_safe_resubmission_with_same_key=False,
    )
    # Create fake media file
    fake_video = tmp_path / "recovered_video.mp4"
    fake_video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100)

    mock_adapter.get_status.return_value = MagicMock(
        outcome_type=ProviderOutcomeType.SUCCESS,
        status="succeeded",
        file_path=str(fake_video),
        raw_response={"status": "completed"},
    )

    mock_registry = MagicMock()
    mock_registry.get_adapter.return_value = mock_adapter

    recovery_service = AttemptRecoveryService(adapter_registry=mock_registry)

    with session_factory() as session:
        result = recovery_service.recover_run(
            run_id="run_abn_01",
            session=session,
            storage_base_dir=str(tmp_path),
        )

        assert result["resolved_count"] == 1
        assert result["overall_status"] == ExecutionStatus.COMPLETED.value

        # Verify DB states
        exec_repo = ExecutionRepository(session)
        updated_exec = exec_repo.get_shot_execution("se_abn_01")
        assert updated_exec.status == ShotExecutionStatus.SUCCEEDED

        updated_attempt = exec_repo.get_execution_attempt("att_abn_01")
        assert updated_attempt.status == ExecutionAttemptStatus.SUCCEEDED


def test_same_key_resubmission_reuses_existing_attempt_request_idempotency_key(session_factory, tmp_path):
    """
    P0-4 requirement: If provider supports safe same-key resubmission, ShotExecutionService
    reuses the exact same AttemptRequest.idempotency_key instead of generating a new one.
    """
    with session_factory() as session:
        shot, rev = _create_test_shot_and_revision(session, "shot_same_01", "rev_same_01")
        exec_repo = ExecutionRepository(session)

        # Existing attempt and request with established idempotency key
        attempt = ExecutionAttempt(
            execution_attempt_id="att_same_01",
            execution_run_id="run_same_01",
            attempt_number=1,
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            provider="safe_provider",
            model="default",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            status=ExecutionAttemptStatus.SUBMITTING,
        )
        exec_repo.add_execution_attempt(attempt)

        original_idemp_key = "idemp_safe_key_prior_crash_789"
        att_req = AttemptRequest(
            execution_attempt_id="att_same_01",
            idempotency_key=original_idemp_key,
            request_hash="test_hash_same",
            provider="safe_provider",
            model="default",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            payload={"prompt": "prompt"},
        )
        exec_repo.save_attempt_request(att_req)

        cand = AssetRouteCandidate(
            capability_id="cap_same_01",
            provider="safe_provider",
            model="default",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            requested_visual_type=VisualType.STOCK_VIDEO,
            is_eligible=True,
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
        # ShotExecution referencing the prior attempt
        shot_exec = ShotExecution(
            shot_execution_id="se_same_01",
            execution_run_id="run_same_01",
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            route_decision=dec,
            route_candidate_capability_ids=(cand.capability_id,),
            status=ShotExecutionStatus.NEEDS_RECOVERY,
            attempt_ids=("att_same_01",),
        )
        exec_repo.add_shot_execution(shot_exec)
        session.commit()

    req = create_routing_request_from_shot_revision(rev)

    submitted_idempotency_keys = []
    mock_adapter = MagicMock()
    mock_adapter.get_capabilities.return_value = MagicMock(
        supports_safe_resubmission_with_same_key=True,
    )

    fake_video = tmp_path / "same_key_video.mp4"
    fake_video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100)

    def mock_submit(req, cand, storage_path, idempotency_key=None):
        submitted_idempotency_keys.append(idempotency_key)
        return AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            remote_task_id="remote_same_key_task",
            status="succeeded",
            file_path=str(fake_video),
            raw_response={},
        )

    mock_adapter.submit.side_effect = mock_submit
    mock_adapter.get_status.return_value = AdapterExecutionResult(
        outcome_type=ProviderOutcomeType.SUCCESS,
        remote_task_id="remote_same_key_task",
        status="running",
        file_path=None,
    )

    mock_registry = MagicMock()
    mock_registry.get_adapter.return_value = mock_adapter

    service = ShotExecutionService(
        adapter_registry=mock_registry, session_factory=session_factory
    )

    service.execute_shot(
        shot_execution=shot_exec,
        request=req,
        storage_base_dir=str(tmp_path),
    )

    # Verify adapter.submit received the EXACT original idempotency key
    assert len(submitted_idempotency_keys) == 1
    assert submitted_idempotency_keys[0] == original_idemp_key


def test_route_plan_execution_completes_safe_same_key_recovery(session_factory, tmp_path):
    """Plan orchestration must act on SAFE_TO_RESUBMIT instead of remaining stuck."""
    with session_factory() as session:
        shot, rev = _create_test_shot_and_revision(
            session, "shot_plan_recovery", "rev_plan_recovery"
        )
        candidate = AssetRouteCandidate(
            capability_id="cap_plan_recovery",
            provider="safe_provider",
            model="default",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            requested_visual_type=VisualType.STOCK_VIDEO,
            is_eligible=True,
        )
        decision = AssetRouteDecision(
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            requested_visual_type=rev.visual_type,
            routing_strategy=RoutingStrategy.BALANCED,
            selected_candidate=candidate,
            eligible_candidates=(candidate,),
            rejected_candidates=(),
        )
        request = create_routing_request_from_shot_revision(rev)
        entry = ShotRoutePlanEntry(
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            beat_lineage_id=rev.beat_lineage_id,
            requested_visual_type=rev.visual_type,
            asset_routing_request=request,
            route_decision=decision,
            route_status=ShotRoutePlanStatus.ROUTED,
        )
        plan = AssetRoutePlan(
            asset_route_plan_id="plan_safe_recovery",
            storyboard_snapshot_id="snapshot_safe_recovery",
            content_plan_revision_id="content_safe_recovery",
            routing_strategy=RoutingStrategy.BALANCED,
            routing_policy_version="v1",
            model_selection_mode=ModelSelectionMode.AUTO,
            shot_routes=(entry,),
            status=AssetRoutePlanStatus.READY,
            total_shots=1,
            routed_shots=1,
        )
        repo = ExecutionRepository(session)
        repo.add_execution_run(
            ExecutionRun(
                execution_run_id="run_safe_recovery",
                asset_route_plan_id=plan.asset_route_plan_id,
                storyboard_snapshot_id=plan.storyboard_snapshot_id,
                status=ExecutionStatus.NEEDS_RECOVERY,
                total_shots=1,
                recovery_required_shots=1,
            )
        )
        attempt = ExecutionAttempt(
            execution_attempt_id="attempt_safe_recovery",
            execution_run_id="run_safe_recovery",
            attempt_number=1,
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            provider="safe_provider",
            model="default",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            status=ExecutionAttemptStatus.SUBMITTING,
        )
        repo.add_execution_attempt(attempt)
        repo.save_attempt_request(
            AttemptRequest(
                execution_attempt_id=attempt.execution_attempt_id,
                idempotency_key="original-safe-key",
                request_hash="safe-recovery-hash",
                provider="safe_provider",
                model="default",
                generation_mode=GenerationMode.TEXT_TO_IMAGE,
                payload={"prompt": "test"},
            )
        )
        repo.add_shot_execution(
            ShotExecution(
                shot_execution_id="shot_exec_safe_recovery",
                execution_run_id="run_safe_recovery",
                shot_id=shot.shot_id,
                shot_revision_id=rev.shot_revision_id,
                route_decision=decision,
                route_candidate_capability_ids=(candidate.capability_id,),
                status=ShotExecutionStatus.NEEDS_RECOVERY,
                attempt_ids=(attempt.execution_attempt_id,),
            )
        )
        session.commit()

    submitted_keys = []
    output_file = tmp_path / "safe-recovered.mp4"
    output_file.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100)
    adapter = MagicMock()
    adapter.get_capabilities.return_value = MagicMock(
        supports_async_status=False,
        supports_safe_resubmission_with_same_key=True,
    )
    adapter.submit.side_effect = lambda *_args, **kwargs: (
        submitted_keys.append(kwargs["idempotency_key"])
        or AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            status="succeeded",
            file_path=str(output_file),
            raw_response={},
        )
    )
    registry = MagicMock()
    registry.has_adapter.return_value = True
    registry.get_adapter.return_value = adapter
    shot_service = ShotExecutionService(
        adapter_registry=registry, session_factory=session_factory
    )

    result = AssetRoutePlanExecutionService(
        session_factory=session_factory,
        shot_execution_service=shot_service,
    ).execute_route_plan(plan=plan, storage_base_dir=tmp_path)

    assert result.state == ExecutionStatus.COMPLETED
    assert submitted_keys == ["original-safe-key"]


def test_unsafe_resubmission_provider_halts_without_calling_provider(session_factory, tmp_path):
    """
    P0-4 requirement: If provider does NOT support safe same-key resubmission and remote status
    is unknown, execute_shot halts in NEEDS_RECOVERY without calling provider or charging twice.
    """
    with session_factory() as session:
        shot, rev = _create_test_shot_and_revision(session, "shot_unsafe_01", "rev_unsafe_01")
        exec_repo = ExecutionRepository(session)

        # Existing attempt in NEEDS_RECOVERY / SUBMITTING state
        attempt = ExecutionAttempt(
            execution_attempt_id="att_unsafe_01",
            execution_run_id="run_unsafe_01",
            attempt_number=1,
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            provider="unsafe_provider",
            model="default",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            status=ExecutionAttemptStatus.SUBMITTING,
        )
        exec_repo.add_execution_attempt(attempt)

        att_req = AttemptRequest(
            execution_attempt_id="att_unsafe_01",
            idempotency_key="idemp_unsafe_key",
            request_hash="test_hash_unsafe",
            provider="unsafe_provider",
            model="default",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            payload={"prompt": "prompt"},
        )
        exec_repo.save_attempt_request(att_req)

        cand = AssetRouteCandidate(
            capability_id="cap_unsafe_01",
            provider="unsafe_provider",
            model="default",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            requested_visual_type=VisualType.STOCK_VIDEO,
            is_eligible=True,
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
            shot_execution_id="se_unsafe_01",
            execution_run_id="run_unsafe_01",
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            route_decision=dec,
            route_candidate_capability_ids=(cand.capability_id,),
            status=ShotExecutionStatus.PENDING,
            attempt_ids=("att_unsafe_01",),
        )
        exec_repo.add_shot_execution(shot_exec)
        session.commit()

    req = create_routing_request_from_shot_revision(rev)

    mock_adapter = MagicMock()
    # Provider does NOT support safe resubmission with same key
    mock_adapter.get_capabilities.return_value = MagicMock(
        supports_safe_resubmission_with_same_key=False,
    )

    mock_registry = MagicMock()
    mock_registry.get_adapter.return_value = mock_adapter

    service = ShotExecutionService(
        adapter_registry=mock_registry, session_factory=session_factory
    )

    res_exec, _attempts, _asset = service.execute_shot(
        shot_execution=shot_exec,
        request=req,
        storage_base_dir=str(tmp_path),
    )

    # Must halt in NEEDS_RECOVERY
    assert res_exec.status == ShotExecutionStatus.NEEDS_RECOVERY
    # Adapter submit must NOT have been called!
    mock_adapter.submit.assert_not_called()


def test_blind_resubmission_forbidden_when_needs_recovery(session_factory, tmp_path):
    """
    P0-4 requirement: If ShotExecution is already marked NEEDS_RECOVERY, execute_shot
    strictly raises BlindSubmissionForbiddenError and never calls external adapter.
    """
    from app.domain.asset_execution import BlindSubmissionForbiddenError

    with session_factory() as session:
        shot, rev = _create_test_shot_and_revision(session, "shot_blind_01", "rev_blind_01")

    cand = AssetRouteCandidate(
        capability_id="cap_blind_01",
        provider="any_provider",
        model="default",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
        requested_visual_type=VisualType.STOCK_VIDEO,
        is_eligible=True,
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
        shot_execution_id="se_blind_01",
        execution_run_id="run_blind_01",
        shot_id=shot.shot_id,
        shot_revision_id=rev.shot_revision_id,
        route_decision=dec,
        route_candidate_capability_ids=(cand.capability_id,),
        status=ShotExecutionStatus.NEEDS_RECOVERY,
    )
    req = create_routing_request_from_shot_revision(rev)

    mock_adapter = MagicMock()
    mock_registry = MagicMock()
    mock_registry.get_adapter.return_value = mock_adapter

    service = ShotExecutionService(
        adapter_registry=mock_registry, session_factory=session_factory
    )

    with pytest.raises(BlindSubmissionForbiddenError):
        service.execute_shot(
            shot_execution=shot_exec,
            request=req,
            storage_base_dir=str(tmp_path),
        )

    mock_adapter.submit.assert_not_called()
