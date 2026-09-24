from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.domain.asset_execution import (
    AssetMediaType,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRun,
    ShotAssetVersion,
)
from app.domain.enums import VisualType
from app.domain.evaluation import (
    DimensionEvaluationStatus,
    EvaluationDecision,
    EvaluationDimension,
    EvaluationReasonCode,
    create_evaluation_target_from_shot,
)
from app.domain.evaluation_policy import EvaluationPolicy
from app.domain.shot import Shot, ShotRevision
from app.persistence.models import Base
from app.persistence.repositories import (
    EvaluationRepository,
    ExecutionRepository,
    ShotRepository,
)
from app.services.evaluation.mpt_multimodal_adapter import (
    MultimodalEvaluationObserverResult,
    MultimodalEvaluatorAdapter,
)
from app.services.evaluation.shot_asset_evaluation_service import (
    ShotAssetEvaluationService,
)


def _setup_test_db_and_data(tmp_path: Path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    shot_repo = ShotRepository(session)
    exec_repo = ExecutionRepository(session)
    eval_repo = EvaluationRepository(session)

    # Create Shot and ShotRevision
    shot = Shot(shot_id="shot_1", beat_lineage_id="lin_1", local_order=1)
    shot_repo.add_shot(shot)

    shot_rev = ShotRevision(
        shot_revision_id="rev_1",
        shot_id="shot_1",
        revision_number=1,
        beat_lineage_id="lin_1",
        created_from_beat_instance_id="b_1",
        target_duration=3.0,
        narration="Photosynthesis occurs in plant chloroplasts.",
        visual_goal="Plant leaf cell absorbing sunlight",
        visual_type=VisualType.AI_IMAGE,
        scene_description="Macro shot of vibrant chloroplasts",
        generation_prompt="microscopic chloroplasts in plant cells",
        camera_movement="Static",
        evidence_refs=("Chloroplasts absorb photons to produce energy",),
    )
    shot_repo.add_revision(shot_rev)

    # Real test image file on disk
    img_file = tmp_path / "asset_image.png"
    Image.new("RGB", (800, 600), "green").save(str(img_file))

    # Execution run and attempt
    exec_run = ExecutionRun(
        execution_run_id="run_1",
        asset_route_plan_id="plan_1",
        storyboard_snapshot_id="snap_1",
        total_shots=1,
    )
    exec_repo.add_execution_run(exec_run)

    attempt = ExecutionAttempt(
        execution_attempt_id="att_1",
        execution_run_id="run_1",
        shot_id="shot_1",
        shot_revision_id="rev_1",
        attempt_number=1,
        provider="dummy",
        model="dummy-model",
        generation_mode="TEXT_TO_IMAGE",
        status=ExecutionAttemptStatus.SUCCEEDED,
    )
    exec_repo.add_execution_attempt(attempt)

    from app.services.asset_probe_service import calculate_sha256

    asset_ver = ShotAssetVersion(
        shot_asset_version_id="asset_v1",
        shot_id="shot_1",
        shot_revision_id="rev_1",
        execution_attempt_id="att_1",
        file_path=str(img_file),
        file_hash=calculate_sha256(img_file),
        file_size_bytes=os.path.getsize(str(img_file)),
        media_type=AssetMediaType.IMAGE,
        mime_type="image/png",
        width=800,
        height=600,
        duration_seconds=0.0,
        provider="dummy",
        model="dummy-model",
        generation_mode="TEXT_TO_IMAGE",
    )
    exec_repo.add_shot_asset_version(asset_ver)

    target = create_evaluation_target_from_shot(shot_rev, asset_ver)
    eval_repo.save_target(target)
    session.commit()

    return session, target, shot_rev, asset_ver


def test_full_evaluation_pipeline_success(tmp_path: Path):
    session, target, _, _ = _setup_test_db_and_data(tmp_path)

    # Mock adapter returning high scores
    mock_adapter = MagicMock(spec=MultimodalEvaluatorAdapter)
    mock_adapter.evaluator_version = "mock-provider/v1:evaluator-v1"
    mock_adapter.evaluate_observation.return_value = MultimodalEvaluationObserverResult(
        status=DimensionEvaluationStatus.SCORED,
        score=0.90,
        reason_codes=("HIGH_QUALITY",),
        concise_summary="Excellent alignment and composition.",
        evaluator_version="mock-provider/v1:evaluator-v1",
    )

    service = ShotAssetEvaluationService(session)
    snapshot = service.evaluate_target(
        evaluation_target_id=target.evaluation_target_id,
        evaluator_adapter=mock_adapter,
    )

    assert snapshot.decision == EvaluationDecision.PASS
    assert snapshot.overall_score is not None
    assert snapshot.overall_score >= 0.85
    assert len(snapshot.dimension_result_ids) == 4
    assert mock_adapter.evaluate_observation.call_count == 4


def test_partial_reuse_and_zero_calls_on_policy_change(tmp_path: Path):
    session, target, _, _ = _setup_test_db_and_data(tmp_path)

    mock_adapter = MagicMock(spec=MultimodalEvaluatorAdapter)
    mock_adapter.evaluator_version = "mock-provider/v1:evaluator-v1"
    mock_adapter.evaluate_observation.return_value = MultimodalEvaluationObserverResult(
        status=DimensionEvaluationStatus.SCORED,
        score=0.85,
        reason_codes=("GOOD",),
        concise_summary="Good result.",
        evaluator_version="mock-provider/v1:evaluator-v1",
    )

    service = ShotAssetEvaluationService(session)
    # First run evaluates all 4 dimensions
    snap1 = service.evaluate_target(
        evaluation_target_id=target.evaluation_target_id,
        evaluator_adapter=mock_adapter,
    )
    assert mock_adapter.evaluate_observation.call_count == 4
    assert snap1.decision == EvaluationDecision.PASS

    # Second run with different policy thresholds (e.g. stricter minimum overall score 0.95 -> FAIL)
    mock_adapter.reset_mock()
    stricter_policy = EvaluationPolicy(
        policy_version="evaluation-policy-v2-strict",
        minimum_overall_score=0.95,
    )

    snap2 = service.evaluate_target(
        evaluation_target_id=target.evaluation_target_id,
        policy=stricter_policy,
        evaluator_adapter=mock_adapter,
    )

    # Crucial: 0 model calls made! All 4 dimensions reused from DB!
    assert mock_adapter.evaluate_observation.call_count == 0
    assert snap2.decision == EvaluationDecision.FAIL
    assert snap2.policy_version == "evaluation-policy-v2-strict"
    # Historical snap1 is completely preserved
    assert snap1.evaluation_snapshot_id != snap2.evaluation_snapshot_id
    assert snap1.decision == EvaluationDecision.PASS


def test_unreadable_media_yields_indeterminate(tmp_path: Path):
    session, _target, _, _asset_ver = _setup_test_db_and_data(tmp_path)

    # Point asset to non-existent file
    exec_repo = ExecutionRepository(session)
    broken_asset = ShotAssetVersion(
        shot_asset_version_id="asset_broken",
        shot_id="shot_1",
        shot_revision_id="rev_1",
        execution_attempt_id="att_1",
        file_path="/non_existent/broken_media.png",
        file_hash="b" * 64,
        file_size_bytes=100,
        media_type=AssetMediaType.IMAGE,
        mime_type="image/png",
        width=800,
        height=600,
        duration_seconds=0.0,
        provider="dummy",
        model="dummy-model",
        generation_mode="TEXT_TO_IMAGE",
    )
    exec_repo.add_shot_asset_version(broken_asset)

    rev_broken = ShotRevision(
        shot_revision_id="rev_broken",
        shot_id="shot_1",
        revision_number=2,
        beat_lineage_id="lin_1",
        created_from_beat_instance_id="b_1",
        target_duration=3.0,
        narration="test",
        visual_goal="test",
        visual_type=VisualType.AI_IMAGE,
        scene_description="test",
        generation_prompt="test",
        camera_movement="Static",
        evidence_refs=(),
    )
    target_broken = create_evaluation_target_from_shot(rev_broken, broken_asset)
    eval_repo = EvaluationRepository(session)
    shot_repo = ShotRepository(session)
    shot_repo.add_revision(rev_broken)
    eval_repo.save_target(target_broken)
    session.commit()

    service = ShotAssetEvaluationService(session)
    mock_adapter = MagicMock(spec=MultimodalEvaluatorAdapter)
    mock_adapter.evaluator_version = "mock/v1:evaluator-v1"

    snap = service.evaluate_target(
        evaluation_target_id=target_broken.evaluation_target_id,
        evaluator_adapter=mock_adapter,
    )

    assert snap.decision == EvaluationDecision.INDETERMINATE
    assert snap.overall_score is None
    # Model was not called
    assert mock_adapter.evaluate_observation.call_count == 0


def test_missing_evidence_context_propagates_to_indeterminate(tmp_path: Path):
    session, _, _, asset_ver = _setup_test_db_and_data(tmp_path)

    # Shot revision with NO evidence refs
    shot_repo = ShotRepository(session)
    shot_rev_no_ev = ShotRevision(
        shot_revision_id="rev_no_ev",
        shot_id="shot_1",
        revision_number=3,
        beat_lineage_id="lin_1",
        created_from_beat_instance_id="b_1",
        target_duration=3.0,
        narration="Some unverified assertion",
        visual_goal="Visual assertion",
        visual_type=VisualType.AI_IMAGE,
        scene_description="Description",
        generation_prompt="Prompt",
        camera_movement="Static",
        evidence_refs=(),  # EMPTY
    )
    shot_repo.add_revision(shot_rev_no_ev)

    eval_repo = EvaluationRepository(session)
    target_no_ev = eval_repo.save_target(
        create_evaluation_target_from_shot(shot_rev_no_ev, asset_ver)
    )
    session.commit()

    mock_adapter = MagicMock(spec=MultimodalEvaluatorAdapter)
    mock_adapter.evaluator_version = "mock/v1:evaluator-v1"
    mock_adapter.evaluate_observation.return_value = MultimodalEvaluationObserverResult(
        status=DimensionEvaluationStatus.SCORED,
        score=0.95,
        reason_codes=("PERFECT",),
        concise_summary="Perfect score.",
        evaluator_version="mock/v1:evaluator-v1",
    )

    service = ShotAssetEvaluationService(session)
    snap = service.evaluate_target(
        evaluation_target_id=target_no_ev.evaluation_target_id,
        evaluator_adapter=mock_adapter,
    )

    # Knowledge dimension is INDETERMINATE -> Rule 1 triggers -> Snapshot is INDETERMINATE
    assert snap.decision == EvaluationDecision.INDETERMINATE
    assert snap.overall_score is None
    assert "KNOWLEDGE_ACCURACY_INDETERMINATE" in snap.summary_reason_codes

    # Check that the underlying knowledge dimension result has MISSING_EVIDENCE_CONTEXT
    results = eval_repo.get_snapshot_dimension_results(snap.evaluation_snapshot_id)
    knowledge_res = next(r for r in results if r.dimension == EvaluationDimension.KNOWLEDGE_ACCURACY)
    assert knowledge_res.status == DimensionEvaluationStatus.INDETERMINATE
    assert EvaluationReasonCode.MISSING_EVIDENCE_CONTEXT.value in knowledge_res.reason_codes


def test_fail_decision_has_zero_phase6_3_side_effects(tmp_path: Path):
    """
    Architectural boundary verification:
    When evaluation yields FAIL, the service persists the EvaluationSnapshot
    and strictly terminates. It NEVER triggers asset regeneration, provider fallback,
    attempt retry, or storyboard replanning.
    """
    session, target, shot_rev, _asset_ver = _setup_test_db_and_data(tmp_path)
    exec_repo = ExecutionRepository(session)

    # Initial state
    initial_assets_count = len(exec_repo.list_asset_versions_for_shot_revision(shot_rev.shot_revision_id))
    initial_attempts_count = len(exec_repo.list_attempts_for_run("run_1"))

    # Mock adapter producing FAIL scores (semantic alignment 0.30 < 0.80 threshold)
    mock_adapter = MagicMock(spec=MultimodalEvaluatorAdapter)
    mock_adapter.evaluator_version = "mock/v1:evaluator-v1"
    mock_adapter.evaluate_observation.return_value = MultimodalEvaluationObserverResult(
        status=DimensionEvaluationStatus.SCORED,
        score=0.30,
        reason_codes=("POOR_ALIGNMENT",),
        concise_summary="Completely mismatched subject.",
        evaluator_version="mock/v1:evaluator-v1",
    )

    service = ShotAssetEvaluationService(session)
    snap = service.evaluate_target(
        evaluation_target_id=target.evaluation_target_id,
        evaluator_adapter=mock_adapter,
    )

    # Policy decision is FAIL
    assert snap.decision == EvaluationDecision.FAIL
    assert "SEMANTIC_ALIGNMENT_BELOW_THRESHOLD" in snap.summary_reason_codes

    # Strict invariant: zero regeneration, zero new attempts, zero new assets
    final_assets_count = len(exec_repo.list_asset_versions_for_shot_revision(shot_rev.shot_revision_id))
    final_attempts_count = len(exec_repo.list_attempts_for_run("run_1"))
    assert initial_assets_count == final_assets_count
    assert initial_attempts_count == final_attempts_count

