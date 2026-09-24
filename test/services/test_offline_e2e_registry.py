"""
Unit and integration tests for Offline E2E Registry and StageWorker offline mode.

Verifies:
1. Default registry maintains production semantics across all 10 stages.
2. Offline registry registers all 10 real production StageExecutors.
3. Offline callers for KNOWLEDGE_PLAN, SCRIPT, and STORYBOARD produce valid deterministic output.
4. QUALITY_REVIEW executor in offline registry uses DeterministicOfflineEvaluatorAdapter.
5. StageWorker detects MPT_OFFLINE_E2E environment variable and selects offline registry when set.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.application.quality_review_stage_executor import QualityReviewStageExecutor
from app.application.stage_executor_registry import (
    StageExecutorRegistry,
    get_default_executor_registry,
    get_offline_e2e_executor_registry,
)
from app.domain.workflow_state import Stage
from app.services.benchmark.offline_adapters import (
    DeterministicOfflineEvaluatorAdapter,
    create_offline_e2e_planner_caller,
    create_offline_e2e_script_caller,
    create_offline_e2e_storyboard_caller,
)
from app.workers.stage_worker import StageWorker, is_offline_e2e_mode


ALL_TEN_STAGES = {
    Stage.EVIDENCE,
    Stage.KNOWLEDGE_PLAN,
    Stage.SCRIPT,
    Stage.STORYBOARD,
    Stage.PRODUCTION_PLAN,
    Stage.ASSET,
    Stage.AUDIO,
    Stage.COMPOSITION,
    Stage.QUALITY_REVIEW,
    Stage.DELIVERY,
}


def test_default_executor_registry_production_contract():
    """Default registry must support all 10 stages with production executors."""
    registry = get_default_executor_registry()
    assert isinstance(registry, StageExecutorRegistry)
    supported = registry.list_supported_stages()
    assert supported == ALL_TEN_STAGES

    # QualityReview default executor must not use offline evaluator adapter
    qr_executor = registry.get_executor(Stage.QUALITY_REVIEW)
    assert isinstance(qr_executor, QualityReviewStageExecutor)
    assert qr_executor._evaluator_adapter is None or not isinstance(
        qr_executor._evaluator_adapter, DeterministicOfflineEvaluatorAdapter
    )


def test_offline_e2e_executor_registry_contract():
    """Offline E2E registry must support all 10 stages and wire offline adapters."""
    registry = get_offline_e2e_executor_registry()
    assert isinstance(registry, StageExecutorRegistry)
    supported = registry.list_supported_stages()
    assert supported == ALL_TEN_STAGES

    # QualityReview executor must use DeterministicOfflineEvaluatorAdapter
    qr_executor = registry.get_executor(Stage.QUALITY_REVIEW)
    assert isinstance(qr_executor, QualityReviewStageExecutor)
    assert isinstance(qr_executor._evaluator_adapter, DeterministicOfflineEvaluatorAdapter)


def test_offline_e2e_planner_caller():
    caller = create_offline_e2e_planner_caller()
    sample_prompt = (
        "Topic: Python Programming\n"
        "Target Video Duration: 20.0\n"
        "Evidence: [ev_12345] Python was created by Guido van Rossum.\n"
    )
    res_str = caller(sample_prompt)
    data = json.loads(res_str)
    assert "beats" in data
    assert len(data["beats"]) > 0
    assert data["beats"][0]["intent"]
    assert "ev_12345" in data["beats"][0]["evidence_refs"]


def test_offline_e2e_script_caller():
    caller = create_offline_e2e_script_caller()
    sample_prompt = (
        "Draft script for content plan:\n"
        "- Beat 1: content_beat_id: 'beat-1', target_duration: 10.0, evidence: [ev_1]\n"
        "- Beat 2: content_beat_id: 'beat-2', target_duration: 10.0, evidence: [ev_2]\n"
    )
    res_str = caller(sample_prompt)
    data = json.loads(res_str)
    assert "segments" in data
    assert len(data["segments"]) == 2
    assert data["segments"][0]["content_beat_id"] == "beat-1"
    assert data["segments"][0]["narration_text"]
    assert data["segments"][1]["content_beat_id"] == "beat-2"


def test_offline_e2e_storyboard_caller():
    caller = create_offline_e2e_storyboard_caller()
    sample_prompt = (
        '## Beat Context:\n"beat_ref": "beat-1"\n'
        "Target Duration: 10.0\n"
        'Allowed Evidence IDs: ["ev_1"]\n'
        '## Authoritative Script Narration: "Python is an interpreted high-level language."\n'
    )
    res_str = caller(sample_prompt)
    data = json.loads(res_str)
    assert data["beat_ref"] == "beat-1"
    assert len(data["shots"]) == 1
    shot = data["shots"][0]
    assert shot["visual_type"] == "AI_IMAGE"
    assert shot["evidence_refs"] == ["ev_1"]
    assert shot["narration"] == "Python is an interpreted high-level language."


def test_stage_worker_mode_selection():
    engine = create_engine("sqlite:///:memory:")
    session_factory = sessionmaker(bind=engine)

    # 1. Unset MPT_OFFLINE_E2E -> production mode
    with patch.dict(os.environ, {}, clear=True):
        assert not is_offline_e2e_mode()
        worker = StageWorker(session_factory=session_factory)
        qr_exec = worker.registry.get_executor(Stage.QUALITY_REVIEW)
        assert not isinstance(qr_exec._evaluator_adapter, DeterministicOfflineEvaluatorAdapter)

    # 2. MPT_OFFLINE_E2E=1 -> offline E2E mode
    with patch.dict(os.environ, {"MPT_OFFLINE_E2E": "1"}):
        assert is_offline_e2e_mode()
        worker = StageWorker(session_factory=session_factory)
        qr_exec = worker.registry.get_executor(Stage.QUALITY_REVIEW)
        assert isinstance(qr_exec._evaluator_adapter, DeterministicOfflineEvaluatorAdapter)


def test_offline_e2e_full_workflow_all_ten_stages():
    """
    Executes a real end-to-end task through all 10 stages using offline E2E mode:
    EVIDENCE -> KNOWLEDGE_PLAN -> SCRIPT -> STORYBOARD -> PRODUCTION_PLAN ->
    ASSET -> AUDIO -> COMPOSITION -> QUALITY_REVIEW -> DELIVERY -> COMPLETED.
    """
    from sqlalchemy.pool import StaticPool
    from app.application.task_command_service import TaskCommandService
    from app.domain.evidence import SourceType
    from app.domain.workflow_state import TaskStatus, WorkflowPolicyType
    from app.persistence.models import Base
    from app.persistence.repositories import (
        KnowledgeVideoTaskRepository,
        StageExecutionRepository,
        TaskArtifactRepository,
    )
    from app.controllers.v1.knowledge_video import InitialEvidenceItemRequest

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    task_id = "test_offline_e2e_task_1"
    evidence_text = (
        "Python is a high-level, general-purpose programming language. "
        "Its design philosophy emphasizes code readability with the use of significant indentation. "
        "Python is dynamically typed and garbage-collected. "
        "It supports multiple programming paradigms, including structured, object-oriented and functional programming. "
        "Created by Guido van Rossum and first released in 1991."
    )

    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        task = cmd_service.create_task(
            task_id=task_id,
            topic="Python Programming Language History and Features",
            target_duration=15.0,
            aspect_ratio="16:9",
            language="zh",
            workflow_policy=WorkflowPolicyType.AUTO,
            task_metadata={"voice_name": "none"},
            allow_research=False,
            initial_evidence=[
                InitialEvidenceItemRequest(
                    source_type=SourceType.TEXT,
                    text_content=evidence_text,
                    title="Python Reference",
                )
            ],
        )
        session.commit()

    with patch.dict(os.environ, {"MPT_OFFLINE_E2E": "1"}):
        worker = StageWorker(
            session_factory=session_factory,
            worker_id="test-offline-e2e-worker",
        )

        max_iterations = 30
        iterations = 0
        while iterations < max_iterations:
            processed = worker.run_once()
            iterations += 1
            if not processed:
                break

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        final_task = task_repo.get_task(task_id)
        assert final_task is not None
        assert final_task.task_status == TaskStatus.COMPLETED

        exec_repo = StageExecutionRepository(session)
        executions = exec_repo.list_executions_for_task(task_id)
        executed_stages = [ex.stage for ex in executions]

        for st in ALL_TEN_STAGES:
            assert st in executed_stages, f"Stage {st.value} was not executed!"

        art_repo = TaskArtifactRepository(session)
        artifacts = art_repo.list_artifact_refs_for_task(task_id)
        artifact_stages = {art.stage for art in artifacts}
        for st in ALL_TEN_STAGES:
            assert st in artifact_stages, f"Stage {st.value} produced no artifact!"

