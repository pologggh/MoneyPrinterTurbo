from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.evidence_service import TaskEvidenceCommandService
from app.application.evidence_stage_executor import EvidenceStageExecutor
from app.application.knowledge_plan_stage_executor import KnowledgePlanStageExecutor
from app.application.knowledge_video_workflow import KnowledgeVideoWorkflow
from app.application.stage_executor_protocol import StageExecutorProtocol
from app.application.stage_executor_registry import (
    StageExecutorRegistry,
    get_default_executor_registry,
)
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType
from app.domain.evidence import (
    EvidenceItem,
    EvidenceSnapshot,
    SourceDocument,
    SourceType,
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.planner import ContentPlanner, PlannerInput
from app.domain.task_artifact import ArtifactType, TaskArtifactRef
from app.domain.trace import TraceEventType
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    JobErrorType,
    JobStatus,
    Stage,
    TaskStatus,
    WorkflowPolicyType,
)
from app.persistence.models import (
    Base,
    ContentBeatORM,
    ContentPlanRevisionORM,
    ShotORM,
    StoryboardSnapshotORM,
    TaskArtifactRefORM,
    WorkflowJobORM,
)
from app.persistence.repositories import (
    ContentPlanRepository,
    EvidenceRepository,
    KnowledgeVideoTaskRepository,
    StageExecutionRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)
from app.services.trace_service import TraceWriter
from app.workers.stage_worker import StageWorker


# =============================================================================
# Fixtures & Helpers
# =============================================================================


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


def _make_sample_llm_json(
    title: str = "Self-Attention in Transformers",
    beats: list[dict] | None = None,
    evidence_id: str = "ev_transformer_1",
) -> str:
    if beats is None:
        beats = [
            {
                "planner_ref": "b1",
                "beat_type": "HOOK",
                "intent": "Hook viewer on how AI understands text relationships",
                "order": 1,
                "target_duration": 5.0,
                "importance": 0.9,
                "evidence_refs": [],
                "inherit_from": None,
            },
            {
                "planner_ref": "b2",
                "beat_type": "KNOWLEDGE",
                "intent": "Explain that self-attention builds context-dependent representations",
                "order": 2,
                "target_duration": 20.0,
                "importance": 1.0,
                "evidence_refs": [evidence_id],
                "inherit_from": None,
            },
            {
                "planner_ref": "b3",
                "beat_type": "SUMMARY",
                "intent": "Summarize how attention enables massive parallelism",
                "order": 3,
                "target_duration": 5.0,
                "importance": 0.8,
                "evidence_refs": [],
                "inherit_from": None,
            },
        ]
    return json.dumps({
        "title": title,
        "beats": beats,
        "planner_notes": "Focused plan for 30s knowledge video",
    })


def _setup_task_with_evidence(
    session_factory,
    topic: str = "Why does Transformer use self-attention?",
    evidence_text: str = (
        "Self-attention allows each token to compute relationships with other tokens "
        "and build context-dependent representations."
    ),
    workflow_policy: WorkflowPolicyType = WorkflowPolicyType.AUTO,
    allow_research: bool = False,
) -> tuple[KnowledgeVideoTask, WorkflowJob, EvidenceSnapshot, EvidenceItem, SourceDocument, TaskArtifactRef]:
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        ev_repo = EvidenceRepository(session)
        art_repo = TaskArtifactRepository(session)

        # 1. Create Task
        task = KnowledgeVideoTask.create(
            topic=topic,
            target_duration=30.0,
            workflow_policy=workflow_policy,
            allow_research=allow_research,
        )
        task.transition_to(TaskStatus.RUNNING)
        task.advance_stage(Stage.KNOWLEDGE_PLAN)
        saved_task = task_repo.save_task(task)

        # 2. Create SourceDocument & EvidenceItem
        source_doc = SourceDocument.create_text(
            text=evidence_text,
            title="Attention Is All You Need",
        )
        saved_source = ev_repo.save_source_document(source_doc)
        ev_repo.associate_task_source(saved_task.task_id, saved_source.source_document_id)

        ev_item = EvidenceItem.create(
            source_document=saved_source,
            original_excerpt=evidence_text,
            normalized_fact="Self-attention builds context-dependent token representations.",
        )
        saved_item = ev_repo.save_evidence_item(ev_item)

        # 3. Create EvidenceSnapshot
        snapshot = EvidenceSnapshot.create(
            task_id=saved_task.task_id,
            source_document_ids=[saved_source.source_document_id],
            evidence_ids=[saved_item.evidence_id],
            snapshot_version=1,
        )
        saved_snapshot = ev_repo.save_evidence_snapshot(snapshot)

        # 4. Create TaskArtifactRef for EVIDENCE stage
        art_ref = TaskArtifactRef.create(
            task_id=saved_task.task_id,
            stage=Stage.EVIDENCE,
            artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
            artifact_id=saved_snapshot.evidence_snapshot_id,
            artifact_version="1",
            metadata_json={"retrieval_snapshot_id": f"rs_{uuid4().hex[:16]}"},
        )
        saved_ref = art_repo.save_artifact_ref(art_ref)

        # 5. Create KNOWLEDGE_PLAN WorkflowJob
        job = WorkflowJob.create(
            task_id=saved_task.task_id,
            stage=Stage.KNOWLEDGE_PLAN,
            idempotency_key=f"idemp_{saved_task.task_id}_knowledge_plan_1",
            input_task_artifact_ref_id=saved_ref.task_artifact_ref_id,
        )
        saved_job = job_repo.create_job(job)
        session.commit()

        return saved_task, saved_job, saved_snapshot, saved_item, saved_source, saved_ref


# =============================================================================
# 1. REGISTRY TESTS
# =============================================================================


def test_production_registry_contains_real_knowledge_plan_executor(session_factory):
    registry = get_default_executor_registry(session_factory=session_factory)
    assert registry.has_executor(Stage.KNOWLEDGE_PLAN)
    executor = registry.get_executor(Stage.KNOWLEDGE_PLAN)
    assert executor is not None
    assert isinstance(executor, KnowledgePlanStageExecutor)
    assert isinstance(executor, StageExecutorProtocol)


def test_production_registry_contains_real_evidence_executor(session_factory):
    registry = get_default_executor_registry(session_factory=session_factory)
    assert registry.has_executor(Stage.EVIDENCE)
    executor = registry.get_executor(Stage.EVIDENCE)
    assert executor is not None
    assert isinstance(executor, EvidenceStageExecutor)
    assert isinstance(executor, StageExecutorProtocol)


def test_production_registry_contains_no_fake_executors(session_factory):
    registry = get_default_executor_registry(session_factory=session_factory)
    for stage in registry.list_supported_stages():
        executor = registry.get_executor(stage)
        assert executor is not None
        assert not executor.__class__.__name__.startswith(("Mock", "Fake", "Dummy"))


def test_script_remains_unsupported(session_factory):
    registry = get_default_executor_registry(session_factory=session_factory)
    assert not registry.has_executor(Stage.SCRIPT)
    assert registry.get_executor(Stage.SCRIPT) is None
    assert Stage.SCRIPT not in registry.list_supported_stages()


# =============================================================================
# 2. INPUT RESOLUTION & TASK LINEAGE TESTS
# =============================================================================


def test_executor_resolves_exact_snapshot_through_task_artifact_ref(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    fake_llm = lambda prompt: _make_sample_llm_json(evidence_id=item.evidence_id)
    executor = KnowledgePlanStageExecutor(
        session_factory=session_factory,
        llm_caller=fake_llm,
    )

    result = executor.execute(task, job)
    assert result.success is True
    assert result.output_artifact_ref is not None
    assert result.metadata_json["evidence_snapshot_id"] == snapshot.evidence_snapshot_id


def test_evidence_snapshot_task_id_mismatch_fails(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    # Different task
    other_task = KnowledgeVideoTask.create(topic="Different Topic")

    executor = KnowledgePlanStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: "",
    )

    # Calling with other_task should fail task_id validation
    result = executor.execute(other_task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "task_id mismatch" in result.error_message


def test_missing_input_artifact_fails_explicitly(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    broken_job = WorkflowJob.create(
        task_id=task.task_id,
        stage=Stage.KNOWLEDGE_PLAN,
        idempotency_key="broken_job_key",
        input_task_artifact_ref_id="non_existent_ref_id",
    )

    executor = KnowledgePlanStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: "",
    )

    result = executor.execute(task, broken_job)
    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "not found" in result.error_message


def test_planner_receives_only_evidence_from_frozen_snapshot(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    # Insert an unrelated evidence item into DB
    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        other_src = ev_repo.save_source_document(
            SourceDocument.create_text(text="Unrelated text", title="Other Doc")
        )
        other_item = ev_repo.save_evidence_item(
            EvidenceItem.create(source_document=other_src, original_excerpt="Unrelated fact")
        )
        session.commit()
        unrelated_id = other_item.evidence_id

    captured_prompts = []

    def tracking_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return _make_sample_llm_json(evidence_id=item.evidence_id)

    executor = KnowledgePlanStageExecutor(
        session_factory=session_factory,
        llm_caller=tracking_llm,
    )

    result = executor.execute(task, job)
    assert result.success is True
    assert len(captured_prompts) == 1
    prompt_text = captured_prompts[0]

    # Grounded item is present, unrelated item is strictly absent
    assert item.evidence_id in prompt_text
    assert unrelated_id not in prompt_text


# =============================================================================
# 3. CONTENT PLAN & BEAT INTEGRITY TESTS
# =============================================================================


def test_successful_planner_creates_real_content_plan_revision(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    executor = KnowledgePlanStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: _make_sample_llm_json(evidence_id=item.evidence_id),
    )

    result = executor.execute(task, job)
    assert result.success is True

    plan_id = result.output_artifact_revision_id
    with session_factory() as session:
        plan_repo = ContentPlanRepository(session)
        plan = plan_repo.get_revision(plan_id)
        assert plan is not None
        assert plan.topic == "Self-Attention in Transformers"
        assert plan.revision_number == 1
        assert len(plan.beats) == 3

        # Verify beats in DB
        stmt = select(ContentBeatORM).where(ContentBeatORM.content_plan_revision_id == plan_id)
        beats = session.scalars(stmt).all()
        assert len(beats) == 3
        orders = [b.order for b in beats]
        assert orders == [1, 2, 3]


def test_replan_creates_new_revision_without_mutating_old(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    worker = StageWorker(
        session_factory=session_factory,
        registry=StageExecutorRegistry(),
        worker_id="test-replan-worker",
    )
    exec_kp = KnowledgePlanStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: _make_sample_llm_json(evidence_id=item.evidence_id),
    )
    worker.registry.register(Stage.KNOWLEDGE_PLAN, exec_kp)

    # 1. Run first plan revision
    worker.run_once()

    with session_factory() as session:
        plan_repo = ContentPlanRepository(session)
        art_repo = TaskArtifactRepository(session)
        job_repo = WorkflowJobRepository(session)
        task_repo = KnowledgeVideoTaskRepository(session)

        first_ref = art_repo.get_latest_artifact_ref(task.task_id, stage=Stage.KNOWLEDGE_PLAN)
        assert first_ref is not None
        first_plan = plan_repo.get_revision(first_ref.artifact_id)
        assert first_plan.revision_number == 1

        # Simulate replan job
        current_task = task_repo.get_task(task.task_id)
        current_task.current_stage = Stage.KNOWLEDGE_PLAN
        task_repo.save_task(current_task)

        replan_job = WorkflowJob.create(
            task_id=task.task_id,
            stage=Stage.KNOWLEDGE_PLAN,
            idempotency_key=f"idemp_{task.task_id}_knowledge_plan_2",
            input_task_artifact_ref_id=art_ref.task_artifact_ref_id,
            attempt_number=1,
        )
        job_repo.create_job(replan_job)
        session.commit()

    # 2. Run replan
    worker.run_once()

    with session_factory() as session:
        plan_repo = ContentPlanRepository(session)
        art_repo = TaskArtifactRepository(session)

        second_ref = art_repo.get_latest_artifact_ref(task.task_id, stage=Stage.KNOWLEDGE_PLAN)
        assert second_ref.artifact_id != first_ref.artifact_id

        second_plan = plan_repo.get_revision(second_ref.artifact_id)
        assert second_plan.revision_number == 2
        # First plan remains unchanged
        old_plan = plan_repo.get_revision(first_ref.artifact_id)
        assert old_plan.revision_number == 1


# =============================================================================
# 4. EVIDENCE GROUNDING RULES
# =============================================================================


def test_factual_knowledge_beat_without_evidence_returns_needs_evidence(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    # Propose KNOWLEDGE beat with empty evidence_refs
    unsupported_beats = [
        {
            "planner_ref": "b1",
            "beat_type": "KNOWLEDGE",
            "intent": "Factual claim with no evidence reference",
            "order": 1,
            "target_duration": 30.0,
            "importance": 1.0,
            "evidence_refs": [],
            "inherit_from": None,
        }
    ]
    executor = KnowledgePlanStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: _make_sample_llm_json(beats=unsupported_beats),
    )

    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.NEEDS_EVIDENCE.value
    assert "requires evidence references" in result.error_message or "requires valid evidence references" in result.error_message


def test_invalid_evidence_ref_returns_needs_evidence(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    # Propose KNOWLEDGE beat citing a non-existent / unauthorized evidence ref
    invalid_beats = [
        {
            "planner_ref": "b1",
            "beat_type": "KNOWLEDGE",
            "intent": "Factual claim citing fake reference",
            "order": 1,
            "target_duration": 30.0,
            "importance": 1.0,
            "evidence_refs": ["fabricated_evidence_id_999"],
            "inherit_from": None,
        }
    ]
    executor = KnowledgePlanStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: _make_sample_llm_json(beats=invalid_beats),
    )

    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.NEEDS_EVIDENCE.value
    assert "fabricated_evidence_id_999" in result.error_message


def test_rhetorical_hook_and_transition_allow_empty_evidence(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    valid_mixed_beats = [
        {
            "planner_ref": "b1",
            "beat_type": "HOOK",
            "intent": "Rhetorical question to grab attention",
            "order": 1,
            "target_duration": 5.0,
            "importance": 0.8,
            "evidence_refs": [],
            "inherit_from": None,
        },
        {
            "planner_ref": "b2",
            "beat_type": "KNOWLEDGE",
            "intent": "Grounded core mechanism",
            "order": 2,
            "target_duration": 20.0,
            "importance": 1.0,
            "evidence_refs": [item.evidence_id],
            "inherit_from": None,
        },
        {
            "planner_ref": "b3",
            "beat_type": "TRANSITION",
            "intent": "Smooth rhetorical pivot",
            "order": 3,
            "target_duration": 5.0,
            "importance": 0.5,
            "evidence_refs": [],
            "inherit_from": None,
        },
    ]

    executor = KnowledgePlanStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: _make_sample_llm_json(beats=valid_mixed_beats),
    )

    result = executor.execute(task, job)
    assert result.success is True


# =============================================================================
# 5. ARTIFACT REF & AUDIT TRAIL TESTS
# =============================================================================


def test_successful_execution_creates_task_artifact_ref_and_stage_execution(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    worker = StageWorker(
        session_factory=session_factory,
        registry=StageExecutorRegistry(),
        worker_id="test-worker-artifact",
    )
    executor = KnowledgePlanStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: _make_sample_llm_json(evidence_id=item.evidence_id),
    )
    worker.registry.register(Stage.KNOWLEDGE_PLAN, executor)

    worker.run_once()

    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        exec_repo = StageExecutionRepository(session)
        job_repo = WorkflowJobRepository(session)

        # 1. Output TaskArtifactRef
        plan_ref = art_repo.get_latest_artifact_ref(
            task_id=task.task_id,
            stage=Stage.KNOWLEDGE_PLAN,
            artifact_type=ArtifactType.CONTENT_PLAN_REVISION,
        )
        assert plan_ref is not None
        assert plan_ref.task_id == task.task_id
        assert plan_ref.stage == Stage.KNOWLEDGE_PLAN
        assert plan_ref.artifact_type == ArtifactType.CONTENT_PLAN_REVISION
        assert plan_ref.metadata_json["beat_count"] == 3
        assert plan_ref.metadata_json["evidence_snapshot_id"] == snapshot.evidence_snapshot_id

        # 2. StageExecution
        executions = exec_repo.list_executions_for_task(task.task_id)
        assert len(executions) == 1
        exec_record = executions[0]
        assert exec_record.task_id == task.task_id
        assert exec_record.stage == Stage.KNOWLEDGE_PLAN
        assert exec_record.status == "SUCCEEDED"
        assert exec_record.input_task_artifact_ref_id == art_ref.task_artifact_ref_id
        assert exec_record.output_task_artifact_ref_id == plan_ref.task_artifact_ref_id


# =============================================================================
# 6. WORKFLOW POLICY (AUTO vs REVIEW)
# =============================================================================


def test_auto_policy_automatically_creates_queued_script_job(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(
        session_factory, workflow_policy=WorkflowPolicyType.AUTO
    )

    worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="test-worker-auto",
    )
    # Inject deterministic stub LLM
    exec_kp = worker.registry.get_executor(Stage.KNOWLEDGE_PLAN)
    exec_kp.llm_caller = lambda p: _make_sample_llm_json(evidence_id=item.evidence_id)

    processed = worker.run_once()
    assert processed is True

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        current_task = task_repo.get_task(task.task_id)
        assert current_task.task_status == TaskStatus.RUNNING
        assert current_task.current_stage == Stage.SCRIPT

        # SCRIPT job created automatically
        script_job = job_repo.get_job_by_idempotency_key(f"idemp_{task.task_id}_script_1")
        assert script_job is not None
        assert script_job.stage == Stage.SCRIPT
        assert script_job.status == JobStatus.QUEUED
        assert script_job.lease_owner is None

    # Worker runs again: SCRIPT is unsupported -> remains QUEUED
    second_processed = worker.run_once()
    assert second_processed is False


def test_review_policy_pauses_in_waiting_user_until_approved(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(
        session_factory, workflow_policy=WorkflowPolicyType.REVIEW
    )

    worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="test-worker-review",
    )
    exec_kp = worker.registry.get_executor(Stage.KNOWLEDGE_PLAN)
    exec_kp.llm_caller = lambda p: _make_sample_llm_json(evidence_id=item.evidence_id)

    processed = worker.run_once()
    assert processed is True

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        current_task = task_repo.get_task(task.task_id)
        # Pauses at review checkpoint
        assert current_task.task_status == TaskStatus.WAITING_USER
        assert "Review required after KNOWLEDGE_PLAN" in (current_task.waiting_reason or "")

        # SCRIPT job must NOT exist before approval
        script_job = job_repo.get_job_by_idempotency_key(f"idemp_{task.task_id}_script_1")
        assert script_job is None

        # Approve task
        workflow = KnowledgeVideoWorkflow(session)
        resumed_job = workflow.resume_after_approval(current_task)
        session.commit()

        assert resumed_job is not None
        assert resumed_job.stage == Stage.SCRIPT
        assert resumed_job.status == JobStatus.QUEUED

        # Task resumes to RUNNING at stage SCRIPT
        updated_task = task_repo.get_task(task.task_id)
        assert updated_task.task_status == TaskStatus.RUNNING
        assert updated_task.current_stage == Stage.SCRIPT

    # Worker runs again: SCRIPT is unsupported -> remains QUEUED
    second_processed = worker.run_once()
    assert second_processed is False


# =============================================================================
# 7. NO STORYBOARD OR SHOT GENERATION IN K1
# =============================================================================


def test_k1_creates_no_storyboard_or_shot_revisions(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    executor = KnowledgePlanStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: _make_sample_llm_json(evidence_id=item.evidence_id),
    )

    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        storyboard_count = session.scalars(select(StoryboardSnapshotORM)).all()
        shot_count = session.scalars(select(ShotORM)).all()
        assert len(storyboard_count) == 0, "K1 must NOT create any StoryboardSnapshot."
        assert len(shot_count) == 0, "K1 must NOT create any Shot records."


# =============================================================================
# 8. ERROR & RETRY SEMANTICS
# =============================================================================


def test_temporary_malformed_llm_output_is_retryable(session_factory):
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    executor = KnowledgePlanStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: "MALFORMED_NON_JSON_STRING",
        max_retries=1,
    )

    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.RETRYABLE.value
    assert result.is_retryable is True


# =============================================================================
# 9. FOUR MANDATORY DEMONSTRATION TESTS
# =============================================================================


def test_mandatory_success_integration(session_factory):
    """MANDATORY SUCCESS INTEGRATION TEST

    Topic: "Why does Transformer use self-attention?"
    Evidence text: "Self-attention allows each token to compute relationships with other tokens and build context-dependent representations."
    Flow:
    KnowledgeVideoTask -> KNOWLEDGE_PLAN WorkflowJob -> StageWorker -> KnowledgePlanStageExecutor -> ContentPlanner -> ContentPlanRevision -> ContentBeat
    Verify:
    - Beat.evidence_refs contains the real EvidenceItem ID
    - ContentBeat.evidence_refs -> EvidenceItem -> SourceDocument
    - AUTO policy creates SCRIPT job in QUEUED state
    - SCRIPT job remains queued/unclaimed
    - No StoryboardSnapshot created
    """
    topic = "Why does Transformer use self-attention?"
    evidence_text = (
        "Self-attention allows each token to compute relationships with other tokens "
        "and build context-dependent representations."
    )
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(
        session_factory,
        topic=topic,
        evidence_text=evidence_text,
        workflow_policy=WorkflowPolicyType.AUTO,
    )

    worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="test-mandatory-success-worker",
    )
    exec_kp = worker.registry.get_executor(Stage.KNOWLEDGE_PLAN)
    exec_kp.llm_caller = lambda p: _make_sample_llm_json(
        title="Why Transformers Use Self-Attention",
        evidence_id=item.evidence_id,
    )

    # Execute KNOWLEDGE_PLAN
    processed = worker.run_once()
    assert processed is True, "Worker must process KNOWLEDGE_PLAN successfully."

    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        plan_repo = ContentPlanRepository(session)
        ev_repo = EvidenceRepository(session)
        job_repo = WorkflowJobRepository(session)

        # 1. Output plan ref
        plan_ref = art_repo.get_latest_artifact_ref(task.task_id, stage=Stage.KNOWLEDGE_PLAN)
        assert plan_ref is not None
        assert plan_ref.artifact_type == ArtifactType.CONTENT_PLAN_REVISION

        # 2. ContentPlanRevision
        plan = plan_repo.get_revision(plan_ref.artifact_id)
        assert plan is not None
        assert plan.topic == "Why Transformers Use Self-Attention"

        # 3. Factual KNOWLEDGE beat citations
        knowledge_beats = [b for b in plan.beats if b.beat_type == BeatType.KNOWLEDGE]
        assert len(knowledge_beats) >= 1
        core_beat = knowledge_beats[0]
        assert item.evidence_id in core_beat.evidence_refs

        # 4. Traversal to EvidenceItem and SourceDocument
        cited_item = ev_repo.get_evidence_item(item.evidence_id)
        assert cited_item is not None
        assert cited_item.source_document_id == source.source_document_id
        src_doc = ev_repo.get_source_document(cited_item.source_document_id)
        assert src_doc is not None
        assert "context-dependent representations" in src_doc.content_snapshot

        # 5. SCRIPT WorkflowJob created and QUEUED
        script_job = job_repo.get_job_by_idempotency_key(f"idemp_{task.task_id}_script_1")
        assert script_job is not None
        assert script_job.stage == Stage.SCRIPT
        assert script_job.status == JobStatus.QUEUED

        # 6. No StoryboardSnapshot created
        storyboards = session.scalars(select(StoryboardSnapshotORM)).all()
        assert len(storyboards) == 0

    # 7. SCRIPT remains unclaimed
    second_processed = worker.run_once()
    assert second_processed is False


def test_mandatory_review_policy(session_factory):
    """MANDATORY REVIEW-POLICY TEST

    Same evidence, workflow_policy = REVIEW.
    1. Execute KNOWLEDGE_PLAN.
    2. Real ContentPlanRevision created.
    3. Task enters WAITING_USER.
    4. SCRIPT job does NOT exist yet.
    5. Approve task.
    6. Workflow resumes, durable SCRIPT job created.
    7. SCRIPT remains queued.
    """
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(
        session_factory,
        workflow_policy=WorkflowPolicyType.REVIEW,
    )

    worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="test-mandatory-review-worker",
    )
    exec_kp = worker.registry.get_executor(Stage.KNOWLEDGE_PLAN)
    exec_kp.llm_caller = lambda p: _make_sample_llm_json(evidence_id=item.evidence_id)

    # 1. Execute
    processed = worker.run_once()
    assert processed is True

    # 2-4. Check pause in WAITING_USER
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        art_repo = TaskArtifactRepository(session)

        current_task = task_repo.get_task(task.task_id)
        assert current_task.task_status == TaskStatus.WAITING_USER
        assert current_task.current_stage == Stage.KNOWLEDGE_PLAN

        plan_ref = art_repo.get_latest_artifact_ref(task.task_id, stage=Stage.KNOWLEDGE_PLAN)
        assert plan_ref is not None

        script_job = job_repo.get_job_by_idempotency_key(f"idemp_{task.task_id}_script_1")
        assert script_job is None, "SCRIPT job must NOT exist before approval."

        # 5. Approve task
        workflow = KnowledgeVideoWorkflow(session)
        resumed_job = workflow.resume_after_approval(current_task)
        session.commit()

        # 6. SCRIPT job created
        assert resumed_job is not None
        assert resumed_job.stage == Stage.SCRIPT
        assert resumed_job.status == JobStatus.QUEUED

        updated_task = task_repo.get_task(task.task_id)
        assert updated_task.task_status == TaskStatus.RUNNING
        assert updated_task.current_stage == Stage.SCRIPT

    # 7. SCRIPT remains queued / unsupported
    second_processed = worker.run_once()
    assert second_processed is False


def test_mandatory_unsupported_fact(session_factory):
    """MANDATORY UNSUPPORTED FACT TEST

    Planner test output attempts to produce factual KNOWLEDGE Beat without valid evidence_refs.
    Expected:
    - Plan not accepted as successful production artifact
    - No successful KNOWLEDGE_PLAN completion
    - No SCRIPT job
    - No fabricated evidence
    - Maps to NEEDS_EVIDENCE
    """
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    unsupported_beats = [
        {
            "planner_ref": "b1",
            "beat_type": "KNOWLEDGE",
            "intent": "Factual claim with no supporting evidence",
            "order": 1,
            "target_duration": 30.0,
            "importance": 1.0,
            "evidence_refs": [],
            "inherit_from": None,
        }
    ]

    worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="test-mandatory-unsupported-worker",
    )
    exec_kp = worker.registry.get_executor(Stage.KNOWLEDGE_PLAN)
    exec_kp.llm_caller = lambda p: _make_sample_llm_json(beats=unsupported_beats)

    processed = worker.run_once()
    assert processed is True

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        art_repo = TaskArtifactRepository(session)

        current_task = task_repo.get_task(task.task_id)
        assert current_task.task_status == TaskStatus.NEEDS_EVIDENCE

        # No plan artifact created
        plan_ref = art_repo.get_latest_artifact_ref(task.task_id, stage=Stage.KNOWLEDGE_PLAN)
        assert plan_ref is None

        # No SCRIPT job created
        script_job = job_repo.get_job_by_idempotency_key(f"idemp_{task.task_id}_script_1")
        assert script_job is None


def test_mandatory_lineage(session_factory):
    """MANDATORY LINEAGE TEST

    Explicit traversal verification:
    KnowledgeVideoTask
    -> EvidenceSnapshot TaskArtifactRef
    -> KNOWLEDGE_PLAN WorkflowJob
    -> StageExecution
    -> ContentPlanRevision
    -> ContentBeat
    -> evidence_refs
    -> EvidenceItem
    -> SourceDocument
    and:
    ContentPlanRevision -> TaskArtifactRef -> same task_id
    """
    task, job, snapshot, item, source, art_ref = _setup_task_with_evidence(session_factory)

    worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="test-mandatory-lineage-worker",
    )
    exec_kp = worker.registry.get_executor(Stage.KNOWLEDGE_PLAN)
    exec_kp.llm_caller = lambda p: _make_sample_llm_json(evidence_id=item.evidence_id)

    worker.run_once()

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        art_repo = TaskArtifactRepository(session)
        exec_repo = StageExecutionRepository(session)
        plan_repo = ContentPlanRepository(session)
        ev_repo = EvidenceRepository(session)

        # 1. Start from Task
        assert task.task_id is not None

        # 2. EvidenceSnapshot TaskArtifactRef
        input_ref = art_repo.get_artifact_ref(job.input_task_artifact_ref_id)
        assert input_ref is not None
        assert input_ref.task_id == task.task_id
        assert input_ref.stage == Stage.EVIDENCE

        # 3. KNOWLEDGE_PLAN WorkflowJob
        completed_job = job_repo.get_job(job.job_id)
        assert completed_job is not None
        assert completed_job.task_id == task.task_id
        assert completed_job.status == JobStatus.SUCCEEDED

        # 4. StageExecution
        executions = exec_repo.list_executions_for_task(task.task_id)
        assert len(executions) == 1
        exec_record = executions[0]
        assert exec_record.task_id == task.task_id
        assert exec_record.stage == Stage.KNOWLEDGE_PLAN
        assert exec_record.output_task_artifact_ref_id == completed_job.output_task_artifact_ref_id

        # 5. ContentPlanRevision TaskArtifactRef
        output_ref = art_repo.get_artifact_ref(exec_record.output_task_artifact_ref_id)
        assert output_ref is not None
        assert output_ref.task_id == task.task_id
        assert output_ref.stage == Stage.KNOWLEDGE_PLAN
        assert output_ref.artifact_type == ArtifactType.CONTENT_PLAN_REVISION

        # 6. ContentPlanRevision
        plan = plan_repo.get_revision(output_ref.artifact_id)
        assert plan is not None
        assert plan.content_plan_revision_id == output_ref.artifact_id

        # 7. ContentBeat
        assert len(plan.beats) >= 1
        cited_beats = [b for b in plan.beats if b.evidence_refs]
        assert len(cited_beats) >= 1
        first_cited_beat = cited_beats[0]

        # 8. evidence_refs -> EvidenceItem
        assert item.evidence_id in first_cited_beat.evidence_refs
        cited_item = ev_repo.get_evidence_item(item.evidence_id)
        assert cited_item is not None
        assert cited_item.evidence_id == item.evidence_id

        # 9. EvidenceItem -> SourceDocument
        assert cited_item.source_document_id == source.source_document_id
        src_doc = ev_repo.get_source_document(cited_item.source_document_id)
        assert src_doc is not None
        assert src_doc.source_document_id == source.source_document_id
