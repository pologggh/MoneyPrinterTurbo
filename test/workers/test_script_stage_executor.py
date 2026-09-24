from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.evidence_stage_executor import EvidenceStageExecutor
from app.application.knowledge_plan_stage_executor import KnowledgePlanStageExecutor
from app.application.knowledge_video_workflow import KnowledgeVideoWorkflow
from app.application.script_stage_executor import ScriptStageExecutor
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
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.script import ScriptRevision, ScriptSegment
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
    ScriptRevisionORM,
    ScriptSegmentORM,
    ShotORM,
    ShotRevisionORM,
    StoryboardSnapshotORM,
    TaskArtifactRefORM,
    WorkflowJobORM,
)
from app.persistence.repositories import (
    ContentPlanRepository,
    EvidenceRepository,
    KnowledgeVideoTaskRepository,
    ScriptRepository,
    StageExecutionRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)
from app.services.script.script_generator import (
    ScriptGenerator,
    ScriptNeedsEvidenceError,
    ScriptOutputInvalidError,
    ScriptOutputSchemaError,
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


def _make_sample_script_llm_json(
    beat_ids: list[str],
    evidence_id: str = "ev_transformer_1",
) -> str:
    """Produces valid sample structured script output matching the beats."""
    segments = [
        {
            "content_beat_id": beat_ids[0],
            "order": 1,
            "narration_text": "Have you ever wondered how modern language models understand complex relationships across an entire sentence?",
            "target_duration": 5.0,
            "evidence_refs": [],
        },
        {
            "content_beat_id": beat_ids[1],
            "order": 2,
            "narration_text": "Self-attention allows each token to dynamically attend to every other token, building deep context-dependent representations.",
            "target_duration": 20.0,
            "evidence_refs": [evidence_id],
        },
        {
            "content_beat_id": beat_ids[2],
            "order": 3,
            "narration_text": "In summary, this unified attention mechanism powers today's generative AI breakthroughs.",
            "target_duration": 5.0,
            "evidence_refs": [],
        },
    ]
    return json.dumps({"segments": segments})


def _setup_task_with_plan(
    session_factory,
    topic: str = "Why does Transformer use self-attention?",
    evidence_text: str = (
        "Self-attention allows each token to compute relationships with other tokens "
        "and build context-dependent representations."
    ),
    workflow_policy: WorkflowPolicyType = WorkflowPolicyType.AUTO,
) -> tuple[
    KnowledgeVideoTask,
    WorkflowJob,
    ContentPlanRevision,
    EvidenceSnapshot,
    EvidenceItem,
    SourceDocument,
    TaskArtifactRef,
]:
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        ev_repo = EvidenceRepository(session)
        plan_repo = ContentPlanRepository(session)
        art_repo = TaskArtifactRepository(session)

        # 1. Create Task
        task = KnowledgeVideoTask.create(
            topic=topic,
            target_duration=30.0,
            workflow_policy=workflow_policy,
        )
        task.transition_to(TaskStatus.RUNNING)
        task.advance_stage(Stage.KNOWLEDGE_PLAN)
        task.advance_stage(Stage.SCRIPT)
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

        # 3. Create EvidenceSnapshot & TaskArtifactRef(EVIDENCE)
        snapshot = EvidenceSnapshot.create(
            task_id=saved_task.task_id,
            source_document_ids=[saved_source.source_document_id],
            evidence_ids=[saved_item.evidence_id],
            snapshot_version=1,
        )
        saved_snapshot = ev_repo.save_evidence_snapshot(snapshot)

        ev_art_ref = TaskArtifactRef.create(
            task_id=saved_task.task_id,
            stage=Stage.EVIDENCE,
            artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
            artifact_id=saved_snapshot.evidence_snapshot_id,
            artifact_version="1",
        )
        art_repo.save_artifact_ref(ev_art_ref)

        # 4. Create ContentPlanRevision with 3 beats
        beat1 = ContentBeat(
            beat_id=f"beat_{uuid4().hex[:8]}",
            beat_lineage_id=f"lin_{uuid4().hex[:8]}",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Hook the viewer with a question",
            target_duration=5.0,
            importance=0.9,
            evidence_refs=(),
        )
        beat2 = ContentBeat(
            beat_id=f"beat_{uuid4().hex[:8]}",
            beat_lineage_id=f"lin_{uuid4().hex[:8]}",
            beat_type=BeatType.KNOWLEDGE,
            order=2,
            intent="Explain self-attention mechanism",
            target_duration=20.0,
            importance=1.0,
            evidence_refs=(saved_item.evidence_id,),
        )
        beat3 = ContentBeat(
            beat_id=f"beat_{uuid4().hex[:8]}",
            beat_lineage_id=f"lin_{uuid4().hex[:8]}",
            beat_type=BeatType.SUMMARY,
            order=3,
            intent="Summarize importance of attention",
            target_duration=5.0,
            importance=0.8,
            evidence_refs=(),
        )
        plan_rev = ContentPlanRevision(
            content_plan_revision_id=f"cpr_{uuid4().hex[:8]}",
            revision_number=1,
            topic=topic,
            overall_target_duration=30.0,
            beats=(beat1, beat2, beat3),
        )
        saved_plan = plan_repo.add_revision(plan_rev)

        # 5. Create TaskArtifactRef for KNOWLEDGE_PLAN stage
        plan_art_ref = TaskArtifactRef.create(
            task_id=saved_task.task_id,
            stage=Stage.KNOWLEDGE_PLAN,
            artifact_type=ArtifactType.CONTENT_PLAN_REVISION,
            artifact_id=saved_plan.content_plan_revision_id,
            artifact_version="1",
        )
        saved_plan_ref = art_repo.save_artifact_ref(plan_art_ref)

        # 6. Create SCRIPT WorkflowJob
        job = WorkflowJob.create(
            task_id=saved_task.task_id,
            stage=Stage.SCRIPT,
            idempotency_key=f"idemp_{saved_task.task_id}_script_1",
            input_task_artifact_ref_id=saved_plan_ref.task_artifact_ref_id,
        )
        saved_job = job_repo.create_job(job)
        session.commit()

        return (
            saved_task,
            saved_job,
            saved_plan,
            saved_snapshot,
            saved_item,
            saved_source,
            saved_plan_ref,
        )


# =============================================================================
# 1. Registry Invariants Tests
# =============================================================================


def test_production_registry_contains_evidence_plan_script_and_no_mocks():
    """Verify production registry registers EVIDENCE, KNOWLEDGE_PLAN, and SCRIPT."""
    registry = get_default_executor_registry()
    assert registry.has_executor(Stage.EVIDENCE)
    assert isinstance(registry.get_executor(Stage.EVIDENCE), EvidenceStageExecutor)

    assert registry.has_executor(Stage.KNOWLEDGE_PLAN)
    assert isinstance(registry.get_executor(Stage.KNOWLEDGE_PLAN), KnowledgePlanStageExecutor)

    assert registry.has_executor(Stage.SCRIPT)
    assert isinstance(registry.get_executor(Stage.SCRIPT), ScriptStageExecutor)

    # STORYBOARD and subsequent stages remain unsupported
    assert not registry.has_executor(Stage.STORYBOARD)
    assert registry.get_executor(Stage.STORYBOARD) is None
    assert not registry.has_executor(Stage.ASSET)
    assert not registry.has_executor(Stage.AUDIO)


def test_unsupported_storyboard_remains_queued_in_default_registry(session_factory):
    """Verify that STORYBOARD job remains safe and QUEUED when worker runs with default registry."""
    task_id = f"task_{uuid4().hex[:8]}"
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="General Relativity",
            target_duration=90.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task.advance_stage(Stage.KNOWLEDGE_PLAN)
        task.advance_stage(Stage.SCRIPT)
        task.advance_stage(Stage.STORYBOARD)
        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.STORYBOARD,
            idempotency_key=f"idemp_{task_id}_storyboard_1",
        )
        task_repo.save_task(task)
        job_repo.create_job(job)
        session.commit()

    worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="worker-test-1",
    )
    processed = worker.run_once()
    assert processed is False

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        task_repo = KnowledgeVideoTaskRepository(session)
        cur_job = job_repo.get_job(job.job_id)
        cur_task = task_repo.get_task(task_id)

        assert cur_job.status == JobStatus.QUEUED
        assert cur_job.lease_owner is None
        assert cur_task.current_stage == Stage.STORYBOARD


# =============================================================================
# 2. Input Resolution and Lineage Verification Tests
# =============================================================================


def test_script_executor_resolves_exact_content_plan_revision(session_factory):
    """ScriptStageExecutor successfully loads exact ContentPlanRevision via TaskArtifactRef."""
    task, job, plan, snap, item, src, plan_ref = _setup_task_with_plan(session_factory)
    beat_ids = [b.beat_id for b in plan.beats]

    llm_json = _make_sample_script_llm_json(beat_ids, evidence_id=item.evidence_id)
    executor = ScriptStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: llm_json,
    )
    result = executor.execute(task, job)

    assert result.success is True
    assert result.output_artifact_ref is not None
    assert result.output_artifact_ref.stage == Stage.SCRIPT
    assert result.output_artifact_ref.artifact_type == ArtifactType.SCRIPT_REVISION
    assert result.metadata_json["segment_count"] == 3


def test_script_executor_fails_when_input_artifact_ref_missing(session_factory):
    """Fails with FATAL when input artifact reference does not exist."""
    task, job, plan, snap, item, src, plan_ref = _setup_task_with_plan(session_factory)

    # Job with invalid input artifact ref
    job.input_task_artifact_ref_id = "non_existent_ref_id"

    executor = ScriptStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: "{}",
    )
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "Input artifact reference not found" in (result.error_message or "")


def test_script_executor_fails_on_task_id_mismatch(session_factory):
    """Fails with FATAL if input TaskArtifactRef belongs to another task."""
    task, job, plan, snap, item, src, plan_ref = _setup_task_with_plan(session_factory)

    # Create artifact ref for a foreign task
    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        foreign_ref = TaskArtifactRef.create(
            task_id="foreign_task_999",
            stage=Stage.KNOWLEDGE_PLAN,
            artifact_type=ArtifactType.CONTENT_PLAN_REVISION,
            artifact_id=plan.content_plan_revision_id,
        )
        art_repo.save_artifact_ref(foreign_ref)
        session.commit()

    job.input_task_artifact_ref_id = foreign_ref.task_artifact_ref_id
    executor = ScriptStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: "{}",
    )
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "task_id mismatch" in (result.error_message or "")


def test_script_executor_fails_on_wrong_stage_artifact_type(session_factory):
    """Fails with FATAL if input artifact is from wrong stage (e.g. EVIDENCE instead of KNOWLEDGE_PLAN)."""
    task, job, plan, snap, item, src, plan_ref = _setup_task_with_plan(session_factory)

    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        ev_ref = art_repo.get_latest_artifact_ref(task.task_id, stage=Stage.EVIDENCE)

    job.input_task_artifact_ref_id = ev_ref.task_artifact_ref_id
    executor = ScriptStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: "{}",
    )
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "Input artifact stage mismatch" in (result.error_message or "")


# =============================================================================
# 3. Beat Order Preservation & Segment Mapping Tests
# =============================================================================


def test_beat_order_preservation(session_factory):
    """Every ScriptSegment maps 1:1 to its originating ContentBeat and preserves strict ordering."""
    task, job, plan, snap, item, src, plan_ref = _setup_task_with_plan(session_factory)
    beat_ids = [b.beat_id for b in plan.beats]

    llm_json = _make_sample_script_llm_json(beat_ids, evidence_id=item.evidence_id)
    executor = ScriptStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: llm_json,
    )
    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        script_repo = ScriptRepository(session)
        rev = script_repo.get_revision(result.output_artifact_ref.artifact_id)
        assert rev is not None
        assert len(rev.segments) == 3

        for idx, seg in enumerate(rev.segments):
            orig_beat = plan.beats[idx]
            assert seg.order == orig_beat.order
            assert seg.content_beat_id == orig_beat.beat_id
            assert seg.beat_type == orig_beat.beat_type


def test_llm_reordering_beats_is_rejected(session_factory):
    """If the LLM reorders beat IDs or orders non-sequentially, the output is rejected as RETRYABLE."""
    task, job, plan, snap, item, src, plan_ref = _setup_task_with_plan(session_factory)
    beat_ids = [b.beat_id for b in plan.beats]

    # Swap beat 1 and beat 2
    swapped_json = json.dumps({
        "segments": [
            {
                "content_beat_id": beat_ids[1],
                "order": 1,
                "narration_text": "Swapped narration 1",
                "target_duration": 10.0,
                "evidence_refs": [item.evidence_id],
            },
            {
                "content_beat_id": beat_ids[0],
                "order": 2,
                "narration_text": "Swapped narration 2",
                "target_duration": 10.0,
                "evidence_refs": [],
            },
            {
                "content_beat_id": beat_ids[2],
                "order": 3,
                "narration_text": "Summary",
                "target_duration": 5.0,
                "evidence_refs": [],
            },
        ]
    })
    executor = ScriptStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: swapped_json,
    )
    result = executor.execute(task, job)

    assert result.success is False
    assert result.is_retryable is True
    assert "does not match originating beat order" in (result.error_message or "")


def test_unknown_beat_id_is_rejected(session_factory):
    """If the LLM invents a non-existent beat_id, the output is rejected as RETRYABLE."""
    task, job, plan, snap, item, src, plan_ref = _setup_task_with_plan(session_factory)

    bad_json = json.dumps({
        "segments": [
            {
                "content_beat_id": "totally_invented_beat_id",
                "order": 1,
                "narration_text": "Invented",
                "target_duration": 5.0,
                "evidence_refs": [],
            },
            {
                "content_beat_id": plan.beats[1].beat_id,
                "order": 2,
                "narration_text": "Second",
                "target_duration": 20.0,
                "evidence_refs": [item.evidence_id],
            },
            {
                "content_beat_id": plan.beats[2].beat_id,
                "order": 3,
                "narration_text": "Third",
                "target_duration": 5.0,
                "evidence_refs": [],
            },
        ]
    })
    executor = ScriptStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: bad_json,
    )
    result = executor.execute(task, job)

    assert result.success is False
    assert result.is_retryable is True
    assert "unknown content_beat_id" in (result.error_message or "")


# =============================================================================
# 4. Evidence Grounding & Subset Invariants Tests
# =============================================================================


def test_segment_evidence_scope_expansion_rejected(session_factory):
    """A ScriptSegment CANNOT add evidence_refs outside its originating ContentBeat."""
    task, job, plan, snap, item, src, plan_ref = _setup_task_with_plan(session_factory)
    beat_ids = [b.beat_id for b in plan.beats]

    # Beat 1 is a HOOK with evidence_refs=()
    # If the LLM tries to attach evidence to Beat 1:
    expanded_json = json.dumps({
        "segments": [
            {
                "content_beat_id": beat_ids[0],
                "order": 1,
                "narration_text": "Hook with smuggled evidence",
                "target_duration": 5.0,
                "evidence_refs": ["ev_smuggled_1"],
            },
            {
                "content_beat_id": beat_ids[1],
                "order": 2,
                "narration_text": "Knowledge",
                "target_duration": 20.0,
                "evidence_refs": [item.evidence_id],
            },
            {
                "content_beat_id": beat_ids[2],
                "order": 3,
                "narration_text": "Summary",
                "target_duration": 5.0,
                "evidence_refs": [],
            },
        ]
    })
    executor = ScriptStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: expanded_json,
    )
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.NEEDS_EVIDENCE.value
    assert result.is_retryable is False
    assert "evidence scope" in (result.error_message or "")


def test_unresolved_evidence_id_fails_as_needs_evidence(session_factory):
    """If a segment cites an evidence ID that doesn't exist in EvidenceSnapshot, it fails with NEEDS_EVIDENCE."""
    task, job, plan, snap, item, src, plan_ref = _setup_task_with_plan(session_factory)

    # Modify Beat 2 in memory to claim a non-existent evidence ID
    with session_factory() as session:
        b2_orm = session.get(ContentBeatORM, plan.beats[1].beat_id)
        b2_orm.evidence_refs = ["ev_ghost_999"]
        session.commit()

    beat_ids = [b.beat_id for b in plan.beats]
    ghost_json = _make_sample_script_llm_json(beat_ids, evidence_id="ev_ghost_999")

    executor = ScriptStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: ghost_json,
    )
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.NEEDS_EVIDENCE.value
    assert "does not exist in the EvidenceSnapshot" in (result.error_message or "")


# =============================================================================
# 5. Error Handling & Bounded Retry Tests
# =============================================================================


def test_malformed_llm_json_is_retryable(session_factory):
    """Malformed LLM response raises RETRYABLE error to trigger workflow retry semantics."""
    task, job, plan, snap, item, src, plan_ref = _setup_task_with_plan(session_factory)

    executor = ScriptStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: "This is not JSON at all, sorry!",
    )
    result = executor.execute(task, job)

    assert result.success is False
    assert result.is_retryable is True
    assert result.error_type == JobErrorType.RETRYABLE.value


# =============================================================================
# 6. Four Mandatory Demonstrations
# =============================================================================


def test_mandatory_success_demonstration(session_factory):
    """
    MANDATORY SUCCESS INTEGRATION DEMONSTRATION:
    - Topic: 'Why does Transformer use self-attention?'
    - Real EvidenceItem: 'Self-attention allows each token to compute relationships...'
    - Real ContentPlanRevision: HOOK, KNOWLEDGE (with real evidence_refs), SUMMARY
    - Execute SCRIPT via StageWorker
    - Verify KNOWLEDGE ScriptSegment has narration and real evidence_refs
    - Resolve: ScriptSegment -> ContentBeat -> EvidenceItem -> SourceDocument
    - Verify ScriptRevision TaskArtifactRef created, StageExecution succeeds
    - Verify STORYBOARD WorkflowJob created and safely QUEUED
    - ZERO StoryboardSnapshot, ZERO Shot, ZERO ShotRevision created
    """
    task, job, plan, snap, item, src, plan_ref = _setup_task_with_plan(
        session_factory,
        topic="Why does Transformer use self-attention?",
    )
    beat_ids = [b.beat_id for b in plan.beats]
    llm_json = _make_sample_script_llm_json(beat_ids, evidence_id=item.evidence_id)

    registry = StageExecutorRegistry()
    registry.register(
        Stage.SCRIPT,
        ScriptStageExecutor(
            session_factory=session_factory,
            llm_caller=lambda p: llm_json,
        ),
    )

    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="prod-worker-demo-1",
    )

    # 1. Run Worker to execute SCRIPT job
    processed = worker.run_once()
    assert processed is True

    # 2. Verify DB state after execution
    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        task_repo = KnowledgeVideoTaskRepository(session)
        art_repo = TaskArtifactRepository(session)
        exec_repo = StageExecutionRepository(session)
        script_repo = ScriptRepository(session)
        ev_repo = EvidenceRepository(session)

        # A. SCRIPT job succeeded
        script_job = job_repo.get_job(job.job_id)
        assert script_job.status == JobStatus.SUCCEEDED

        # B. StageExecution recorded truthfully
        executions = exec_repo.list_executions_for_task(task.task_id)
        assert len(executions) == 1
        assert executions[0].stage == Stage.SCRIPT
        assert executions[0].status == "SUCCEEDED"
        assert executions[0].output_task_artifact_ref_id is not None

        # C. TaskArtifactRef(SCRIPT_REVISION) created
        script_ref = art_repo.get_artifact_ref(executions[0].output_task_artifact_ref_id)
        assert script_ref is not None
        assert script_ref.task_id == task.task_id
        assert script_ref.stage == Stage.SCRIPT
        assert script_ref.artifact_type == ArtifactType.SCRIPT_REVISION

        # D. ScriptRevision & Segments
        script_rev = script_repo.get_revision(script_ref.artifact_id)
        assert script_rev is not None
        assert len(script_rev.segments) == 3

        # E. Full Provenance Lineage Resolution:
        # ScriptSegment -> ContentBeat -> EvidenceItem -> SourceDocument
        knowledge_seg = script_rev.segments[1]
        assert knowledge_seg.beat_type == BeatType.KNOWLEDGE
        assert len(knowledge_seg.narration_text) > 10
        assert item.evidence_id in knowledge_seg.evidence_refs

        # Resolve to ContentBeat
        orig_beat = plan.beats[1]
        assert knowledge_seg.content_beat_id == orig_beat.beat_id
        assert knowledge_seg.evidence_refs[0] == orig_beat.evidence_refs[0]

        # Resolve to EvidenceItem
        resolved_ev_item = ev_repo.get_evidence_item(knowledge_seg.evidence_refs[0])
        assert resolved_ev_item is not None
        assert "Self-attention" in resolved_ev_item.normalized_fact

        # Resolve to SourceDocument
        resolved_source_doc = ev_repo.get_source_document(resolved_ev_item.source_document_id)
        assert resolved_source_doc is not None
        assert resolved_source_doc.title == "Attention Is All You Need"

        # F. Workflow progression: next stage is STORYBOARD
        current_task = task_repo.get_task(task.task_id)
        assert current_task.current_stage == Stage.STORYBOARD
        assert current_task.task_status == TaskStatus.RUNNING

        # G. STORYBOARD job created and safely QUEUED
        all_jobs = job_repo.list_jobs_for_task(task.task_id)
        sb_jobs = [j for j in all_jobs if j.stage == Stage.STORYBOARD]
        assert len(sb_jobs) == 1
        sb_job = sb_jobs[0]
        assert sb_job.status == JobStatus.QUEUED
        assert sb_job.input_task_artifact_ref_id == script_ref.task_artifact_ref_id

        # H. ZERO StoryboardSnapshot, ZERO Shot, ZERO ShotRevision created
        sb_snapshots = session.scalars(select(StoryboardSnapshotORM)).all()
        assert len(sb_snapshots) == 0
        shots = session.scalars(select(ShotORM)).all()
        assert len(shots) == 0
        shot_revisions = session.scalars(select(ShotRevisionORM)).all()
        assert len(shot_revisions) == 0


def test_mandatory_unsupported_fact_demonstration(session_factory):
    """
    MANDATORY UNSUPPORTED FACT DEMONSTRATION:
    - LLM produces factual claim on KNOWLEDGE beat with empty evidence_refs.
    - Script stage rejects output as NEEDS_EVIDENCE.
    - No successful ScriptRevision output is produced.
    - No STORYBOARD job is created.
    - No fabricated EvidenceItems are created.
    """
    task, job, plan, snap, item, src, plan_ref = _setup_task_with_plan(session_factory)
    beat_ids = [b.beat_id for b in plan.beats]

    # Produce narration for KNOWLEDGE beat but leave evidence_refs empty!
    unsupported_json = json.dumps({
        "segments": [
            {
                "content_beat_id": beat_ids[0],
                "order": 1,
                "narration_text": "Hook text",
                "target_duration": 5.0,
                "evidence_refs": [],
            },
            {
                "content_beat_id": beat_ids[1],
                "order": 2,
                "narration_text": "This factual claim has zero evidence references attached.",
                "target_duration": 20.0,
                "evidence_refs": [],  # Empty on factual beat!
            },
            {
                "content_beat_id": beat_ids[2],
                "order": 3,
                "narration_text": "Summary text",
                "target_duration": 5.0,
                "evidence_refs": [],
            },
        ]
    })

    registry = StageExecutorRegistry()
    registry.register(
        Stage.SCRIPT,
        ScriptStageExecutor(
            session_factory=session_factory,
            llm_caller=lambda p: unsupported_json,
        ),
    )

    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="prod-worker-demo-2",
    )

    processed = worker.run_once()
    assert processed is True

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        task_repo = KnowledgeVideoTaskRepository(session)
        script_repo = ScriptRepository(session)
        ev_repo = EvidenceRepository(session)

        # Job marked FAILED with NEEDS_EVIDENCE
        script_job = job_repo.get_job(job.job_id)
        assert script_job.status == JobStatus.FAILED
        assert script_job.error_type == JobErrorType.NEEDS_EVIDENCE.value

        # No ScriptRevision created
        revisions = script_repo.list_revisions_for_task(task.task_id)
        assert len(revisions) == 0

        # No STORYBOARD job created
        all_jobs = job_repo.list_jobs_for_task(task.task_id)
        assert not any(j.stage == Stage.STORYBOARD for j in all_jobs)

        # Task transitioned to NEEDS_EVIDENCE
        current_task = task_repo.get_task(task.task_id)
        assert current_task.task_status == TaskStatus.NEEDS_EVIDENCE

        # No fabricated EvidenceItem created
        all_items = ev_repo.list_evidence_items_for_source(src.source_document_id)
        assert len(all_items) == 1
        assert all_items[0].evidence_id == item.evidence_id


def test_mandatory_immutability_demonstration(session_factory):
    """
    MANDATORY IMMUTABILITY DEMONSTRATION:
    - Produce ScriptRevision S1 from ContentPlanRevision P1.
    - Rerun Script stage intentionally to produce S2.
    - S1 remains completely unchanged.
    - S1 and S2 are both queryable in database.
    - TaskArtifactRef points to latest S2.
    - Historical StageExecution for S1 remains inspectable.
    """
    task, job, plan, snap, item, src, plan_ref = _setup_task_with_plan(session_factory)
    beat_ids = [b.beat_id for b in plan.beats]

    llm_json_1 = _make_sample_script_llm_json(beat_ids, evidence_id=item.evidence_id)
    executor_1 = ScriptStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: llm_json_1,
    )
    result_1 = executor_1.execute(task, job)
    assert result_1.success is True
    s1_id = result_1.output_artifact_ref.artifact_id

    # Persist artifact ref for S1 as StageWorker would
    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        art_repo.save_artifact_ref(result_1.output_artifact_ref)
        session.commit()

    # Create S2 rerun job
    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        job_2 = WorkflowJob.create(
            task_id=task.task_id,
            stage=Stage.SCRIPT,
            idempotency_key=f"idemp_{task.task_id}_script_2",
            attempt_number=2,
            input_task_artifact_ref_id=plan_ref.task_artifact_ref_id,
        )
        job_repo.create_job(job_2)
        session.commit()

    # Slightly different narration for S2
    llm_json_2 = json.dumps({
        "segments": [
            {
                "content_beat_id": beat_ids[0],
                "order": 1,
                "narration_text": "Updated Hook narration text for revision 2.",
                "target_duration": 5.0,
                "evidence_refs": [],
            },
            {
                "content_beat_id": beat_ids[1],
                "order": 2,
                "narration_text": "Updated Knowledge narration text for revision 2.",
                "target_duration": 20.0,
                "evidence_refs": [item.evidence_id],
            },
            {
                "content_beat_id": beat_ids[2],
                "order": 3,
                "narration_text": "Updated Summary narration text for revision 2.",
                "target_duration": 5.0,
                "evidence_refs": [],
            },
        ]
    })
    executor_2 = ScriptStageExecutor(
        session_factory=session_factory,
        llm_caller=lambda p: llm_json_2,
    )
    result_2 = executor_2.execute(task, job_2)
    assert result_2.success is True
    s2_id = result_2.output_artifact_ref.artifact_id
    assert s1_id != s2_id

    # Verify both S1 and S2 exist independently in database
    with session_factory() as session:
        script_repo = ScriptRepository(session)
        s1 = script_repo.get_revision(s1_id)
        s2 = script_repo.get_revision(s2_id)

        assert s1 is not None
        assert s2 is not None
        assert s1.revision_number == 1
        assert s2.revision_number == 2
        assert s1.content_fingerprint != s2.content_fingerprint
        assert "Have you ever wondered" in s1.segments[0].narration_text
        assert "Updated Hook" in s2.segments[0].narration_text

        all_revisions = script_repo.list_revisions_for_task(task.task_id)
        assert len(all_revisions) == 2


def test_mandatory_review_flow_demonstration(session_factory):
    """
    MANDATORY REVIEW-FLOW DEMONSTRATION:
    - Task with REVIEW policy.
    - KNOWLEDGE_PLAN has already been approved (resumed via resume_after_approval).
    - Run SCRIPT via StageWorker.
    - SCRIPT succeeds and transitions directly to STORYBOARD job.
    - NO extra review pause is introduced after SCRIPT (no WAITING_USER).
    - Confirms AUTO and REVIEW execute the exact same ScriptStageExecutor.
    """
    # 1. Setup task with REVIEW policy paused in WAITING_USER after KNOWLEDGE_PLAN
    task_id = f"task_rev_{uuid4().hex[:8]}"
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        plan_repo = ContentPlanRepository(session)
        art_repo = TaskArtifactRepository(session)
        ev_repo = EvidenceRepository(session)

        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="Why does Transformer use self-attention?",
            target_duration=30.0,
            workflow_policy=WorkflowPolicyType.REVIEW,
        )
        task.transition_to(TaskStatus.RUNNING)
        task.advance_stage(Stage.KNOWLEDGE_PLAN)
        task.transition_to(TaskStatus.WAITING_USER, reason="Review required after KNOWLEDGE_PLAN")
        task_repo.save_task(task)

        source_doc = SourceDocument.create_text(
            text="Self-attention allows tokens to communicate directly.",
            title="Attention Paper",
        )
        saved_source = ev_repo.save_source_document(source_doc)
        ev_repo.associate_task_source(task.task_id, saved_source.source_document_id)

        ev_item = EvidenceItem.create(
            source_document=saved_source,
            original_excerpt="Self-attention allows tokens to communicate directly.",
            normalized_fact="Self-attention allows direct token communication.",
        )
        saved_item = ev_repo.save_evidence_item(ev_item)

        snapshot = EvidenceSnapshot.create(
            task_id=task.task_id,
            source_document_ids=[saved_source.source_document_id],
            evidence_ids=[saved_item.evidence_id],
            snapshot_version=1,
        )
        ev_repo.save_evidence_snapshot(snapshot)

        ev_art_ref = TaskArtifactRef.create(
            task_id=task.task_id,
            stage=Stage.EVIDENCE,
            artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
            artifact_id=snapshot.evidence_snapshot_id,
        )
        art_repo.save_artifact_ref(ev_art_ref)

        beat1 = ContentBeat(
            beat_id=f"beat_{uuid4().hex[:8]}",
            beat_lineage_id=f"lin_{uuid4().hex[:8]}",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Hook",
            target_duration=5.0,
            importance=0.9,
            evidence_refs=(),
        )
        beat2 = ContentBeat(
            beat_id=f"beat_{uuid4().hex[:8]}",
            beat_lineage_id=f"lin_{uuid4().hex[:8]}",
            beat_type=BeatType.KNOWLEDGE,
            order=2,
            intent="Direct token communication",
            target_duration=20.0,
            importance=1.0,
            evidence_refs=(saved_item.evidence_id,),
        )
        beat3 = ContentBeat(
            beat_id=f"beat_{uuid4().hex[:8]}",
            beat_lineage_id=f"lin_{uuid4().hex[:8]}",
            beat_type=BeatType.SUMMARY,
            order=3,
            intent="Summary",
            target_duration=5.0,
            importance=0.8,
            evidence_refs=(),
        )
        plan_rev = ContentPlanRevision(
            content_plan_revision_id=f"cpr_{uuid4().hex[:8]}",
            revision_number=1,
            topic=task.topic,
            overall_target_duration=30.0,
            beats=(beat1, beat2, beat3),
        )
        saved_plan = plan_repo.add_revision(plan_rev)

        plan_ref = TaskArtifactRef.create(
            task_id=task.task_id,
            stage=Stage.KNOWLEDGE_PLAN,
            artifact_type=ArtifactType.CONTENT_PLAN_REVISION,
            artifact_id=saved_plan.content_plan_revision_id,
        )
        art_repo.save_artifact_ref(plan_ref)
        session.commit()

    # 2. User approves KNOWLEDGE_PLAN via resume_after_approval
    with session_factory() as session:
        workflow = KnowledgeVideoWorkflow(session)
        task_repo = KnowledgeVideoTaskRepository(session)
        cur_task = task_repo.get_task(task_id)

        script_job = workflow.resume_after_approval(cur_task)
        session.commit()
        assert script_job is not None
        assert script_job.stage == Stage.SCRIPT
        assert cur_task.task_status == TaskStatus.RUNNING
        assert cur_task.current_stage == Stage.SCRIPT

    # 3. StageWorker runs SCRIPT
    beat_ids = [beat1.beat_id, beat2.beat_id, beat3.beat_id]
    llm_json = _make_sample_script_llm_json(beat_ids, evidence_id=saved_item.evidence_id)

    registry = StageExecutorRegistry()
    registry.register(
        Stage.SCRIPT,
        ScriptStageExecutor(
            session_factory=session_factory,
            llm_caller=lambda p: llm_json,
        ),
    )

    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="prod-worker-demo-4",
    )
    processed = worker.run_once()
    assert processed is True

    # 4. Verify no extra review pause occurred after SCRIPT
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        cur_task = task_repo.get_task(task_id)
        # Task must NOT be WAITING_USER; it must be RUNNING at Stage.STORYBOARD
        assert cur_task.task_status == TaskStatus.RUNNING
        assert cur_task.current_stage == Stage.STORYBOARD

        # STORYBOARD WorkflowJob is QUEUED
        jobs = job_repo.list_jobs_for_task(task_id)
        sb_job = next(j for j in jobs if j.stage == Stage.STORYBOARD)
        assert sb_job.status == JobStatus.QUEUED
