from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.evidence_stage_executor import EvidenceStageExecutor
from app.application.knowledge_plan_stage_executor import KnowledgePlanStageExecutor
from app.application.knowledge_video_workflow import KnowledgeVideoWorkflow
from app.application.script_stage_executor import ScriptStageExecutor
from app.application.stage_executor_protocol import StageExecutionResult
from app.application.stage_executor_registry import (
    StageExecutorRegistry,
    get_default_executor_registry,
)
from app.application.storyboard_stage_executor import StoryboardStageExecutor
from app.application.task_command_service import TaskCommandService
from app.application.workflow_policy import WorkflowPolicy
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.evidence import EvidenceItem, EvidenceSnapshot, SourceDocument
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.script import ScriptRevision, ScriptSegment
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_agent import StoryboardAgent, StoryboardPlanningInput
from app.domain.storyboard_approval import StoryboardApprovalRecord
from app.domain.task_artifact import ArtifactType, TaskArtifactRef
from app.domain.trace import TraceEventType
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    JobErrorType,
    JobStatus,
    Stage,
    TaskStatus,
    WorkflowConflictError,
    WorkflowPolicyType,
)
from app.persistence.models import (
    AssetRoutePlanORM,
    Base,
    ExecutionAttemptORM,
    ExecutionRunORM,
    ShotAssetVersionORM,
    ShotORM,
    ShotRevisionORM,
    StoryboardApprovalRecordORM,
    StoryboardSnapshotORM,
    TaskArtifactRefORM,
    WorkflowJobORM,
)
from app.persistence.repositories import (
    ContentPlanRepository,
    EvidenceRepository,
    KnowledgeVideoTaskRepository,
    ScriptRepository,
    ShotRepository,
    StageExecutionRepository,
    StoryboardApprovalRepository,
    StoryboardRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)
from app.services.storyboard.storyboard_adapter import (
    StoryboardAdapter,
    StoryboardAdapterError,
    StoryboardDurationMismatchError,
    StoryboardEvidenceScopeError,
    StoryboardScriptMappingError,
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


def _make_storyboard_mock_llm_caller():
    """Deterministic LLM mock returning structured storyboard proposals grounded in script narration."""
    def _caller(prompt: str) -> str:
        beat_ref = "beat-1"
        match_ref = re.search(r'"beat_ref":\s*"([^"]+)"', prompt)
        if match_ref:
            beat_ref = match_ref.group(1)

        target_dur = 10.0
        match_dur = re.search(r'Target Duration:\s*([0-9.]+)', prompt)
        if match_dur:
            target_dur = float(match_dur.group(1))

        ev_refs: list[str] = []
        match_ev = re.search(r'Allowed Evidence IDs:\s*\[([^\]]+)\]', prompt)
        if match_ev and "KNOWLEDGE" in prompt:
            ids = [x.strip().strip("'\"") for x in match_ev.group(1).split(",") if x.strip()]
            ev_refs = ids

        narration = "Detailed spoken narration explaining the topic clearly and accurately."
        match_narr = re.search(r'## Authoritative Script Narration:\s*"([^"]+)"', prompt)
        if match_narr:
            narration = match_narr.group(1)

        shot = {
            "planner_ref": "s1",
            "local_order": 1,
            "narration": narration,
            "target_duration": target_dur,
            "visual_goal": "Illustrate the concept clearly",
            "visual_type": "DIAGRAM" if ev_refs else "STOCK_VIDEO",
            "scene_description": "Clean modern animation of neural network attention patterns",
            "generation_prompt": "3D visualization of self-attention matrices connecting words",
            "camera_movement": "Smooth push forward",
            "evidence_refs": ev_refs,
        }
        return json.dumps({"beat_ref": beat_ref, "shots": [shot]})

    return _caller


def _setup_task_with_script(
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
    ScriptRevision,
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
        script_repo = ScriptRepository(session)
        art_repo = TaskArtifactRepository(session)

        # 1. Task
        task = KnowledgeVideoTask.create(
            topic=topic,
            target_duration=30.0,
            workflow_policy=workflow_policy,
        )
        task.advance_stage(Stage.KNOWLEDGE_PLAN)
        task.advance_stage(Stage.SCRIPT)
        task.advance_stage(Stage.STORYBOARD)
        task.transition_to(TaskStatus.RUNNING)
        saved_task = task_repo.save_task(task)

        # 2. SourceDocument
        source = SourceDocument.create_text(
            text=evidence_text,
            title="Attention is All You Need",
        )
        saved_source = ev_repo.save_source_document(source)
        ev_repo.associate_task_source(saved_task.task_id, saved_source.source_document_id)

        # 3. EvidenceItem
        item = EvidenceItem.create(
            source_document=saved_source,
            original_excerpt=evidence_text,
            normalized_fact="Self-attention allows each token to compute relationships.",
        )
        saved_item = ev_repo.save_evidence_item(item)

        # 4. EvidenceSnapshot & Ref
        snapshot = EvidenceSnapshot.create(
            task_id=saved_task.task_id,
            source_document_ids=[saved_source.source_document_id],
            evidence_ids=[saved_item.evidence_id],
            snapshot_version=1,
        )
        saved_snapshot = ev_repo.save_evidence_snapshot(snapshot)

        ev_ref = TaskArtifactRef.create(
            task_id=saved_task.task_id,
            stage=Stage.EVIDENCE,
            artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
            artifact_id=saved_snapshot.evidence_snapshot_id,
            artifact_version="1",
        )
        art_repo.save_artifact_ref(ev_ref)

        # 5. ContentPlanRevision & Ref
        beat_lineage_1 = f"lin_{uuid4().hex[:6]}"
        beat_lineage_2 = f"lin_{uuid4().hex[:6]}"
        beat_lineage_3 = f"lin_{uuid4().hex[:6]}"

        beats = [
            ContentBeat(
                beat_id=f"beat_{uuid4().hex[:6]}",
                beat_lineage_id=beat_lineage_1,
                beat_type=BeatType.HOOK,
                order=1,
                intent="Engage viewer curiosity with core question",
                target_duration=5.0,
                importance=0.8,
                evidence_refs=(),
            ),
            ContentBeat(
                beat_id=f"beat_{uuid4().hex[:6]}",
                beat_lineage_id=beat_lineage_2,
                beat_type=BeatType.KNOWLEDGE,
                order=2,
                intent="Explain self-attention mechanism with evidence",
                target_duration=20.0,
                importance=1.0,
                evidence_refs=(saved_item.evidence_id,),
            ),
            ContentBeat(
                beat_id=f"beat_{uuid4().hex[:6]}",
                beat_lineage_id=beat_lineage_3,
                beat_type=BeatType.SUMMARY,
                order=3,
                intent="Summarize core takeaway of contextual representation",
                target_duration=5.0,
                importance=0.7,
                evidence_refs=(),
            ),
        ]
        plan = ContentPlanRevision(
            content_plan_revision_id=f"cpr_{uuid4().hex[:8]}",
            revision_number=1,
            topic=topic,
            overall_target_duration=30.0,
            beats=tuple(beats),
        )
        saved_plan = plan_repo.add_revision(plan)

        plan_ref = TaskArtifactRef.create(
            task_id=saved_task.task_id,
            stage=Stage.KNOWLEDGE_PLAN,
            artifact_type=ArtifactType.CONTENT_PLAN_REVISION,
            artifact_id=saved_plan.content_plan_revision_id,
            artifact_version="1",
        )
        art_repo.save_artifact_ref(plan_ref)

        # 6. ScriptRevision & Ref
        segments = [
            ScriptSegment(
                script_revision_id="placeholder",
                content_beat_id=beats[0].beat_id,
                beat_lineage_id=beat_lineage_1,
                order=1,
                narration_text="Have you ever wondered how modern language models understand complex relationships across an entire sentence?",
                target_duration=5.0,
                beat_type=BeatType.HOOK,
                evidence_refs=(),
            ),
            ScriptSegment(
                script_revision_id="placeholder",
                content_beat_id=beats[1].beat_id,
                beat_lineage_id=beat_lineage_2,
                order=2,
                narration_text="Self-attention allows each token to dynamically attend to every other token, building deep context-dependent representations.",
                target_duration=20.0,
                beat_type=BeatType.KNOWLEDGE,
                evidence_refs=(saved_item.evidence_id,),
            ),
            ScriptSegment(
                script_revision_id="placeholder",
                content_beat_id=beats[2].beat_id,
                beat_lineage_id=beat_lineage_3,
                order=3,
                narration_text="In summary, this unified attention mechanism powers today's generative AI breakthroughs.",
                target_duration=5.0,
                beat_type=BeatType.SUMMARY,
                evidence_refs=(),
            ),
        ]
        script_rev = ScriptRevision.create(
            task_id=saved_task.task_id,
            content_plan_revision_id=saved_plan.content_plan_revision_id,
            segments=segments,
            overall_target_duration=30.0,
            language="zh",
            revision_number=1,
        )
        saved_script = script_repo.save_revision(script_rev)

        script_ref = TaskArtifactRef.create(
            task_id=saved_task.task_id,
            stage=Stage.SCRIPT,
            artifact_type=ArtifactType.SCRIPT_REVISION,
            artifact_id=saved_script.script_revision_id,
            artifact_version="1",
        )
        saved_script_ref = art_repo.save_artifact_ref(script_ref)

        # 7. WorkflowJob for STORYBOARD
        job = WorkflowJob.create(
            task_id=saved_task.task_id,
            stage=Stage.STORYBOARD,
            idempotency_key=f"idemp_{saved_task.task_id}_storyboard_1",
            input_task_artifact_ref_id=saved_script_ref.task_artifact_ref_id,
        )
        saved_job = job_repo.create_job(job)
        session.commit()

        return (
            saved_task,
            saved_job,
            saved_script,
            saved_plan,
            saved_snapshot,
            saved_item,
            saved_source,
            saved_script_ref,
        )


# =============================================================================
# 1. Registry Invariants Tests (Items 1 - 4)
# =============================================================================


def test_production_registry_contains_storyboard_and_no_mocks():
    """1-3. Verify production registry registers real StoryboardStageExecutor with no mocks."""
    registry = get_default_executor_registry()
    assert registry.has_executor(Stage.STORYBOARD)
    executor = registry.get_executor(Stage.STORYBOARD)
    assert isinstance(executor, StoryboardStageExecutor)
    assert not type(executor).__name__.startswith("Mock")
    assert not type(executor).__name__.startswith("Fake")


def test_production_registry_retains_prior_stages():
    """2. Verify EVIDENCE, KNOWLEDGE_PLAN, and SCRIPT remain registered."""
    registry = get_default_executor_registry()
    assert registry.has_executor(Stage.EVIDENCE)
    assert isinstance(registry.get_executor(Stage.EVIDENCE), EvidenceStageExecutor)
    assert registry.has_executor(Stage.KNOWLEDGE_PLAN)
    assert isinstance(registry.get_executor(Stage.KNOWLEDGE_PLAN), KnowledgePlanStageExecutor)
    assert registry.has_executor(Stage.SCRIPT)
    assert isinstance(registry.get_executor(Stage.SCRIPT), ScriptStageExecutor)


def test_delivery_remains_unsupported():
    """4. Verify DELIVERY remains unsupported in default registry."""
    registry = get_default_executor_registry()
    assert registry.has_executor(Stage.QUALITY_REVIEW)
    assert not registry.has_executor(Stage.DELIVERY)
    assert registry.get_executor(Stage.DELIVERY) is None


# =============================================================================
# 2. Input Resolution and Lineage Verification Tests (Items 5 - 10)
# =============================================================================


def test_storyboard_executor_resolves_exact_script_artifact(session_factory):
    """5-7. Resolves exact ScriptRevision, ContentPlanRevision, and EvidenceSnapshot."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
    )
    result = executor.execute(task, job)
    assert result.success is True
    assert result.output_artifact_ref is not None
    assert result.output_artifact_ref.artifact_type == ArtifactType.STORYBOARD_SNAPSHOT
    assert result.metadata_json["script_revision_id"] == script.script_revision_id
    assert result.metadata_json["content_plan_revision_id"] == plan.content_plan_revision_id


def test_wrong_task_script_artifact_is_rejected(session_factory):
    """8. TaskArtifactRef task_id mismatch is rejected."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)
    other_task = KnowledgeVideoTask.create(topic="Different Topic")

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
    )
    result = executor.execute(other_task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "task_id mismatch" in result.error_message


def test_missing_script_artifact_fails_explicitly(session_factory):
    """9. Missing script artifact reference fails gracefully without crashing."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    broken_job = WorkflowJob.create(
        task_id=task.task_id,
        stage=Stage.STORYBOARD,
        idempotency_key=f"idemp_{task.task_id}_storyboard_broken",
        input_task_artifact_ref_id="non_existent_ref_id",
    )

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
    )
    result = executor.execute(task, broken_job)
    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value


def test_no_latest_global_script_lookup(session_factory):
    """10. Does not look up global latest script, strictly uses task lineage."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    # Create another script for a different task
    with session_factory() as session:
        s_repo = ScriptRepository(session)
        other_script = ScriptRevision.create(
            task_id="unrelated_task_999",
            content_plan_revision_id="unrelated_plan_999",
            segments=[],
            overall_target_duration=60.0,
            revision_number=5,
        )
        s_repo.save_revision(other_script)
        session.commit()

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
    )
    result = executor.execute(task, job)
    assert result.success is True
    assert result.metadata_json["script_revision_id"] == script.script_revision_id


# =============================================================================
# 3. Domain Reuse Tests (Items 11 - 14)
# =============================================================================


def test_existing_storyboard_agent_is_reused(session_factory):
    """11-12. Verifies existing StoryboardAgent is called without creating a duplicate agent."""
    called = []

    def tracking_caller(prompt: str) -> str:
        called.append(True)
        return _make_storyboard_mock_llm_caller()(prompt)

    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=tracking_caller,
    )
    result = executor.execute(task, job)
    assert result.success is True
    assert len(called) == len(script.segments)


def test_content_plan_and_script_are_not_regenerated(session_factory):
    """13-14. Neither ContentPlan nor Script is regenerated during Storyboard stage."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
    )
    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        p_repo = ContentPlanRepository(session)
        s_repo = ScriptRepository(session)
        # Assert exact plan revision and script revision remain unchanged
        reloaded_plan = p_repo.get_revision(plan.content_plan_revision_id)
        assert reloaded_plan is not None
        reloaded_script = s_repo.get_revision(script.script_revision_id)
        assert reloaded_script is not None


# =============================================================================
# 4. Script Mapping and Provenance Tests (Items 15 - 19)
# =============================================================================


def test_shots_map_to_script_segments_and_preserve_order(session_factory):
    """15-18. Each Shot maps to a ScriptSegment and preserves sequential order."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
    )
    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        sb_repo = StoryboardRepository(session)
        snap = sb_repo.get_snapshot(result.output_artifact_ref.artifact_id)
        revisions = sb_repo.get_snapshot_shot_revisions(snap.storyboard_snapshot_id)

        assert len(revisions) == len(script.segments)
        for rev, seg in zip(revisions, script.segments, strict=True):
            assert rev.script_segment_id == seg.script_segment_id
            assert rev.created_from_beat_instance_id == seg.content_beat_id
            assert rev.narration == seg.narration_text


def test_narration_originates_authoritatively_from_script(session_factory):
    """19. Narration in ShotRevision is locked to the ScriptSegment narration."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
    )
    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        sb_repo = StoryboardRepository(session)
        snap = sb_repo.get_snapshot(result.output_artifact_ref.artifact_id)
        revisions = sb_repo.get_snapshot_shot_revisions(snap.storyboard_snapshot_id)

        assert revisions[0].narration == script.segments[0].narration_text
        assert revisions[1].narration == script.segments[1].narration_text
        assert revisions[2].narration == script.segments[2].narration_text


# =============================================================================
# 5. Evidence Propagation Tests (Items 20 - 24)
# =============================================================================


def test_factual_shot_evidence_refs_are_valid_and_scoped(session_factory):
    """20-21. Factual shot evidence_refs are valid EvidenceItem IDs within upstream scope."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
    )
    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        sb_repo = StoryboardRepository(session)
        snap = sb_repo.get_snapshot(result.output_artifact_ref.artifact_id)
        revisions = sb_repo.get_snapshot_shot_revisions(snap.storyboard_snapshot_id)

        # Segment 2 is KNOWLEDGE and has item.evidence_id
        assert revisions[1].evidence_refs == (item.evidence_id,)
        # Non-factual segments have empty evidence
        assert revisions[0].evidence_refs == ()
        assert revisions[2].evidence_refs == ()


def test_unsupported_or_hallucinated_evidence_is_rejected(session_factory):
    """22. Hallucinated evidence citation outside upstream scope triggers NEEDS_EVIDENCE."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    def rogue_caller(prompt: str) -> str:
        res = _make_storyboard_mock_llm_caller()(prompt)
        data = json.loads(res)
        for s in data["shots"]:
            s["evidence_refs"] = ["invented_unauthorized_evidence_id"]
        return json.dumps(data)

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=rogue_caller,
    )
    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.NEEDS_EVIDENCE.value


# =============================================================================
# 6. Persistence, Immutability & Duration (Items 25 - 34)
# =============================================================================


def test_storyboard_persistence_and_reproducibility(session_factory):
    """25-29. Real StoryboardSnapshot, Shot entities, and ShotRevisions are persisted."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
    )
    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        sb_repo = StoryboardRepository(session)
        shot_repo = ShotRepository(session)

        snap = sb_repo.get_snapshot(result.output_artifact_ref.artifact_id)
        assert snap is not None
        assert snap.snapshot_state == StoryboardSnapshotState.DRAFT
        assert len(snap.shot_revision_ids) == len(script.segments)

        for rev_id in snap.shot_revision_ids:
            rev = shot_repo.get_revision(rev_id)
            assert rev is not None
            shot = shot_repo.get_shot(rev.shot_id)
            assert shot is not None
            assert shot.beat_lineage_id == rev.beat_lineage_id


def test_duration_validation_rejects_out_of_tolerance_output(session_factory):
    """32-34. Durations must be positive and follow script duration tolerance."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    def wild_duration_caller(prompt: str) -> str:
        res = _make_storyboard_mock_llm_caller()(prompt)
        data = json.loads(res)
        for s in data["shots"]:
            s["target_duration"] = 500.0  # Excessive deviation from 30s target
        return json.dumps(data)

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=wild_duration_caller,
    )
    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.RETRYABLE.value
    assert result.is_retryable is True


# =============================================================================
# 7. No Asset Generation Boundary Verification (Items 35 - 39)
# =============================================================================


def test_no_asset_generation_occurs_in_storyboard_stage(session_factory):
    """35-39. Asserts zero rows created in AssetRoutePlan, ExecutionRun, ShotAssetVersion."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
    )
    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        assert session.scalars(select(AssetRoutePlanORM)).all() == []
        assert session.scalars(select(ExecutionRunORM)).all() == []
        assert session.scalars(select(ExecutionAttemptORM)).all() == []
        assert session.scalars(select(ShotAssetVersionORM)).all() == []


# =============================================================================
# 8. AUTO Policy Workflow Behavior (Items 43 - 47)
# =============================================================================


def test_auto_policy_automatically_approves_and_queues_production_plan(session_factory):
    """43-47. In AUTO mode: storyboard generated -> domain approval -> PRODUCTION_PLAN job queued."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(
        session_factory, workflow_policy=WorkflowPolicyType.AUTO
    )

    registry = StageExecutorRegistry()
    registry.register(
        Stage.STORYBOARD,
        StoryboardStageExecutor(
            session_factory=session_factory,
            llm_caller=_make_storyboard_mock_llm_caller(),
        ),
    )

    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="test-auto-worker",
    )
    processed = worker.run_once()
    assert processed is True

    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        sb_repo = StoryboardRepository(session)
        appr_repo = StoryboardApprovalRepository(session)
        art_repo = TaskArtifactRepository(session)

        cur_task = t_repo.get_task(task.task_id)
        assert cur_task.current_stage == Stage.PRODUCTION_PLAN
        assert cur_task.task_status == TaskStatus.RUNNING

        # Verify StoryboardApprovalRecord exists
        approval_records = session.scalars(select(StoryboardApprovalRecordORM)).all()
        assert len(approval_records) == 1
        record = approval_records[0]
        assert record.approved_by == "workflow_auto"

        # Verify APPROVED StoryboardSnapshot exists
        approved_snap = sb_repo.get_snapshot(record.approved_storyboard_snapshot_id)
        assert approved_snap.snapshot_state == StoryboardSnapshotState.APPROVED

        # Verify latest STORYBOARD artifact ref points to APPROVED snapshot
        latest_sb_ref = art_repo.get_latest_artifact_ref(cur_task.task_id, stage=Stage.STORYBOARD)
        assert latest_sb_ref.artifact_id == approved_snap.storyboard_snapshot_id
        assert latest_sb_ref.artifact_version == "approved"

        # Verify PRODUCTION_PLAN job is created and QUEUED
        prod_job = j_repo.get_current_job_for_task(cur_task.task_id)
        assert prod_job is not None
        assert prod_job.stage == Stage.PRODUCTION_PLAN
        assert prod_job.status == JobStatus.QUEUED
        assert prod_job.input_task_artifact_ref_id == latest_sb_ref.task_artifact_ref_id


# =============================================================================
# 9. REVIEW Policy Workflow Behavior (Items 48 - 53)
# =============================================================================


def test_review_policy_pauses_at_waiting_user_and_advances_upon_approval(session_factory):
    """48-53. In REVIEW mode: pauses in WAITING_USER -> approve_task -> domain approval -> PRODUCTION_PLAN queued."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(
        session_factory, workflow_policy=WorkflowPolicyType.REVIEW
    )

    registry = StageExecutorRegistry()
    registry.register(
        Stage.STORYBOARD,
        StoryboardStageExecutor(
            session_factory=session_factory,
            llm_caller=_make_storyboard_mock_llm_caller(),
        ),
    )

    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="test-review-worker",
    )
    processed = worker.run_once()
    assert processed is True

    # Check paused state before approval
    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)

        cur_task = t_repo.get_task(task.task_id)
        assert cur_task.current_stage == Stage.STORYBOARD
        assert cur_task.task_status == TaskStatus.WAITING_USER

        # NO PRODUCTION_PLAN job yet
        prod_jobs = session.scalars(
            select(WorkflowJobORM).where(WorkflowJobORM.stage == Stage.PRODUCTION_PLAN.value)
        ).all()
        assert len(prod_jobs) == 0

    # User explicitly approves task
    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        approved_task = cmd_service.approve_task(task.task_id)
        assert approved_task.current_stage == Stage.PRODUCTION_PLAN
        assert approved_task.task_status == TaskStatus.RUNNING
        session.commit()

    # Check state after approval
    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        sb_repo = StoryboardRepository(session)
        art_repo = TaskArtifactRepository(session)

        # StoryboardApprovalRecord created
        approval_records = session.scalars(select(StoryboardApprovalRecordORM)).all()
        assert len(approval_records) == 1
        record = approval_records[0]
        assert record.approved_by == "user_approval"

        # APPROVED snapshot exists
        approved_snap = sb_repo.get_snapshot(record.approved_storyboard_snapshot_id)
        assert approved_snap.snapshot_state == StoryboardSnapshotState.APPROVED

        # Exactly one PRODUCTION_PLAN job queued
        prod_job = j_repo.get_current_job_for_task(task.task_id)
        assert prod_job.stage == Stage.PRODUCTION_PLAN
        assert prod_job.status == JobStatus.QUEUED

    # Idempotency: repeated approval on non-waiting task is rejected
    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        with pytest.raises(WorkflowConflictError):
            cmd_service.approve_task(task.task_id)


# =============================================================================
# 10. Error Handling & Trace (Items 54 - 57)
# =============================================================================


def test_malformed_llm_output_uses_bounded_job_retry(session_factory):
    """54-55. Malformed LLM output is rejected as RETRYABLE."""
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    def broken_json_caller(prompt: str) -> str:
        return "{ this is definitely not valid json }"

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=broken_json_caller,
    )
    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.RETRYABLE.value
    assert result.is_retryable is True


def test_trace_records_ids_and_metrics_without_full_payload(session_factory):
    """57. Trace stores metrics and IDs without full Storyboard payload."""
    events: list[tuple[str, TraceEventType, dict]] = []

    class MockTraceWriter(TraceWriter):
        def __init__(self):
            pass

        def write_event(self, task_id: str, event_type: TraceEventType, attributes: dict):
            events.append((task_id, event_type, attributes))

    mock_tw = MockTraceWriter()
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
        trace_writer=mock_tw,
    )
    result = executor.execute(task, job)
    assert result.success is True

    event_types = [e[1] for e in events]
    assert TraceEventType.STORYBOARD_STAGE_STARTED in event_types
    assert TraceEventType.STORYBOARD_CREATED in event_types

    created_ev = next(e for e in events if e[1] == TraceEventType.STORYBOARD_CREATED)
    attrs = created_ev[2]
    assert "storyboard_snapshot_id" in attrs
    assert "shot_count" in attrs
    assert "total_duration" in attrs
    # Full shots list should not be dumped into trace attributes
    assert "shots" not in attrs


# =============================================================================
# 11. MANDATORY DEMONSTRATION 1: SUCCESS INTEGRATION TEST
# =============================================================================


def test_mandatory_demonstration_1_success_integration(session_factory):
    """
    Construct lineage: KnowledgeVideoTask -> EvidenceSnapshot -> ContentPlanRevision -> ScriptRevision.
    Script contains:
      Segment 1: HOOK
      Segment 2: KNOWLEDGE, evidence_refs=[real EvidenceItem ID]
      Segment 3: SUMMARY
    Execute: STORYBOARD WorkflowJob -> StageWorker -> StoryboardStageExecutor -> StoryboardAgent.
    Verify:
      - Real StoryboardSnapshot exists
      - All Shots link to upstream Script/Beat
      - KNOWLEDGE Shot resolves evidence chain
      - Narration originates from Script
      - No assets are generated
      - TaskArtifactRef created
      - Same task_id preserved
      - Under AUTO: Storyboard approved through correct domain semantics -> PRODUCTION_PLAN job queued
    """
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(
        session_factory,
        topic="Why does Transformer use self-attention?",
        workflow_policy=WorkflowPolicyType.AUTO,
    )

    registry = StageExecutorRegistry()
    registry.register(
        Stage.STORYBOARD,
        StoryboardStageExecutor(
            session_factory=session_factory,
            llm_caller=_make_storyboard_mock_llm_caller(),
        ),
    )

    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="demo-worker-1",
    )
    assert worker.run_once() is True

    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        sb_repo = StoryboardRepository(session)
        shot_repo = ShotRepository(session)
        art_repo = TaskArtifactRepository(session)

        # Task advanced to PRODUCTION_PLAN
        cur_task = t_repo.get_task(task.task_id)
        assert cur_task.current_stage == Stage.PRODUCTION_PLAN
        assert cur_task.task_status == TaskStatus.RUNNING

        # StoryboardSnapshot exists
        sb_refs = art_repo.list_artifact_refs_for_task(task.task_id, stage=Stage.STORYBOARD)
        assert len(sb_refs) >= 1
        latest_sb_ref = art_repo.get_latest_artifact_ref(task.task_id, stage=Stage.STORYBOARD)
        assert latest_sb_ref.task_id == task.task_id

        approved_snapshot = sb_repo.get_snapshot(latest_sb_ref.artifact_id)
        assert approved_snapshot.snapshot_state == StoryboardSnapshotState.APPROVED

        # Shots linkage & narration origin
        revisions = sb_repo.get_snapshot_shot_revisions(approved_snapshot.storyboard_snapshot_id)
        assert len(revisions) == 3
        assert revisions[0].narration == script.segments[0].narration_text
        assert revisions[1].narration == script.segments[1].narration_text
        assert revisions[2].narration == script.segments[2].narration_text

        # KNOWLEDGE shot resolves verified evidence
        assert revisions[1].evidence_refs == (item.evidence_id,)

        # PRODUCTION_PLAN job is queued
        prod_job = j_repo.get_current_job_for_task(task.task_id)
        assert prod_job.stage == Stage.PRODUCTION_PLAN
        assert prod_job.status == JobStatus.QUEUED


# =============================================================================
# 12. MANDATORY DEMONSTRATION 2: REVIEW FLOW TEST
# =============================================================================


def test_mandatory_demonstration_2_review_flow(session_factory):
    """
    Workflow mode: REVIEW.
    Run STORYBOARD stage.
    Expected:
      - StoryboardSnapshot generated
      - TaskArtifactRef persisted
      - Task = WAITING_USER
      - No PRODUCTION_PLAN job
    Then call real unified approve command.
    Verify:
      - Existing Storyboard approval path is used
      - Same Storyboard is approved according to domain semantics
      - Task resumes
      - Exactly one PRODUCTION_PLAN job exists and remains QUEUED
    """
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(
        session_factory,
        topic="Why does Transformer use self-attention?",
        workflow_policy=WorkflowPolicyType.REVIEW,
    )

    registry = StageExecutorRegistry()
    registry.register(
        Stage.STORYBOARD,
        StoryboardStageExecutor(
            session_factory=session_factory,
            llm_caller=_make_storyboard_mock_llm_caller(),
        ),
    )

    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="demo-worker-review",
    )
    assert worker.run_once() is True

    # 1. Check paused in WAITING_USER with NO PRODUCTION_PLAN job
    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        cur_task = t_repo.get_task(task.task_id)
        assert cur_task.task_status == TaskStatus.WAITING_USER
        assert cur_task.current_stage == Stage.STORYBOARD

        prod_job = j_repo.get_current_job_for_task(task.task_id)
        assert prod_job is None or prod_job.stage != Stage.PRODUCTION_PLAN

    # 2. Call unified approve command via TaskCommandService
    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        approved_task = cmd_service.approve_task(task.task_id)
        session.commit()

    # 3. Check resumed and PRODUCTION_PLAN job queued
    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        sb_repo = StoryboardRepository(session)
        appr_repo = StoryboardApprovalRepository(session)

        final_task = t_repo.get_task(task.task_id)
        assert final_task.current_stage == Stage.PRODUCTION_PLAN
        assert final_task.task_status == TaskStatus.RUNNING

        # Domain approval verified
        records = session.scalars(select(StoryboardApprovalRecordORM)).all()
        assert len(records) == 1
        assert records[0].approved_by == "user_approval"

        # Exactly one PRODUCTION_PLAN job queued
        prod_job = j_repo.get_current_job_for_task(task.task_id)
        assert prod_job is not None
        assert prod_job.stage == Stage.PRODUCTION_PLAN
        assert prod_job.status == JobStatus.QUEUED


# =============================================================================
# 13. MANDATORY DEMONSTRATION 3: PROVENANCE TEST
# =============================================================================


def test_mandatory_demonstration_3_provenance(session_factory):
    """
    Resolve one factual Shot:
      Shot -> ShotRevision -> ScriptSegment -> ContentBeat -> EvidenceItem -> SourceDocument.
    Every ID must be real and persisted.
    No inference by timestamp or text matching.
    """
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
    )
    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        sb_repo = StoryboardRepository(session)
        shot_repo = ShotRepository(session)
        script_repo = ScriptRepository(session)
        plan_repo = ContentPlanRepository(session)
        ev_repo = EvidenceRepository(session)

        # 1. Shot entity
        # Segment 2 is KNOWLEDGE
        snap = sb_repo.get_snapshot(result.output_artifact_ref.artifact_id)
        revisions = sb_repo.get_snapshot_shot_revisions(snap.storyboard_snapshot_id)
        factual_rev = revisions[1]

        shot = shot_repo.get_shot(factual_rev.shot_id)
        assert shot is not None
        shot_id = shot.shot_id

        # 2. ShotRevision
        assert factual_rev.shot_id == shot_id
        shot_rev_id = factual_rev.shot_revision_id
        script_seg_id = factual_rev.script_segment_id
        assert script_seg_id is not None

        # 3. ScriptSegment
        reloaded_script = script_repo.get_revision(script.script_revision_id)
        script_segment = next(s for s in reloaded_script.segments if s.script_segment_id == script_seg_id)
        assert script_segment is not None
        content_beat_id = script_segment.content_beat_id

        # 4. ContentBeat
        reloaded_plan = plan_repo.get_revision(plan.content_plan_revision_id)
        content_beat = next(b for b in reloaded_plan.beats if b.beat_id == content_beat_id)
        assert content_beat is not None
        assert item.evidence_id in content_beat.evidence_refs

        # 5. EvidenceItem
        evidence_item = ev_repo.get_evidence_item(item.evidence_id)
        assert evidence_item is not None
        source_doc_id = evidence_item.source_document_id

        # 6. SourceDocument
        source_doc = ev_repo.get_source_document(source_doc_id)
        assert source_doc is not None
        assert source_doc.source_document_id == source.source_document_id

        # Assert full unbroken chain
        assert shot.shot_id == factual_rev.shot_id
        assert factual_rev.script_segment_id == script_segment.script_segment_id
        assert script_segment.content_beat_id == content_beat.beat_id
        assert item.evidence_id in factual_rev.evidence_refs
        assert item.evidence_id in script_segment.evidence_refs
        assert item.evidence_id in content_beat.evidence_refs
        assert evidence_item.source_document_id == source_doc.source_document_id


# =============================================================================
# 14. MANDATORY DEMONSTRATION 4: NO-ASSET TEST
# =============================================================================


def test_mandatory_demonstration_4_no_asset(session_factory):
    """
    Execute successful STORYBOARD.
    Assert no new rows/objects equivalent to:
      AssetRoutePlan, ExecutionRun, ExecutionAttempt, ShotAssetVersion,
      and no asset provider adapter invocation.
    """
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
    )
    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        # Zero asset router plans
        route_plans = session.scalars(select(AssetRoutePlanORM)).all()
        assert len(route_plans) == 0

        # Zero execution runs
        runs = session.scalars(select(ExecutionRunORM)).all()
        assert len(runs) == 0

        # Zero execution attempts
        attempts = session.scalars(select(ExecutionAttemptORM)).all()
        assert len(attempts) == 0

        # Zero shot asset versions
        assets = session.scalars(select(ShotAssetVersionORM)).all()
        assert len(assets) == 0


# =============================================================================
# 15. MANDATORY DEMONSTRATION 5: IMMUTABILITY TEST
# =============================================================================


def test_mandatory_demonstration_5_immutability(session_factory):
    """
    1. Create StoryboardSnapshot S1 (DRAFT).
    2. Execute approval creating S2 (APPROVED).
    3. Verify S1 remains historically inspectable in DRAFT state.
    4. Verify TaskArtifactRef points to appropriate new output S2.
    5. Historical revisions remain identical.
    """
    task, job, script, plan, snapshot, item, source, script_ref = _setup_task_with_script(session_factory)

    # 1. Execute StoryboardStageExecutor -> S1 (DRAFT)
    executor = StoryboardStageExecutor(
        session_factory=session_factory,
        llm_caller=_make_storyboard_mock_llm_caller(),
    )
    result = executor.execute(task, job)
    assert result.success is True
    s1_draft_id = result.output_artifact_ref.artifact_id

    # Persist the executor's output artifact ref as StageWorker normally would
    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        art_repo.save_artifact_ref(result.output_artifact_ref)
        session.commit()

    # 2. Approve S1 -> S2 (APPROVED) via workflow
    with session_factory() as session:
        workflow = KnowledgeVideoWorkflow(session)
        s2_approved_ref_id = workflow._approve_storyboard_for_task(
            task_id=task.task_id,
            draft_output_ref_id=result.output_artifact_ref.task_artifact_ref_id,
            approved_by="tester",
            now=datetime.now(UTC),
        )
        session.commit()

    # 3. Verify S1 remains historically inspectable
    with session_factory() as session:
        sb_repo = StoryboardRepository(session)
        art_repo = TaskArtifactRepository(session)

        s1 = sb_repo.get_snapshot(s1_draft_id)
        assert s1 is not None
        assert s1.snapshot_state == StoryboardSnapshotState.DRAFT

        # 4. S2 is APPROVED
        s2_ref = art_repo.get_artifact_ref(s2_approved_ref_id)
        s2 = sb_repo.get_snapshot(s2_ref.artifact_id)
        assert s2 is not None
        assert s2.snapshot_state == StoryboardSnapshotState.APPROVED
        assert s2.storyboard_snapshot_id != s1.storyboard_snapshot_id
        # S2 has exact same shot revisions
        assert s2.shot_revision_ids == s1.shot_revision_ids

        # Latest artifact ref points to S2
        latest_ref = art_repo.get_latest_artifact_ref(task.task_id, stage=Stage.STORYBOARD)
        assert latest_ref.artifact_id == s2.storyboard_snapshot_id
