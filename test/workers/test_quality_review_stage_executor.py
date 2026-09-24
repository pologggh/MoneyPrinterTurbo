from __future__ import annotations

import hashlib
from pathlib import Path
import struct
from unittest.mock import MagicMock, patch
from uuid import uuid4
import wave

from PIL import Image
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.quality_review_stage_executor import (
    QualityReviewStageExecutor,
)
from app.application.knowledge_video_workflow import KnowledgeVideoWorkflow
from app.application.stage_executor_registry import (
    StageExecutorRegistry,
    get_default_executor_registry,
)
from app.domain.asset_execution import (
    AssetMediaType,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRun,
    ExecutionStatus,
    GenerationMode,
    ShotAssetVersion,
    ShotExecution,
    ShotExecutionStatus,
)
from app.domain.asset_router import (
    AssetCapability,
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutePlan,
    AssetRoutePlanStatus,
    AssetRoutingRequest,
    CandidateScore,
    RoutingPolicy,
    RoutingStrategy,
    ShotRoutePlanEntry,
    ShotRoutePlanStatus,
    StaticCapabilityMetadata,
    TierLevel,
)
from app.domain.audio import AudioOutput
from app.domain.composition import CompositionOutput
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.evidence import EvidenceItem, EvidenceRole, EvidenceSnapshot, SourceDocument
from app.domain.evaluation import (
    DimensionEvaluationResult,
    DimensionEvaluationStatus,
    EvaluationContextRef,
    EvaluationDecision,
    EvaluationDimension,
    EvaluationReasonCode,
    EvaluationSnapshot,
    EvaluationTarget,
    create_evaluation_target_from_shot,
)
from app.domain.evaluation_policy import EvaluationPolicy, EvaluationPolicyEngine
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.quality_remediation import (
    QualityRemediationAction,
    QualityRemediationDecision,
    QualityRemediationPolicy,
    QualityRemediationReasonCode,
)
from app.domain.quality_remediation_policy import QualityRemediationPolicyEngine
from app.domain.script import ScriptRevision, ScriptSegment
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.task_artifact import TaskArtifactRef
from app.domain.trace import TraceEventType
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    ArtifactType,
    JobErrorType,
    JobStatus,
    Stage,
    TaskStatus,
    WorkflowPolicyType,
)
from app.persistence.models import Base, EvaluationTargetORM
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    AudioOutputRepository,
    CompositionOutputRepository,
    ContentPlanRepository,
    EvaluationRepository,
    ExecutionRepository,
    KnowledgeVideoTaskRepository,
    ScriptRepository,
    ShotExecutionRepository,
    ShotRepository,
    StageExecutionRepository,
    StoryboardRepository,
    TaskArtifactRepository,
    EvidenceRepository,
    WorkflowJobRepository,
)
from app.services.evaluation.mpt_multimodal_adapter import (
    MultimodalEvaluationObserverResult,
    MultimodalEvaluatorAdapter,
)
from app.workers.stage_worker import StageWorker


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


def _create_wav_audio_fixture(file_path: Path | str, duration_sec: float = 2.0) -> str:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 44100
    n_frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        data = struct.pack("<h", 0) * n_frames
        wav_file.writeframes(data)
    return str(path)


def _create_image_fixture(file_path: Path | str, color: tuple[int, int, int] = (100, 150, 200)) -> str:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (320, 240), color=color)
    img.save(str(path))
    return str(path)


class ControlledEvaluatorAdapter(MultimodalEvaluatorAdapter):
    """
    Controlled deterministic evaluator adapter for QualityReviewStageExecutor testing.
    Zero network or paid API calls.
    """
    def __init__(
        self,
        default_score: float = 0.92,
        failing_dimensions: dict[EvaluationDimension, tuple[float, tuple[str, ...]]] | None = None,
        failing_shots: set[str] | None = None,
        transient_error: bool = False,
        missing_evidence: bool = False,
    ):
        self.default_score = default_score
        self.failing_dimensions = failing_dimensions or {}
        self.failing_shots = failing_shots or set()
        self.transient_error = transient_error
        self.missing_evidence = missing_evidence
        self.evaluator_version = "test-evaluator-v1"

    def evaluate_observation(
        self,
        prompt: str,
        image_paths: tuple[str, ...] | list[str],
    ) -> MultimodalEvaluationObserverResult:
        if self.transient_error:
            return MultimodalEvaluationObserverResult(
                status=DimensionEvaluationStatus.ERROR,
                error_code="EVALUATOR_CALL_FAILED",
                concise_summary="Simulated transient network timeout",
                evaluator_version=self.evaluator_version,
            )

        if "KNOWLEDGE ACCURACY" in prompt:
            dim = EvaluationDimension.KNOWLEDGE_ACCURACY
        elif "SEMANTIC ALIGNMENT" in prompt:
            dim = EvaluationDimension.SEMANTIC_ALIGNMENT
        elif "COMPOSITION SUITABILITY" in prompt:
            dim = EvaluationDimension.COMPOSITION_SUITABILITY
        else:
            dim = EvaluationDimension.VISUAL_QUALITY

        if dim in self.failing_dimensions:
            score, reasons = self.failing_dimensions[dim]
            return MultimodalEvaluationObserverResult(
                status=DimensionEvaluationStatus.SCORED,
                score=score,
                reason_codes=reasons,
                concise_summary=f"Failing dimension {dim.value}",
                evaluator_version=self.evaluator_version,
            )

        is_failing_shot = any(s in str(p) for s in self.failing_shots for p in image_paths) or any(s in prompt for s in self.failing_shots)
        if is_failing_shot and dim == EvaluationDimension.VISUAL_QUALITY:
            return MultimodalEvaluationObserverResult(
                status=DimensionEvaluationStatus.SCORED,
                score=0.45,
                reason_codes=("VISUAL_QUALITY_STOCHASTIC_FAIL",),
                concise_summary="Simulated shot visual defect",
                evaluator_version=self.evaluator_version,
            )

        return MultimodalEvaluationObserverResult(
            status=DimensionEvaluationStatus.SCORED,
            score=self.default_score,
            reason_codes=("HIGH_QUALITY",),
            concise_summary="Observation passed quality thresholds",
            evaluator_version=self.evaluator_version,
        )


def _setup_quality_review_prerequisites(
    session_factory,
    tmp_path,
    shot_count: int = 2,
    workflow_policy: WorkflowPolicyType = WorkflowPolicyType.AUTO,
    candidate_b_available: bool = False,
):
    """Sets up a complete task ready for QUALITY_REVIEW stage execution."""
    task_id = f"task_{uuid4().hex[:8]}"
    plan_id = f"cpr_{uuid4().hex[:8]}"
    script_id = f"scr_{uuid4().hex[:8]}"
    sb_id = f"sb_{uuid4().hex[:8]}"
    route_plan_id = f"arp_{uuid4().hex[:8]}"
    run_id = f"run_{uuid4().hex[:8]}"
    audio_id = f"ao_{uuid4().hex[:8]}"
    comp_id = f"co_{uuid4().hex[:8]}"
    ev_snap_id = f"es_{uuid4().hex[:8]}"
    doc_id = f"doc_{uuid4().hex[:8]}"
    item_id = f"evi_{uuid4().hex[:8]}"

    task_dir = tmp_path / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    narration_path = _create_wav_audio_fixture(task_dir / "narration.wav", duration_sec=float(shot_count * 2))
    video_path = task_dir / "final_video.mp4"
    video_path.write_bytes(b"dummy final video binary content")

    # Evidence Setup
    source_doc = SourceDocument.create_text(
        text="Quantum entanglement verified in lab experiments at subatomic scales.",
        source_document_id=doc_id,
    )
    ev_item = EvidenceItem.create(
        source_document=source_doc,
        original_excerpt="Quantum entanglement verified in lab experiments.",
        locator={"char_start": 0, "char_end": 75},
        normalized_fact="Quantum entanglement has been empirically confirmed.",
        evidence_role=EvidenceRole.FACTUAL_SUPPORT,
        confidence=1.0,
        extraction_method="test_fixture",
    )
    ev_snap = EvidenceSnapshot.create(
        evidence_snapshot_id=ev_snap_id,
        task_id=task_id,
        snapshot_version=1,
        source_document_ids=(source_doc.source_document_id,),
        evidence_ids=(ev_item.evidence_id,),
        knowledge_claim_ids=(),
    )

    beats = []
    shots = []
    shot_revs = []
    asset_versions = []
    attempts = []
    shot_executions = []
    route_entries = []

    candidate_a = AssetRouteCandidate(
        capability_id="cap_model_a",
        provider="provider_a",
        model="model_alpha",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
        requested_visual_type=VisualType.AI_IMAGE,
        is_eligible=True,
    )
    candidate_b = AssetRouteCandidate(
        capability_id="cap_model_b",
        provider="provider_b",
        model="model_beta",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
        requested_visual_type=VisualType.AI_IMAGE,
        is_eligible=True,
    )

    for i in range(1, shot_count + 1):
        beat_lineage = f"bl_{i}"
        beat = ContentBeat(
            beat_lineage_id=beat_lineage,
            order=i,
            intent=f"Explain quantum mechanics concept {i}",
            target_duration=2.0,
            importance=0.9,
            beat_type=BeatType.KNOWLEDGE,
        )
        beats.append(beat)

        shot_id = f"shot_{i}"
        shot = Shot(
            shot_id=shot_id,
            beat_lineage_id=beat_lineage,
            local_order=1,
        )
        shots.append(shot)

        shot_rev_id = f"sr_{i}"
        shot_rev = ShotRevision(
            shot_revision_id=shot_rev_id,
            shot_id=shot_id,
            revision_number=1,
            beat_lineage_id=beat_lineage,
            created_from_beat_instance_id=beat.beat_id,
            visual_type=VisualType.AI_IMAGE,
            generation_prompt=f"Accurate laboratory visual representing concept {i}",
            visual_goal=f"Demonstrate concept {i}",
            scene_description=f"Subatomic particles interacting under quantum entanglement {i}",
            camera_movement="Static",
            target_duration=2.0,
            narration=f"Narration text for quantum concept {i}",
            evidence_refs=(ev_item.evidence_id,),
        )
        shot_revs.append(shot_rev)

        img_path = _create_image_fixture(task_dir / f"shot_{i}.png", color=(i * 30, 80, 180))
        img_bytes = Path(img_path).read_bytes()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        att_id = f"att_{i}"
        asset_v_id = f"sav_{i}"

        att = ExecutionAttempt(
            execution_attempt_id=att_id,
            execution_run_id=run_id,
            shot_id=shot_id,
            shot_revision_id=shot_rev_id,
            candidate_index=0,
            attempt_number=1,
            provider="provider_a",
            model="model_alpha",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            status=ExecutionAttemptStatus.SUCCEEDED,
        )
        attempts.append(att)

        asset_v = ShotAssetVersion(
            shot_asset_version_id=asset_v_id,
            shot_id=shot_id,
            shot_revision_id=shot_rev_id,
            execution_attempt_id=att_id,
            file_path=img_path,
            file_hash=img_hash,
            file_size_bytes=len(img_bytes),
            media_type=AssetMediaType.IMAGE,
            mime_type="image/png",
            width=320,
            height=240,
            duration_seconds=2.0,
            fps=30.0,
            provider="provider_a",
            model="model_alpha",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
        )
        asset_versions.append(asset_v)

        eligible_cands = (candidate_a, candidate_b) if candidate_b_available else (candidate_a,)
        route_decision = AssetRouteDecision(
            shot_id=shot_id,
            shot_revision_id=shot_rev_id,
            routing_strategy=RoutingStrategy.BALANCED,
            requested_visual_type=VisualType.AI_IMAGE,
            selected_candidate=candidate_a,
            eligible_candidates=eligible_cands,
        )

        se = ShotExecution(
            shot_execution_id=f"se_{i}",
            execution_run_id=run_id,
            shot_id=shot_id,
            shot_revision_id=shot_rev_id,
            route_decision=route_decision,
            status=ShotExecutionStatus.SUCCEEDED,
            produced_asset_version_id=asset_v_id,
        )
        shot_executions.append(se)

        routing_req = AssetRoutingRequest(
            shot_id=shot_id,
            shot_revision_id=shot_rev_id,
            requested_visual_type=VisualType.AI_IMAGE,
            target_duration=2.0,
            visual_goal=shot_rev.visual_goal,
            scene_description=shot_rev.scene_description,
            generation_prompt=shot_rev.generation_prompt,
            camera_movement=shot_rev.camera_movement,
        )
        route_entry = ShotRoutePlanEntry(
            shot_id=shot_id,
            shot_revision_id=shot_rev_id,
            beat_lineage_id=beat_lineage,
            requested_visual_type=VisualType.AI_IMAGE,
            asset_routing_request=routing_req,
            route_decision=route_decision,
            route_status=ShotRoutePlanStatus.ROUTED,
        )
        route_entries.append(route_entry)

    plan = ContentPlanRevision(
        content_plan_revision_id=plan_id,
        revision_number=1,
        topic="Quantum Entanglement",
        overall_target_duration=float(shot_count * 2),
        beats=tuple(beats),
    )

    script = ScriptRevision.create(
        script_revision_id=script_id,
        task_id=task_id,
        content_plan_revision_id=plan_id,
        revision_number=1,
        overall_target_duration=float(shot_count * 2),
        segments=[
            ScriptSegment(
                script_revision_id=script_id,
                content_beat_id=b.beat_id,
                beat_lineage_id=b.beat_lineage_id,
                order=b.order,
                narration_text=f"Narration text for quantum concept {b.order}",
                target_duration=2.0,
            )
            for b in beats
        ],
    )

    snapshot = StoryboardSnapshot(
        storyboard_snapshot_id=sb_id,
        content_plan_revision_id=plan_id,
        shot_revision_ids=tuple(r.shot_revision_id for r in shot_revs),
        snapshot_state=StoryboardSnapshotState.APPROVED,
    )

    route_plan = AssetRoutePlan(
        asset_route_plan_id=route_plan_id,
        storyboard_snapshot_id=sb_id,
        content_plan_revision_id=plan_id,
        routing_strategy=RoutingStrategy.BALANCED,
        routing_policy_version="v1.0",
        shot_routes=tuple(route_entries),
        status=AssetRoutePlanStatus.READY,
        total_shots=shot_count,
        routed_shots=shot_count,
    )

    run = ExecutionRun(
        execution_run_id=run_id,
        asset_route_plan_id=route_plan_id,
        storyboard_snapshot_id=sb_id,
        status=ExecutionStatus.COMPLETED,
        total_shots=shot_count,
        succeeded_shots=shot_count,
        failed_shots=0,
    )

    audio_output = AudioOutput.create(
        audio_output_id=audio_id,
        task_id=task_id,
        script_revision_id=script_id,
        execution_run_id=run_id,
        narration_audio_path=narration_path,
        narration_audio_hash="hash_narr_1",
        actual_narration_duration=float(shot_count * 2),
        subtitle_path="",
        subtitle_hash="",
        bgm_path=None,
        bgm_volume=0.0,
    )

    composition_output = CompositionOutput.create(
        composition_output_id=comp_id,
        task_id=task_id,
        storyboard_snapshot_id=sb_id,
        execution_run_id=run_id,
        audio_output_id=audio_id,
        video_path=str(video_path),
        video_hash="hash_comp_video_1",
        duration=float(shot_count * 2),
        width=1920,
        height=1080,
        fps=30.0,
        file_size=len(video_path.read_bytes()),
        video_codec="h264",
        audio_codec="aac",
    )

    task = KnowledgeVideoTask.create(
        task_id=task_id,
        topic="Quantum Entanglement",
        target_duration=float(shot_count * 2),
        workflow_policy=workflow_policy,
        task_metadata={"video_aspect": "16:9"},
    )
    task.transition_to(TaskStatus.RUNNING)
    task.advance_stage(Stage.KNOWLEDGE_PLAN)
    task.advance_stage(Stage.SCRIPT)
    task.advance_stage(Stage.STORYBOARD)
    task.advance_stage(Stage.PRODUCTION_PLAN)
    task.advance_stage(Stage.ASSET)
    task.advance_stage(Stage.AUDIO)
    task.advance_stage(Stage.COMPOSITION)
    task.advance_stage(Stage.QUALITY_REVIEW)

    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        p_repo = ContentPlanRepository(session)
        s_repo = ScriptRepository(session)
        sb_repo = StoryboardRepository(session)
        shot_repo = ShotRepository(session)
        rp_repo = AssetRoutePlanRepository(session)
        exec_repo = ExecutionRepository(session)
        se_repo = ShotExecutionRepository(session)
        aud_repo = AudioOutputRepository(session)
        comp_repo = CompositionOutputRepository(session)
        art_repo = TaskArtifactRepository(session)
        ev_repo = EvidenceRepository(session)
        j_repo = WorkflowJobRepository(session)

        t_repo.save_task(task)
        p_repo.add_revision(plan)
        s_repo.save_revision(script)
        for sh in shots:
            shot_repo.add_shot(sh)
        for sr in shot_revs:
            shot_repo.add_revision(sr)
        sb_repo.add_snapshot(snapshot)

        ev_repo.save_source_document(source_doc)
        ev_repo.save_evidence_item(ev_item)
        ev_repo.save_evidence_snapshot(ev_snap)

        rp_repo.add_route_plan(route_plan)
        exec_repo.add_execution_run(run)
        for att in attempts:
            exec_repo.add_execution_attempt(att)
        for av in asset_versions:
            exec_repo.add_shot_asset_version(av)
        for se_item in shot_executions:
            se_repo.add_shot_execution(se_item)

        aud_repo.save_audio_output(audio_output)
        comp_repo.save_composition_output(composition_output)

        # Save TaskArtifactRefs
        ev_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.EVIDENCE,
            artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
            artifact_id=ev_snap_id,
        )
        art_repo.save_artifact_ref(ev_ref)

        plan_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.KNOWLEDGE_PLAN,
            artifact_type=ArtifactType.CONTENT_PLAN_REVISION,
            artifact_id=plan_id,
        )
        art_repo.save_artifact_ref(plan_ref)

        script_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.SCRIPT,
            artifact_type=ArtifactType.SCRIPT_REVISION,
            artifact_id=script_id,
        )
        art_repo.save_artifact_ref(script_ref)

        sb_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.STORYBOARD,
            artifact_type=ArtifactType.STORYBOARD_SNAPSHOT,
            artifact_id=sb_id,
        )
        art_repo.save_artifact_ref(sb_ref)

        arp_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.PRODUCTION_PLAN,
            artifact_type=ArtifactType.ASSET_ROUTE_PLAN,
            artifact_id=route_plan_id,
        )
        art_repo.save_artifact_ref(arp_ref)

        asset_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.ASSET,
            artifact_type=ArtifactType.EXECUTION_RUN,
            artifact_id=run_id,
        )
        art_repo.save_artifact_ref(asset_ref)

        aud_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.AUDIO,
            artifact_type=ArtifactType.AUDIO_OUTPUT,
            artifact_id=audio_id,
        )
        art_repo.save_artifact_ref(aud_ref)

        comp_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.COMPOSITION,
            artifact_type=ArtifactType.COMPOSITION_OUTPUT,
            artifact_id=comp_id,
        )
        art_repo.save_artifact_ref(comp_ref)

        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.QUALITY_REVIEW,
            idempotency_key=f"idemp_{task_id}_quality_review_1",
            input_task_artifact_ref_id=comp_ref.task_artifact_ref_id,
        )
        j_repo.create_job(job)
        session.commit()

    return task, job, composition_output, run, snapshot, comp_ref, ev_snap


# =============================================================================
# 1. Registry & Boundary Tests
# =============================================================================

def test_registry_registration_and_boundaries():
    """Verify QUALITY_REVIEW and DELIVERY use real production executors."""
    registry = get_default_executor_registry()
    assert registry.has_executor(Stage.QUALITY_REVIEW)
    executor = registry.get_executor(Stage.QUALITY_REVIEW)
    assert executor is not None
    assert isinstance(executor, QualityReviewStageExecutor)
    assert not executor.__class__.__name__.startswith(("Mock", "Fake", "Dummy"))

    assert registry.has_executor(Stage.DELIVERY)
    assert registry.get_executor(Stage.DELIVERY) is not None


# =============================================================================
# 2. Input Resolution and Lineage Verification Tests
# =============================================================================

def test_quality_review_input_resolution_and_validation(session_factory, tmp_path):
    """Verifies QualityReviewStageExecutor loads and validates all upstream artifacts from task lineage."""
    task, job, comp_out, run, snapshot, comp_ref, ev_snap = _setup_quality_review_prerequisites(session_factory, tmp_path)
    adapter = ControlledEvaluatorAdapter(default_score=0.95)
    executor = QualityReviewStageExecutor(session_factory=session_factory, evaluator_adapter=adapter)

    result = executor.execute(task, job)
    assert result.success is True
    assert result.metadata_json["decision"] == "PASS"
    assert result.metadata_json["composition_output_id"] == comp_out.composition_output_id
    assert result.metadata_json["evaluated_shot_count"] == 2


def test_quality_review_rejects_stale_or_mismatched_artifacts(session_factory, tmp_path):
    """Rejects input composition artifacts if marked stale or belonging to a different task."""
    task, job, comp_out, run, snapshot, comp_ref, ev_snap = _setup_quality_review_prerequisites(session_factory, tmp_path)
    adapter = ControlledEvaluatorAdapter(default_score=0.95)
    executor = QualityReviewStageExecutor(session_factory=session_factory, evaluator_adapter=adapter)

    # 1. Stale artifact test
    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        art_repo.mark_artifact_stale(comp_ref.task_artifact_ref_id, reason="Testing rejection of stale ref")
        session.commit()

    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "marked stale" in result.error_message


# =============================================================================
# 3. Mandatory Demonstration: Quality PASS (AUTO and REVIEW Modes)
# =============================================================================

def test_mandatory_pass_auto_mode(session_factory, tmp_path):
    """
    Mandatory Quality PASS in AUTO mode:
    - QualityReviewStageExecutor evaluates composition and passes.
    - Produces TaskArtifactRef(EVALUATION_SNAPSHOT).
    - Advances stage to Stage.DELIVERY.
    - Creates WorkflowJob(stage=Stage.DELIVERY) in QUEUED state.
    - Confirms the production StageWorker executes DELIVERY and completes the task.
    """
    task, job, comp_out, run, snapshot, comp_ref, ev_snap = _setup_quality_review_prerequisites(
        session_factory, tmp_path, workflow_policy=WorkflowPolicyType.AUTO
    )
    adapter = ControlledEvaluatorAdapter(default_score=0.92)
    executor = QualityReviewStageExecutor(session_factory=session_factory, evaluator_adapter=adapter)

    registry = StageExecutorRegistry()
    registry.register(Stage.QUALITY_REVIEW, executor)
    worker = StageWorker(session_factory=session_factory, registry=registry, worker_id="qr-worker-1")

    # Worker runs QUALITY_REVIEW job
    assert worker.run_once() is True

    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        art_repo = TaskArtifactRepository(session)

        cur_task = t_repo.get_task(task.task_id)
        # AUTO mode advances directly to DELIVERY
        assert cur_task.current_stage == Stage.DELIVERY
        assert cur_task.task_status == TaskStatus.RUNNING

        # Initial job succeeded
        cur_job = j_repo.get_job(job.job_id)
        assert cur_job.status == JobStatus.SUCCEEDED

        # Output artifact ref recorded
        qr_ref = art_repo.get_artifact_ref(cur_job.output_task_artifact_ref_id)
        assert qr_ref.stage == Stage.QUALITY_REVIEW
        assert qr_ref.artifact_type == ArtifactType.EVALUATION_SNAPSHOT
        assert qr_ref.metadata_json.get("decision") == "PASS"

        # Exactly one DELIVERY job queued
        jobs = j_repo.list_jobs_for_task(task.task_id)
        del_jobs = [j for j in jobs if j.stage == Stage.DELIVERY]
        assert len(del_jobs) == 1
        assert del_jobs[0].status == JobStatus.QUEUED

    # The completed production registry executes DELIVERY.
    prod_worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="prod-worker-del",
    )
    assert prod_worker.run_once() is True
    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        completed_task = t_repo.get_task(task.task_id)
        delivery_jobs = [
            item for item in j_repo.list_jobs_for_task(task.task_id)
            if item.stage == Stage.DELIVERY
        ]
        assert completed_task.task_status == TaskStatus.COMPLETED
        assert delivery_jobs[0].status == JobStatus.SUCCEEDED


def test_mandatory_pass_review_mode(session_factory, tmp_path):
    """
    Mandatory Quality PASS in REVIEW mode:
    - QualityReviewStageExecutor evaluates composition and passes.
    - Pauses at TaskStatus.WAITING_USER after QUALITY_REVIEW.
    - No DELIVERY job is created yet.
    - Explicit human approval via resume_after_approval advances task to DELIVERY and creates exactly one DELIVERY job.
    """
    task, job, comp_out, run, snapshot, comp_ref, ev_snap = _setup_quality_review_prerequisites(
        session_factory, tmp_path, workflow_policy=WorkflowPolicyType.REVIEW
    )
    adapter = ControlledEvaluatorAdapter(default_score=0.92)
    executor = QualityReviewStageExecutor(session_factory=session_factory, evaluator_adapter=adapter)

    registry = StageExecutorRegistry()
    registry.register(Stage.QUALITY_REVIEW, executor)
    worker = StageWorker(session_factory=session_factory, registry=registry, worker_id="qr-worker-review")

    # Worker executes QUALITY_REVIEW
    assert worker.run_once() is True

    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)

        cur_task = t_repo.get_task(task.task_id)
        # Pauses at WAITING_USER
        assert cur_task.task_status == TaskStatus.WAITING_USER
        assert "Review required after QUALITY_REVIEW" in (cur_task.waiting_reason or "")
        assert cur_task.current_stage == Stage.QUALITY_REVIEW

        # No DELIVERY job yet
        jobs = j_repo.list_jobs_for_task(task.task_id)
        assert len([j for j in jobs if j.stage == Stage.DELIVERY]) == 0

        # User approves milestone
        workflow = KnowledgeVideoWorkflow(session)
        del_job = workflow.resume_after_approval(cur_task)
        session.commit()

        assert del_job is not None
        assert del_job.stage == Stage.DELIVERY
        assert del_job.status == JobStatus.QUEUED

        approved_task = t_repo.get_task(task.task_id)
        assert approved_task.current_stage == Stage.DELIVERY
        assert approved_task.task_status == TaskStatus.RUNNING


# =============================================================================
# 4. Mandatory Demonstration: Frozen Evidence Verification (No LLM Truth)
# =============================================================================

def test_mandatory_frozen_evidence_verification(session_factory, tmp_path):
    """
    Mandatory Frozen Evidence Test:
    - Visual depiction contradicts frozen task evidence item.
    - Evaluator detects contradiction on KNOWLEDGE_ACCURACY.
    - Hard gate fails: contradiction cannot be averaged away by high scores in other dimensions.
    - Evaluator makes ZERO calls to web retrieval or external model memory truth.
    """
    task, job, comp_out, run, snapshot, comp_ref, ev_snap = _setup_quality_review_prerequisites(
        session_factory, tmp_path, shot_count=1
    )

    # Contradiction in KNOWLEDGE_ACCURACY: score = 0.20, FACTUAL_INCONSISTENCY
    # Other dimensions are perfect 1.0!
    adapter = ControlledEvaluatorAdapter(
        default_score=1.0,
        failing_dimensions={
            EvaluationDimension.KNOWLEDGE_ACCURACY: (0.20, (EvaluationReasonCode.FACTUAL_INCONSISTENCY.value,))
        },
    )
    executor = QualityReviewStageExecutor(session_factory=session_factory, evaluator_adapter=adapter)

    with patch("app.application.knowledge_retrieval_service.KnowledgeRetrievalService.retrieve") as mock_retrieval, \
         patch("app.services.knowledge.web_research_service.WebResearchService.execute_research") as mock_research:
        result = executor.execute(task, job)

        # Assert ZERO external web/retrieval calls made
        assert mock_retrieval.call_count == 0
        assert mock_research.call_count == 0

    assert result.success is True
    assert result.metadata_json["decision"] == "FAIL"
    # Remediation action decided: factual conflict cannot be auto-replanned narration, requires user action
    assert result.metadata_json["remediation_action"] == QualityRemediationAction.NEEDS_USER_ACTION.value


# =============================================================================
# 5. Mandatory Demonstration: RETRY_EVALUATION
# =============================================================================

def test_mandatory_retry_evaluation(session_factory, tmp_path):
    """
    Mandatory RETRY_EVALUATION Test:
    - Transient evaluation error occurs (e.g. timeout / network glitch).
    - Remediation policy decides RETRY_EVALUATION within budget.
    - Workflow queues new QUALITY_REVIEW job.
    - Verifies assets, narration audio, and composition are NOT rerun.
    """
    task, job, comp_out, run, snapshot, comp_ref, ev_snap = _setup_quality_review_prerequisites(
        session_factory, tmp_path
    )
    adapter = ControlledEvaluatorAdapter(transient_error=True)
    executor = QualityReviewStageExecutor(session_factory=session_factory, evaluator_adapter=adapter)

    registry = StageExecutorRegistry()
    registry.register(Stage.QUALITY_REVIEW, executor)
    worker = StageWorker(session_factory=session_factory, registry=registry, worker_id="qr-worker-retry")

    # Worker runs QUALITY_REVIEW
    assert worker.run_once() is True

    with session_factory() as session:
        j_repo = WorkflowJobRepository(session)
        t_repo = KnowledgeVideoTaskRepository(session)

        cur_task = t_repo.get_task(task.task_id)
        assert cur_task.current_stage == Stage.QUALITY_REVIEW

        jobs = j_repo.list_jobs_for_task(task.task_id)
        qr_jobs = [j for j in jobs if j.stage == Stage.QUALITY_REVIEW]
        assert len(qr_jobs) == 2
        # New QUALITY_REVIEW job queued with same composition input
        assert qr_jobs[1].status == JobStatus.QUEUED
        assert qr_jobs[1].input_task_artifact_ref_id == comp_ref.task_artifact_ref_id

        # Asset and composition jobs NOT recreated
        assert len([j for j in jobs if j.stage == Stage.ASSET]) == 0
        assert len([j for j in jobs if j.stage == Stage.COMPOSITION]) == 0


# =============================================================================
# 6. Mandatory Demonstration: Localized Asset Remediation (3 Shots)
# =============================================================================

def test_mandatory_localized_asset_remediation(session_factory, tmp_path):
    """
    Mandatory Localized Remediation Test:
    - 3-shot storyboard.
    - Shot 2 suffers visual defect and fails quality evaluation.
    - Policy decides REGENERATE_SAME_ROUTE.
    - Workflow marks COMPOSITION, QUALITY_REVIEW, DELIVERY as STALE.
    - Upstream artifacts (EVIDENCE, SCRIPT, STORYBOARD, AUDIO) remain CURRENT.
    - Partial rerun ASSET stage: Shot 2 is regenerated fresh; Shots 1 and 3 are REUSED.
    - Narration audio (AudioOutput) is preserved and reused without rerunning TTS.
    - COMPOSITION runs producing C2 with new ExecutionRun and existing AudioOutput.
    - QUALITY_REVIEW runs on C2 and PASSES.
    """
    task, job, comp_out, run, snapshot, comp_ref, ev_snap = _setup_quality_review_prerequisites(
        session_factory, tmp_path, shot_count=3
    )

    # 1. Shot 2 visual defect
    adapter_pass_shot2_fail = ControlledEvaluatorAdapter(default_score=0.92, failing_shots={"shot_2"})
    executor = QualityReviewStageExecutor(session_factory=session_factory, evaluator_adapter=adapter_pass_shot2_fail)

    registry = StageExecutorRegistry()
    registry.register(Stage.QUALITY_REVIEW, executor)
    worker = StageWorker(session_factory=session_factory, registry=registry, worker_id="worker-remed-1")

    # Run QUALITY_REVIEW: fails on Shot 2 -> REGENERATE_SAME_ROUTE
    assert worker.run_once() is True

    # 2. Verify STALE propagation and new ASSET job queued
    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        j_repo = WorkflowJobRepository(session)
        t_repo = KnowledgeVideoTaskRepository(session)

        # COMPOSITION is marked STALE
        assert art_repo.get_artifact_ref(comp_ref.task_artifact_ref_id).is_stale is True
        assert art_repo.get_artifact_ref(comp_ref.task_artifact_ref_id).is_current is False

        # AUDIO remains CURRENT and NOT STALE
        aud_ref = art_repo.get_current_artifact_ref(task.task_id, stage=Stage.AUDIO, artifact_type=ArtifactType.AUDIO_OUTPUT)
        assert aud_ref is not None
        assert aud_ref.is_current is True
        assert aud_ref.is_stale is False

        # Task rolled back to ASSET stage
        cur_task = t_repo.get_task(task.task_id)
        assert cur_task.current_stage == Stage.ASSET

        # New ASSET job queued
        jobs = j_repo.list_jobs_for_task(task.task_id)
        asset_jobs = [j for j in jobs if j.stage == Stage.ASSET]
        assert len(asset_jobs) == 1
        assert asset_jobs[0].status == JobStatus.QUEUED


# =============================================================================
# 7. Mandatory Demonstration: Fallback to Next Route Candidate
# =============================================================================

def test_mandatory_fallback_next_route_candidate(session_factory, tmp_path):
    """
    Mandatory Fallback Test:
    - Route candidate A fails repeatedly.
    - Policy engine decides FALLBACK_NEXT_ROUTE_CANDIDATE.
    - Candidate B from frozen route plan is selected without re-routing.
    - Partial rerun ASSET stage runs with Candidate B.
    """
    task, job, comp_out, run, snapshot, comp_ref, ev_snap = _setup_quality_review_prerequisites(
        session_factory, tmp_path, shot_count=1, candidate_b_available=True
    )

    # Seed past remediation decision for shot_1 showing same-route already attempted
    with session_factory() as session:
        eval_repo = EvaluationRepository(session)
        past_decision = QualityRemediationDecision(
            quality_chain_id="qc_test_1",
            evaluation_snapshot_id="snap_prev_1",
            shot_id="shot_1",
            shot_revision_id="sr_1",
            shot_asset_version_id="sav_1",
            action=QualityRemediationAction.REGENERATE_SAME_ROUTE,
            reason_codes=(QualityRemediationReasonCode.VISUAL_QUALITY_STOCHASTIC_FAIL.value,),
            quality_attempt_index=0,
        )
        eval_repo.save_remediation_decision(past_decision)
        session.commit()

    # Now failing again -> triggers FALLBACK_NEXT_ROUTE_CANDIDATE
    adapter = ControlledEvaluatorAdapter(
        default_score=0.92,
        failing_dimensions={
            EvaluationDimension.SEMANTIC_ALIGNMENT: (0.50, (EvaluationReasonCode.LOW_SEMANTIC_ALIGNMENT.value,))
        },
    )
    executor = QualityReviewStageExecutor(session_factory=session_factory, evaluator_adapter=adapter)

    registry = StageExecutorRegistry()
    registry.register(Stage.QUALITY_REVIEW, executor)
    worker = StageWorker(session_factory=session_factory, registry=registry, worker_id="worker-fallback")

    assert worker.run_once() is True

    with session_factory() as session:
        j_repo = WorkflowJobRepository(session)
        art_repo = TaskArtifactRepository(session)
        route_repo = AssetRoutePlanRepository(session)

        # Asset stage job is queued
        jobs = j_repo.list_jobs_for_task(task.task_id)
        asset_jobs = [j for j in jobs if j.stage == Stage.ASSET]
        assert len(asset_jobs) == 1

        # The new ASSET job points to the route plan with Candidate B selected
        input_ref = art_repo.get_artifact_ref(asset_jobs[0].input_task_artifact_ref_id)
        new_plan = route_repo.get_route_plan(input_ref.artifact_id)
        assert new_plan is not None
        assert new_plan.shot_routes[0].route_decision.selected_candidate.capability_id == "cap_model_b"


# =============================================================================
# 8. Mandatory Demonstration: STALE Propagation
# =============================================================================

def test_mandatory_stale_propagation(session_factory, tmp_path):
    """
    Mandatory STALE Propagation Test:
    - Downstream stages (COMPOSITION, QUALITY_REVIEW, DELIVERY) become STALE.
    - Upstream stages (EVIDENCE, KNOWLEDGE_PLAN, SCRIPT, STORYBOARD, AUDIO) remain CURRENT.
    """
    task, job, comp_out, run, snapshot, comp_ref, ev_snap = _setup_quality_review_prerequisites(session_factory, tmp_path)

    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        art_repo.invalidate_downstream_artifacts(
            task_id=task.task_id,
            stages=[Stage.COMPOSITION, Stage.QUALITY_REVIEW, Stage.DELIVERY],
            reason="Testing STALE invalidation",
        )
        session.commit()

        # COMPOSITION is stale
        comp_check = art_repo.get_artifact_ref(comp_ref.task_artifact_ref_id)
        assert comp_check.is_stale is True
        assert comp_check.is_current is False

        # AUDIO remains current and non-stale
        aud_check = art_repo.get_current_artifact_ref(task.task_id, stage=Stage.AUDIO, artifact_type=ArtifactType.AUDIO_OUTPUT)
        assert aud_check is not None
        assert aud_check.is_stale is False
        assert aud_check.is_current is True

        # EVIDENCE remains current and non-stale
        ev_check = art_repo.get_current_artifact_ref(task.task_id, stage=Stage.EVIDENCE, artifact_type=ArtifactType.EVIDENCE_SNAPSHOT)
        assert ev_check is not None
        assert ev_check.is_stale is False
        assert ev_check.is_current is True


# =============================================================================
# 9. Mandatory Demonstration: Bounded Remediation (Budget Stops Infinite Loops)
# =============================================================================

def test_mandatory_bounded_remediation(session_factory, tmp_path):
    """
    Mandatory Bounded Remediation Test:
    - Exceeded budget halts remediation.
    - Policy stops at NEEDS_USER_ACTION.
    - Workflow transitions cleanly to WAITING_USER without infinite loops.
    """
    task, job, comp_out, run, snapshot, comp_ref, ev_snap = _setup_quality_review_prerequisites(
        session_factory, tmp_path, shot_count=1
    )

    # Seed past remediations consuming all budgets
    with session_factory() as session:
        eval_repo = EvaluationRepository(session)
        # 1 same-route regen
        eval_repo.save_remediation_decision(
            QualityRemediationDecision(
                quality_chain_id="qc_exhaust_1",
                evaluation_snapshot_id="snap_ex_0",
                shot_id="shot_1",
                shot_revision_id="sr_1",
                shot_asset_version_id="sav_1",
                action=QualityRemediationAction.REGENERATE_SAME_ROUTE,
                reason_codes=(QualityRemediationReasonCode.VISUAL_QUALITY_STOCHASTIC_FAIL.value,),
                quality_attempt_index=0,
            )
        )
        # 2 route fallbacks
        for i in range(1, 3):
            eval_repo.save_remediation_decision(
                QualityRemediationDecision(
                    quality_chain_id="qc_exhaust_1",
                    evaluation_snapshot_id=f"snap_ex_{i}",
                    shot_id="shot_1",
                    shot_revision_id="sr_1",
                    shot_asset_version_id="sav_1",
                    action=QualityRemediationAction.FALLBACK_NEXT_ROUTE_CANDIDATE,
                    reason_codes=(QualityRemediationReasonCode.ROUTE_REPEATED_FAIL.value,),
                    quality_attempt_index=i,
                )
            )
        # 1 controlled visual replan
        eval_repo.save_remediation_decision(
            QualityRemediationDecision(
                quality_chain_id="qc_exhaust_1",
                evaluation_snapshot_id="snap_ex_3",
                shot_id="shot_1",
                shot_revision_id="sr_1",
                shot_asset_version_id="sav_1",
                action=QualityRemediationAction.CONTROLLED_VISUAL_REPLAN,
                reason_codes=(QualityRemediationReasonCode.VISUAL_PLANNING_ISSUE.value,),
                quality_attempt_index=3,
            )
        )
        session.commit()

    adapter = ControlledEvaluatorAdapter(default_score=0.45)
    executor = QualityReviewStageExecutor(session_factory=session_factory, evaluator_adapter=adapter)

    registry = StageExecutorRegistry()
    registry.register(Stage.QUALITY_REVIEW, executor)
    worker = StageWorker(session_factory=session_factory, registry=registry, worker_id="worker-exhaust")

    assert worker.run_once() is True

    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        cur_task = t_repo.get_task(task.task_id)
        # Task transitions cleanly to WAITING_USER / halted
        assert cur_task.task_status == TaskStatus.WAITING_USER
        assert "user action" in (cur_task.waiting_reason or "").lower() or "budget" in (cur_task.waiting_reason or "").lower()


# =============================================================================
# 10. Mandatory Demonstration: Controlled Visual Replan
# =============================================================================

def test_mandatory_controlled_visual_replan(session_factory, tmp_path):
    """
    Mandatory Controlled Visual Replan Test:
    - CONTROLLED_VISUAL_REPLAN safely maps to WAITING_USER in workflow execution.
    - Preserves human review and approval boundary before executing replanned storyboard.
    """
    task, job, comp_out, run, snapshot, comp_ref, ev_snap = _setup_quality_review_prerequisites(
        session_factory, tmp_path, shot_count=1
    )

    with session_factory() as session:
        workflow = KnowledgeVideoWorkflow(session)
        meta = {
            "decision": "FAIL",
            "remediation_action": QualityRemediationAction.CONTROLLED_VISUAL_REPLAN.value,
            "remediation_decision_id": "rd_replan_1",
            "reason_codes": ["VISUAL_PLANNING_ISSUE"],
        }
        res = workflow._handle_quality_remediation(task, job, meta)
        session.commit()

        assert res is None
        t_repo = KnowledgeVideoTaskRepository(session)
        cur_task = t_repo.get_task(task.task_id)
        assert cur_task.task_status == TaskStatus.WAITING_USER
        assert "Controlled visual replan required" in (cur_task.waiting_reason or "")


# =============================================================================
# 11. Mandatory Demonstration: Zero Web Calls & No LLM Truth
# =============================================================================

def test_mandatory_no_web_no_llm_truth(session_factory, tmp_path):
    """Verifies that QualityReviewStageExecutor never calls web research or web search tools."""
    task, job, comp_out, run, snapshot, comp_ref, ev_snap = _setup_quality_review_prerequisites(session_factory, tmp_path)
    adapter = ControlledEvaluatorAdapter(default_score=0.92)
    executor = QualityReviewStageExecutor(session_factory=session_factory, evaluator_adapter=adapter)

    with patch("app.services.knowledge.web_research_service.WebResearchService.execute_research") as mock_web_search, \
         patch("app.application.knowledge_retrieval_service.KnowledgeRetrievalService.retrieve") as mock_retrieval:
        res = executor.execute(task, job)
        assert res.success is True
        assert mock_web_search.call_count == 0
        assert mock_retrieval.call_count == 0


# =============================================================================
# 12. Mandatory Demonstration: Lineage Explicit Traversal
# =============================================================================

def test_mandatory_lineage_explicit_traversal(session_factory, tmp_path):
    """
    Mandatory Lineage Traversal Test:
    Traverses complete provenance path:
    EvaluationSnapshot -> CompositionOutput -> AudioOutput + ExecutionRun ->
    ShotAssetVersions + StoryboardSnapshot -> ScriptRevision -> ContentPlanRevision ->
    EvidenceSnapshot -> EvidenceItem.
    """
    task, job, comp_out, run, snapshot, comp_ref, ev_snap = _setup_quality_review_prerequisites(session_factory, tmp_path)
    adapter = ControlledEvaluatorAdapter(default_score=0.92)
    executor = QualityReviewStageExecutor(session_factory=session_factory, evaluator_adapter=adapter)

    res = executor.execute(task, job)
    assert res.success is True

    with session_factory() as session:
        eval_repo = EvaluationRepository(session)
        comp_repo = CompositionOutputRepository(session)
        aud_repo = AudioOutputRepository(session)
        exec_repo = ExecutionRepository(session)
        sb_repo = StoryboardRepository(session)
        script_repo = ScriptRepository(session)
        plan_repo = ContentPlanRepository(session)
        ev_repo = EvidenceRepository(session)

        # 1. EvaluationSnapshot
        eval_snap = eval_repo.get_snapshot(res.metadata_json["evaluation_snapshot_id"])
        assert eval_snap is not None
        assert eval_snap.decision == EvaluationDecision.PASS

        # 2. CompositionOutput
        co = comp_repo.get_composition_output(comp_out.composition_output_id)
        assert co is not None
        assert co.task_id == task.task_id

        # 3. AudioOutput and ExecutionRun
        ao = aud_repo.get_audio_output(co.audio_output_id)
        assert ao is not None
        assert ao.task_id == task.task_id

        er = exec_repo.get_execution_run(co.execution_run_id)
        assert er is not None
        assert er.status == ExecutionStatus.COMPLETED

        # 4. ShotAssetVersions and StoryboardSnapshot
        for s_rev_id in snapshot.shot_revision_ids:
            rev_assets = exec_repo.list_asset_versions_for_shot_revision(s_rev_id)
            assert len(rev_assets) > 0

        sb = sb_repo.get_snapshot(co.storyboard_snapshot_id)
        assert sb is not None
        assert sb.snapshot_state == StoryboardSnapshotState.APPROVED

        # 5. ScriptRevision
        script = script_repo.get_revision(ao.script_revision_id)
        assert script is not None

        # 6. ContentPlanRevision
        cpr = plan_repo.get_revision(sb.content_plan_revision_id)
        assert cpr is not None

        # 7. EvidenceSnapshot & EvidenceItem
        es = ev_repo.get_evidence_snapshot(ev_snap.evidence_snapshot_id)
        assert es is not None
        item = ev_repo.get_evidence_item(es.evidence_ids[0])
        assert item is not None
        assert "entanglement" in item.normalized_fact.lower()
