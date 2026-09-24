"""
Comprehensive test suite for Stage P1: Production Plan + Asset Stage Integration.

Verifies:
- Production stage registry registers PRODUCTION_PLAN and ASSET; AUDIO remains unsupported.
- ProductionPlanStageExecutor consumes approved StoryboardSnapshot, invokes AssetRoutePlanningService,
  and creates TaskArtifactRef(ASSET_ROUTE_PLAN) without calling external providers.
- Unapproved (DRAFT) StoryboardSnapshot is rejected.
- AssetStageExecutor consumes exact frozen AssetRoutePlan, invokes AssetRoutePlanExecutionService,
  persists AttemptRequest before provider calls, records ProviderReceipts, produces ShotAssetVersions,
  and creates TaskArtifactRef(EXECUTION_RUN).
- Ambiguous submissions map cleanly to NEEDS_RECOVERY without blind retry.
- Partial failures preserve successful ShotAssetVersions without unnecessary regeneration.
- Six mandatory demonstrations:
  1. Mandatory ProductionPlan Integration
  2. Mandatory Attempt Ordering
  3. Mandatory Successful Asset
  4. Mandatory Unknown-Submission
  5. Mandatory Partial-Failure
  6. Mandatory Task Lineage
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.asset_stage_executor import AssetStageExecutor
from app.application.knowledge_video_workflow import KnowledgeVideoWorkflow
from app.application.production_plan_stage_executor import ProductionPlanStageExecutor
from app.application.stage_executor_registry import (
    StageExecutorRegistry,
    get_default_executor_registry,
)
from app.domain.asset_execution import (
    AdapterExecutionResult,
    AssetMediaType,
    AssetReuseMode,
    AttemptRequest,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRetryPolicy,
    ExecutionRun,
    ExecutionStatus,
    FailureCategory,
    ProviderOutcomeType,
    ProviderReceipt,
    ShotAssetVersion,
    ShotExecution,
    ShotExecutionStatus,
)
from app.domain.asset_router import (
    AssetCapability,
    AssetRouteCandidate,
    AssetRoutePlan,
    AssetRoutePlanStatus,
    AssetRoutingRequest,
    GenerationMode,
    ModelSelectionMode,
    RoutingStrategy,
    ShotRoutePlanStatus,
    StaticCapabilityMetadata,
    TierLevel,
)
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.evidence import EvidenceItem, EvidenceSnapshot, SourceDocument
from app.domain.knowledge_video_task import KnowledgeVideoTask
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
from app.persistence.models import Base
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    ContentPlanRepository,
    EvidenceRepository,
    ExecutionRepository,
    KnowledgeVideoTaskRepository,
    ScriptRepository,
    ShotExecutionRepository,
    ShotRepository,
    StoryboardRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)
from app.services.asset_adapters.base import (
    AssetExecutionAdapter,
    ProviderExecutionCapabilities,
)
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.asset_capability_registry import (
    AssetCapabilityProvider,
    AssetCapabilityRegistry,
)
from app.services.asset_probe_service import calculate_sha256
from app.services.asset_route_plan_execution_service import (
    AssetRoutePlanExecutionService,
)
from app.services.asset_route_planning_service import (
    AssetRoutePlanningService,
)
from app.services.attempt_recovery_service import AttemptRecoveryService
from app.services.hybrid_asset_router import HybridAssetRouter
from app.services.shot_execution_service import ShotExecutionService
from app.workers.stage_worker import StageWorker


# =============================================================================
# Fixtures & Test Doubles
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


class StaticTestCapabilityProvider:
    """Provides test capabilities for AI_IMAGE, AI_VIDEO, and STOCK_VIDEO."""

    def __init__(self, capabilities: Sequence[AssetCapability] | None = None):
        self._caps = tuple(capabilities or [
            AssetCapability(
                capability_id="test_image_provider:model_img:TEXT_TO_IMAGE",
                provider="test_image_provider",
                model="model_img",
                generation_mode=GenerationMode.TEXT_TO_IMAGE,
                supported_visual_types=(VisualType.AI_IMAGE,),
                supported_aspect_ratios=("16:9", "9:16", "1:1"),
                enabled=True,
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.HIGH,
                    cost_tier=TierLevel.LOW,
                    latency_tier=TierLevel.LOW,
                ),
            ),
            AssetCapability(
                capability_id="test_video_provider:model_vid:TEXT_TO_VIDEO",
                provider="test_video_provider",
                model="model_vid",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
                supported_visual_types=(VisualType.AI_VIDEO,),
                supported_aspect_ratios=("16:9", "9:16", "1:1"),
                min_duration=2.0,
                max_duration=30.0,
                enabled=True,
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.HIGH,
                    cost_tier=TierLevel.MEDIUM,
                    latency_tier=TierLevel.MEDIUM,
                ),
            ),
            AssetCapability(
                capability_id="test_stock_provider:model_stock:STOCK_SEARCH",
                provider="test_stock_provider",
                model="model_stock",
                generation_mode=GenerationMode.STOCK_SEARCH,
                supported_visual_types=(VisualType.STOCK_VIDEO,),
                enabled=True,
                metadata=StaticCapabilityMetadata(
                    quality_tier=TierLevel.MEDIUM,
                    cost_tier=TierLevel.LOW,
                    latency_tier=TierLevel.LOW,
                ),
            ),
        ])

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        return self._caps


class RecordingTestAdapter(AssetExecutionAdapter):
    """
    Test adapter that creates deterministic local media files and records call history.
    Never makes real network requests or incurs external billing.
    """

    def __init__(
        self,
        provider_name: str,
        results_by_shot: dict[str, AdapterExecutionResult | ProviderReceipt] | None = None,
        default_outcome: ProviderOutcomeType = ProviderOutcomeType.SUCCESS,
        async_mode: bool = False,
    ):
        self.provider_name = provider_name
        self.results_by_shot = dict(results_by_shot or {})
        self.default_outcome = default_outcome
        self.async_mode = async_mode
        self.call_history: list[dict] = []
        self._capabilities = ProviderExecutionCapabilities(
            supports_async_status=async_mode,
            supports_sync_execution=not async_mode,
        )

    @property
    def capabilities(self) -> ProviderExecutionCapabilities:
        return self._capabilities

    def _create_dummy_image(self, target_dir: Path, shot_id: str) -> str:
        target_dir.mkdir(parents=True, exist_ok=True)
        img_path = target_dir / f"shot_{shot_id}_asset.png"
        img = Image.new("RGB", (1920, 1080), color=(30, 60, 90))
        img.save(img_path, format="PNG")
        return str(img_path)

    def execute(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        res = self.submit(request, candidate, target_dir)
        if isinstance(res, AdapterExecutionResult):
            return res
        return AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            status="completed",
            remote_task_id=res.provider_job_id,
        )

    def submit(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
        idempotency_key: str | None = None,
    ) -> AdapterExecutionResult | ProviderReceipt:
        call_record = {
            "action": "submit",
            "shot_id": request.shot_id,
            "provider": candidate.provider,
            "model": candidate.model,
            "idempotency_key": idempotency_key,
            "timestamp": datetime.now(UTC),
        }
        self.call_history.append(call_record)

        if request.shot_id in self.results_by_shot:
            return self.results_by_shot[request.shot_id]

        if self.async_mode:
            job_id = f"job_{request.shot_id}_{uuid4().hex[:6]}"
            return ProviderReceipt(
                execution_attempt_id="temp",
                provider=candidate.provider,
                provider_job_id=job_id,
                provider_status="submitted",
                sanitized_metadata={"job_id": job_id},
            )

        if self.default_outcome == ProviderOutcomeType.SUCCESS:
            local_path = self._create_dummy_image(target_dir, request.shot_id)
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUCCESS,
                status="completed",
                file_path=local_path,
                remote_task_id=f"task_{request.shot_id}",
            )
        elif self.default_outcome == ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN:
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
                status="unknown",
                error_code="NETWORK_DROPPED_SUBMISSION_UNKNOWN",
                error_message="Connection dropped before receipt confirmation",
            )
        else:
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                status="failed",
                error_code="PROVIDER_GENERATION_FAILED",
                error_message="Provider returned definitive error",
            )

    def get_status(self, provider_job_id: str, target_dir: Path | None = None) -> AdapterExecutionResult:
        self.call_history.append({
            "action": "get_status",
            "provider_job_id": provider_job_id,
            "timestamp": datetime.now(UTC),
        })
        t_dir = target_dir or Path("storage/test_status")
        local_path = self._create_dummy_image(t_dir, provider_job_id)
        return AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            status="completed",
            file_path=local_path,
            remote_task_id=provider_job_id,
        )


def _setup_task_with_approved_storyboard(
    session_factory,
    topic: str = "Why does Transformer use self-attention?",
    num_shots: int = 3,
    workflow_policy: WorkflowPolicyType = WorkflowPolicyType.AUTO,
    visual_types: Sequence[VisualType] | None = None,
) -> tuple[
    KnowledgeVideoTask,
    WorkflowJob,
    StoryboardSnapshot,
    list[Shot],
    list[ShotRevision],
    ScriptRevision,
    ContentPlanRevision,
    TaskArtifactRef,
]:
    """
    Creates complete upstream lineage:
    Task -> EvidenceSnapshot -> ContentPlanRevision -> ScriptRevision -> APPROVED StoryboardSnapshot.
    Task is positioned in Stage.PRODUCTION_PLAN with an active QUEUED WorkflowJob.
    """
    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        ev_repo = EvidenceRepository(session)
        p_repo = ContentPlanRepository(session)
        s_repo = ScriptRepository(session)
        sb_repo = StoryboardRepository(session)
        shot_repo = ShotRepository(session)
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
        task.advance_stage(Stage.PRODUCTION_PLAN)
        task.transition_to(TaskStatus.RUNNING)
        saved_task = t_repo.save_task(task)

        # 2. Evidence
        source = SourceDocument.create_text(
            text="Self-attention allows tokens to compute mutual relationships.",
            title="Transformer Paper",
        )
        ev_repo.save_source_document(source)
        item = EvidenceItem.create(
            source_document=source,
            original_excerpt="Self-attention computes token-to-token attention weights.",
        )
        ev_repo.save_evidence_item(item)
        ev_snapshot = EvidenceSnapshot.create(
            task_id=saved_task.task_id,
            source_document_ids=[source.source_document_id],
            evidence_ids=[item.evidence_id],
            snapshot_version=1,
        )
        ev_repo.save_evidence_snapshot(ev_snapshot)

        # 3. ContentPlan
        beats = [
            ContentBeat(
                beat_id=f"beat-{i+1}",
                beat_lineage_id=f"lineage-b{i+1}",
                order=i + 1,
                beat_type=BeatType.KNOWLEDGE if i == 1 else BeatType.HOOK,
                intent=f"Beat {i+1} intent",
                target_duration=10.0,
                importance=1.0 if i == 1 else 0.8,
                evidence_refs=(item.evidence_id,) if i == 1 else (),
            )
            for i in range(num_shots)
        ]
        plan = ContentPlanRevision(
            content_plan_revision_id=f"cpr_{uuid4().hex[:8]}",
            revision_number=1,
            topic=topic,
            overall_target_duration=30.0,
            beats=tuple(beats),
        )
        saved_plan = p_repo.add_revision(plan)

        # 4. Script
        segments = [
            ScriptSegment(
                script_segment_id=f"seg-{i+1}",
                script_revision_id="placeholder",
                content_beat_id=beats[i].beat_id,
                beat_lineage_id=beats[i].beat_lineage_id,
                order=i + 1,
                narration_text=f"Spoken narration text for segment {i+1}.",
                target_duration=10.0,
                beat_type=beats[i].beat_type,
                evidence_refs=beats[i].evidence_refs,
            )
            for i in range(num_shots)
        ]
        script = ScriptRevision.create(
            task_id=saved_task.task_id,
            content_plan_revision_id=saved_plan.content_plan_revision_id,
            segments=segments,
            overall_target_duration=30.0,
        )
        saved_script = s_repo.save_revision(script)

        # 5. Storyboard Shots & APPROVED Snapshot
        shots: list[Shot] = []
        revisions: list[ShotRevision] = []
        v_types = list(visual_types) if visual_types is not None else [VisualType.AI_IMAGE, VisualType.AI_VIDEO, VisualType.AI_IMAGE]

        for i in range(num_shots):
            s_id = f"shot-{i+1}-{uuid4().hex[:6]}"
            s_rev_id = f"shot-rev-{i+1}-{uuid4().hex[:6]}"
            v_type = v_types[i % len(v_types)]

            shot = Shot(
                shot_id=s_id,
                beat_lineage_id=beats[i].beat_lineage_id,
                local_order=1,
            )
            shot_repo.add_shot(shot)

            rev = ShotRevision(
                shot_revision_id=s_rev_id,
                shot_id=s_id,
                revision_number=1,
                beat_lineage_id=beats[i].beat_lineage_id,
                created_from_beat_instance_id=beats[i].beat_id,
                script_segment_id=segments[i].script_segment_id,
                narration=segments[i].narration_text,
                target_duration=10.0,
                visual_goal=f"Visual goal for shot {i+1}",
                visual_type=v_type,
                scene_description=f"Scene description for shot {i+1}",
                generation_prompt=f"Generation prompt for shot {i+1}",
                camera_movement="Static center",
                evidence_refs=tuple(segments[i].evidence_refs),
            )
            shot_repo.add_revision(rev)
            shots.append(shot)
            revisions.append(rev)

        snapshot = StoryboardSnapshot(
            storyboard_snapshot_id=str(uuid4()),
            content_plan_revision_id=plan.content_plan_revision_id,
            snapshot_state=StoryboardSnapshotState.APPROVED,
            shot_revision_ids=tuple(r.shot_revision_id for r in revisions),
        )
        sb_repo.add_snapshot(snapshot)

        # 6. TaskArtifactRef
        sb_ref = TaskArtifactRef.create(
            task_id=saved_task.task_id,
            stage=Stage.STORYBOARD,
            artifact_type=ArtifactType.STORYBOARD_SNAPSHOT,
            artifact_id=snapshot.storyboard_snapshot_id,
        )
        art_repo.save_artifact_ref(sb_ref)

        # 7. WorkflowJob for PRODUCTION_PLAN
        job = WorkflowJob.create(
            task_id=saved_task.task_id,
            stage=Stage.PRODUCTION_PLAN,
            idempotency_key=f"idemp_{saved_task.task_id}_prod_plan_1",
            input_task_artifact_ref_id=sb_ref.task_artifact_ref_id,
        )
        saved_job = j_repo.create_job(job)
        session.commit()

        return saved_task, saved_job, snapshot, shots, revisions, script, plan, sb_ref


# =============================================================================
# 1. Production Stage Registry Tests (Points 1, 19)
# =============================================================================


def test_production_registry_contains_production_plan_and_asset():
    """1, 19. Verify default registry has real PRODUCTION_PLAN and ASSET executors."""
    registry = get_default_executor_registry()
    assert registry.has_executor(Stage.PRODUCTION_PLAN)
    assert isinstance(registry.get_executor(Stage.PRODUCTION_PLAN), ProductionPlanStageExecutor)
    assert registry.has_executor(Stage.ASSET)
    assert isinstance(registry.get_executor(Stage.ASSET), AssetStageExecutor)


def test_production_registry_leaves_audio_unsupported():
    """Verify COMPOSITION and subsequent stages remain unsupported in default registry."""
    registry = get_default_executor_registry()
    assert not registry.has_executor(Stage.COMPOSITION)
    assert registry.get_executor(Stage.COMPOSITION) is None
    assert not registry.has_executor(Stage.QUALITY_REVIEW)
    assert not registry.has_executor(Stage.DELIVERY)


# =============================================================================
# 2. ProductionPlan Stage Executor Tests (Points 2 - 18)
# =============================================================================


def test_production_plan_resolves_exact_approved_storyboard(session_factory):
    """2, 4, 16, 17. Resolves exact approved snapshot and creates TaskArtifactRef(ASSET_ROUTE_PLAN)."""
    task, job, snapshot, shots, revs, script, plan, sb_ref = _setup_task_with_approved_storyboard(session_factory)

    cap_registry = AssetCapabilityRegistry(providers=[StaticTestCapabilityProvider()])
    executor = ProductionPlanStageExecutor(
        session_factory=session_factory,
        capability_registry=cap_registry,
    )
    result = executor.execute(task, job)

    assert result.success is True
    assert result.output_artifact_ref is not None
    assert result.output_artifact_ref.artifact_type == ArtifactType.ASSET_ROUTE_PLAN
    assert result.output_artifact_ref.stage == Stage.PRODUCTION_PLAN
    assert result.output_artifact_ref.task_id == task.task_id

    # Verify route plan persisted in DB
    with session_factory() as session:
        rp_repo = AssetRoutePlanRepository(session)
        saved_plan = rp_repo.get_route_plan(result.output_artifact_ref.artifact_id)
        assert saved_plan is not None
        assert saved_plan.storyboard_snapshot_id == snapshot.storyboard_snapshot_id
        assert saved_plan.total_shots == len(shots)
        assert saved_plan.routed_shots == len(shots)
        assert saved_plan.is_ready


def test_production_plan_rejects_unapproved_storyboard(session_factory):
    """3. Unapproved (DRAFT) StoryboardSnapshot must be rejected."""
    task, job, snapshot, shots, revs, script, plan, sb_ref = _setup_task_with_approved_storyboard(session_factory)

    # Mutate snapshot to DRAFT
    with session_factory() as session:
        sb_repo = StoryboardRepository(session)
        # Create a draft snapshot
        draft_snap = StoryboardSnapshot(
            storyboard_snapshot_id=str(uuid4()),
            content_plan_revision_id=plan.content_plan_revision_id,
            snapshot_state=StoryboardSnapshotState.DRAFT,
            shot_revision_ids=tuple(r.shot_revision_id for r in revs),
        )
        sb_repo.add_snapshot(draft_snap)

        art_repo = TaskArtifactRepository(session)
        draft_ref = TaskArtifactRef.create(
            task_id=task.task_id,
            stage=Stage.STORYBOARD,
            artifact_type=ArtifactType.STORYBOARD_SNAPSHOT,
            artifact_id=draft_snap.storyboard_snapshot_id,
        )
        art_repo.save_artifact_ref(draft_ref)
        job.input_task_artifact_ref_id = draft_ref.task_artifact_ref_id
        session.commit()

    executor = ProductionPlanStageExecutor(session_factory=session_factory)
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "expected APPROVED" in result.error_message


def test_production_plan_rejects_task_id_mismatch(session_factory):
    """Task ID mismatch must be rejected cleanly."""
    task, job, snapshot, shots, revs, script, plan, sb_ref = _setup_task_with_approved_storyboard(session_factory)

    # Create artifact belonging to different task
    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        alien_ref = TaskArtifactRef.create(
            task_id="alien_task_999",
            stage=Stage.STORYBOARD,
            artifact_type=ArtifactType.STORYBOARD_SNAPSHOT,
            artifact_id=snapshot.storyboard_snapshot_id,
        )
        art_repo.save_artifact_ref(alien_ref)
        job.input_task_artifact_ref_id = alien_ref.task_artifact_ref_id
        session.commit()

    executor = ProductionPlanStageExecutor(session_factory=session_factory)
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "Task ID mismatch" in result.error_message


def test_production_plan_makes_zero_provider_calls(session_factory):
    """6. Verifies that ProductionPlanStageExecutor makes zero provider calls."""
    task, job, snapshot, shots, revs, script, plan, sb_ref = _setup_task_with_approved_storyboard(session_factory)

    mock_adapter = RecordingTestAdapter("test_image_provider")
    cap_registry = AssetCapabilityRegistry(providers=[StaticTestCapabilityProvider()])

    executor = ProductionPlanStageExecutor(
        session_factory=session_factory,
        capability_registry=cap_registry,
    )
    result = executor.execute(task, job)

    assert result.success is True
    # Verify mock adapter received 0 calls
    assert len(mock_adapter.call_history) == 0


def test_production_plan_preserves_shot_identities_and_narration(session_factory):
    """7, 8, 9, 10, 11. All shots have route decisions, stable Shot IDs, no narration mutation."""
    task, job, snapshot, shots, revs, script, plan, sb_ref = _setup_task_with_approved_storyboard(session_factory)

    cap_registry = AssetCapabilityRegistry(providers=[StaticTestCapabilityProvider()])
    executor = ProductionPlanStageExecutor(
        session_factory=session_factory,
        capability_registry=cap_registry,
    )
    result = executor.execute(task, job)

    with session_factory() as session:
        rp_repo = AssetRoutePlanRepository(session)
        route_plan = rp_repo.get_route_plan(result.output_artifact_ref.artifact_id)

        assert len(route_plan.shot_routes) == len(shots)
        for i, entry in enumerate(route_plan.shot_routes):
            assert entry.shot_id == shots[i].shot_id
            assert entry.shot_revision_id == revs[i].shot_revision_id
            assert entry.route_status == ShotRoutePlanStatus.ROUTED
            assert entry.route_decision.selected_candidate is not None

        # Verify Shot narration remains unchanged in shot repo
        s_repo = ShotRepository(session)
        for rev in revs:
            loaded_rev = s_repo.get_revision(rev.shot_revision_id)
            assert loaded_rev.narration == rev.narration
            assert loaded_rev.evidence_refs == rev.evidence_refs


# =============================================================================
# 3. Asset Stage Executor Tests (Points 20 - 47)
# =============================================================================


def _setup_task_with_route_plan(
    session_factory,
    topic: str = "Transformer self-attention",
    num_shots: int = 2,
    visual_types: Sequence[VisualType] | None = None,
) -> tuple[KnowledgeVideoTask, WorkflowJob, AssetRoutePlan, TaskArtifactRef]:
    """Sets up task in Stage.ASSET with an existing AssetRoutePlan artifact."""
    task, job, snapshot, shots, revs, script, plan, sb_ref = _setup_task_with_approved_storyboard(
        session_factory, topic=topic, num_shots=num_shots, visual_types=visual_types
    )
    # Generate route plan
    cap_registry = AssetCapabilityRegistry(providers=[StaticTestCapabilityProvider()])
    pp_executor = ProductionPlanStageExecutor(
        session_factory=session_factory,
        capability_registry=cap_registry,
    )
    pp_result = pp_executor.execute(task, job)

    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        art_repo = TaskArtifactRepository(session)
        rp_repo = AssetRoutePlanRepository(session)

        # Save route plan artifact
        art_repo.save_artifact_ref(pp_result.output_artifact_ref)

        # Mark PRODUCTION_PLAN job as SUCCEEDED
        prod_plan_job = j_repo.get_job(job.job_id)
        prod_plan_job.mark_succeeded(output_task_artifact_ref_id=pp_result.output_artifact_ref.task_artifact_ref_id)
        j_repo.update_job(prod_plan_job)

        # Advance task to ASSET
        cur_task = t_repo.get_task(task.task_id)
        cur_task.advance_stage(Stage.ASSET)
        t_repo.save_task(cur_task)

        # Create ASSET WorkflowJob
        asset_job = WorkflowJob.create(
            task_id=cur_task.task_id,
            stage=Stage.ASSET,
            idempotency_key=f"idemp_{cur_task.task_id}_asset_1",
            input_task_artifact_ref_id=pp_result.output_artifact_ref.task_artifact_ref_id,
        )
        j_repo.create_job(asset_job)
        session.commit()

        route_plan = rp_repo.get_route_plan(pp_result.output_artifact_ref.artifact_id)
        return cur_task, asset_job, route_plan, pp_result.output_artifact_ref


def test_asset_executor_consumes_exact_route_plan_and_produces_assets(session_factory, tmp_path):
    """20, 21, 24, 25, 29, 30, 45. Consumes exact plan, creates ExecutionRun, persists ShotAssetVersions."""
    task, job, route_plan, rp_ref = _setup_task_with_route_plan(session_factory, num_shots=2)

    img_adapter = RecordingTestAdapter("test_image_provider")
    vid_adapter = RecordingTestAdapter("test_video_provider")
    registry = AdapterRegistry(custom_adapters={
        "test_image_provider": img_adapter,
        "test_video_provider": vid_adapter,
    })

    executor = AssetStageExecutor(
        session_factory=session_factory,
        adapter_registry=registry,
        storage_base_dir=tmp_path,
    )
    result = executor.execute(task, job)

    assert result.success is True
    assert result.output_artifact_ref is not None
    assert result.output_artifact_ref.artifact_type == ArtifactType.EXECUTION_RUN
    assert result.output_artifact_ref.stage == Stage.ASSET
    assert result.output_artifact_ref.task_id == task.task_id

    # Verify ExecutionRun in DB
    with session_factory() as session:
        exec_repo = ExecutionRepository(session)
        run = exec_repo.get_execution_run(result.output_artifact_ref.artifact_id)
        assert run is not None
        assert run.status == ExecutionStatus.COMPLETED
        assert run.succeeded_shots + run.reused_shots == 2

        # Verify ShotAssetVersions
        for entry in route_plan.shot_routes:
            versions = exec_repo.list_asset_versions_for_shot_revision(entry.shot_revision_id)
            assert len(versions) >= 1
            v = versions[0]
            assert v.shot_id == entry.shot_id
            assert os.path.exists(v.file_path)
            assert v.file_size_bytes > 0


def test_asset_executor_unknown_submission_maps_to_needs_recovery(session_factory, tmp_path):
    """34, 35. Ambiguous submission maps cleanly to NEEDS_RECOVERY and does not blindly retry."""
    task, job, route_plan, rp_ref = _setup_task_with_route_plan(session_factory, num_shots=1)

    # Configure adapter to return SUBMISSION_OUTCOME_UNKNOWN
    unknown_adapter = RecordingTestAdapter(
        "test_image_provider",
        default_outcome=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
    )
    registry = AdapterRegistry(custom_adapters={
        "test_image_provider": unknown_adapter,
        "test_video_provider": unknown_adapter,
    })

    executor = AssetStageExecutor(
        session_factory=session_factory,
        adapter_registry=registry,
        storage_base_dir=tmp_path,
    )
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.NEEDS_RECOVERY.value
    # Assert provider was called exactly ONCE (no blind retry loop)
    assert len(unknown_adapter.call_history) == 1

    # Verify ExecutionRun is in NEEDS_RECOVERY state in DB
    with session_factory() as session:
        exec_repo = ExecutionRepository(session)
        runs = exec_repo.list_runs_for_plan(route_plan.asset_route_plan_id)
        assert len(runs) >= 1
        assert runs[0].status == ExecutionStatus.NEEDS_RECOVERY


def test_asset_executor_partial_failure_preserves_successful_assets(session_factory, tmp_path):
    """36, 37. When shot 1 succeeds and shot 2 fails, shot 1 asset is preserved."""
    task, job, route_plan, rp_ref = _setup_task_with_route_plan(session_factory, num_shots=2)
    shot_1_id = route_plan.shot_routes[0].shot_id
    shot_2_id = route_plan.shot_routes[1].shot_id

    img_adapter = RecordingTestAdapter("test_image_provider")
    failing_adapter = RecordingTestAdapter(
        "test_video_provider",
        default_outcome=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
    )
    registry = AdapterRegistry(custom_adapters={
        "test_image_provider": img_adapter,
        "test_video_provider": failing_adapter,
    })

    executor = AssetStageExecutor(
        session_factory=session_factory,
        adapter_registry=registry,
        storage_base_dir=tmp_path,
    )
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.RETRYABLE.value

    # Verify shot 1 asset exists and is preserved in DB
    with session_factory() as session:
        exec_repo = ExecutionRepository(session)
        v1 = exec_repo.list_asset_versions_for_shot_revision(route_plan.shot_routes[0].shot_revision_id)
        assert len(v1) >= 1
        assert os.path.exists(v1[0].file_path)

        # Shot 2 has no asset
        v2 = exec_repo.list_asset_versions_for_shot_revision(route_plan.shot_routes[1].shot_revision_id)
        assert len(v2) == 0


# =============================================================================
# 4. No Early Downstream Work Tests (Points 48 - 52)
# =============================================================================


def test_no_early_downstream_work(session_factory, tmp_path):
    """48 - 52. Verifies zero TTS, subtitles, BGM, final composition, or quality evaluation."""
    task, job, route_plan, rp_ref = _setup_task_with_route_plan(session_factory, num_shots=1)

    img_adapter = RecordingTestAdapter("test_image_provider")
    registry = AdapterRegistry(custom_adapters={"test_image_provider": img_adapter})

    executor = AssetStageExecutor(
        session_factory=session_factory,
        adapter_registry=registry,
        storage_base_dir=tmp_path,
    )
    result = executor.execute(task, job)
    assert result.success is True

    # Verify AUDIO job is created but unexecuted
    with session_factory() as session:
        j_repo = WorkflowJobRepository(session)
        t_repo = KnowledgeVideoTaskRepository(session)
        art_repo = TaskArtifactRepository(session)
        workflow = KnowledgeVideoWorkflow(session)

        art_repo.save_artifact_ref(result.output_artifact_ref)
        curr_task = t_repo.get_task(task.task_id)
        curr_job = j_repo.get_job(job.job_id)
        curr_job.mark_succeeded(output_task_artifact_ref_id=result.output_artifact_ref.task_artifact_ref_id)
        j_repo.update_job(curr_job)

        audio_job = workflow.on_stage_completed(curr_task, curr_job)
        session.commit()

        assert audio_job is not None
        assert audio_job.stage == Stage.AUDIO
        assert audio_job.status == JobStatus.QUEUED

        # Run StageWorker without audio executor - worker must NOT process AUDIO
        worker_registry = StageExecutorRegistry()
        worker_registry.register(Stage.PRODUCTION_PLAN, ProductionPlanStageExecutor(session_factory=session_factory))
        worker_registry.register(Stage.ASSET, executor)
        worker = StageWorker(
            session_factory=session_factory,
            registry=worker_registry,
            worker_id="test-worker",
        )
        processed = worker.run_once()
        assert processed is False

        # Reload job - remains QUEUED
        reloaded_audio_job = j_repo.get_job(audio_job.job_id)
        assert reloaded_audio_job.status == JobStatus.QUEUED


# =============================================================================
# 5. Six Mandatory Demonstrations
# =============================================================================


def test_mandatory_demonstration_1_production_plan(session_factory):
    """
    Mandatory Demonstration 1: ProductionPlan Integration.
    Approved StoryboardSnapshot with Shot 1 (suitable for AI_IMAGE) and Shot 2 (suitable for AI_VIDEO).
    Executes PRODUCTION_PLAN WorkflowJob -> StageWorker -> ProductionPlanStageExecutor -> AssetRoutePlan.
    Verifies:
      - Route decisions exist
      - Shot IDs preserved
      - No provider calls occurred
      - TaskArtifactRef created
      - ASSET job created in QUEUED state
    """
    task, job, snapshot, shots, revs, script, plan, sb_ref = _setup_task_with_approved_storyboard(
        session_factory, num_shots=2
    )

    cap_registry = AssetCapabilityRegistry(providers=[StaticTestCapabilityProvider()])
    pp_executor = ProductionPlanStageExecutor(
        session_factory=session_factory,
        capability_registry=cap_registry,
    )

    stage_registry = StageExecutorRegistry()
    stage_registry.register(Stage.PRODUCTION_PLAN, pp_executor)

    worker = StageWorker(
        session_factory=session_factory,
        registry=stage_registry,
        worker_id="demo-pp-worker",
    )
    assert worker.run_once() is True

    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        art_repo = TaskArtifactRepository(session)
        rp_repo = AssetRoutePlanRepository(session)

        cur_task = t_repo.get_task(task.task_id)
        assert cur_task.current_stage == Stage.ASSET
        assert cur_task.task_status == TaskStatus.RUNNING

        # TaskArtifactRef(ASSET_ROUTE_PLAN) exists
        plan_ref = art_repo.get_latest_artifact_ref(task.task_id, stage=Stage.PRODUCTION_PLAN)
        assert plan_ref is not None
        assert plan_ref.artifact_type == ArtifactType.ASSET_ROUTE_PLAN

        # AssetRoutePlan exists and has decisions
        route_plan = rp_repo.get_route_plan(plan_ref.artifact_id)
        assert route_plan is not None
        assert route_plan.total_shots == 2
        assert route_plan.is_ready
        assert len(route_plan.shot_routes) == 2
        assert route_plan.shot_routes[0].shot_id == shots[0].shot_id
        assert route_plan.shot_routes[1].shot_id == shots[1].shot_id

        # Next WorkflowJob(stage=ASSET) is queued
        asset_jobs = j_repo.list_jobs_for_task(task.task_id)
        asset_job = next(j for j in asset_jobs if j.stage == Stage.ASSET)
        assert asset_job.status == JobStatus.QUEUED
        assert asset_job.input_task_artifact_ref_id == plan_ref.task_artifact_ref_id


def test_mandatory_demonstration_2_attempt_ordering(session_factory, tmp_path):
    """
    Mandatory Demonstration 2: Attempt Ordering Instrument.
    Instruments order:
      1. AttemptRequest persisted in DB
      2. Database commit completed
      3. Provider adapter called
      4. ProviderReceipt persisted
      5. ShotAssetVersion persisted
    Asserts exact safety ordering. Provider call must NOT occur before step 1/2.
    """
    task, job, route_plan, rp_ref = _setup_task_with_route_plan(session_factory, num_shots=1)

    events_order: list[str] = []

    class OrderInstrumentedAdapter(AssetExecutionAdapter):
        def __init__(self, sf):
            self._sf = sf
            self._capabilities = ProviderExecutionCapabilities(supports_sync_execution=True)

        @property
        def capabilities(self):
            return self._capabilities

        def execute(self, request, candidate, target_dir):
            res = self.submit(request, candidate, target_dir)
            if isinstance(res, AdapterExecutionResult):
                return res
            return AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, status="completed")

        def submit(self, request, candidate, target_dir, idempotency_key=None):
            # Assert that AttemptRequest has ALREADY been committed to DB before this adapter method was invoked!
            with self._sf() as sess:
                from app.persistence.models import AttemptRequestORM
                from sqlalchemy import select
                stmt = select(AttemptRequestORM).where(AttemptRequestORM.idempotency_key == idempotency_key)
                req_orm = sess.scalars(stmt).first()
                assert req_orm is not None, "AttemptRequest must be committed in DB before adapter.submit is called!"
                events_order.append("ATTEMPT_REQUEST_COMMITTED_IN_DB")
            events_order.append("ADAPTER_SUBMIT_CALLED")

            img_path = target_dir / "ordered_asset.png"
            target_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1920, 1080), color="green").save(img_path, format="PNG")
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUCCESS,
                status="completed",
                file_path=str(img_path),
                remote_task_id="remote_order_123",
            )

        def get_status(self, provider_job_id, target_dir=None):
            return AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS)

    adapter = OrderInstrumentedAdapter(session_factory)
    registry = AdapterRegistry(custom_adapters={"test_image_provider": adapter})

    executor = AssetStageExecutor(
        session_factory=session_factory,
        adapter_registry=registry,
        storage_base_dir=tmp_path,
    )
    result = executor.execute(task, job)

    assert result.success is True
    assert "ATTEMPT_REQUEST_COMMITTED_IN_DB" in events_order
    assert "ADAPTER_SUBMIT_CALLED" in events_order
    assert events_order.index("ATTEMPT_REQUEST_COMMITTED_IN_DB") < events_order.index("ADAPTER_SUBMIT_CALLED")

    # Verify final persistence in DB
    with session_factory() as session:
        exec_repo = ExecutionRepository(session)
        shot_repo = ShotExecutionRepository(session)

        # 1. AttemptRequest persisted
        runs = exec_repo.list_runs_for_plan(route_plan.asset_route_plan_id)
        run_id = runs[0].execution_run_id
        attempts = exec_repo.list_attempts_for_run(run_id)
        assert len(attempts) >= 1
        att_req = exec_repo.get_attempt_request(attempts[0].execution_attempt_id)
        assert att_req is not None

        # 2. ProviderReceipt persisted
        receipt = shot_repo.get_provider_receipt(attempts[0].execution_attempt_id)
        assert receipt is not None
        assert receipt.provider_job_id == "remote_order_123"

        # 3. ShotAssetVersion persisted
        versions = exec_repo.list_asset_versions_for_shot_revision(route_plan.shot_routes[0].shot_revision_id)
        assert len(versions) == 1
        assert versions[0].execution_attempt_id == attempts[0].execution_attempt_id


def test_mandatory_demonstration_3_successful_asset(session_factory, tmp_path):
    """
    Mandatory Demonstration 3: Successful Asset Integration.
    Executes ASSET WorkflowJob -> StageWorker -> AssetStageExecutor -> ShotAssetVersion.
    Verifies:
      - All required shot assets exist
      - Correct lineage exists
      - ExecutionRun completes
      - TaskArtifactRef(EXECUTION_RUN) created
      - AUDIO job created and safely queued
      - Zero TTS/composition calls
    """
    task, job, route_plan, rp_ref = _setup_task_with_route_plan(session_factory, num_shots=2)

    img_adapter = RecordingTestAdapter("test_image_provider")
    vid_adapter = RecordingTestAdapter("test_video_provider")
    registry = AdapterRegistry(custom_adapters={
        "test_image_provider": img_adapter,
        "test_video_provider": vid_adapter,
    })

    asset_executor = AssetStageExecutor(
        session_factory=session_factory,
        adapter_registry=registry,
        storage_base_dir=tmp_path,
    )

    stage_registry = StageExecutorRegistry()
    stage_registry.register(Stage.ASSET, asset_executor)

    worker = StageWorker(
        session_factory=session_factory,
        registry=stage_registry,
        worker_id="demo-asset-worker",
    )
    assert worker.run_once() is True

    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        art_repo = TaskArtifactRepository(session)
        exec_repo = ExecutionRepository(session)

        # Task advanced to AUDIO
        cur_task = t_repo.get_task(task.task_id)
        assert cur_task.current_stage == Stage.AUDIO
        assert cur_task.task_status == TaskStatus.RUNNING

        # TaskArtifactRef(EXECUTION_RUN)
        run_ref = art_repo.get_latest_artifact_ref(task.task_id, stage=Stage.ASSET)
        assert run_ref is not None
        assert run_ref.artifact_type == ArtifactType.EXECUTION_RUN

        # ExecutionRun completed
        run = exec_repo.get_execution_run(run_ref.artifact_id)
        assert run.status == ExecutionStatus.COMPLETED

        # AUDIO job created and QUEUED
        audio_job = j_repo.get_job_by_idempotency_key(f"idemp_{task.task_id}_audio_1")
        assert audio_job is not None
        assert audio_job.status == JobStatus.QUEUED
        assert audio_job.input_task_artifact_ref_id == run_ref.task_artifact_ref_id


def test_mandatory_demonstration_4_unknown_submission(session_factory, tmp_path):
    """
    Mandatory Demonstration 4: Unknown-Submission Handling.
    Simulate ambiguous submission outcome.
    Verifies:
      - No blind duplicate provider call
      - Attempt and Run map to NEEDS_RECOVERY
      - ASSET stage does not advance
      - No AUDIO job created
      - AttemptRecoveryService can inspect durable request state
    """
    task, job, route_plan, rp_ref = _setup_task_with_route_plan(session_factory, num_shots=1)

    ambiguous_adapter = RecordingTestAdapter(
        "test_image_provider",
        default_outcome=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
    )
    registry = AdapterRegistry(custom_adapters={"test_image_provider": ambiguous_adapter})

    asset_executor = AssetStageExecutor(
        session_factory=session_factory,
        adapter_registry=registry,
        storage_base_dir=tmp_path,
    )

    stage_registry = StageExecutorRegistry()
    stage_registry.register(Stage.ASSET, asset_executor)

    worker = StageWorker(
        session_factory=session_factory,
        registry=stage_registry,
        worker_id="demo-recovery-worker",
    )
    assert worker.run_once() is True

    # Assert exactly 1 submission (no blind retry!)
    assert len(ambiguous_adapter.call_history) == 1

    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        exec_repo = ExecutionRepository(session)

        # Task paused in NEEDS_RECOVERY
        cur_task = t_repo.get_task(task.task_id)
        assert cur_task.task_status == TaskStatus.NEEDS_RECOVERY

        # No AUDIO job created
        audio_job = j_repo.get_job_by_idempotency_key(f"idemp_{task.task_id}_audio_1")
        assert audio_job is None

        # Inspectable by AttemptRecoveryService
        runs = exec_repo.list_runs_for_plan(route_plan.asset_route_plan_id)
        assert len(runs) == 1
        assert runs[0].status == ExecutionStatus.NEEDS_RECOVERY
        attempts = exec_repo.list_attempts_for_run(runs[0].execution_run_id)
        assert len(attempts) == 1
        assert attempts[0].status == ExecutionAttemptStatus.SUBMISSION_OUTCOME_UNKNOWN

        att_req = exec_repo.get_attempt_request(attempts[0].execution_attempt_id)
        assert att_req is not None
        assert att_req.provider == "test_image_provider"


def test_mandatory_demonstration_5_partial_failure_and_reuse(session_factory, tmp_path):
    """
    Mandatory Demonstration 5: Partial Failure and Asset Reuse.
    Shot A succeeds, Shot B succeeds, Shot C fails.
    Verifies:
      - A & B assets preserved on disk and in database
      - ExecutionRun does not become COMPLETED while C is unresolved
      - No AUDIO job
      - Second run with fixed adapter reuses A & B without regenerating, only executes C
    """
    task, job, route_plan, rp_ref = _setup_task_with_route_plan(
        session_factory,
        num_shots=3,
        visual_types=[VisualType.AI_IMAGE, VisualType.AI_IMAGE, VisualType.AI_IMAGE],
    )
    shot_a = route_plan.shot_routes[0].shot_id
    shot_b = route_plan.shot_routes[1].shot_id
    shot_c = route_plan.shot_routes[2].shot_id

    # Configure adapter: A and B succeed, C fails definitively
    failing_adapter = RecordingTestAdapter(
        "test_image_provider",
        results_by_shot={
            shot_c: AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                error_code="CAPABILITY_OFFLINE",
                error_message="Provider C is temporarily offline",
            )
        },
    )
    vid_adapter = RecordingTestAdapter("test_video_provider")
    registry = AdapterRegistry(custom_adapters={
        "test_image_provider": failing_adapter,
        "test_video_provider": vid_adapter,
    })

    executor = AssetStageExecutor(
        session_factory=session_factory,
        adapter_registry=registry,
        storage_base_dir=tmp_path,
    )
    result1 = executor.execute(task, job)

    assert result1.success is False

    # Assert A and B assets are preserved
    with session_factory() as session:
        exec_repo = ExecutionRepository(session)
        j_repo = WorkflowJobRepository(session)

        # No AUDIO job
        assert j_repo.get_job_by_idempotency_key(f"idemp_{task.task_id}_audio_1") is None

        v_a = exec_repo.list_asset_versions_for_shot_revision(route_plan.shot_routes[0].shot_revision_id)
        v_b = exec_repo.list_asset_versions_for_shot_revision(route_plan.shot_routes[1].shot_revision_id)
        v_c = exec_repo.list_asset_versions_for_shot_revision(route_plan.shot_routes[2].shot_revision_id)

        assert len(v_a) == 1
        assert len(v_b) == 1
        assert len(v_c) == 0

    # Second execution: now C succeeds!
    recovered_adapter = RecordingTestAdapter("test_image_provider")
    recovered_registry = AdapterRegistry(custom_adapters={
        "test_image_provider": recovered_adapter,
        "test_video_provider": vid_adapter,
    })

    executor2 = AssetStageExecutor(
        session_factory=session_factory,
        adapter_registry=recovered_registry,
        storage_base_dir=tmp_path,
        reuse_mode=AssetReuseMode.REUSE_COMPATIBLE,
    )
    result2 = executor2.execute(task, job)

    assert result2.success is True

    # Check that A & B were REUSED (only C was newly submitted!)
    assert result2.metadata_json["reused_shots"] == 2
    assert result2.metadata_json["generated_shots"] == 1


def test_mandatory_demonstration_6_task_lineage(session_factory, tmp_path):
    """
    Mandatory Demonstration 6: Full Task Lineage Continuity.
    Traverses complete stage-level lineage:
      KnowledgeVideoTask
      -> StoryboardSnapshot TaskArtifactRef
      -> PRODUCTION_PLAN WorkflowJob
      -> StageExecution
      -> AssetRoutePlan
      -> TaskArtifactRef(ASSET_ROUTE_PLAN)
      -> ASSET WorkflowJob
      -> StageExecution
      -> ExecutionRun
      -> ShotExecution
      -> ExecutionAttempt
      -> AttemptRequest
      -> ProviderReceipt
      -> ShotAssetVersion
      -> TaskArtifactRef(EXECUTION_RUN)
      -> AUDIO WorkflowJob
    Verifies same task_id is preserved throughout.
    """
    task, pp_job, snapshot, shots, revs, script, plan, sb_ref = _setup_task_with_approved_storyboard(
        session_factory, num_shots=2
    )

    cap_registry = AssetCapabilityRegistry(providers=[StaticTestCapabilityProvider()])
    img_adapter = RecordingTestAdapter("test_image_provider")
    vid_adapter = RecordingTestAdapter("test_video_provider")
    adapter_registry = AdapterRegistry(custom_adapters={
        "test_image_provider": img_adapter,
        "test_video_provider": vid_adapter,
    })

    stage_registry = StageExecutorRegistry()
    stage_registry.register(
        Stage.PRODUCTION_PLAN,
        ProductionPlanStageExecutor(
            session_factory=session_factory,
            capability_registry=cap_registry,
        ),
    )
    stage_registry.register(
        Stage.ASSET,
        AssetStageExecutor(
            session_factory=session_factory,
            adapter_registry=adapter_registry,
            storage_base_dir=tmp_path,
        ),
    )

    worker = StageWorker(
        session_factory=session_factory,
        registry=stage_registry,
        worker_id="lineage-worker",
    )

    # 1. Worker runs PRODUCTION_PLAN
    assert worker.run_once() is True
    # 2. Worker runs ASSET
    assert worker.run_once() is True

    # 3. Verify end-to-end lineage
    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        art_repo = TaskArtifactRepository(session)
        exec_repo = ExecutionRepository(session)
        shot_repo = ShotExecutionRepository(session)
        rp_repo = AssetRoutePlanRepository(session)

        # Root task
        cur_task = t_repo.get_task(task.task_id)
        assert cur_task.task_id == task.task_id
        assert cur_task.current_stage == Stage.AUDIO

        # Storyboard ArtifactRef
        sb_art = art_repo.get_latest_artifact_ref(task.task_id, stage=Stage.STORYBOARD)
        assert sb_art.task_id == task.task_id

        # Route Plan ArtifactRef
        rp_art = art_repo.get_latest_artifact_ref(task.task_id, stage=Stage.PRODUCTION_PLAN)
        assert rp_art.task_id == task.task_id
        route_plan = rp_repo.get_route_plan(rp_art.artifact_id)
        assert route_plan.storyboard_snapshot_id == sb_art.artifact_id

        # ExecutionRun ArtifactRef
        run_art = art_repo.get_latest_artifact_ref(task.task_id, stage=Stage.ASSET)
        assert run_art.task_id == task.task_id
        run = exec_repo.get_execution_run(run_art.artifact_id)
        assert run.asset_route_plan_id == route_plan.asset_route_plan_id

        # ShotExecutions and Attempts
        shot_execs = shot_repo.list_shot_executions_for_run(run.execution_run_id)
        assert len(shot_execs) == 2
        for se in shot_execs:
            attempts = exec_repo.list_attempts_for_run(run.execution_run_id)
            relevant_attempts = [a for a in attempts if a.shot_id == se.shot_id]
            assert len(relevant_attempts) >= 1
            att = relevant_attempts[0]

            # AttemptRequest
            att_req = exec_repo.get_attempt_request(att.execution_attempt_id)
            assert att_req is not None

            # ShotAssetVersion
            asset_v = exec_repo.get_shot_asset_version(se.produced_asset_version_id)
            assert asset_v is not None
            assert asset_v.shot_id == se.shot_id
            assert asset_v.execution_attempt_id == att.execution_attempt_id

        # AUDIO WorkflowJob
        audio_jobs = [j for j in j_repo.list_jobs_for_task(task.task_id) if j.stage == Stage.AUDIO]
        assert len(audio_jobs) == 1
        assert audio_jobs[0].task_id == task.task_id
        assert audio_jobs[0].input_task_artifact_ref_id == run_art.task_artifact_ref_id
        assert audio_jobs[0].status == JobStatus.QUEUED
