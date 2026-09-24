from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.domain.asset_execution import (
    AdapterExecutionResult,
    AssetMediaType,
    AssetReuseDecisionType,
    AssetReuseMode,
    AssetReuseReasonCode,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRetryPolicy,
    ExecutionRun,
    ExecutionRunResult,
    ExecutionStatus,
    ProviderOutcomeType,
    RoutePlanNotReadyError,
    ShotAssetVersion,
    ShotExecutionStatus,
)
from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutePlan,
    AssetRoutePlanStatus,
    AssetRoutingRequest,
    GenerationMode,
    RoutingStrategy,
    ShotRoutePlanEntry,
    ShotRoutePlanStatus,
)
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.persistence.models import Base
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    ContentPlanRepository,
    ExecutionRepository,
    ShotExecutionRepository,
    ShotRepository,
    StoryboardRepository,
)
from app.services.asset_adapters.base import AssetExecutionAdapter
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.asset_probe_service import calculate_sha256
from app.services.asset_reuse_policy import AssetReusePolicy
from app.services.asset_route_plan_execution_service import (
    AssetRoutePlanExecutionService,
)
from app.services.shot_execution_service import ShotExecutionService


class ConfigurableMockAdapter(AssetExecutionAdapter):
    """Mock adapter supporting queued or scripted responses."""

    def __init__(self, outcomes: list[AdapterExecutionResult] | None = None):
        self.outcomes = list(outcomes or [])
        self.call_count = 0
        self.last_candidate = None
        self.last_request = None
        self.on_execute_hook = None

    def execute(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        self.last_candidate = candidate
        self.last_request = request
        if self.on_execute_hook:
            self.on_execute_hook()

        if self.call_count < len(self.outcomes):
            res = self.outcomes[self.call_count]
        elif self.outcomes:
            res = self.outcomes[-1]
        else:
            # Default success file
            img_path = target_dir / f"mock_{uuid4().hex[:6]}.png"
            img = Image.new("RGB", (1920, 1080), color="blue")
            img.save(str(img_path))
            res = AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUCCESS,
                file_path=str(img_path),
            )

        self.call_count += 1
        return res


@pytest.fixture
def db_session_factory():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    Base.metadata.drop_all(engine)


def _create_real_image(target_dir: Path, filename: str = "asset.png", size=(1920, 1080), color="blue") -> tuple[str, str, int]:
    target_dir.mkdir(parents=True, exist_ok=True)
    p = target_dir / filename
    img = Image.new("RGB", size, color=color)
    img.save(str(p))
    sha = calculate_sha256(p)
    return str(p), sha, p.stat().st_size


def _setup_test_environment(
    session_factory,
    shot_count: int = 3,
    status: AssetRoutePlanStatus = AssetRoutePlanStatus.READY,
) -> tuple[AssetRoutePlan, list[ShotRevision]]:
    """Seeds DB with ContentPlan, Shots, ShotRevisions, StoryboardSnapshot, and AssetRoutePlan."""
    with session_factory() as session:
        plan_repo = ContentPlanRepository(session)
        shot_repo = ShotRepository(session)
        sb_repo = StoryboardRepository(session)
        route_repo = AssetRoutePlanRepository(session)

        beat = ContentBeat(
            beat_id="b-test-1",
            beat_lineage_id="lin-test-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Test Hook",
            target_duration=float(shot_count * 5),
            importance=1.0,
        )
        c_plan = ContentPlanRevision(
            content_plan_revision_id="cplan-1",
            revision_number=1,
            topic="Plan Execution Invariant Tests",
            overall_target_duration=float(shot_count * 5),
            beats=(beat,),
        )
        plan_repo.add_revision(c_plan)

        revisions: list[ShotRevision] = []
        entries: list[ShotRoutePlanEntry] = []

        for i in range(1, shot_count + 1):
            shot = Shot(shot_id=f"shot-{i}", beat_lineage_id="lin-test-1", local_order=i)
            shot_repo.add_shot(shot)

            rev = ShotRevision(
                shot_revision_id=f"rev-{i}",
                shot_id=f"shot-{i}",
                revision_number=1,
                beat_lineage_id="lin-test-1",
                created_from_beat_instance_id="b-test-1",
                narration=f"Narration for shot {i}",
                target_duration=5.0,
                visual_goal=f"Goal {i}",
                visual_type=VisualType.AI_IMAGE,
                scene_description=f"Scene {i}",
                generation_prompt=f"Prompt {i}",
                camera_movement="Static",
            )
            shot_repo.add_revision(rev)
            revisions.append(rev)

            req = AssetRoutingRequest(
                shot_id=f"shot-{i}",
                shot_revision_id=f"rev-{i}",
                requested_visual_type=VisualType.AI_IMAGE,
                target_duration=5.0,
                visual_goal=f"Goal {i}",
                scene_description=f"Scene {i}",
                generation_prompt=f"Prompt {i}",
                camera_movement="Static",
                aspect_ratio="16:9",
            )
            cand = AssetRouteCandidate(
                capability_id="mock_prov:model_1:TEXT_TO_IMAGE",
                provider="mock_prov",
                model="model_1",
                generation_mode=GenerationMode.TEXT_TO_IMAGE,
                requested_visual_type=VisualType.AI_IMAGE,
                is_eligible=True,
            )
            dec = AssetRouteDecision(
                shot_id=f"shot-{i}",
                shot_revision_id=f"rev-{i}",
                routing_strategy=RoutingStrategy.BALANCED,
                requested_visual_type=VisualType.AI_IMAGE,
                selected_candidate=cand,
                eligible_candidates=(cand,),
            )
            entry = ShotRoutePlanEntry(
                shot_id=f"shot-{i}",
                shot_revision_id=f"rev-{i}",
                beat_lineage_id="lin-test-1",
                requested_visual_type=VisualType.AI_IMAGE,
                asset_routing_request=req,
                route_decision=dec,
                route_status=ShotRoutePlanStatus.ROUTED if status == AssetRoutePlanStatus.READY else ShotRoutePlanStatus.BLOCKED,
            )
            entries.append(entry)

        snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap-approved-1",
            content_plan_revision_id="cplan-1",
            shot_revision_ids=tuple(r.shot_revision_id for r in revisions),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        sb_repo.add_snapshot(snap)

        route_plan = AssetRoutePlan(
            asset_route_plan_id="plan-route-1",
            storyboard_snapshot_id="snap-approved-1",
            content_plan_revision_id="cplan-1",
            routing_strategy=RoutingStrategy.BALANCED,
            routing_policy_version="v1.0",
            shot_routes=tuple(entries),
            status=status,
            total_shots=len(entries),
            routed_shots=len(entries) if status == AssetRoutePlanStatus.READY else 0,
            blocked_shots=0 if status == AssetRoutePlanStatus.READY else len(entries),
        )
        route_repo.add_route_plan(route_plan)
        session.commit()

    return route_plan, revisions


def _seed_compatible_asset(
    session_factory,
    plan: AssetRoutePlan,
    rev: ShotRevision,
    tmp_path: Path,
    filename: str = "seed.png",
    version_id: str = "asset-hist-1",
    attempt_id: str = "att-seed-1",
    width: int = 1920,
    height: int = 1080,
    media_type: AssetMediaType = AssetMediaType.IMAGE,
    duration: float = 0.0,
    gen_mode: GenerationMode = GenerationMode.TEXT_TO_IMAGE,
) -> ShotAssetVersion:
    """Helper to seed a valid ShotAssetVersion respecting DB foreign keys."""
    img_path, img_hash, img_size = _create_real_image(tmp_path, filename, size=(width, height))
    with session_factory() as session:
        exec_repo = ExecutionRepository(session)
        dummy_run = ExecutionRun(
            execution_run_id=f"run-{attempt_id}",
            asset_route_plan_id=plan.asset_route_plan_id,
            storyboard_snapshot_id=plan.storyboard_snapshot_id,
            status=ExecutionStatus.COMPLETED,
            total_shots=1,
        )
        exec_repo.add_execution_run(dummy_run)
        dummy_attempt = ExecutionAttempt(
            execution_attempt_id=attempt_id,
            execution_run_id=dummy_run.execution_run_id,
            shot_id=rev.shot_id,
            shot_revision_id=rev.shot_revision_id,
            attempt_number=1,
            provider="mock_prov",
            model="model_1",
            generation_mode=gen_mode,
            status=ExecutionAttemptStatus.SUCCEEDED,
        )
        exec_repo.add_execution_attempt(dummy_attempt)
        v = ShotAssetVersion(
            shot_asset_version_id=version_id,
            shot_id=rev.shot_id,
            shot_revision_id=rev.shot_revision_id,
            execution_attempt_id=attempt_id,
            file_path=img_path,
            file_hash=img_hash,
            file_size_bytes=img_size,
            media_type=media_type,
            mime_type="image/png" if media_type == AssetMediaType.IMAGE else "video/mp4",
            width=width,
            height=height,
            duration_seconds=duration,
            provider="mock_prov",
            model="model_1",
            generation_mode=gen_mode,
        )
        exec_repo.add_shot_asset_version(v)
        session.commit()
    return v


# ==============================================================================
# 26 BUSINESS INVARIANT TESTS FOR PHASE 5.3
# ==============================================================================


def test_01_execute_route_plan_requires_ready_status(db_session_factory, tmp_path):
    """Invariant 1: Non-READY route plan raises RoutePlanNotReadyError."""
    blocked_plan, _ = _setup_test_environment(db_session_factory, shot_count=2, status=AssetRoutePlanStatus.BLOCKED)
    service = AssetRoutePlanExecutionService(session_factory=db_session_factory)

    with pytest.raises(RoutePlanNotReadyError) as exc_info:
        service.execute_route_plan(blocked_plan, storage_base_dir=tmp_path)

    assert "BLOCKED" in str(exc_info.value)
    assert "expected READY" in str(exc_info.value)


def test_02_brand_new_plan_executes_all_shots(db_session_factory, tmp_path):
    """Invariant 2: Brand new plan with no historical assets executes all shots via provider adapter."""
    plan, _ = _setup_test_environment(db_session_factory, shot_count=3)

    img_path, _, _ = _create_real_image(tmp_path)
    mock_adapter = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path)
    ])
    shot_service = ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": mock_adapter}))
    service = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=shot_service,
    )

    result = service.execute_route_plan(plan, storage_base_dir=tmp_path)

    assert result.state == ExecutionStatus.COMPLETED
    assert result.total_shots == 3
    assert result.generated_shots == 3
    assert result.reused_shots == 0
    assert result.failed_shots == 0
    assert result.recovery_required_shots == 0
    assert mock_adapter.call_count == 3


def test_03_reused_shot_does_not_invoke_provider(db_session_factory, tmp_path):
    """Invariant 3: Reused shot never invokes provider adapter submit/execute."""
    plan, revs = _setup_test_environment(db_session_factory, shot_count=1)
    _seed_compatible_asset(db_session_factory, plan, revs[0], tmp_path)

    mock_adapter = ConfigurableMockAdapter()
    shot_service = ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": mock_adapter}))
    service = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=shot_service,
    )

    result = service.execute_route_plan(plan, storage_base_dir=tmp_path)

    assert result.state == ExecutionStatus.COMPLETED
    assert result.reused_shots == 1
    assert result.generated_shots == 0
    assert mock_adapter.call_count == 0


def test_04_reused_shot_creates_no_execution_attempts(db_session_factory, tmp_path):
    """Invariant 4: Reused shot has empty attempt_ids (no faked attempts)."""
    plan, revs = _setup_test_environment(db_session_factory, shot_count=1)
    _seed_compatible_asset(db_session_factory, plan, revs[0], tmp_path)

    service = AssetRoutePlanExecutionService(session_factory=db_session_factory)
    result = service.execute_route_plan(plan, storage_base_dir=tmp_path)

    with db_session_factory() as session:
        shot_repo = ShotExecutionRepository(session)
        shot_execs = shot_repo.list_shot_executions_for_run(result.execution_run_id)
        assert len(shot_execs) == 1
        assert shot_execs[0].attempt_ids == []
        assert result.shot_results[0].attempts_count == 0


def test_05_reused_shot_preserves_historical_asset_version_id(db_session_factory, tmp_path):
    """Invariant 5: Reused shot references original historical ShotAssetVersion ID."""
    plan, revs = _setup_test_environment(db_session_factory, shot_count=1)
    seeded_v = _seed_compatible_asset(
        db_session_factory, plan, revs[0], tmp_path, version_id="asset-original-uuid-12345"
    )

    service = AssetRoutePlanExecutionService(session_factory=db_session_factory)
    result = service.execute_route_plan(plan, storage_base_dir=tmp_path)

    assert result.shot_results[0].asset_version_id == seeded_v.shot_asset_version_id
    with db_session_factory() as session:
        shot_repo = ShotExecutionRepository(session)
        shot_exec = shot_repo.list_shot_executions_for_run(result.execution_run_id)[0]
        assert shot_exec.produced_asset_version_id == seeded_v.shot_asset_version_id


def test_06_reused_shot_marked_with_status_reused(db_session_factory, tmp_path):
    """Invariant 6: Reused shot status is ShotExecutionStatus.REUSED."""
    plan, revs = _setup_test_environment(db_session_factory, shot_count=1)
    _seed_compatible_asset(db_session_factory, plan, revs[0], tmp_path)

    service = AssetRoutePlanExecutionService(session_factory=db_session_factory)
    result = service.execute_route_plan(plan, storage_base_dir=tmp_path)

    assert result.shot_results[0].status == ShotExecutionStatus.REUSED
    assert result.shot_results[0].result_type == "REUSED"


def test_07_incremental_rerun_reuses_compatible_and_regenerates_failed(db_session_factory, tmp_path):
    """Invariant 7: Rerun reuses previously succeeded shots and executes only failed shots."""
    plan, _ = _setup_test_environment(db_session_factory, shot_count=3)
    img_path, _, _ = _create_real_image(tmp_path)

    single_attempt_policy = ExecutionRetryPolicy(max_attempts_per_candidate=1, max_total_attempts_per_shot=1)

    # Run 1: shot 1 succeeds, shot 2 fails, shot 3 succeeds
    adapter_run_1 = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE, error_code="NET_ERR"),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
    ])
    service_1 = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(
            adapter_registry=AdapterRegistry({"mock_prov": adapter_run_1}),
            policy=single_attempt_policy,
        ),
    )
    result_1 = service_1.execute_route_plan(plan, storage_base_dir=tmp_path)

    assert result_1.state == ExecutionStatus.FAILED
    assert result_1.generated_shots == 2
    assert result_1.failed_shots == 1
    assert adapter_run_1.call_count == 3

    # Run 2 (Rerun): shot 1 & 3 should be reused, only shot 2 should execute!
    adapter_run_2 = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
    ])
    service_2 = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(
            adapter_registry=AdapterRegistry({"mock_prov": adapter_run_2}),
            policy=single_attempt_policy,
        ),
    )
    result_2 = service_2.execute_route_plan(plan, storage_base_dir=tmp_path)

    assert result_2.state == ExecutionStatus.COMPLETED
    assert result_2.reused_shots == 2
    assert result_2.generated_shots == 1
    assert result_2.failed_shots == 0
    # Exactly 1 provider call for the previously failed shot!
    assert adapter_run_2.call_count == 1


def test_08_force_regenerate_bypasses_all_reuse(db_session_factory, tmp_path):
    """Invariant 8: FORCE_REGENERATE mode executes all shots despite existing compatible assets."""
    plan, _ = _setup_test_environment(db_session_factory, shot_count=2)
    img_path, _, _ = _create_real_image(tmp_path)

    # First run creates compatible assets
    adapter_1 = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
    ])
    service_1 = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": adapter_1})),
    )
    res_1 = service_1.execute_route_plan(plan, storage_base_dir=tmp_path)
    assert res_1.state == ExecutionStatus.COMPLETED

    # Second run with FORCE_REGENERATE
    adapter_2 = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
    ])
    service_2 = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": adapter_2})),
    )
    res_2 = service_2.execute_route_plan(
        plan,
        storage_base_dir=tmp_path,
        reuse_mode=AssetReuseMode.FORCE_REGENERATE,
    )

    assert res_2.state == ExecutionStatus.COMPLETED
    assert res_2.generated_shots == 2
    assert res_2.reused_shots == 0
    assert adapter_2.call_count == 2


def test_09_shot_revision_mismatch_prevents_reuse(tmp_path):
    """Invariant 9: Different shot_revision_id causes SHOT_REVISION_CHANGED and EXECUTE."""
    policy = AssetReusePolicy()
    img_path, img_hash, img_size = _create_real_image(tmp_path)

    candidate = ShotAssetVersion(
        shot_asset_version_id="asset-rev-old",
        shot_id="shot-1",
        shot_revision_id="rev-old-uuid",  # Mismatched revision
        execution_attempt_id="att-1",
        file_path=img_path,
        file_hash=img_hash,
        file_size_bytes=img_size,
        media_type=AssetMediaType.IMAGE,
        mime_type="image/png",
        width=1920,
        height=1080,
        duration_seconds=0.0,
        provider="mock_prov",
        model="model_1",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-new-uuid",
        requested_visual_type=VisualType.AI_IMAGE,
        target_duration=5.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Static",
    )

    decision = policy.evaluate(req, [candidate])
    assert decision.decision == AssetReuseDecisionType.EXECUTE
    assert AssetReuseReasonCode.SHOT_REVISION_CHANGED in decision.reason_codes


def test_10_visual_type_mismatch_prevents_reuse(tmp_path):
    """Invariant 10: Incompatible visual media type causes VISUAL_TYPE_CHANGED."""
    policy = AssetReusePolicy()
    img_path, img_hash, img_size = _create_real_image(tmp_path)

    # Asset is IMAGE, but request is AI_VIDEO
    candidate = ShotAssetVersion(
        shot_asset_version_id="asset-img",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        execution_attempt_id="att-1",
        file_path=img_path,
        file_hash=img_hash,
        file_size_bytes=img_size,
        media_type=AssetMediaType.IMAGE,
        mime_type="image/png",
        width=1920,
        height=1080,
        duration_seconds=0.0,
        provider="mock_prov",
        model="model_1",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_VIDEO,  # Requires VIDEO media type
        target_duration=5.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Static",
    )

    decision = policy.evaluate(req, [candidate])
    assert decision.decision == AssetReuseDecisionType.EXECUTE
    assert AssetReuseReasonCode.VISUAL_TYPE_CHANGED in decision.reason_codes


def test_11_generation_mode_mismatch_prevents_reuse(tmp_path):
    """Invariant 11: Generation mode mismatch causes GENERATION_MODE_CHANGED."""
    policy = AssetReusePolicy()
    img_path, img_hash, img_size = _create_real_image(tmp_path)

    candidate = ShotAssetVersion(
        shot_asset_version_id="asset-mode",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        execution_attempt_id="att-1",
        file_path=img_path,
        file_hash=img_hash,
        file_size_bytes=img_size,
        media_type=AssetMediaType.IMAGE,
        mime_type="image/png",
        width=1920,
        height=1080,
        duration_seconds=0.0,
        provider="mock_prov",
        model="model_1",
        generation_mode=GenerationMode.IMAGE_TO_VIDEO,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_IMAGE,
        target_duration=5.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Static",
    )

    decision = policy.evaluate(
        req,
        [candidate],
        expected_generation_mode=GenerationMode.TEXT_TO_IMAGE,
    )
    assert decision.decision == AssetReuseDecisionType.EXECUTE
    assert AssetReuseReasonCode.GENERATION_MODE_CHANGED in decision.reason_codes


def test_12_aspect_ratio_incompatible_prevents_reuse(tmp_path):
    """Invariant 12: Incompatible aspect ratio (e.g. 9:16 vs 16:9) causes ASPECT_RATIO_INCOMPATIBLE."""
    policy = AssetReusePolicy()
    # Vertical image 1080x1920 (9:16)
    img_path, img_hash, img_size = _create_real_image(tmp_path, "vert.png", size=(1080, 1920))

    candidate = ShotAssetVersion(
        shot_asset_version_id="asset-vert",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        execution_attempt_id="att-1",
        file_path=img_path,
        file_hash=img_hash,
        file_size_bytes=img_size,
        media_type=AssetMediaType.IMAGE,
        mime_type="image/png",
        width=1080,
        height=1920,
        duration_seconds=0.0,
        provider="mock_prov",
        model="model_1",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_IMAGE,
        target_duration=5.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Static",
        aspect_ratio="16:9",  # Horizontal requested!
    )

    decision = policy.evaluate(req, [candidate])
    assert decision.decision == AssetReuseDecisionType.EXECUTE
    assert AssetReuseReasonCode.ASPECT_RATIO_INCOMPATIBLE in decision.reason_codes


def test_13_duration_incompatible_prevents_reuse(tmp_path):
    """Invariant 13: Video duration discrepancy beyond tolerance causes DURATION_INCOMPATIBLE."""
    policy = AssetReusePolicy(duration_tolerance_seconds=2.0)
    img_path, img_hash, img_size = _create_real_image(tmp_path)

    # Video asset duration is 2.0 seconds
    candidate = ShotAssetVersion(
        shot_asset_version_id="asset-short-vid",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        execution_attempt_id="att-1",
        file_path=img_path,
        file_hash=img_hash,
        file_size_bytes=img_size,
        media_type=AssetMediaType.VIDEO,
        mime_type="video/mp4",
        width=1920,
        height=1080,
        duration_seconds=2.0,
        provider="mock_prov",
        model="model_1",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=10.0,  # 10s requested vs 2s asset
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Static",
    )

    decision = policy.evaluate(req, [candidate])
    assert decision.decision == AssetReuseDecisionType.EXECUTE
    assert AssetReuseReasonCode.DURATION_INCOMPATIBLE in decision.reason_codes


def test_14_missing_file_on_disk_prevents_reuse(tmp_path):
    """Invariant 14: Non-existent file on disk causes ASSET_MISSING."""
    policy = AssetReusePolicy()
    missing_file = str(tmp_path / "non_existent.png")

    candidate = ShotAssetVersion(
        shot_asset_version_id="asset-missing",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        execution_attempt_id="att-1",
        file_path=missing_file,
        file_hash="a" * 64,
        file_size_bytes=1024,
        media_type=AssetMediaType.IMAGE,
        mime_type="image/png",
        width=1920,
        height=1080,
        duration_seconds=0.0,
        provider="mock_prov",
        model="model_1",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_IMAGE,
        target_duration=5.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Static",
    )

    decision = policy.evaluate(req, [candidate])
    assert decision.decision == AssetReuseDecisionType.EXECUTE
    assert AssetReuseReasonCode.ASSET_MISSING in decision.reason_codes


def test_15_empty_file_on_disk_prevents_reuse(tmp_path):
    """Invariant 15: Zero-byte file on disk causes ASSET_MISSING / ASSET_INTEGRITY_FAILED."""
    policy = AssetReusePolicy()
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")

    candidate = ShotAssetVersion(
        shot_asset_version_id="asset-empty",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        execution_attempt_id="att-1",
        file_path=str(empty_file),
        file_hash="b" * 64,
        file_size_bytes=1024,
        media_type=AssetMediaType.IMAGE,
        mime_type="image/png",
        width=1920,
        height=1080,
        duration_seconds=0.0,
        provider="mock_prov",
        model="model_1",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_IMAGE,
        target_duration=5.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Static",
    )

    decision = policy.evaluate(req, [candidate])
    assert decision.decision == AssetReuseDecisionType.EXECUTE
    assert (
        AssetReuseReasonCode.ASSET_MISSING in decision.reason_codes
        or AssetReuseReasonCode.ASSET_INTEGRITY_FAILED in decision.reason_codes
    )


def test_16_corrupted_file_hash_prevents_reuse(tmp_path):
    """Invariant 16: Checksum mismatch on disk causes ASSET_INTEGRITY_FAILED."""
    policy = AssetReusePolicy(verify_hash=True)
    img_path, _, img_size = _create_real_image(tmp_path)

    # Candidate specifies wrong hash
    candidate = ShotAssetVersion(
        shot_asset_version_id="asset-corrupt-hash",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        execution_attempt_id="att-1",
        file_path=img_path,
        file_hash="0" * 64,  # Intentionally corrupt
        file_size_bytes=img_size,
        media_type=AssetMediaType.IMAGE,
        mime_type="image/png",
        width=1920,
        height=1080,
        duration_seconds=0.0,
        provider="mock_prov",
        model="model_1",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_IMAGE,
        target_duration=5.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Static",
    )

    decision = policy.evaluate(req, [candidate])
    assert decision.decision == AssetReuseDecisionType.EXECUTE
    assert AssetReuseReasonCode.ASSET_INTEGRITY_FAILED in decision.reason_codes


def test_17_technical_failure_continues_independent_shots(db_session_factory, tmp_path):
    """Invariant 17: Technical failure on Shot 1 does not abort Shot 2 and Shot 3."""
    plan, _ = _setup_test_environment(db_session_factory, shot_count=3)
    img_path, _, _ = _create_real_image(tmp_path)

    single_attempt_policy = ExecutionRetryPolicy(max_attempts_per_candidate=1, max_total_attempts_per_shot=1)

    # Shot 1 fails, Shot 2 succeeds, Shot 3 succeeds
    adapter = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE, error_code="SYS_ERROR"),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
    ])
    service = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(
            adapter_registry=AdapterRegistry({"mock_prov": adapter}),
            policy=single_attempt_policy,
        ),
    )

    result = service.execute_route_plan(plan, storage_base_dir=tmp_path)

    assert result.state == ExecutionStatus.FAILED
    assert result.failed_shots == 1
    assert result.generated_shots == 2
    # All 3 shots were attempted
    assert adapter.call_count == 3
    assert result.shot_results[0].status in (ShotExecutionStatus.FAILED, ShotExecutionStatus.EXHAUSTED)
    assert result.shot_results[1].status == ShotExecutionStatus.SUCCEEDED
    assert result.shot_results[2].status == ShotExecutionStatus.SUCCEEDED


def test_18_needs_recovery_shot_halts_without_blind_retry(db_session_factory, tmp_path):
    """Invariant 18: Shot encountering SUBMISSION_OUTCOME_UNKNOWN halts that shot and marks NEEDS_RECOVERY."""
    plan, _ = _setup_test_environment(db_session_factory, shot_count=1)

    adapter = ConfigurableMockAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
            remote_task_id="unconfirmed-task-999",
            error_message="Gateway timeout waiting for submission ack",
        )
    ])
    service = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": adapter})),
    )

    result = service.execute_route_plan(plan, storage_base_dir=tmp_path)

    assert result.state == ExecutionStatus.NEEDS_RECOVERY
    assert result.recovery_required_shots == 1
    # Exactly 1 call: NO blind retry attempt!
    assert adapter.call_count == 1
    assert result.shot_results[0].status == ShotExecutionStatus.NEEDS_RECOVERY


def test_19_needs_recovery_marks_run_needs_recovery(db_session_factory, tmp_path):
    """Invariant 19: Any shot with NEEDS_RECOVERY causes overall run state to be NEEDS_RECOVERY."""
    plan, _ = _setup_test_environment(db_session_factory, shot_count=2)
    img_path, _, _ = _create_real_image(tmp_path)

    adapter = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN, remote_task_id="task-x"),
    ])
    service = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": adapter})),
    )

    result = service.execute_route_plan(plan, storage_base_dir=tmp_path)

    assert result.state == ExecutionStatus.NEEDS_RECOVERY
    assert result.generated_shots == 1
    assert result.recovery_required_shots == 1


def test_20_all_shots_succeeded_or_reused_aggregates_completed(db_session_factory, tmp_path):
    """Invariant 20: succeeded_shots + reused_shots == total_shots -> COMPLETED."""
    plan, _ = _setup_test_environment(db_session_factory, shot_count=2)
    img_path, _, _ = _create_real_image(tmp_path)

    adapter = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
    ])
    service = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": adapter})),
    )

    result = service.execute_route_plan(plan, storage_base_dir=tmp_path)
    assert result.state == ExecutionStatus.COMPLETED


def test_21_any_shot_failed_aggregates_failed(db_session_factory, tmp_path):
    """Invariant 21: One or more failed shots (and no recovery needed) aggregates to FAILED."""
    plan, _ = _setup_test_environment(db_session_factory, shot_count=2)
    img_path, _, _ = _create_real_image(tmp_path)

    single_attempt_policy = ExecutionRetryPolicy(max_attempts_per_candidate=1, max_total_attempts_per_shot=1)
    adapter = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE, error_code="FAIL"),
    ])
    service = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(
            adapter_registry=AdapterRegistry({"mock_prov": adapter}),
            policy=single_attempt_policy,
        ),
    )

    result = service.execute_route_plan(plan, storage_base_dir=tmp_path)
    assert result.state == ExecutionStatus.FAILED


def test_22_execution_run_persisted_with_counters(db_session_factory, tmp_path):
    """Invariant 22: ExecutionRun in database has exact counts for total, succeeded, reused, failed, recovery."""
    plan, _ = _setup_test_environment(db_session_factory, shot_count=3)
    img_path, _, _ = _create_real_image(tmp_path)

    single_attempt_policy = ExecutionRetryPolicy(max_attempts_per_candidate=1, max_total_attempts_per_shot=1)
    adapter = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE, error_code="ERR"),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN, remote_task_id="t1"),
    ])
    service = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(
            adapter_registry=AdapterRegistry({"mock_prov": adapter}),
            policy=single_attempt_policy,
        ),
    )

    result = service.execute_route_plan(plan, storage_base_dir=tmp_path)

    with db_session_factory() as session:
        exec_repo = ExecutionRepository(session)
        persisted_run = exec_repo.get_execution_run(result.execution_run_id)

        assert persisted_run is not None
        assert persisted_run.total_shots == 3
        assert persisted_run.succeeded_shots == 1
        assert persisted_run.reused_shots == 0
        assert persisted_run.failed_shots == 1
        assert persisted_run.recovery_required_shots == 1
        assert persisted_run.status == ExecutionStatus.NEEDS_RECOVERY


def test_23_rerun_creates_new_execution_run_id(db_session_factory, tmp_path):
    """Invariant 23: Subsequent rerun creates a new ExecutionRun entity, preserving historical run."""
    plan, _ = _setup_test_environment(db_session_factory, shot_count=1)
    img_path, _, _ = _create_real_image(tmp_path)

    adapter = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
    ])
    service = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": adapter})),
    )

    res_1 = service.execute_route_plan(plan, storage_base_dir=tmp_path)
    res_2 = service.execute_route_plan(plan, storage_base_dir=tmp_path)

    assert res_1.execution_run_id != res_2.execution_run_id

    with db_session_factory() as session:
        exec_repo = ExecutionRepository(session)
        run_1 = exec_repo.get_execution_run(res_1.execution_run_id)
        run_2 = exec_repo.get_execution_run(res_2.execution_run_id)
        assert run_1 is not None
        assert run_2 is not None
        assert run_1.execution_run_id != run_2.execution_run_id


def test_24_no_open_db_transaction_across_provider_calls(db_session_factory, tmp_path):
    """Invariant 24: No database transaction is open when provider adapter is executed."""
    plan, _ = _setup_test_environment(db_session_factory, shot_count=1)
    img_path, _, _ = _create_real_image(tmp_path)

    active_tx_detected = []

    def check_tx_hook():
        with db_session_factory() as session:
            active_tx_detected.append(session.in_transaction())

    adapter = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
    ])
    adapter.on_execute_hook = check_tx_hook

    service = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": adapter})),
    )

    service.execute_route_plan(plan, storage_base_dir=tmp_path)

    assert len(active_tx_detected) == 1
    assert active_tx_detected[0] is False


def test_25_audit_transitions_recorded_for_all_shots(db_session_factory, tmp_path):
    """Invariant 25: ShotExecution transitions are recorded in execution_transitions."""
    plan, _ = _setup_test_environment(db_session_factory, shot_count=2)
    img_path, _, _ = _create_real_image(tmp_path)

    single_attempt_policy = ExecutionRetryPolicy(max_attempts_per_candidate=1, max_total_attempts_per_shot=1)
    adapter = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE, error_code="ERR"),
    ])
    service = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(
            adapter_registry=AdapterRegistry({"mock_prov": adapter}),
            policy=single_attempt_policy,
        ),
    )

    result = service.execute_route_plan(plan, storage_base_dir=tmp_path)

    with db_session_factory() as session:
        shot_repo = ShotExecutionRepository(session)
        for s in result.shot_results:
            trans = shot_repo.list_transitions(s.shot_execution_id)
            assert len(trans) >= 1
            assert any(t.to_state == s.status.value for t in trans)


def test_26_deterministic_execution_run_result_returned(db_session_factory, tmp_path):
    """Invariant 26: Return object is a typed ExecutionRunResult with accurate structure."""
    plan, _ = _setup_test_environment(db_session_factory, shot_count=2)
    img_path, _, _ = _create_real_image(tmp_path)

    adapter = ConfigurableMockAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path),
    ])
    service = AssetRoutePlanExecutionService(
        session_factory=db_session_factory,
        shot_execution_service=ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": adapter})),
    )

    result = service.execute_route_plan(plan, storage_base_dir=tmp_path)

    assert isinstance(result, ExecutionRunResult)
    assert result.asset_route_plan_id == plan.asset_route_plan_id
    assert result.storyboard_snapshot_id == plan.storyboard_snapshot_id
    assert result.state == ExecutionStatus.COMPLETED
    assert result.total_shots == 2
    assert result.generated_shots == 2
    assert result.reused_shots == 0
    assert result.failed_shots == 0
    assert result.recovery_required_shots == 0
    assert len(result.shot_results) == 2
    assert result.started_at <= result.finished_at
