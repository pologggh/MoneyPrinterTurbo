from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.domain.asset_execution import (
    AdapterExecutionResult,
    AssetReuseMode,
    ExecutionRetryPolicy,
    ExecutionStatus,
    ProviderOutcomeType,
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
    ShotRepository,
    StoryboardRepository,
)
from app.services.asset_adapters.base import AssetExecutionAdapter
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.asset_probe_service import calculate_sha256
from app.services.asset_route_plan_execution_service import (
    AssetRoutePlanExecutionService,
)
from app.services.shot_execution_service import ShotExecutionService


class ScriptedMockAdapter(AssetExecutionAdapter):
    """Adapter with explicit response mapping per shot_id or sequence."""

    def __init__(self, response_map: dict[str, list[AdapterExecutionResult]] | None = None):
        self.response_map = response_map or {}
        self.calls_by_shot: dict[str, int] = {}
        self.total_calls = 0

    def execute(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        shot_id = request.shot_id
        count = self.calls_by_shot.get(shot_id, 0)
        self.calls_by_shot[shot_id] = count + 1
        self.total_calls += 1

        queue = self.response_map.get(shot_id, [])
        if count < len(queue):
            return queue[count]

        # Default success generator
        target_dir.mkdir(parents=True, exist_ok=True)
        img_p = target_dir / f"{shot_id}_output_{uuid4().hex[:6]}.png"
        img = Image.new("RGB", (1920, 1080), color="purple")
        img.save(str(img_p))
        return AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path=str(img_p),
        )


@pytest.fixture
def rerun_db_factory():
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


def _create_real_image(target_dir: Path, filename: str = "asset.png", size=(1920, 1080)) -> tuple[str, str, int]:
    target_dir.mkdir(parents=True, exist_ok=True)
    p = target_dir / filename
    img = Image.new("RGB", size, color="teal")
    img.save(str(p))
    sha = calculate_sha256(p)
    return str(p), sha, p.stat().st_size


def _setup_storyboard_plan(
    session_factory,
    shot_count: int = 4,
) -> tuple[AssetRoutePlan, list[ShotRevision]]:
    with session_factory() as session:
        plan_repo = ContentPlanRepository(session)
        shot_repo = ShotRepository(session)
        sb_repo = StoryboardRepository(session)
        route_repo = AssetRoutePlanRepository(session)

        beat = ContentBeat(
            beat_id="beat-rerun-1",
            beat_lineage_id="lin-rerun-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Incremental Rerun Testing",
            target_duration=float(shot_count * 5),
            importance=1.0,
        )
        c_plan = ContentPlanRevision(
            content_plan_revision_id="cplan-rerun-1",
            revision_number=1,
            topic="Incremental Rerun Scenarios",
            overall_target_duration=float(shot_count * 5),
            beats=(beat,),
        )
        plan_repo.add_revision(c_plan)

        revs: list[ShotRevision] = []
        entries: list[ShotRoutePlanEntry] = []

        for i in range(1, shot_count + 1):
            s = Shot(shot_id=f"shot-{i}", beat_lineage_id="lin-rerun-1", local_order=i)
            shot_repo.add_shot(s)

            r = ShotRevision(
                shot_revision_id=f"rev-{i}",
                shot_id=f"shot-{i}",
                revision_number=1,
                beat_lineage_id="lin-rerun-1",
                created_from_beat_instance_id="beat-rerun-1",
                narration=f"Narration {i}",
                target_duration=5.0,
                visual_goal=f"Goal {i}",
                visual_type=VisualType.AI_IMAGE,
                scene_description=f"Scene {i}",
                generation_prompt=f"Prompt {i}",
                camera_movement="Static",
            )
            shot_repo.add_revision(r)
            revs.append(r)

            cand = AssetRouteCandidate(
                capability_id="mock_prov:v1:TEXT_TO_IMAGE",
                provider="mock_prov",
                model="v1",
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
                beat_lineage_id="lin-rerun-1",
                requested_visual_type=VisualType.AI_IMAGE,
                asset_routing_request=AssetRoutingRequest(
                    shot_id=f"shot-{i}",
                    shot_revision_id=f"rev-{i}",
                    requested_visual_type=VisualType.AI_IMAGE,
                    target_duration=5.0,
                    visual_goal=f"Goal {i}",
                    scene_description=f"Scene {i}",
                    generation_prompt=f"Prompt {i}",
                    camera_movement="Static",
                    aspect_ratio="16:9",
                ),
                route_decision=dec,
                route_status=ShotRoutePlanStatus.ROUTED,
            )
            entries.append(entry)

        snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap-rerun-approved",
            content_plan_revision_id="cplan-rerun-1",
            shot_revision_ids=tuple(r.shot_revision_id for r in revs),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        sb_repo.add_snapshot(snap)

        route_plan = AssetRoutePlan(
            asset_route_plan_id="plan-rerun-1",
            storyboard_snapshot_id="snap-rerun-approved",
            content_plan_revision_id="cplan-rerun-1",
            routing_strategy=RoutingStrategy.BALANCED,
            routing_policy_version="v1.0",
            shot_routes=tuple(entries),
            status=AssetRoutePlanStatus.READY,
            total_shots=len(entries),
            routed_shots=len(entries),
            blocked_shots=0,
        )
        route_repo.add_route_plan(route_plan)
        session.commit()

    return route_plan, revs


# ==============================================================================
# INCREMENTAL RERUN SCENARIOS
# ==============================================================================


def test_scenario_e1_failure_then_e2_rerun_partial_recovery(rerun_db_factory, tmp_path):
    """
    Scenario:
    - 4-shot storyboard plan.
    - Run E1: Shot 1, 2, 4 succeed; Shot 3 fails due to provider network error.
      -> Run E1 outcome: FAILED.
    - Run E2: Rerun same plan with REUSE_COMPATIBLE.
      -> Shots 1, 2, 4 are reused with 0 provider calls.
      -> Shot 3 is re-submitted and succeeds.
      -> Run E2 outcome: COMPLETED.
      -> Provider called exactly 1 time across all 4 shots in Run E2.
      -> Both Run E1 and Run E2 are preserved in DB with distinct IDs and statuses.
    """
    plan, _ = _setup_storyboard_plan(rerun_db_factory, shot_count=4)
    img_path, _, _ = _create_real_image(tmp_path, "success.png")

    single_attempt_policy = ExecutionRetryPolicy(max_attempts_per_candidate=1, max_total_attempts_per_shot=1)

    # Configure Run E1 adapter responses
    adapter_e1 = ScriptedMockAdapter({
        "shot-1": [AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path)],
        "shot-2": [AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path)],
        "shot-3": [AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE, error_code="NET_CONN_REFUSED")],
        "shot-4": [AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path)],
    })
    service_e1 = AssetRoutePlanExecutionService(
        session_factory=rerun_db_factory,
        shot_execution_service=ShotExecutionService(
            adapter_registry=AdapterRegistry({"mock_prov": adapter_e1}),
            policy=single_attempt_policy,
        ),
    )

    result_e1 = service_e1.execute_route_plan(plan, storage_base_dir=tmp_path)

    assert result_e1.state == ExecutionStatus.FAILED
    assert result_e1.total_shots == 4
    assert result_e1.generated_shots == 3
    assert result_e1.failed_shots == 1
    assert result_e1.reused_shots == 0
    assert adapter_e1.total_calls == 4

    # Configure Run E2 adapter responses: Shot 3 will now succeed!
    adapter_e2 = ScriptedMockAdapter({
        "shot-3": [AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=img_path)],
    })
    service_e2 = AssetRoutePlanExecutionService(
        session_factory=rerun_db_factory,
        shot_execution_service=ShotExecutionService(
            adapter_registry=AdapterRegistry({"mock_prov": adapter_e2}),
            policy=single_attempt_policy,
        ),
    )

    result_e2 = service_e2.execute_route_plan(plan, storage_base_dir=tmp_path, reuse_mode=AssetReuseMode.REUSE_COMPATIBLE)

    assert result_e2.state == ExecutionStatus.COMPLETED
    assert result_e2.total_shots == 4
    assert result_e2.reused_shots == 3
    assert result_e2.generated_shots == 1
    assert result_e2.failed_shots == 0
    # Strict efficiency: Adapter was called EXACTLY ONCE!
    assert adapter_e2.total_calls == 1
    assert adapter_e2.calls_by_shot == {"shot-3": 1}

    # Verify both runs coexist in database
    with rerun_db_factory() as session:
        exec_repo = ExecutionRepository(session)
        runs = exec_repo.list_runs_for_plan(plan.asset_route_plan_id)
        assert len(runs) == 2
        run_e2_db = next(r for r in runs if r.execution_run_id == result_e2.execution_run_id)
        run_e1_db = next(r for r in runs if r.execution_run_id == result_e1.execution_run_id)

        assert run_e2_db.status == ExecutionStatus.COMPLETED
        assert run_e2_db.reused_shots == 3
        assert run_e2_db.succeeded_shots == 1

        assert run_e1_db.status == ExecutionStatus.FAILED
        assert run_e1_db.failed_shots == 1
        assert run_e1_db.succeeded_shots == 3


def test_scenario_rerun_with_force_regenerate_preserves_history(rerun_db_factory, tmp_path):
    """
    Scenario:
    - Run E1 creates compatible assets for all shots.
    - Run E2 is executed with FORCE_REGENERATE.
    - All shots are re-executed via provider.
    - New versions are produced without overwriting or destroying historical versions.
    """
    plan, revs = _setup_storyboard_plan(rerun_db_factory, shot_count=2)
    _img_path, _, _ = _create_real_image(tmp_path)

    # Run E1
    adapter_e1 = ScriptedMockAdapter()
    service = AssetRoutePlanExecutionService(
        session_factory=rerun_db_factory,
        shot_execution_service=ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": adapter_e1})),
    )
    result_e1 = service.execute_route_plan(plan, storage_base_dir=tmp_path)
    assert result_e1.state == ExecutionStatus.COMPLETED
    assert result_e1.generated_shots == 2
    assert adapter_e1.total_calls == 2

    # Run E2: FORCE_REGENERATE
    adapter_e2 = ScriptedMockAdapter()
    service_e2 = AssetRoutePlanExecutionService(
        session_factory=rerun_db_factory,
        shot_execution_service=ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": adapter_e2})),
    )
    result_e2 = service_e2.execute_route_plan(
        plan,
        storage_base_dir=tmp_path,
        reuse_mode=AssetReuseMode.FORCE_REGENERATE,
    )
    assert result_e2.state == ExecutionStatus.COMPLETED
    assert result_e2.generated_shots == 2
    assert result_e2.reused_shots == 0
    assert adapter_e2.total_calls == 2

    # Verify versions in DB: 2 revisions each have 2 distinct ShotAssetVersion records!
    with rerun_db_factory() as session:
        exec_repo = ExecutionRepository(session)
        for r in revs:
            v_list = exec_repo.list_asset_versions_for_shot_revision(r.shot_revision_id)
            assert len(v_list) == 2
            # Both versions are preserved with unique IDs
            assert v_list[0].shot_asset_version_id != v_list[1].shot_asset_version_id


def test_scenario_rerun_after_single_shot_editing(rerun_db_factory, tmp_path):
    """
    Scenario:
    - 3 shots in base plan.
    - Run E1 executes and succeeds for shots 1, 2, 3.
    - Shot 2 narration is edited, creating new revision rev-2-v2.
    - A new route plan is created with rev-1, rev-2-v2, rev-3.
    - Execution of new plan reuses shot 1 and shot 3, and only calls provider for modified shot 2!
    """
    plan, _revs = _setup_storyboard_plan(rerun_db_factory, shot_count=3)
    _img_path, _, _ = _create_real_image(tmp_path)

    # Run E1 on base plan
    adapter_e1 = ScriptedMockAdapter()
    service_e1 = AssetRoutePlanExecutionService(
        session_factory=rerun_db_factory,
        shot_execution_service=ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": adapter_e1})),
    )
    res_e1 = service_e1.execute_route_plan(plan, storage_base_dir=tmp_path)
    assert res_e1.state == ExecutionStatus.COMPLETED

    # Now simulate Shot 2 edit: create new ShotRevision for shot-2
    with rerun_db_factory() as session:
        shot_repo = ShotRepository(session)
        sb_repo = StoryboardRepository(session)
        route_repo = AssetRoutePlanRepository(session)

        rev_2_edited = ShotRevision(
            shot_revision_id="rev-2-v2",
            shot_id="shot-2",
            revision_number=2,
            beat_lineage_id="lin-rerun-1",
            created_from_beat_instance_id="beat-rerun-1",
            narration="Updated narration for shot 2",
            target_duration=5.0,
            visual_goal="Updated visual goal",
            visual_type=VisualType.AI_IMAGE,
            scene_description="Updated scene",
            generation_prompt="Updated prompt",
            camera_movement="Static",
        )
        shot_repo.add_revision(rev_2_edited)

        # Create new approved snapshot containing rev-1, rev-2-v2, rev-3
        snap_v2 = StoryboardSnapshot(
            storyboard_snapshot_id="snap-v2-approved",
            content_plan_revision_id="cplan-rerun-1",
            shot_revision_ids=("rev-1", "rev-2-v2", "rev-3"),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        sb_repo.add_snapshot(snap_v2)

        # Create route plan for new snapshot
        entries_v2 = []
        for r_id, s_id in [("rev-1", "shot-1"), ("rev-2-v2", "shot-2"), ("rev-3", "shot-3")]:
            cand = AssetRouteCandidate(
                capability_id="mock_prov:v1:TEXT_TO_IMAGE",
                provider="mock_prov",
                model="v1",
                generation_mode=GenerationMode.TEXT_TO_IMAGE,
                requested_visual_type=VisualType.AI_IMAGE,
                is_eligible=True,
            )
            dec = AssetRouteDecision(
                shot_id=s_id,
                shot_revision_id=r_id,
                routing_strategy=RoutingStrategy.BALANCED,
                requested_visual_type=VisualType.AI_IMAGE,
                selected_candidate=cand,
                eligible_candidates=(cand,),
            )
            entries_v2.append(
                ShotRoutePlanEntry(
                    shot_id=s_id,
                    shot_revision_id=r_id,
                    beat_lineage_id="lin-rerun-1",
                    requested_visual_type=VisualType.AI_IMAGE,
                    asset_routing_request=AssetRoutingRequest(
                        shot_id=s_id,
                        shot_revision_id=r_id,
                        requested_visual_type=VisualType.AI_IMAGE,
                        target_duration=5.0,
                        visual_goal="Goal",
                        scene_description="Desc",
                        generation_prompt="Prompt",
                        camera_movement="Static",
                        aspect_ratio="16:9",
                    ),
                    route_decision=dec,
                    route_status=ShotRoutePlanStatus.ROUTED,
                )
            )

        plan_v2 = AssetRoutePlan(
            asset_route_plan_id="plan-rerun-v2",
            storyboard_snapshot_id="snap-v2-approved",
            content_plan_revision_id="cplan-rerun-1",
            routing_strategy=RoutingStrategy.BALANCED,
            routing_policy_version="v1.0",
            shot_routes=tuple(entries_v2),
            status=AssetRoutePlanStatus.READY,
            total_shots=3,
            routed_shots=3,
            blocked_shots=0,
        )
        route_repo.add_route_plan(plan_v2)
        session.commit()

    # Execute plan_v2 with REUSE_COMPATIBLE
    adapter_e2 = ScriptedMockAdapter()
    service_e2 = AssetRoutePlanExecutionService(
        session_factory=rerun_db_factory,
        shot_execution_service=ShotExecutionService(adapter_registry=AdapterRegistry({"mock_prov": adapter_e2})),
    )
    result_e2 = service_e2.execute_route_plan(plan_v2, storage_base_dir=tmp_path)

    assert result_e2.state == ExecutionStatus.COMPLETED
    assert result_e2.reused_shots == 2  # Shot 1 and Shot 3 reused!
    assert result_e2.generated_shots == 1  # Shot 2 re-generated due to revision change!
    assert result_e2.failed_shots == 0
    assert adapter_e2.total_calls == 1
    assert adapter_e2.calls_by_shot == {"shot-2": 1}
