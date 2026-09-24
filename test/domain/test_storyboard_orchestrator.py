from uuid import uuid4
import pytest

from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.plan_diff import BeatChangeType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_agent import (
    StoryboardAgent,
    StoryboardError,
    StoryboardExecutionResult,
)
from app.domain.storyboard_orchestrator import (
    BeatStoryboardAction,
    StoryboardBaselineMismatchError,
    StoryboardBeatGenerationError,
    StoryboardBuildInput,
    StoryboardBuildResult,
    StoryboardOrchestrator,
)


class InMemoryRepositories:
    """Mock repository layer covering ContentPlan, Shot, and Storyboard repositories."""

    def __init__(self):
        self.plans: dict[str, ContentPlanRevision] = {}
        self.shots: dict[str, Shot] = {}
        self.revisions: dict[str, ShotRevision] = {}
        self.snapshots: dict[str, StoryboardSnapshot] = {}

    def add_plan(self, plan: ContentPlanRevision) -> ContentPlanRevision:
        self.plans[plan.content_plan_revision_id] = plan
        return plan

    def get_revision(self, content_plan_revision_id: str) -> ContentPlanRevision | None:
        return self.plans.get(content_plan_revision_id)

    def add_shot(self, shot: Shot) -> Shot:
        self.shots[shot.shot_id] = shot
        return shot

    def add_revision(self, revision: ShotRevision) -> ShotRevision:
        self.revisions[revision.shot_revision_id] = revision
        return revision

    def get_shot(self, shot_id: str) -> Shot | None:
        return self.shots.get(shot_id)

    def get_shot_revision(self, shot_revision_id: str) -> ShotRevision | None:
        return self.revisions.get(shot_revision_id)

    def add_snapshot(self, snapshot: StoryboardSnapshot) -> StoryboardSnapshot:
        self.snapshots[snapshot.storyboard_snapshot_id] = snapshot
        return snapshot

    def get_snapshot(self, storyboard_snapshot_id: str) -> StoryboardSnapshot | None:
        return self.snapshots.get(storyboard_snapshot_id)

    def get_snapshot_shot_revisions(
        self, storyboard_snapshot_id: str
    ) -> tuple[ShotRevision, ...]:
        snap = self.get_snapshot(storyboard_snapshot_id)
        if snap is None:
            return ()
        return tuple(
            self.revisions[rev_id]
            for rev_id in snap.shot_revision_ids
            if rev_id in self.revisions
        )


def _make_shot_pair(
    beat_lineage_id: str,
    created_from_beat_instance_id: str,
    local_order: int = 1,
    revision_number: int = 1,
    shot_id: str | None = None,
    shot_revision_id: str | None = None,
    narration: str = "Test narration",
) -> tuple[Shot, ShotRevision]:
    s_id = shot_id or str(uuid4())
    sr_id = shot_revision_id or str(uuid4())
    shot = Shot(
        shot_id=s_id,
        beat_lineage_id=beat_lineage_id,
        local_order=local_order,
    )
    rev = ShotRevision(
        shot_revision_id=sr_id,
        shot_id=s_id,
        revision_number=revision_number,
        beat_lineage_id=beat_lineage_id,
        created_from_beat_instance_id=created_from_beat_instance_id,
        narration=narration,
        target_duration=5.0,
        visual_goal="Goal",
        visual_type=VisualType.STOCK_VIDEO,
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Pan",
        evidence_refs=(),
    )
    return shot, rev


def _setup_mock_agent(agent_fn=None):
    agent = StoryboardAgent()

    def default_fn(input_data, shot_repository=None):
        shot, rev = _make_shot_pair(
            beat_lineage_id=input_data.beat.beat_lineage_id,
            created_from_beat_instance_id=input_data.beat.beat_id,
            local_order=1,
        )
        return StoryboardExecutionResult(shots=(shot,), shot_revisions=(rev,))

    agent.generate_shots_for_beat = agent_fn or default_fn
    return agent


# 1. Brand-new ContentPlan generates Shots for every Beat and produces one DRAFT StoryboardSnapshot
def test_brand_new_content_plan_generates_shots_and_draft_snapshot():
    repos = InMemoryRepositories()
    beat1 = ContentBeat(
        beat_id="b1",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=10.0,
        importance=0.9,
    )
    beat2 = ContentBeat(
        beat_id="b2",
        beat_lineage_id="l2",
        beat_type=BeatType.SUMMARY,
        order=2,
        intent="Summary",
        target_duration=10.0,
        importance=0.8,
    )
    plan = ContentPlanRevision(
        content_plan_revision_id="plan-1",
        revision_number=1,
        topic="Physics",
        overall_target_duration=20.0,
        beats=(beat1, beat2),
    )
    repos.add_plan(plan)

    agent = _setup_mock_agent()
    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=agent,
    )

    result = orchestrator.build_storyboard(
        StoryboardBuildInput(new_content_plan_revision_id="plan-1")
    )

    assert isinstance(result, StoryboardBuildResult)
    assert result.storyboard_snapshot.snapshot_state == StoryboardSnapshotState.DRAFT
    assert result.storyboard_snapshot.content_plan_revision_id == "plan-1"
    assert len(result.storyboard_snapshot.shot_revision_ids) == 2
    assert result.statistics.total_beats == 2
    assert result.statistics.added_beats == 2
    assert result.statistics.reused_beats == 0
    assert result.statistics.generated_shots == 2
    assert result.statistics.reused_shots == 0


# 2. UNCHANGED Beat reuses exact historical ShotRevision IDs
def test_unchanged_beat_reuses_exact_historical_shot_revision_ids():
    repos = InMemoryRepositories()

    # Setup Plan 1 & Baseline Storyboard
    b1 = ContentBeat(
        beat_id="b1",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=10.0,
        importance=0.9,
    )
    plan1 = ContentPlanRevision(
        content_plan_revision_id="plan-1",
        revision_number=1,
        topic="Physics",
        overall_target_duration=10.0,
        beats=(b1,),
    )
    repos.add_plan(plan1)

    s1, r1 = _make_shot_pair(beat_lineage_id="l1", created_from_beat_instance_id="b1")
    repos.add_shot(s1)
    repos.add_revision(r1)

    snap1 = StoryboardSnapshot(
        storyboard_snapshot_id="snap-1",
        content_plan_revision_id="plan-1",
        shot_revision_ids=(r1.shot_revision_id,),
        snapshot_state=StoryboardSnapshotState.APPROVED,
    )
    repos.add_snapshot(snap1)

    # Setup Plan 2 where Beat 1 is UNCHANGED
    b1_revised = ContentBeat(
        beat_id="b1_new_instance",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",  # Same content
        target_duration=10.0,
        importance=0.9,
    )
    plan2 = ContentPlanRevision(
        content_plan_revision_id="plan-2",
        revision_number=2,
        topic="Physics",
        overall_target_duration=10.0,
        beats=(b1_revised,),
    )
    repos.add_plan(plan2)

    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=_setup_mock_agent(),
    )

    result = orchestrator.build_storyboard(
        StoryboardBuildInput(
            new_content_plan_revision_id="plan-2",
            previous_content_plan_revision_id="plan-1",
            previous_storyboard_snapshot_id="snap-1",
        )
    )

    assert result.storyboard_snapshot.shot_revision_ids == (r1.shot_revision_id,)
    assert result.statistics.reused_beats == 1
    assert result.statistics.reused_shots == 1
    assert result.statistics.generated_shots == 0
    assert result.beat_results[0].action == BeatStoryboardAction.REUSED
    assert result.beat_results[0].change_type == BeatChangeType.UNCHANGED


# 3. MOVED Beat reuses exact historical ShotRevision IDs
def test_moved_beat_reuses_exact_historical_shot_revision_ids():
    repos = InMemoryRepositories()

    # Plan 1: Beat A (order 1), Beat B (order 2)
    bA = ContentBeat(
        beat_id="bA",
        beat_lineage_id="lA",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=5.0,
        importance=0.9,
    )
    bB = ContentBeat(
        beat_id="bB",
        beat_lineage_id="lB",
        beat_type=BeatType.SUMMARY,
        order=2,
        intent="Summary",
        target_duration=5.0,
        importance=0.8,
    )
    plan1 = ContentPlanRevision(
        content_plan_revision_id="p1",
        revision_number=1,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(bA, bB),
    )
    repos.add_plan(plan1)

    sA, rA = _make_shot_pair(beat_lineage_id="lA", created_from_beat_instance_id="bA")
    sB, rB = _make_shot_pair(beat_lineage_id="lB", created_from_beat_instance_id="bB")
    repos.add_shot(sA)
    repos.add_revision(rA)
    repos.add_shot(sB)
    repos.add_revision(rB)

    snap1 = StoryboardSnapshot(
        storyboard_snapshot_id="snap-1",
        content_plan_revision_id="p1",
        shot_revision_ids=(rA.shot_revision_id, rB.shot_revision_id),
    )
    repos.add_snapshot(snap1)

    # Plan 2: Swapped orders (Beat B order 1, Beat A order 2)
    bA_moved = ContentBeat(
        beat_id="bA2",
        beat_lineage_id="lA",
        beat_type=BeatType.HOOK,
        order=2,  # Moved to 2
        intent="Hook",
        target_duration=5.0,
        importance=0.9,
    )
    bB_moved = ContentBeat(
        beat_id="bB2",
        beat_lineage_id="lB",
        beat_type=BeatType.SUMMARY,
        order=1,  # Moved to 1
        intent="Summary",
        target_duration=5.0,
        importance=0.8,
    )
    plan2 = ContentPlanRevision(
        content_plan_revision_id="p2",
        revision_number=2,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(bB_moved, bA_moved),
    )
    repos.add_plan(plan2)

    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=_setup_mock_agent(),
    )

    result = orchestrator.build_storyboard(
        StoryboardBuildInput(
            new_content_plan_revision_id="p2",
            previous_content_plan_revision_id="p1",
            previous_storyboard_snapshot_id="snap-1",
        )
    )

    # Both beats are reused
    assert result.statistics.reused_beats == 2
    assert result.statistics.generated_shots == 0
    assert set(result.storyboard_snapshot.shot_revision_ids) == {
        rA.shot_revision_id,
        rB.shot_revision_id,
    }


# 4. MOVED Beat appears in the correct new global position
def test_moved_beat_appears_in_correct_new_global_position():
    repos = InMemoryRepositories()

    # Plan 1: Beat A (order 1), Beat B (order 2)
    bA = ContentBeat(
        beat_id="bA",
        beat_lineage_id="lA",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=5.0,
        importance=0.9,
    )
    bB = ContentBeat(
        beat_id="bB",
        beat_lineage_id="lB",
        beat_type=BeatType.SUMMARY,
        order=2,
        intent="Summary",
        target_duration=5.0,
        importance=0.8,
    )
    plan1 = ContentPlanRevision(
        content_plan_revision_id="p1",
        revision_number=1,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(bA, bB),
    )
    repos.add_plan(plan1)

    sA, rA = _make_shot_pair(beat_lineage_id="lA", created_from_beat_instance_id="bA")
    sB, rB = _make_shot_pair(beat_lineage_id="lB", created_from_beat_instance_id="bB")
    repos.add_shot(sA)
    repos.add_revision(rA)
    repos.add_shot(sB)
    repos.add_revision(rB)

    snap1 = StoryboardSnapshot(
        storyboard_snapshot_id="snap-1",
        content_plan_revision_id="p1",
        shot_revision_ids=(rA.shot_revision_id, rB.shot_revision_id),
    )
    repos.add_snapshot(snap1)

    # Plan 2: Beat B is order 1, Beat A is order 2
    bA_moved = ContentBeat(
        beat_id="bA2",
        beat_lineage_id="lA",
        beat_type=BeatType.HOOK,
        order=2,
        intent="Hook",
        target_duration=5.0,
        importance=0.9,
    )
    bB_moved = ContentBeat(
        beat_id="bB2",
        beat_lineage_id="lB",
        beat_type=BeatType.SUMMARY,
        order=1,
        intent="Summary",
        target_duration=5.0,
        importance=0.8,
    )
    plan2 = ContentPlanRevision(
        content_plan_revision_id="p2",
        revision_number=2,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(bB_moved, bA_moved),
    )
    repos.add_plan(plan2)

    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=_setup_mock_agent(),
    )

    result = orchestrator.build_storyboard(
        StoryboardBuildInput(
            new_content_plan_revision_id="p2",
            previous_content_plan_revision_id="p1",
            previous_storyboard_snapshot_id="snap-1",
        )
    )

    # Must be ordered by new Beat.order (Beat B shots first, then Beat A shots)
    assert result.storyboard_snapshot.shot_revision_ids == (
        rB.shot_revision_id,
        rA.shot_revision_id,
    )


# 5. MODIFIED Beat produces completely NEW Shot IDs
def test_modified_beat_produces_completely_new_shot_ids():
    repos = InMemoryRepositories()

    b1 = ContentBeat(
        beat_id="b1",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Original intent",
        target_duration=10.0,
        importance=0.9,
    )
    plan1 = ContentPlanRevision(
        content_plan_revision_id="p1",
        revision_number=1,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1,),
    )
    repos.add_plan(plan1)

    s1_old, r1_old = _make_shot_pair(
        beat_lineage_id="l1", created_from_beat_instance_id="b1"
    )
    repos.add_shot(s1_old)
    repos.add_revision(r1_old)

    snap1 = StoryboardSnapshot(
        storyboard_snapshot_id="snap-1",
        content_plan_revision_id="p1",
        shot_revision_ids=(r1_old.shot_revision_id,),
    )
    repos.add_snapshot(snap1)

    # Modified intent
    b1_modified = ContentBeat(
        beat_id="b1_v2",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Modified radically different intent",
        target_duration=10.0,
        importance=0.9,
    )
    plan2 = ContentPlanRevision(
        content_plan_revision_id="p2",
        revision_number=2,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1_modified,),
    )
    repos.add_plan(plan2)

    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=_setup_mock_agent(),
    )

    result = orchestrator.build_storyboard(
        StoryboardBuildInput(
            new_content_plan_revision_id="p2",
            previous_content_plan_revision_id="p1",
            previous_storyboard_snapshot_id="snap-1",
        )
    )

    assert result.statistics.regenerated_beats == 1
    assert result.statistics.reused_beats == 0
    assert result.statistics.generated_shots == 1
    new_rev_id = result.storyboard_snapshot.shot_revision_ids[0]
    assert new_rev_id != r1_old.shot_revision_id

    # The revision exists and has a new shot_id
    new_rev = repos.get_shot_revision(new_rev_id)
    assert new_rev.shot_id != s1_old.shot_id


# 6. MODIFIED Beat does NOT reuse old Shot IDs
def test_modified_beat_does_not_reuse_old_shot_ids():
    repos = InMemoryRepositories()
    b1 = ContentBeat(
        beat_id="b1",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Old",
        target_duration=10.0,
        importance=0.9,
    )
    plan1 = ContentPlanRevision(
        content_plan_revision_id="p1",
        revision_number=1,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1,),
    )
    repos.add_plan(plan1)

    s_old, r_old = _make_shot_pair(
        beat_lineage_id="l1", created_from_beat_instance_id="b1"
    )
    repos.add_shot(s_old)
    repos.add_revision(r_old)

    snap1 = StoryboardSnapshot(
        storyboard_snapshot_id="snap-1",
        content_plan_revision_id="p1",
        shot_revision_ids=(r_old.shot_revision_id,),
    )
    repos.add_snapshot(snap1)

    b1_mod = ContentBeat(
        beat_id="b1_v2",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="New intent",
        target_duration=10.0,
        importance=0.9,
    )
    plan2 = ContentPlanRevision(
        content_plan_revision_id="p2",
        revision_number=2,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1_mod,),
    )
    repos.add_plan(plan2)

    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=_setup_mock_agent(),
    )

    result = orchestrator.build_storyboard(
        StoryboardBuildInput(
            new_content_plan_revision_id="p2",
            previous_content_plan_revision_id="p1",
            previous_storyboard_snapshot_id="snap-1",
        )
    )

    new_rev_id = result.storyboard_snapshot.shot_revision_ids[0]
    new_rev = repos.get_shot_revision(new_rev_id)
    assert new_rev.shot_id != s_old.shot_id


# 7. ADDED Beat produces new Shots
def test_added_beat_produces_new_shots():
    repos = InMemoryRepositories()

    # Plan 1 with Beat 1
    b1 = ContentBeat(
        beat_id="b1",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=10.0,
        importance=0.9,
    )
    plan1 = ContentPlanRevision(
        content_plan_revision_id="p1",
        revision_number=1,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1,),
    )
    repos.add_plan(plan1)
    s1, r1 = _make_shot_pair(beat_lineage_id="l1", created_from_beat_instance_id="b1")
    repos.add_shot(s1)
    repos.add_revision(r1)
    snap1 = StoryboardSnapshot(
        storyboard_snapshot_id="snap-1",
        content_plan_revision_id="p1",
        shot_revision_ids=(r1.shot_revision_id,),
    )
    repos.add_snapshot(snap1)

    # Plan 2 keeps Beat 1 and adds Beat 2
    b1_kept = ContentBeat(
        beat_id="b1_v2",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=10.0,
        importance=0.9,
    )
    b2_added = ContentBeat(
        beat_id="b2",
        beat_lineage_id="l2",
        beat_type=BeatType.SUMMARY,
        order=2,
        intent="Summary",
        target_duration=10.0,
        importance=0.8,
    )
    plan2 = ContentPlanRevision(
        content_plan_revision_id="p2",
        revision_number=2,
        topic="Topic",
        overall_target_duration=20.0,
        beats=(b1_kept, b2_added),
    )
    repos.add_plan(plan2)

    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=_setup_mock_agent(),
    )

    result = orchestrator.build_storyboard(
        StoryboardBuildInput(
            new_content_plan_revision_id="p2",
            previous_content_plan_revision_id="p1",
            previous_storyboard_snapshot_id="snap-1",
        )
    )

    assert result.statistics.reused_beats == 1
    assert result.statistics.added_beats == 1
    assert result.statistics.generated_shots == 1
    assert len(result.storyboard_snapshot.shot_revision_ids) == 2
    assert result.storyboard_snapshot.shot_revision_ids[0] == r1.shot_revision_id


# 8. REMOVED Beat's historical Shots do not appear in the new StoryboardSnapshot
def test_removed_beat_historical_shots_do_not_appear_in_new_snapshot():
    repos = InMemoryRepositories()

    # Plan 1: Beat 1 and Beat 2
    b1 = ContentBeat(
        beat_id="b1",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=10.0,
        importance=0.9,
    )
    b2 = ContentBeat(
        beat_id="b2",
        beat_lineage_id="l2",
        beat_type=BeatType.SUMMARY,
        order=2,
        intent="Summary",
        target_duration=10.0,
        importance=0.8,
    )
    plan1 = ContentPlanRevision(
        content_plan_revision_id="p1",
        revision_number=1,
        topic="Topic",
        overall_target_duration=20.0,
        beats=(b1, b2),
    )
    repos.add_plan(plan1)

    s1, r1 = _make_shot_pair(beat_lineage_id="l1", created_from_beat_instance_id="b1")
    s2, r2 = _make_shot_pair(beat_lineage_id="l2", created_from_beat_instance_id="b2")
    repos.add_shot(s1)
    repos.add_revision(r1)
    repos.add_shot(s2)
    repos.add_revision(r2)

    snap1 = StoryboardSnapshot(
        storyboard_snapshot_id="snap-1",
        content_plan_revision_id="p1",
        shot_revision_ids=(r1.shot_revision_id, r2.shot_revision_id),
    )
    repos.add_snapshot(snap1)

    # Plan 2: Beat 2 is removed, only Beat 1 remains
    b1_kept = ContentBeat(
        beat_id="b1_v2",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=10.0,
        importance=0.9,
    )
    plan2 = ContentPlanRevision(
        content_plan_revision_id="p2",
        revision_number=2,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1_kept,),
    )
    repos.add_plan(plan2)

    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=_setup_mock_agent(),
    )

    result = orchestrator.build_storyboard(
        StoryboardBuildInput(
            new_content_plan_revision_id="p2",
            previous_content_plan_revision_id="p1",
            previous_storyboard_snapshot_id="snap-1",
        )
    )

    assert result.statistics.removed_beats == 1
    assert result.storyboard_snapshot.shot_revision_ids == (r1.shot_revision_id,)
    assert r2.shot_revision_id not in result.storyboard_snapshot.shot_revision_ids

    # Check that removed beat was reported in beat_results
    removed_results = [
        r for r in result.beat_results if r.action == BeatStoryboardAction.REMOVED
    ]
    assert len(removed_results) == 1
    assert removed_results[0].beat_lineage_id == "l2"


# 9. Old StoryboardSnapshot remains unchanged
def test_old_storyboard_snapshot_remains_unchanged():
    repos = InMemoryRepositories()
    b1 = ContentBeat(
        beat_id="b1",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=10.0,
        importance=0.9,
    )
    plan1 = ContentPlanRevision(
        content_plan_revision_id="p1",
        revision_number=1,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1,),
    )
    repos.add_plan(plan1)
    s1, r1 = _make_shot_pair(beat_lineage_id="l1", created_from_beat_instance_id="b1")
    repos.add_shot(s1)
    repos.add_revision(r1)

    snap1 = StoryboardSnapshot(
        storyboard_snapshot_id="snap-1",
        content_plan_revision_id="p1",
        shot_revision_ids=(r1.shot_revision_id,),
        snapshot_state=StoryboardSnapshotState.APPROVED,
    )
    repos.add_snapshot(snap1)

    plan2 = ContentPlanRevision(
        content_plan_revision_id="p2",
        revision_number=2,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1,),
    )
    repos.add_plan(plan2)

    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=_setup_mock_agent(),
    )
    orchestrator.build_storyboard(
        StoryboardBuildInput(
            new_content_plan_revision_id="p2",
            previous_content_plan_revision_id="p1",
            previous_storyboard_snapshot_id="snap-1",
        )
    )

    old_snap = repos.get_snapshot("snap-1")
    assert old_snap.snapshot_state == StoryboardSnapshotState.APPROVED
    assert old_snap.content_plan_revision_id == "p1"
    assert old_snap.shot_revision_ids == (r1.shot_revision_id,)


# 10. New StoryboardSnapshot references the new ContentPlanRevision
def test_new_storyboard_snapshot_references_new_content_plan_revision():
    repos = InMemoryRepositories()
    b1 = ContentBeat(
        beat_id="b1",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=10.0,
        importance=0.9,
    )
    plan = ContentPlanRevision(
        content_plan_revision_id="p-unique-123",
        revision_number=1,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1,),
    )
    repos.add_plan(plan)

    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=_setup_mock_agent(),
    )
    result = orchestrator.build_storyboard(
        StoryboardBuildInput(new_content_plan_revision_id="p-unique-123")
    )

    assert result.storyboard_snapshot.content_plan_revision_id == "p-unique-123"


# 11. New StoryboardSnapshot state is DRAFT
def test_new_storyboard_snapshot_state_is_draft():
    repos = InMemoryRepositories()
    b1 = ContentBeat(
        beat_id="b1",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=10.0,
        importance=0.9,
    )
    plan1 = ContentPlanRevision(
        content_plan_revision_id="p1",
        revision_number=1,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1,),
    )
    repos.add_plan(plan1)
    s1, r1 = _make_shot_pair(beat_lineage_id="l1", created_from_beat_instance_id="b1")
    repos.add_shot(s1)
    repos.add_revision(r1)

    # Baseline snapshot was APPROVED
    snap1 = StoryboardSnapshot(
        storyboard_snapshot_id="snap-1",
        content_plan_revision_id="p1",
        shot_revision_ids=(r1.shot_revision_id,),
        snapshot_state=StoryboardSnapshotState.APPROVED,
    )
    repos.add_snapshot(snap1)

    plan2 = ContentPlanRevision(
        content_plan_revision_id="p2",
        revision_number=2,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1,),
    )
    repos.add_plan(plan2)

    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=_setup_mock_agent(),
    )
    result = orchestrator.build_storyboard(
        StoryboardBuildInput(
            new_content_plan_revision_id="p2",
            previous_content_plan_revision_id="p1",
            previous_storyboard_snapshot_id="snap-1",
        )
    )

    # State must be DRAFT, never copied from APPROVED
    assert result.storyboard_snapshot.snapshot_state == StoryboardSnapshotState.DRAFT


# 12. Baseline plan/snapshot mismatch is rejected
def test_baseline_plan_snapshot_mismatch_rejected():
    repos = InMemoryRepositories()
    b1 = ContentBeat(
        beat_id="b1",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=10.0,
        importance=0.9,
    )
    plan1 = ContentPlanRevision(
        content_plan_revision_id="p1",
        revision_number=1,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1,),
    )
    repos.add_plan(plan1)
    plan2 = ContentPlanRevision(
        content_plan_revision_id="p2",
        revision_number=2,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1,),
    )
    repos.add_plan(plan2)

    # Snapshot created for an unrelated plan 'other-plan-999'
    snap_mismatch = StoryboardSnapshot(
        storyboard_snapshot_id="snap-bad",
        content_plan_revision_id="other-plan-999",
        shot_revision_ids=(),
    )
    repos.add_snapshot(snap_mismatch)

    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=_setup_mock_agent(),
    )

    # Case A: Snapshot doesn't match plan
    with pytest.raises(StoryboardBaselineMismatchError) as exc_info:
        orchestrator.build_storyboard(
            StoryboardBuildInput(
                new_content_plan_revision_id="p2",
                previous_content_plan_revision_id="p1",
                previous_storyboard_snapshot_id="snap-bad",
            )
        )
    assert "Baseline mismatch" in str(exc_info.value)

    # Case B: Only one of (previous_content_plan_revision_id, previous_storyboard_snapshot_id) supplied
    with pytest.raises(StoryboardBaselineMismatchError):
        orchestrator.build_storyboard(
            StoryboardBuildInput(
                new_content_plan_revision_id="p2",
                previous_content_plan_revision_id="p1",
                previous_storyboard_snapshot_id=None,
            )
        )


# 13. Reuse uses exact frozen ShotRevision from baseline Snapshot, not latest ShotRevision in database
def test_reuse_uses_exact_frozen_shot_revision_from_baseline_snapshot_not_latest():
    repos = InMemoryRepositories()
    b1 = ContentBeat(
        beat_id="b1",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=10.0,
        importance=0.9,
    )
    plan1 = ContentPlanRevision(
        content_plan_revision_id="p1",
        revision_number=1,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1,),
    )
    repos.add_plan(plan1)

    s1, r1_v1 = _make_shot_pair(
        beat_lineage_id="l1",
        created_from_beat_instance_id="b1",
        revision_number=1,
        shot_revision_id="sr-frozen-v1",
    )
    repos.add_shot(s1)
    repos.add_revision(r1_v1)

    # Later revision 2 added to the database for this shot
    _, r1_v2 = _make_shot_pair(
        beat_lineage_id="l1",
        created_from_beat_instance_id="b1",
        revision_number=2,
        shot_id=s1.shot_id,
        shot_revision_id="sr-later-v2",
    )
    repos.add_revision(r1_v2)

    # Snapshot 1 was frozen when r1_v1 was current
    snap1 = StoryboardSnapshot(
        storyboard_snapshot_id="snap-1",
        content_plan_revision_id="p1",
        shot_revision_ids=("sr-frozen-v1",),
    )
    repos.add_snapshot(snap1)

    # Plan 2 keeps Beat 1 UNCHANGED
    plan2 = ContentPlanRevision(
        content_plan_revision_id="p2",
        revision_number=2,
        topic="Topic",
        overall_target_duration=10.0,
        beats=(b1,),
    )
    repos.add_plan(plan2)

    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=_setup_mock_agent(),
    )

    result = orchestrator.build_storyboard(
        StoryboardBuildInput(
            new_content_plan_revision_id="p2",
            previous_content_plan_revision_id="p1",
            previous_storyboard_snapshot_id="snap-1",
        )
    )

    # Must reuse exact frozen ID "sr-frozen-v1", NOT "sr-later-v2"
    assert result.storyboard_snapshot.shot_revision_ids == ("sr-frozen-v1",)


# 14. Failure generating one required Beat does not produce a successful partial StoryboardSnapshot
def test_failure_generating_one_required_beat_does_not_produce_partial_snapshot():
    repos = InMemoryRepositories()
    b1 = ContentBeat(
        beat_id="b1",
        beat_lineage_id="l1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=10.0,
        importance=0.9,
    )
    b2 = ContentBeat(
        beat_id="b2",
        beat_lineage_id="l2",
        beat_type=BeatType.SUMMARY,
        order=2,
        intent="Summary",
        target_duration=10.0,
        importance=0.8,
    )
    plan = ContentPlanRevision(
        content_plan_revision_id="p1",
        revision_number=1,
        topic="Topic",
        overall_target_duration=20.0,
        beats=(b1, b2),
    )
    repos.add_plan(plan)

    # Agent fails on beat 2
    def failing_agent_fn(input_data, shot_repository=None):
        if input_data.beat.beat_lineage_id == "l2":
            raise StoryboardError("LLM failed on beat 2")
        shot, rev = _make_shot_pair(
            beat_lineage_id="l1", created_from_beat_instance_id="b1"
        )
        return StoryboardExecutionResult(shots=(shot,), shot_revisions=(rev,))

    agent = _setup_mock_agent(failing_agent_fn)
    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=agent,
    )

    with pytest.raises(StoryboardBeatGenerationError) as exc_info:
        orchestrator.build_storyboard(
            StoryboardBuildInput(new_content_plan_revision_id="p1")
        )
    assert exc_info.value.beat_lineage_id == "l2"

    # Verify no snapshot was persisted
    assert len(repos.snapshots) == 0
    # And no shots were persisted
    assert len(repos.shots) == 0


# 15. StoryboardAgent is NOT called for UNCHANGED/MOVED Beats
def test_storyboard_agent_not_called_for_unchanged_or_moved_beats():
    repos = InMemoryRepositories()

    # Plan 1: Beat A (order 1), Beat B (order 2), Beat C (order 3)
    bA = ContentBeat(
        beat_id="bA",
        beat_lineage_id="lA",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=5.0,
        importance=0.9,
    )
    bB = ContentBeat(
        beat_id="bB",
        beat_lineage_id="lB",
        beat_type=BeatType.KNOWLEDGE,
        order=2,
        intent="Knowledge",
        target_duration=5.0,
        importance=0.9,
    )
    bC = ContentBeat(
        beat_id="bC",
        beat_lineage_id="lC",
        beat_type=BeatType.SUMMARY,
        order=3,
        intent="Summary",
        target_duration=5.0,
        importance=0.8,
    )
    plan1 = ContentPlanRevision(
        content_plan_revision_id="p1",
        revision_number=1,
        topic="Topic",
        overall_target_duration=15.0,
        beats=(bA, bB, bC),
    )
    repos.add_plan(plan1)

    sA, rA = _make_shot_pair(beat_lineage_id="lA", created_from_beat_instance_id="bA")
    sB, rB = _make_shot_pair(beat_lineage_id="lB", created_from_beat_instance_id="bB")
    sC, rC = _make_shot_pair(beat_lineage_id="lC", created_from_beat_instance_id="bC")
    for s, r in [(sA, rA), (sB, rB), (sC, rC)]:
        repos.add_shot(s)
        repos.add_revision(r)

    snap1 = StoryboardSnapshot(
        storyboard_snapshot_id="snap-1",
        content_plan_revision_id="p1",
        shot_revision_ids=(
            rA.shot_revision_id,
            rB.shot_revision_id,
            rC.shot_revision_id,
        ),
    )
    repos.add_snapshot(snap1)

    # Plan 2:
    # Beat A is UNCHANGED (order 1)
    # Beat B is MOVED to order 3
    # Beat C is MODIFIED and at order 2
    bA_unchanged = ContentBeat(
        beat_id="bA2",
        beat_lineage_id="lA",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook",
        target_duration=5.0,
        importance=0.9,
    )
    bB_moved = ContentBeat(
        beat_id="bB2",
        beat_lineage_id="lB",
        beat_type=BeatType.KNOWLEDGE,
        order=3,
        intent="Knowledge",
        target_duration=5.0,
        importance=0.9,
    )
    bC_modified = ContentBeat(
        beat_id="bC2",
        beat_lineage_id="lC",
        beat_type=BeatType.SUMMARY,
        order=2,
        intent="Completely new summary intent",
        target_duration=5.0,
        importance=0.8,
    )
    plan2 = ContentPlanRevision(
        content_plan_revision_id="p2",
        revision_number=2,
        topic="Topic",
        overall_target_duration=15.0,
        beats=(bA_unchanged, bC_modified, bB_moved),
    )
    repos.add_plan(plan2)

    agent_calls = []

    def tracking_agent_fn(input_data, shot_repository=None):
        agent_calls.append(input_data.beat.beat_lineage_id)
        shot, rev = _make_shot_pair(
            beat_lineage_id=input_data.beat.beat_lineage_id,
            created_from_beat_instance_id=input_data.beat.beat_id,
        )
        return StoryboardExecutionResult(shots=(shot,), shot_revisions=(rev,))

    agent = _setup_mock_agent(tracking_agent_fn)
    orchestrator = StoryboardOrchestrator(
        plan_repository=repos,
        shot_repository=repos,
        storyboard_repository=repos,
        storyboard_agent=agent,
    )

    result = orchestrator.build_storyboard(
        StoryboardBuildInput(
            new_content_plan_revision_id="p2",
            previous_content_plan_revision_id="p1",
            previous_storyboard_snapshot_id="snap-1",
        )
    )

    # StoryboardAgent MUST be called ONLY for lC (MODIFIED)
    # MUST NOT be called for lA (UNCHANGED) or lB (MOVED)
    assert agent_calls == ["lC"]
    assert result.statistics.reused_beats == 2
    assert result.statistics.regenerated_beats == 1
    assert result.statistics.generated_shots == 1
    assert result.statistics.reused_shots == 2
