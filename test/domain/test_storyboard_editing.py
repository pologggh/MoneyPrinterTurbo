import pytest
from pydantic import ValidationError

from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_editing import (
    CrossBeatMoveNotAllowedError,
    EditShotInput,
    InvalidShotEditError,
    InvalidShotOrderError,
    ReorderShotsInput,
    StaleShotRevisionError,
    StoryboardEditingService,
    StoryboardEditingView,
)


class InMemoryPlanRepo:
    def __init__(self):
        self.plans: dict[str, ContentPlanRevision] = {}

    def add_plan(self, plan: ContentPlanRevision) -> ContentPlanRevision:
        self.plans[plan.content_plan_revision_id] = plan
        return plan

    def get_revision(self, content_plan_revision_id: str) -> ContentPlanRevision | None:
        return self.plans.get(content_plan_revision_id)


class InMemoryShotRepo:
    def __init__(self):
        self.shots: dict[str, Shot] = {}
        self.revisions: dict[str, ShotRevision] = {}

    def add_shot(self, shot: Shot) -> Shot:
        self.shots[shot.shot_id] = shot
        return shot

    def update_shot_local_order(self, shot_id: str, local_order: int) -> Shot | None:
        shot = self.shots.get(shot_id)
        if shot is None:
            return None
        updated = Shot(
            shot_id=shot.shot_id,
            beat_lineage_id=shot.beat_lineage_id,
            local_order=local_order,
            created_at=shot.created_at,
        )
        self.shots[shot_id] = updated
        return updated

    def add_revision(self, revision: ShotRevision) -> ShotRevision:
        self.revisions[revision.shot_revision_id] = revision
        return revision

    def get_shot(self, shot_id: str) -> Shot | None:
        return self.shots.get(shot_id)

    def get_revision(self, shot_revision_id: str) -> ShotRevision | None:
        return self.revisions.get(shot_revision_id)

    def list_revisions(self, shot_id: str) -> tuple[ShotRevision, ...]:
        revs = [r for r in self.revisions.values() if r.shot_id == shot_id]
        revs.sort(key=lambda x: x.revision_number)
        return tuple(revs)


class InMemoryStoryboardRepo:
    def __init__(self, shot_repo: InMemoryShotRepo):
        self.snapshots: dict[str, StoryboardSnapshot] = {}
        self.shot_repo = shot_repo

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
            self.shot_repo.revisions[rev_id]
            for rev_id in snap.shot_revision_ids
            if rev_id in self.shot_repo.revisions
        )


class RepositoriesWrapper:
    def __init__(self):
        self.plan_repo = InMemoryPlanRepo()
        self.shot_repo = InMemoryShotRepo()
        self.storyboard_repo = InMemoryStoryboardRepo(self.shot_repo)

    # Proxy helpers for test backward-compatibility
    @property
    def plans(self):
        return self.plan_repo.plans

    @property
    def shots(self):
        return self.shot_repo.shots

    @property
    def revisions(self):
        return self.shot_repo.revisions

    @property
    def snapshots(self):
        return self.storyboard_repo.snapshots

    def add_plan(self, p):
        return self.plan_repo.add_plan(p)

    def add_shot(self, s):
        return self.shot_repo.add_shot(s)

    def add_revision(self, r):
        return self.shot_repo.add_revision(r)

    def add_snapshot(self, sn):
        return self.storyboard_repo.add_snapshot(sn)

    def get_shot(self, sid):
        return self.shot_repo.get_shot(sid)

    def get_revision(self, rid):
        return self.shot_repo.get_revision(rid)

    def get_snapshot(self, sid):
        return self.storyboard_repo.get_snapshot(sid)

    def list_revisions(self, sid):
        return self.shot_repo.list_revisions(sid)


def _setup_sample_storyboard(snapshot_state=StoryboardSnapshotState.DRAFT):
    repos = RepositoriesWrapper()

    beat1 = ContentBeat(
        beat_id="beat-1",
        beat_lineage_id="lineage-1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook the viewer",
        target_duration=10.0,
        importance=0.9,
    )
    beat2 = ContentBeat(
        beat_id="beat-2",
        beat_lineage_id="lineage-2",
        beat_type=BeatType.SUMMARY,
        order=2,
        intent="Summarize key points",
        target_duration=10.0,
        importance=0.8,
    )
    plan = ContentPlanRevision(
        content_plan_revision_id="plan-1",
        revision_number=1,
        topic="Astronomy",
        overall_target_duration=20.0,
        beats=(beat1, beat2),
    )
    repos.add_plan(plan)

    shot1 = Shot(shot_id="shot-1", beat_lineage_id="lineage-1", local_order=1)
    shot2 = Shot(shot_id="shot-2", beat_lineage_id="lineage-1", local_order=2)
    shot3 = Shot(shot_id="shot-3", beat_lineage_id="lineage-2", local_order=1)
    for s in (shot1, shot2, shot3):
        repos.add_shot(s)

    rev1 = ShotRevision(
        shot_revision_id="rev-1",
        shot_id="shot-1",
        revision_number=1,
        beat_lineage_id="lineage-1",
        created_from_beat_instance_id="beat-1",
        narration="Original narration 1",
        target_duration=5.0,
        visual_goal="Goal 1",
        visual_type=VisualType.STOCK_VIDEO,
        scene_description="Scene 1",
        generation_prompt="Prompt 1",
        camera_movement="Static",
        evidence_refs=("ev1",),
    )
    rev2 = ShotRevision(
        shot_revision_id="rev-2",
        shot_id="shot-2",
        revision_number=1,
        beat_lineage_id="lineage-1",
        created_from_beat_instance_id="beat-1",
        narration="Original narration 2",
        target_duration=5.0,
        visual_goal="Goal 2",
        visual_type=VisualType.AI_VIDEO,
        scene_description="Scene 2",
        generation_prompt="Prompt 2",
        camera_movement="Pan right",
        evidence_refs=(),
    )
    rev3 = ShotRevision(
        shot_revision_id="rev-3",
        shot_id="shot-3",
        revision_number=1,
        beat_lineage_id="lineage-2",
        created_from_beat_instance_id="beat-2",
        narration="Original narration 3",
        target_duration=10.0,
        visual_goal="Goal 3",
        visual_type=VisualType.DIAGRAM,
        scene_description="Scene 3",
        generation_prompt="Prompt 3",
        camera_movement="Zoom in",
        evidence_refs=(),
    )
    for r in (rev1, rev2, rev3):
        repos.add_revision(r)

    snapshot = StoryboardSnapshot(
        storyboard_snapshot_id="snap-1",
        content_plan_revision_id="plan-1",
        shot_revision_ids=("rev-1", "rev-2", "rev-3"),
        snapshot_state=snapshot_state,
    )
    repos.add_snapshot(snapshot)

    return repos, plan, snapshot, (shot1, shot2, shot3), (rev1, rev2, rev3)


# 1. Editing Shot narration creates a NEW ShotRevision
def test_editing_shot_narration_creates_new_shot_revision():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    edit_input = EditShotInput(
        storyboard_snapshot_id="snap-1",
        shot_id="shot-1",
        base_shot_revision_id="rev-1",
        narration="Updated narration text",
    )

    new_snap = service.edit_shot(edit_input)
    assert new_snap.storyboard_snapshot_id != "snap-1"

    revisions = repos.list_revisions("shot-1")
    assert len(revisions) == 2
    new_rev = revisions[1]
    assert new_rev.shot_revision_id != "rev-1"
    assert new_rev.revision_number == 2
    assert new_rev.narration == "Updated narration text"


# 2. Old ShotRevision remains unchanged
def test_old_shot_revision_remains_unchanged():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    service.edit_shot(
        EditShotInput(
            storyboard_snapshot_id="snap-1",
            shot_id="shot-1",
            base_shot_revision_id="rev-1",
            narration="Updated narration text",
        )
    )

    old_rev = repos.get_revision("rev-1")
    assert old_rev.narration == "Original narration 1"
    assert old_rev.revision_number == 1


# 3. New ShotRevision preserves same Shot identity
def test_new_shot_revision_preserves_same_shot_identity():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    new_snap = service.edit_shot(
        EditShotInput(
            storyboard_snapshot_id="snap-1",
            shot_id="shot-1",
            base_shot_revision_id="rev-1",
            visual_goal="Updated Goal",
        )
    )

    new_rev_id = new_snap.shot_revision_ids[0]
    new_rev = repos.get_revision(new_rev_id)

    assert new_rev.shot_id == "shot-1"
    assert new_rev.beat_lineage_id == "lineage-1"
    assert new_rev.created_from_beat_instance_id == "beat-1"
    assert new_rev.evidence_refs == ("ev1",)  # Evidence preserved read-only


# 4. Editing one Shot creates a NEW DRAFT StoryboardSnapshot
def test_editing_one_shot_creates_new_draft_storyboard_snapshot():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    new_snap = service.edit_shot(
        EditShotInput(
            storyboard_snapshot_id="snap-1",
            shot_id="shot-2",
            base_shot_revision_id="rev-2",
            scene_description="New scene description",
        )
    )

    assert new_snap.storyboard_snapshot_id != "snap-1"
    assert new_snap.snapshot_state == StoryboardSnapshotState.DRAFT
    assert new_snap.content_plan_revision_id == "plan-1"


# 5. Old StoryboardSnapshot remains unchanged
def test_old_storyboard_snapshot_remains_unchanged():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    service.edit_shot(
        EditShotInput(
            storyboard_snapshot_id="snap-1",
            shot_id="shot-2",
            base_shot_revision_id="rev-2",
            scene_description="New scene description",
        )
    )

    original_snap = repos.get_snapshot("snap-1")
    assert original_snap.shot_revision_ids == ("rev-1", "rev-2", "rev-3")
    assert original_snap.snapshot_state == StoryboardSnapshotState.DRAFT


# 6. New Snapshot replaces only the edited ShotRevision reference
def test_new_snapshot_replaces_only_the_edited_shot_revision_reference():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    new_snap = service.edit_shot(
        EditShotInput(
            storyboard_snapshot_id="snap-1",
            shot_id="shot-2",
            base_shot_revision_id="rev-2",
            target_duration=6.0,
        )
    )

    new_rev_2_id = new_snap.shot_revision_ids[1]
    assert new_rev_2_id != "rev-2"
    # Exact positional replacement: rev-1 and rev-3 are untouched
    assert new_snap.shot_revision_ids == ("rev-1", new_rev_2_id, "rev-3")


# 7. Client cannot change shot_id
def test_client_cannot_change_shot_id():
    # Attempting to pass extra shot_id modification fields
    with pytest.raises(ValidationError):
        EditShotInput(
            storyboard_snapshot_id="snap-1",
            shot_id="shot-1",
            base_shot_revision_id="rev-1",
            injected_shot_id="malicious-shot-id",  # extra forbidden
        )


# 8. Client cannot change beat_lineage_id
def test_client_cannot_change_beat_lineage_id():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    with pytest.raises(ValidationError):
        EditShotInput(
            storyboard_snapshot_id="snap-1",
            shot_id="shot-1",
            base_shot_revision_id="rev-1",
            beat_lineage_id="new-lineage-hack",  # extra forbidden
        )


# 9. Invalid VisualType/edit input is rejected
def test_invalid_visual_type_or_edit_input_rejected():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    # Invalid VisualType string
    with pytest.raises(ValidationError):
        EditShotInput(
            storyboard_snapshot_id="snap-1",
            shot_id="shot-1",
            base_shot_revision_id="rev-1",
            visual_type="3D_VR_HOLOGRAM",  # Invalid enum
        )

    # Non-positive duration
    with pytest.raises(InvalidShotEditError):
        service.edit_shot(
            EditShotInput(
                storyboard_snapshot_id="snap-1",
                shot_id="shot-1",
                base_shot_revision_id="rev-1",
                target_duration=-2.5,
            )
        )


# 10. Empty narration is rejected
@pytest.mark.parametrize("bad_narration", ["", "   ", "\n\t"])
def test_empty_narration_is_rejected(bad_narration):
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    with pytest.raises(InvalidShotEditError) as exc_info:
        service.edit_shot(
            EditShotInput(
                storyboard_snapshot_id="snap-1",
                shot_id="shot-1",
                base_shot_revision_id="rev-1",
                narration=bad_narration,
            )
        )
    assert "Narration cannot be empty" in str(exc_info.value)


# 11. Reordering within one Beat works
def test_reordering_within_one_beat_works():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    # In lineage-1, original order was shot-1 then shot-2. Reorder to shot-2 then shot-1.
    reorder_input = ReorderShotsInput(
        storyboard_snapshot_id="snap-1",
        beat_lineage_id="lineage-1",
        ordered_shot_ids=("shot-2", "shot-1"),
    )

    new_snap = service.reorder_shots_within_beat(reorder_input)
    assert new_snap.storyboard_snapshot_id != "snap-1"

    # Revisions in snapshot must now place shot-2 first, then shot-1, then shot-3
    assert new_snap.shot_revision_ids == ("rev-2", "rev-1", "rev-3")

    # Verify shot local orders were updated in repository
    assert repos.get_shot("shot-2").local_order == 1
    assert repos.get_shot("shot-1").local_order == 2


# 12. Moving Shot across Beats is rejected
def test_moving_shot_across_beats_is_rejected():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    # Attempt to include shot-3 (which belongs to lineage-2) in lineage-1 reordering
    reorder_cross = ReorderShotsInput(
        storyboard_snapshot_id="snap-1",
        beat_lineage_id="lineage-1",
        ordered_shot_ids=("shot-1", "shot-3"),
    )

    with pytest.raises(CrossBeatMoveNotAllowedError) as exc_info:
        service.reorder_shots_within_beat(reorder_cross)
    assert "Moving shots across different beats" in str(exc_info.value)


# 13. Duplicate local_order within a Beat is rejected
def test_duplicate_local_order_within_a_beat_is_rejected():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    duplicate_reorder = ReorderShotsInput(
        storyboard_snapshot_id="snap-1",
        beat_lineage_id="lineage-1",
        ordered_shot_ids=("shot-1", "shot-1"),  # Duplicate!
    )

    with pytest.raises(InvalidShotOrderError) as exc_info:
        service.reorder_shots_within_beat(duplicate_reorder)
    assert "Duplicate shot_ids" in str(exc_info.value)


# 14. Stale base_shot_revision_id is rejected
def test_stale_base_shot_revision_id_is_rejected():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    # Client submits base_shot_revision_id that does not match current snapshot revision
    stale_input = EditShotInput(
        storyboard_snapshot_id="snap-1",
        shot_id="shot-1",
        base_shot_revision_id="ghost-outdated-rev-id",
        narration="Narration edit",
    )

    with pytest.raises(StaleShotRevisionError) as exc_info:
        service.edit_shot(stale_input)
    assert "Stale edit" in str(exc_info.value)


# 15. Editing an APPROVED snapshot never mutates that historical snapshot
def test_editing_approved_snapshot_never_mutates_that_historical_snapshot():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard(
        snapshot_state=StoryboardSnapshotState.APPROVED
    )
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    new_snap = service.edit_shot(
        EditShotInput(
            storyboard_snapshot_id="snap-1",
            shot_id="shot-1",
            base_shot_revision_id="rev-1",
            narration="Post-approval tweak",
        )
    )

    # Original approved snapshot is untouched
    orig_snap = repos.get_snapshot("snap-1")
    assert orig_snap.snapshot_state == StoryboardSnapshotState.APPROVED
    assert orig_snap.shot_revision_ids == ("rev-1", "rev-2", "rev-3")

    # Newly produced snapshot is DRAFT
    assert new_snap.storyboard_snapshot_id != "snap-1"
    assert new_snap.snapshot_state == StoryboardSnapshotState.DRAFT


# 16. StoryboardEditingView reconstructs exact ShotRevision references from requested Snapshot
def test_storyboard_editing_view_reconstructs_exact_shot_revisions():
    repos, _plan, _snapshot, _shots, _revs = _setup_sample_storyboard()
    service = StoryboardEditingService(
        plan_repository=repos.plan_repo,
        shot_repository=repos.shot_repo,
        storyboard_repository=repos.storyboard_repo,
    )

    # Snapshot 1 view
    view1 = service.get_storyboard_for_editing("snap-1")
    assert isinstance(view1, StoryboardEditingView)
    assert view1.storyboard_snapshot_id == "snap-1"
    assert len(view1.beats) == 2
    assert view1.beats[0].shots[0].shot_revision_id == "rev-1"
    assert view1.beats[0].shots[0].revision_number == 1

    # Apply edit creating Snapshot 2
    snap2 = service.edit_shot(
        EditShotInput(
            storyboard_snapshot_id="snap-1",
            shot_id="shot-1",
            base_shot_revision_id="rev-1",
            narration="Narration V2",
        )
    )

    # Query Snapshot 2 view
    view2 = service.get_storyboard_for_editing(snap2.storyboard_snapshot_id)
    assert view2.storyboard_snapshot_id == snap2.storyboard_snapshot_id
    assert view2.beats[0].shots[0].shot_revision_id != "rev-1"
    assert view2.beats[0].shots[0].revision_number == 2
    assert view2.beats[0].shots[0].narration == "Narration V2"

    # Query Snapshot 1 again -> must still return exact revision 1, not 2!
    view1_repeat = service.get_storyboard_for_editing("snap-1")
    assert view1_repeat.beats[0].shots[0].shot_revision_id == "rev-1"
    assert view1_repeat.beats[0].shots[0].revision_number == 1
    assert view1_repeat.beats[0].shots[0].narration == "Original narration 1"
