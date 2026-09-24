
import pytest

from app.domain.content_plan import ContentBeat
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.evaluation import EvaluationDecision, EvaluationSnapshot
from app.domain.shot import ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.services.evaluation.controlled_visual_replan import (
    ControlledVisualReplanRequest,
    ControlledVisualReplanService,
    EvidenceModificationForbiddenError,
    FactualNarrationModificationForbiddenError,
    StructuralModificationForbiddenError,
)


def _make_beat() -> ContentBeat:
    return ContentBeat(
        beat_id="beat-10",
        beat_lineage_id="lineage-10",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Introduce quantum computing concept",
        target_duration=5.0,
        importance=0.9,
        evidence_refs=("ev-1", "ev-2"),
    )


def _make_shot_revision() -> ShotRevision:
    return ShotRevision(
        shot_revision_id="srev-1",
        shot_id="shot-abc",
        revision_number=1,
        beat_lineage_id="lineage-10",
        created_from_beat_instance_id="beat-10",
        narration="Quantum computers leverage superposition to calculate exponentially faster.",
        target_duration=5.0,
        visual_goal="Show a glowing qubit in superposition state",
        visual_type=VisualType.AI_VIDEO,
        scene_description="A 3D atom sphere oscillating between states 0 and 1 with blue particles",
        generation_prompt="3d glowing quantum sphere oscillating blue particles, hyperrealistic",
        camera_movement="slow pan right",
        evidence_refs=("ev-1",),
    )


def _make_snapshot() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        evaluation_snapshot_id="snap-fail-1",
        evaluation_target_id="target-1",
        dimension_result_ids=("d1", "d2", "d3", "d4"),
        decision=EvaluationDecision.FAIL,
        summary_reason_codes=("UNSUITABLE_COMPOSITION",),
    )


class TestControlledVisualReplan:

    def test_replan_creates_new_revision_for_same_shot_id(self):
        """Invariant 19: Creates NEW ShotRevision for SAME shot_id with incremented revision number."""
        beat = _make_beat()
        orig = _make_shot_revision()
        snap = _make_snapshot()

        req = ControlledVisualReplanRequest(
            shot_revision=orig,
            beat=beat,
            evaluation_snapshot=snap,
            visual_goal="A stabilized top-down view of quantum superposition",
        )

        result = ControlledVisualReplanService.create_candidate_revision(req)

        assert result.candidate_shot_revision.shot_id == orig.shot_id
        assert result.candidate_shot_revision.shot_revision_id != orig.shot_revision_id
        assert result.candidate_shot_revision.revision_number == orig.revision_number + 1
        assert result.candidate_shot_revision.visual_goal == "A stabilized top-down view of quantum superposition"

    def test_replan_preserves_narration_exactly_and_forbids_changes(self):
        """Invariant 20: Narration is locked and cannot be changed by replan."""
        beat = _make_beat()
        orig = _make_shot_revision()
        snap = _make_snapshot()

        # Valid request preserves narration verbatim
        req = ControlledVisualReplanRequest(
            shot_revision=orig,
            beat=beat,
            evaluation_snapshot=snap,
        )
        result = ControlledVisualReplanService.create_candidate_revision(req)
        assert result.candidate_shot_revision.narration == orig.narration

        # Forbids altered narration
        bad_req = ControlledVisualReplanRequest(
            shot_revision=orig,
            beat=beat,
            evaluation_snapshot=snap,
            proposed_narration="Altered factual claim narration here.",
        )
        with pytest.raises(FactualNarrationModificationForbiddenError):
            ControlledVisualReplanService.create_candidate_revision(bad_req)

    def test_replan_preserves_evidence_refs_exactly_and_forbids_changes(self):
        """Invariant 21: Evidence references are locked and cannot be altered."""
        beat = _make_beat()
        orig = _make_shot_revision()
        snap = _make_snapshot()

        bad_req = ControlledVisualReplanRequest(
            shot_revision=orig,
            beat=beat,
            evaluation_snapshot=snap,
            proposed_evidence_refs=("ev-1", "ev-invented"),
        )
        with pytest.raises(EvidenceModificationForbiddenError):
            ControlledVisualReplanService.create_candidate_revision(bad_req)

    def test_replan_preserves_target_duration_and_forbids_changes(self):
        """Invariant 22: Target duration is locked and cannot be altered."""
        beat = _make_beat()
        orig = _make_shot_revision()
        snap = _make_snapshot()

        bad_req = ControlledVisualReplanRequest(
            shot_revision=orig,
            beat=beat,
            evaluation_snapshot=snap,
            proposed_target_duration=orig.target_duration + 3.0,
        )
        with pytest.raises(StructuralModificationForbiddenError):
            ControlledVisualReplanService.create_candidate_revision(bad_req)

    def test_replan_forbids_beat_lineage_mismatch(self):
        """Invariant 25: Cannot change Beat identity or lineage."""
        orig = _make_shot_revision()
        snap = _make_snapshot()

        wrong_beat = ContentBeat(
            beat_id="beat-999",
            beat_lineage_id="different-lineage",
            beat_type=BeatType.KNOWLEDGE,
            order=2,
            intent="Different Goal",
            target_duration=5.0,
            importance=0.5,
        )
        bad_req = ControlledVisualReplanRequest(
            shot_revision=orig,
            beat=wrong_beat,
            evaluation_snapshot=snap,
        )
        with pytest.raises(StructuralModificationForbiddenError):
            ControlledVisualReplanService.create_candidate_revision(bad_req)

    def test_replan_may_change_visual_type(self):
        """Invariant 26: Controlled replan may change VisualType (e.g. AI_VIDEO to AI_IMAGE)."""
        beat = _make_beat()
        orig = _make_shot_revision()
        snap = _make_snapshot()

        req = ControlledVisualReplanRequest(
            shot_revision=orig,
            beat=beat,
            evaluation_snapshot=snap,
            visual_type=VisualType.AI_IMAGE,
            generation_prompt="A high-res still diagram of a Bloch sphere representing qubit superposition",
        )
        result = ControlledVisualReplanService.create_candidate_revision(req)
        assert result.candidate_shot_revision.visual_type == VisualType.AI_IMAGE
        assert orig.visual_type == VisualType.AI_VIDEO

    def test_approved_storyboard_snapshot_remains_immutable(self):
        """Invariant 27 & 28: APPROVED snapshot remains unchanged; replanned revision in candidate DRAFT."""
        beat = _make_beat()
        orig = _make_shot_revision()
        snap = _make_snapshot()

        approved_snapshot = StoryboardSnapshot(
            storyboard_snapshot_id="snap-approved-10",
            content_plan_revision_id="cprev-1",
            shot_revision_ids=(orig.shot_revision_id, "other-rev-9"),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )

        req = ControlledVisualReplanRequest(
            shot_revision=orig,
            beat=beat,
            evaluation_snapshot=snap,
            camera_movement="static centered wide shot",
        )

        result = ControlledVisualReplanService.create_candidate_revision(
            req, existing_approved_snapshot=approved_snapshot
        )

        # Original APPROVED snapshot is 100% untouched
        assert approved_snapshot.shot_revision_ids == (orig.shot_revision_id, "other-rev-9")
        assert approved_snapshot.snapshot_state == StoryboardSnapshotState.APPROVED

        # New snapshot is DRAFT and contains the new candidate revision
        draft_snapshot = result.candidate_draft_snapshot
        assert draft_snapshot.snapshot_state == StoryboardSnapshotState.DRAFT
        assert draft_snapshot.storyboard_snapshot_id != approved_snapshot.storyboard_snapshot_id
        assert result.candidate_shot_revision.shot_revision_id in draft_snapshot.shot_revision_ids
        assert orig.shot_revision_id not in draft_snapshot.shot_revision_ids
        assert "other-rev-9" in draft_snapshot.shot_revision_ids
