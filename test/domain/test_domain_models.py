import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot, derive_deterministic_shot_order


class TestDomainModelFoundation(unittest.TestCase):
    def test_beat_type_enum_contains_required_values(self):
        """Invariant 1: BeatType enum contains exactly the V1 supported beat types."""
        expected_values = {"HOOK", "KNOWLEDGE", "EXAMPLE", "TRANSITION", "SUMMARY"}
        actual_values = {bt.value for bt in BeatType}
        self.assertEqual(actual_values, expected_values)

    def test_visual_type_enum_contains_required_values(self):
        """Invariant 2: VisualType enum contains exactly the 6 required visual types."""
        expected_values = {
            "STOCK_VIDEO",
            "AI_VIDEO",
            "AI_IMAGE",
            "DIAGRAM",
            "SOURCE_ASSET",
            "USER_ASSET",
        }
        actual_values = {vt.value for vt in VisualType}
        self.assertEqual(actual_values, expected_values)

    def test_shot_revision_preserves_provenance_and_data(self):
        """Invariant 3: ShotRevision preserves shot identity, beat lineage, and exact creative data."""
        revision = ShotRevision(
            shot_revision_id="rev-001",
            shot_id="shot-100",
            revision_number=1,
            beat_lineage_id="beat-lineage-200",
            created_from_beat_instance_id="beat-instance-300",
            narration="Here is a concrete example of backpropagation.",
            target_duration=5.0,
            visual_goal="Illustrate loss gradients descending",
            visual_type=VisualType.DIAGRAM,
            scene_description="2D contour plot of gradient descent",
            generation_prompt="gradient descent contour plot, clean vector style",
            camera_movement="slow tilt down",
            evidence_refs=("doc-ref-alpha", "doc-ref-beta"),
        )
        self.assertEqual(revision.shot_id, "shot-100")
        self.assertEqual(revision.beat_lineage_id, "beat-lineage-200")
        self.assertEqual(revision.created_from_beat_instance_id, "beat-instance-300")
        self.assertEqual(revision.revision_number, 1)
        self.assertEqual(revision.visual_type, VisualType.DIAGRAM)
        self.assertEqual(revision.evidence_refs, ("doc-ref-alpha", "doc-ref-beta"))
        self.assertEqual(revision.target_duration, 5.0)

    def test_storyboard_snapshot_stores_exact_shot_revision_references(self):
        """Invariant 4: StoryboardSnapshot freezes exact ShotRevision references, distinguishing DRAFT and APPROVED."""
        snapshot = StoryboardSnapshot(
            storyboard_snapshot_id="snap-001",
            content_plan_revision_id="plan-rev-001",
            shot_revision_ids=("rev-001", "rev-002"),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        self.assertEqual(snapshot.storyboard_snapshot_id, "snap-001")
        self.assertEqual(snapshot.content_plan_revision_id, "plan-rev-001")
        self.assertEqual(snapshot.shot_revision_ids, ("rev-001", "rev-002"))
        self.assertEqual(snapshot.snapshot_state, StoryboardSnapshotState.APPROVED)

    def test_deterministic_ordering_derived_from_beat_order_and_shot_local_order(self):
        """Invariant 5: Deterministic order is derived from Beat.order + Shot.local_order without duplicate global order in snapshot."""
        beat_first = ContentBeat(
            beat_id="b-1",
            beat_lineage_id="lineage-first",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Grab attention",
            target_duration=3.0,
            importance=0.9,
        )
        beat_second = ContentBeat(
            beat_id="b-2",
            beat_lineage_id="lineage-second",
            beat_type=BeatType.KNOWLEDGE,
            order=2,
            intent="Explain core theorem",
            target_duration=10.0,
            importance=1.0,
        )

        shot_b1 = Shot(shot_id="s-b1", beat_lineage_id="lineage-second", local_order=1)
        shot_b2 = Shot(shot_id="s-b2", beat_lineage_id="lineage-second", local_order=2)
        shot_a1 = Shot(shot_id="s-a1", beat_lineage_id="lineage-first", local_order=1)

        # Input is deliberately shuffled
        shuffled_shots = [shot_b2, shot_a1, shot_b1]
        ordered_shots = derive_deterministic_shot_order(
            beats=[beat_second, beat_first],
            shots=shuffled_shots,
        )

        expected_ids = ("s-a1", "s-b1", "s-b2")
        self.assertEqual(tuple(s.shot_id for s in ordered_shots), expected_ids)

    def test_immutable_domain_objects_reject_silent_mutation(self):
        """Invariant 6: Immutable domain objects cannot be silently mutated."""
        beat = ContentBeat(
            beat_id="beat-001",
            beat_lineage_id="lineage-001",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Initial Hook",
            target_duration=3.0,
            importance=0.8,
        )
        with self.assertRaises(ValidationError):
            beat.intent = "mutated intent"

        plan = ContentPlanRevision(
            content_plan_revision_id="plan-001",
            revision_number=1,
            topic="Calculus",
            overall_target_duration=60.0,
            beats=(beat,),
        )
        with self.assertRaises(ValidationError):
            plan.topic = "Linear Algebra"

        snapshot = StoryboardSnapshot(
            storyboard_snapshot_id="snap-001",
            content_plan_revision_id="plan-001",
            shot_revision_ids=("rev-001",),
            snapshot_state=StoryboardSnapshotState.DRAFT,
        )
        with self.assertRaises(ValidationError):
            snapshot.snapshot_state = StoryboardSnapshotState.APPROVED


if __name__ == "__main__":
    unittest.main()
