import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType
from app.domain.fingerprint import compute_beat_content_fingerprint
from app.domain.plan_diff import (
    BeatChangeType,
    DuplicateBeatLineageError,
    ShotReuseDecision,
    decide_shot_reuse,
    diff_content_plans,
)


class TestFingerprintAndPlanDiff(unittest.TestCase):
    def test_fingerprint_deterministic_across_equivalent_construction(self):
        """Case 1: Fingerprint is deterministic across equivalent construction with varying whitespace/evidence order."""
        beat_a = ContentBeat(
            beat_id="beat-inst-1",
            beat_lineage_id="lineage-1",
            beat_type=BeatType.KNOWLEDGE,
            order=1,
            intent="  Explain gradient   descent algorithm.  ",
            target_duration=5.0,
            importance=0.8,
            evidence_refs=("ref-b", "ref-a"),
        )
        beat_b = ContentBeat(
            beat_id="beat-inst-2",
            beat_lineage_id="lineage-2",
            beat_type=BeatType.KNOWLEDGE,
            order=2,
            intent="Explain gradient descent algorithm.",
            target_duration=5.0,
            importance=0.8,
            evidence_refs=("ref-a", "ref-b"),
        )

        fp_a = compute_beat_content_fingerprint(beat_a)
        fp_b = compute_beat_content_fingerprint(beat_b)
        self.assertEqual(fp_a, fp_b)

    def test_fingerprint_ignores_instance_id_lineage_id_order(self):
        """Case 2: Fingerprint does NOT depend on beat instance ID, lineage ID, or order."""
        base_beat = ContentBeat(
            beat_id="inst-alpha",
            beat_lineage_id="lin-alpha",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Attention grabber",
            target_duration=3.0,
            importance=0.9,
            evidence_refs=("ref-1",),
        )
        moved_beat = ContentBeat(
            beat_id="inst-beta",
            beat_lineage_id="lin-beta",
            beat_type=BeatType.HOOK,
            order=5,  # Order changed
            intent="Attention grabber",
            target_duration=3.0,
            importance=0.9,
            evidence_refs=("ref-1",),
        )

        self.assertEqual(
            compute_beat_content_fingerprint(base_beat),
            compute_beat_content_fingerprint(moved_beat),
        )

    def test_fingerprint_changes_on_semantic_content_difference(self):
        """Case 3: Fingerprint DOES change when semantically relevant Beat content changes."""
        base_beat = ContentBeat(
            beat_id="inst-1",
            beat_lineage_id="lin-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Attention grabber",
            target_duration=3.0,
            importance=0.9,
            evidence_refs=("ref-1",),
        )

        modified_intent = base_beat.model_copy(update={"intent": "Different grabber"})
        modified_duration = base_beat.model_copy(update={"target_duration": 4.5})
        modified_type = base_beat.model_copy(update={"beat_type": BeatType.SUMMARY})
        modified_evidence = base_beat.model_copy(update={"evidence_refs": ("ref-1", "ref-2")})

        base_fp = compute_beat_content_fingerprint(base_beat)
        self.assertNotEqual(base_fp, compute_beat_content_fingerprint(modified_intent))
        self.assertNotEqual(base_fp, compute_beat_content_fingerprint(modified_duration))
        self.assertNotEqual(base_fp, compute_beat_content_fingerprint(modified_type))
        self.assertNotEqual(base_fp, compute_beat_content_fingerprint(modified_evidence))

    def test_plan_diff_unchanged_moved_modified(self):
        """Cases 4, 5, 6: Plan diff correctly classifies UNCHANGED, MOVED, and MODIFIED beats."""
        beat_unchanged_old = ContentBeat(
            beat_id="b-u1",
            beat_lineage_id="lin-u",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Hook",
            target_duration=3.0,
            importance=0.8,
        )
        beat_moved_old = ContentBeat(
            beat_id="b-m1",
            beat_lineage_id="lin-m",
            beat_type=BeatType.KNOWLEDGE,
            order=2,
            intent="Core concept",
            target_duration=8.0,
            importance=1.0,
        )
        beat_modified_old = ContentBeat(
            beat_id="b-c1",
            beat_lineage_id="lin-c",
            beat_type=BeatType.EXAMPLE,
            order=3,
            intent="Original example",
            target_duration=5.0,
            importance=0.7,
        )

        old_plan = ContentPlanRevision(
            content_plan_revision_id="plan-1",
            revision_number=1,
            topic="ML",
            overall_target_duration=16.0,
            beats=(beat_unchanged_old, beat_moved_old, beat_modified_old),
        )

        # In new plan:
        # lin-u is UNCHANGED (order 1, same content)
        # lin-m is MOVED (order 3, same content)
        # lin-c is MODIFIED (order 2, intent changed)
        beat_unchanged_new = beat_unchanged_old.model_copy(update={"beat_id": "b-u2"})
        beat_moved_new = beat_moved_old.model_copy(update={"beat_id": "b-m2", "order": 3})
        beat_modified_new = beat_modified_old.model_copy(
            update={"beat_id": "b-c2", "order": 2, "intent": "Rewritten new example"}
        )

        new_plan = ContentPlanRevision(
            content_plan_revision_id="plan-2",
            revision_number=2,
            topic="ML",
            overall_target_duration=16.0,
            beats=(beat_unchanged_new, beat_modified_new, beat_moved_new),
        )

        diff = diff_content_plans(old_plan, new_plan)
        changes_by_lineage = {c.beat_lineage_id: c for c in diff.beat_changes}

        self.assertEqual(changes_by_lineage["lin-u"].change_type, BeatChangeType.UNCHANGED)
        self.assertEqual(changes_by_lineage["lin-m"].change_type, BeatChangeType.MOVED)
        self.assertEqual(changes_by_lineage["lin-c"].change_type, BeatChangeType.MODIFIED)

    def test_plan_diff_added_and_removed(self):
        """Case 7: Lineages existing only in new or old revision are classified as ADDED or REMOVED."""
        beat_old = ContentBeat(
            beat_id="b-old",
            beat_lineage_id="lin-retiring",
            beat_type=BeatType.TRANSITION,
            order=1,
            intent="Bridge",
            target_duration=2.0,
            importance=0.5,
        )
        beat_new = ContentBeat(
            beat_id="b-new",
            beat_lineage_id="lin-incoming",
            beat_type=BeatType.SUMMARY,
            order=1,
            intent="Recap",
            target_duration=4.0,
            importance=0.9,
        )

        old_plan = ContentPlanRevision(
            content_plan_revision_id="p-old",
            revision_number=1,
            topic="Topic",
            overall_target_duration=2.0,
            beats=(beat_old,),
        )
        new_plan = ContentPlanRevision(
            content_plan_revision_id="p-new",
            revision_number=2,
            topic="Topic",
            overall_target_duration=4.0,
            beats=(beat_new,),
        )

        diff = diff_content_plans(old_plan, new_plan)
        changes_by_lineage = {c.beat_lineage_id: c for c in diff.beat_changes}

        self.assertEqual(changes_by_lineage["lin-incoming"].change_type, BeatChangeType.ADDED)
        self.assertEqual(changes_by_lineage["lin-retiring"].change_type, BeatChangeType.REMOVED)

    def test_shot_reuse_and_invalidation_decisions(self):
        """Cases 8, 9: Shot invalidation correctly maps UNCHANGED/MOVED to REUSE and MODIFIED to GENERATE_NEW_SHOTS."""
        self.assertEqual(
            decide_shot_reuse(BeatChangeType.UNCHANGED),
            ShotReuseDecision.REUSE_EXISTING_SHOTS,
        )
        self.assertEqual(
            decide_shot_reuse(BeatChangeType.MOVED),
            ShotReuseDecision.REUSE_EXISTING_SHOTS,
        )
        self.assertEqual(
            decide_shot_reuse(BeatChangeType.MODIFIED),
            ShotReuseDecision.GENERATE_NEW_SHOTS,
        )
        self.assertEqual(
            decide_shot_reuse(BeatChangeType.ADDED),
            ShotReuseDecision.NO_PREVIOUS_SHOTS,
        )
        self.assertEqual(
            decide_shot_reuse(BeatChangeType.REMOVED),
            ShotReuseDecision.NO_PREVIOUS_SHOTS,
        )

    def test_duplicate_beat_lineage_rejected(self):
        """Case 10: Duplicate beat_lineage_id inside one revision is rejected with DuplicateBeatLineageError."""
        shared_lineage = "duplicate-lineage-100"
        beat_1 = ContentBeat(
            beat_id="b-1",
            beat_lineage_id=shared_lineage,
            beat_type=BeatType.HOOK,
            order=1,
            intent="First",
            target_duration=2.0,
            importance=0.5,
        )
        beat_2 = ContentBeat(
            beat_id="b-2",
            beat_lineage_id=shared_lineage,  # Duplicate lineage ID in same revision
            beat_type=BeatType.EXAMPLE,
            order=2,
            intent="Second",
            target_duration=3.0,
            importance=0.6,
        )

        invalid_plan = ContentPlanRevision(
            content_plan_revision_id="plan-dup",
            revision_number=1,
            topic="Invalid",
            overall_target_duration=5.0,
            beats=(beat_1, beat_2),
        )

        with self.assertRaises(DuplicateBeatLineageError):
            diff_content_plans(None, invalid_plan)


if __name__ == "__main__":
    unittest.main()
