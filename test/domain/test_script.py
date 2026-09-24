import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import ValidationError

from app.domain.enums import BeatType
from app.domain.script import ScriptRevision, ScriptSegment, compute_script_fingerprint


class TestScriptDomain(unittest.TestCase):
    def test_script_segment_immutability(self):
        """ScriptSegment is frozen and cannot be mutated."""
        seg = ScriptSegment(
            script_revision_id="rev-1",
            content_beat_id="beat-1",
            order=1,
            narration_text="Self-attention allows tokens to attend to each other.",
            target_duration=5.0,
            beat_type=BeatType.KNOWLEDGE,
            evidence_refs=("ev-1",),
        )
        with self.assertRaises(ValidationError):
            seg.narration_text = "Modified text"

        with self.assertRaises(ValidationError):
            seg.order = 2

    def test_script_revision_immutability(self):
        """ScriptRevision is frozen and cannot be mutated."""
        seg = ScriptSegment(
            script_revision_id="rev-1",
            content_beat_id="beat-1",
            order=1,
            narration_text="Intro",
            target_duration=4.0,
            beat_type=BeatType.HOOK,
            evidence_refs=(),
        )
        rev = ScriptRevision.create(
            task_id="task-1",
            content_plan_revision_id="plan-rev-1",
            segments=[seg],
            overall_target_duration=4.0,
            script_revision_id="rev-1",
        )
        with self.assertRaises(ValidationError):
            rev.overall_target_duration = 10.0

        with self.assertRaises(ValidationError):
            rev.language = "en"

    def test_fingerprint_deterministic_and_stable(self):
        """Fingerprint is identical across multiple runs and ignores timestamps/IDs."""
        seg1_a = ScriptSegment(
            script_segment_id=str(uuid4()),
            script_revision_id="rev-1",
            content_beat_id="beat-1",
            order=1,
            narration_text="  Self-attention computes token relationships.  ",
            target_duration=5.0,
            evidence_refs=("ev-2", "ev-1"),
            created_at=datetime.now(UTC),
        )
        seg1_b = ScriptSegment(
            script_segment_id=str(uuid4()),
            script_revision_id="rev-2",
            content_beat_id="beat-1",
            order=1,
            narration_text="Self-attention computes token relationships.",
            target_duration=5.0,
            evidence_refs=("ev-1", "ev-2"),  # inverted evidence order
            created_at=datetime.now(UTC) + timedelta(days=1),
        )

        fp_a = compute_script_fingerprint("plan-1", [seg1_a])
        fp_b = compute_script_fingerprint("plan-1", [seg1_b])
        self.assertEqual(fp_a, fp_b)
        self.assertEqual(len(fp_a), 64)  # SHA-256 hex length

    def test_fingerprint_changes_on_semantic_difference(self):
        """Fingerprint changes if narration text, beat id, duration, or plan revision id changes."""
        seg_base = ScriptSegment(
            script_revision_id="rev-1",
            content_beat_id="beat-1",
            order=1,
            narration_text="Base text",
            target_duration=5.0,
            evidence_refs=("ev-1",),
        )
        fp_base = compute_script_fingerprint("plan-1", [seg_base])

        # 1. Different plan revision
        fp_diff_plan = compute_script_fingerprint("plan-2", [seg_base])
        self.assertNotEqual(fp_base, fp_diff_plan)

        # 2. Different narration text
        seg_diff_text = ScriptSegment(
            script_revision_id="rev-1",
            content_beat_id="beat-1",
            order=1,
            narration_text="Different text",
            target_duration=5.0,
            evidence_refs=("ev-1",),
        )
        self.assertNotEqual(fp_base, compute_script_fingerprint("plan-1", [seg_diff_text]))

        # 3. Different evidence refs
        seg_diff_ev = ScriptSegment(
            script_revision_id="rev-1",
            content_beat_id="beat-1",
            order=1,
            narration_text="Base text",
            target_duration=5.0,
            evidence_refs=("ev-2",),
        )
        self.assertNotEqual(fp_base, compute_script_fingerprint("plan-1", [seg_diff_ev]))

        # 4. Different duration
        seg_diff_dur = ScriptSegment(
            script_revision_id="rev-1",
            content_beat_id="beat-1",
            order=1,
            narration_text="Base text",
            target_duration=10.0,
            evidence_refs=("ev-1",),
        )
        self.assertNotEqual(fp_base, compute_script_fingerprint("plan-1", [seg_diff_dur]))


if __name__ == "__main__":
    unittest.main()
