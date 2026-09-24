import hashlib
import unittest
from datetime import UTC, datetime

from app.domain.benchmark import (
    BenchmarkCaseRef,
    BenchmarkExecutionMode,
    BenchmarkExpectedConstraints,
    BenchmarkRun,
    BenchmarkRunStatus,
    BenchmarkSuite,
    BenchmarkVariant,
    KnowledgeFixtureRef,
    compute_benchmark_case_fingerprint,
    compute_benchmark_suite_fingerprint,
    evaluate_benchmark_structural_constraints,
)
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState
from app.domain.storyboard import StoryboardSnapshot


class TestBenchmarkContracts(unittest.TestCase):

    def setUp(self):
        self.fixture_ref = KnowledgeFixtureRef(
            fixture_id="fix_transformer",
            fixture_version="v1.0",
            file_path="fixtures/benchmark/knowledge/transformer_attention.md",
            content_sha256=hashlib.sha256(b"Transformer attention mechanism content").hexdigest(),
        )
        self.constraints = BenchmarkExpectedConstraints(
            min_beats=2,
            max_beats=5,
            required_beat_types=(BeatType.HOOK, BeatType.KNOWLEDGE),
            require_evidence_for_knowledge_beats=True,
            target_duration_tolerance=2.0,
            minimum_shots=2,
            maximum_shots=10,
        )

    def test_benchmark_case_fingerprint_is_deterministic(self):
        """Case fingerprint is deterministic and independent of DB IDs / timestamps."""
        fp1 = compute_benchmark_case_fingerprint(
            case_key="transformer-attention",
            case_version="v1",
            topic="Attention Mechanisms",
            target_duration=8.0,
            user_instruction="Focus on self-attention",
            knowledge_fixture_ref=self.fixture_ref,
            expected_constraints=self.constraints,
            tags=("nlp", "ai"),
        )
        fp2 = compute_benchmark_case_fingerprint(
            case_key="transformer-attention",
            case_version="v1",
            topic="Attention Mechanisms",
            target_duration=8.0,
            user_instruction="Focus on self-attention",
            knowledge_fixture_ref=self.fixture_ref,
            expected_constraints=self.constraints,
            tags=("nlp", "ai"),
        )
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)

    def test_changing_case_topic_changes_fingerprint(self):
        """Changing topic modifies fingerprint."""
        fp1 = compute_benchmark_case_fingerprint(
            case_key="transformer-attention",
            case_version="v1",
            topic="Attention Mechanisms",
            target_duration=8.0,
            user_instruction="Focus on self-attention",
            knowledge_fixture_ref=self.fixture_ref,
            expected_constraints=self.constraints,
            tags=("nlp", "ai"),
        )
        fp2 = compute_benchmark_case_fingerprint(
            case_key="transformer-attention",
            case_version="v1",
            topic="Convolutional Networks",
            target_duration=8.0,
            user_instruction="Focus on self-attention",
            knowledge_fixture_ref=self.fixture_ref,
            expected_constraints=self.constraints,
            tags=("nlp", "ai"),
        )
        self.assertNotEqual(fp1, fp2)

    def test_changing_knowledge_fixture_changes_fingerprint(self):
        """Changing fixture SHA-256 modifies case fingerprint."""
        altered_fixture = KnowledgeFixtureRef(
            fixture_id=self.fixture_ref.fixture_id,
            fixture_version=self.fixture_ref.fixture_version,
            file_path=self.fixture_ref.file_path,
            content_sha256=hashlib.sha256(b"Altered fixture content").hexdigest(),
        )
        fp1 = compute_benchmark_case_fingerprint(
            case_key="transformer-attention",
            case_version="v1",
            topic="Attention Mechanisms",
            target_duration=8.0,
            user_instruction=None,
            knowledge_fixture_ref=self.fixture_ref,
            expected_constraints=self.constraints,
            tags=(),
        )
        fp2 = compute_benchmark_case_fingerprint(
            case_key="transformer-attention",
            case_version="v1",
            topic="Attention Mechanisms",
            target_duration=8.0,
            user_instruction=None,
            knowledge_fixture_ref=altered_fixture,
            expected_constraints=self.constraints,
            tags=(),
        )
        self.assertNotEqual(fp1, fp2)

    def test_benchmark_suite_freezes_exact_case_versions(self):
        """Suite fingerprint freezes exact case versions and fingerprints in order."""
        case_ref_1 = BenchmarkCaseRef(case_key="case-1", case_version="v1", content_fingerprint="fp1111")
        case_ref_2 = BenchmarkCaseRef(case_key="case-2", case_version="v1", content_fingerprint="fp2222")

        suite_fp1 = compute_benchmark_suite_fingerprint(
            suite_key="knowledge-video-v1",
            suite_version="v1",
            case_refs=(case_ref_1, case_ref_2),
        )
        suite_fp2 = compute_benchmark_suite_fingerprint(
            suite_key="knowledge-video-v1",
            suite_version="v1",
            case_refs=(case_ref_1, case_ref_2),
        )
        self.assertEqual(suite_fp1, suite_fp2)

        # Swapping case order changes suite fingerprint
        suite_fp_swapped = compute_benchmark_suite_fingerprint(
            suite_key="knowledge-video-v1",
            suite_version="v1",
            case_refs=(case_ref_2, case_ref_1),
        )
        self.assertNotEqual(suite_fp1, suite_fp_swapped)

    def test_old_suite_does_not_resolve_latest_case(self):
        """When case-1 updates to v2, an old suite referencing v1 maintains its frozen reference."""
        case_ref_v1 = BenchmarkCaseRef(case_key="case-1", case_version="v1", content_fingerprint="fp1111")
        case_ref_v2 = BenchmarkCaseRef(case_key="case-1", case_version="v2", content_fingerprint="fp9999")

        suite_v1 = BenchmarkSuite(
            benchmark_suite_id="suite-1",
            suite_key="knowledge-video-v1",
            suite_version="v1",
            benchmark_case_refs=(case_ref_v1,),
            created_at=datetime.now(UTC),
            content_fingerprint=compute_benchmark_suite_fingerprint(
                "knowledge-video-v1", "v1", (case_ref_v1,)
            ),
        )
        self.assertEqual(suite_v1.benchmark_case_refs[0].case_version, "v1")
        self.assertEqual(suite_v1.benchmark_case_refs[0].content_fingerprint, "fp1111")
        self.assertNotEqual(suite_v1.benchmark_case_refs[0].case_version, case_ref_v2.case_version)

    def test_structural_constraints_evaluation_success(self):
        """Valid generated plan and storyboard satisfy structural constraints."""
        beat_hook = ContentBeat(
            beat_id="b1",
            beat_lineage_id="lin_b1",
            order=1,
            beat_type=BeatType.HOOK,
            intent="Engage audience",
            importance=0.8,
            target_duration=4.0,
            evidence_refs=(),
        )
        beat_know = ContentBeat(
            beat_id="b2",
            beat_lineage_id="lin_b2",
            order=2,
            beat_type=BeatType.KNOWLEDGE,
            intent="Explain attention",
            importance=0.9,
            target_duration=4.0,
            evidence_refs=("ev_1",),
        )
        plan_rev = ContentPlanRevision(
            content_plan_revision_id="cprev-1",
            revision_number=1,
            topic="Attention",
            overall_target_duration=8.0,
            beats=(beat_hook, beat_know),
        )
        snapshot = StoryboardSnapshot(
            storyboard_snapshot_id="snap-1",
            content_plan_revision_id="cprev-1",
            snapshot_state=StoryboardSnapshotState.DRAFT,
            shot_revision_ids=("srev-1", "srev-2"),
        )

        valid, errors = evaluate_benchmark_structural_constraints(
            plan_rev=plan_rev,
            storyboard_snapshot=snapshot,
            constraints=self.constraints,
            target_case_duration=8.0,
        )
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_missing_evidence_in_knowledge_beat_fails_validation(self):
        """KNOWLEDGE beat with missing evidence fails structural validation when required."""
        beat_hook = ContentBeat(
            beat_id="b1",
            beat_lineage_id="lin_b1",
            order=1,
            beat_type=BeatType.HOOK,
            intent="Engage",
            importance=0.8,
            target_duration=4.0,
            evidence_refs=(),
        )
        beat_know_no_evidence = ContentBeat(
            beat_id="b2",
            beat_lineage_id="lin_b2",
            order=2,
            beat_type=BeatType.KNOWLEDGE,
            intent="Explain attention without evidence",
            importance=0.9,
            target_duration=4.0,
            evidence_refs=(),  # EMPTY evidence!
        )
        plan_rev = ContentPlanRevision(
            content_plan_revision_id="cprev-1",
            revision_number=1,
            topic="Attention",
            overall_target_duration=8.0,
            beats=(beat_hook, beat_know_no_evidence),
        )
        snapshot = StoryboardSnapshot(
            storyboard_snapshot_id="snap-1",
            content_plan_revision_id="cprev-1",
            snapshot_state=StoryboardSnapshotState.DRAFT,
            shot_revision_ids=("srev-1",),
        )

        valid, errors = evaluate_benchmark_structural_constraints(
            plan_rev=plan_rev,
            storyboard_snapshot=snapshot,
            constraints=self.constraints,
            target_case_duration=8.0,
        )
        self.assertFalse(valid)
        self.assertTrue(any("evidence" in err.lower() for err in errors))

    def test_offline_is_default_mode(self):
        """OFFLINE is the default execution mode."""
        self.assertEqual(BenchmarkExecutionMode.OFFLINE.value, "OFFLINE")
        run = BenchmarkRun(
            benchmark_run_id="brun-1",
            benchmark_suite_id="bsuite-1",
            suite_key="knowledge-video-v1",
            suite_version="v1",
            suite_fingerprint="fp123",
            variant=BenchmarkVariant(variant_key="balanced-auto"),
        )
        self.assertEqual(run.execution_mode, BenchmarkExecutionMode.OFFLINE)
        self.assertEqual(run.status, BenchmarkRunStatus.CREATED)


if __name__ == "__main__":
    unittest.main()
