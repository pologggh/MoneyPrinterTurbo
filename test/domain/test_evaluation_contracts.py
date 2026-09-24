from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from app.domain.asset_execution import AssetMediaType, ShotAssetVersion
from app.domain.asset_router import GenerationMode
from app.domain.enums import VisualType
from app.domain.evaluation import (
    DimensionEvaluationResult,
    DimensionEvaluationStatus,
    DuplicateEvaluationDimensionError,
    EvaluationDecision,
    EvaluationDimension,
    EvaluationDimensionMissingError,
    EvaluationTarget,
    EvaluationTargetMismatchError,
    InvalidEvaluationDecisionError,
    InvalidEvaluationScoreError,
    create_evaluation_snapshot,
    create_evaluation_target_from_shot,
)
from app.domain.shot import ShotRevision

# ==============================================================================
# TEST FIXTURES & HELPERS
# ==============================================================================


def _build_dummy_shot_revision(
    shot_id: str = "shot-101",
    shot_revision_id: str = "rev-101-v1",
    evidence_refs: tuple[str, ...] = ("ev-doc-1", "ev-doc-2"),
) -> ShotRevision:
    return ShotRevision(
        shot_revision_id=shot_revision_id,
        shot_id=shot_id,
        revision_number=1,
        beat_lineage_id="lineage-1",
        created_from_beat_instance_id="beat-inst-1",
        narration="Self-Attention allows tokens to model relationships.",
        target_duration=4.5,
        visual_goal="Show attention connections",
        visual_type=VisualType.DIAGRAM,
        scene_description="Animated graph of tokens and attention lines",
        generation_prompt="Graph network animation",
        camera_movement="Static",
        evidence_refs=evidence_refs,
    )


def _build_dummy_asset_version(
    shot_id: str = "shot-101",
    shot_revision_id: str = "rev-101-v1",
    version_id: str = "asset-ver-101",
) -> ShotAssetVersion:
    return ShotAssetVersion(
        shot_asset_version_id=version_id,
        shot_id=shot_id,
        shot_revision_id=shot_revision_id,
        execution_attempt_id="att-1",
        file_path="storage/assets/test.png",
        file_hash="a" * 64,
        file_size_bytes=1024,
        media_type=AssetMediaType.IMAGE,
        mime_type="image/png",
        width=1920,
        height=1080,
        duration_seconds=0.0,
        provider="mock_prov",
        model="mock_model",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
    )


def _build_dummy_dimension_results(
    target_id: str,
    override_statuses: dict[EvaluationDimension, tuple[DimensionEvaluationStatus, float | None]] | None = None,
) -> list[DimensionEvaluationResult]:
    statuses = override_statuses or {}
    results = []
    for dim in EvaluationDimension:
        status, score = statuses.get(dim, (DimensionEvaluationStatus.SCORED, 0.85))
        res = DimensionEvaluationResult(
            evaluation_target_id=target_id,
            dimension=dim,
            status=status,
            score=score,
            reason_codes=("REASON_OK",),
            concise_summary=f"Evaluation summary for {dim.value}",
            evaluator_version="test-eval-v1",
            dimension_semantics_version="test-sem-v1",
        )
        results.append(res)
    return results


# ==============================================================================
# BUSINESS INVARIANT TESTS (1 to 24)
# ==============================================================================


def test_01_target_freezes_exact_shot_revision_id():
    """Invariant 1: EvaluationTarget freezes exact shot_revision_id."""
    rev = _build_dummy_shot_revision(shot_revision_id="rev-exact-123")
    asset = _build_dummy_asset_version(shot_revision_id="rev-exact-123")

    target = create_evaluation_target_from_shot(rev, asset)

    assert target.shot_revision_id == "rev-exact-123"
    assert target.shot_id == rev.shot_id


def test_02_target_freezes_exact_shot_asset_version_id():
    """Invariant 2: EvaluationTarget freezes exact shot_asset_version_id."""
    rev = _build_dummy_shot_revision()
    asset = _build_dummy_asset_version(version_id="asset-exact-999")

    target = create_evaluation_target_from_shot(rev, asset)

    assert target.shot_asset_version_id == "asset-exact-999"


def test_03_target_does_not_resolve_latest_shot_revision_or_asset():
    """Invariant 3: EvaluationTarget stores explicit IDs, never 'latest' references."""
    target = EvaluationTarget(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        shot_asset_version_id="asset-1",
        evidence_refs_snapshot=("ev-1",),
    )

    # Immutable frozen model
    with pytest.raises((ValidationError, TypeError)):
        target.shot_revision_id = "rev-2"  # type: ignore[misc]

    assert target.shot_revision_id == "rev-1"
    assert target.shot_asset_version_id == "asset-1"


def test_04_evidence_references_frozen_into_target():
    """Invariant 4: Evidence references are frozen into the target using current evidence representation."""
    evidence = ("ev-doc-alpha", "ev-doc-beta")
    rev = _build_dummy_shot_revision(evidence_refs=evidence)
    asset = _build_dummy_asset_version()

    target = create_evaluation_target_from_shot(rev, asset)

    assert target.evidence_refs_snapshot == evidence
    assert isinstance(target.evidence_refs_snapshot, tuple)


def test_05_scored_result_requires_numeric_score():
    """Invariant 5: SCORED DimensionEvaluationResult requires a numeric score (cannot be None)."""
    with pytest.raises((InvalidEvaluationScoreError, ValidationError), match="requires a numeric score"):
        DimensionEvaluationResult(
            evaluation_target_id="target-1",
            dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
            status=DimensionEvaluationStatus.SCORED,
            score=None,
        )


@pytest.mark.parametrize("invalid_score", [-0.01, 1.01, -1.0, 2.5])
def test_06_scored_score_must_be_within_zero_to_one_range(invalid_score):
    """Invariant 6: SCORED score must be normalized within [0.0, 1.0]."""
    with pytest.raises((InvalidEvaluationScoreError, ValidationError), match=r"within \[0.0, 1.0\]"):
        DimensionEvaluationResult(
            evaluation_target_id="target-1",
            dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
            status=DimensionEvaluationStatus.SCORED,
            score=invalid_score,
        )


def test_07_indeterminate_result_must_not_contain_numeric_score():
    """Invariant 7: INDETERMINATE result must not contain a numeric score (must be None)."""
    with pytest.raises((InvalidEvaluationScoreError, ValidationError), match="must not contain a numeric score"):
        DimensionEvaluationResult(
            evaluation_target_id="target-1",
            dimension=EvaluationDimension.KNOWLEDGE_ACCURACY,
            status=DimensionEvaluationStatus.INDETERMINATE,
            score=0.5,
        )


def test_08_error_result_must_not_contain_numeric_score():
    """Invariant 8: ERROR result must not contain a numeric score (must be None)."""
    with pytest.raises((InvalidEvaluationScoreError, ValidationError), match="must not contain a numeric score"):
        DimensionEvaluationResult(
            evaluation_target_id="target-1",
            dimension=EvaluationDimension.VISUAL_QUALITY,
            status=DimensionEvaluationStatus.ERROR,
            score=0.0,
        )


def test_09_result_preserves_evaluator_version():
    """Invariant 9: DimensionEvaluationResult preserves evaluator_version."""
    res = DimensionEvaluationResult(
        evaluation_target_id="target-1",
        dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
        status=DimensionEvaluationStatus.SCORED,
        score=0.9,
        evaluator_version="gemini-vision-v2-preview",
    )

    assert res.evaluator_version == "gemini-vision-v2-preview"


def test_10_result_preserves_dimension_semantics_version():
    """Invariant 10: DimensionEvaluationResult preserves dimension_semantics_version."""
    res = DimensionEvaluationResult(
        evaluation_target_id="target-1",
        dimension=EvaluationDimension.KNOWLEDGE_ACCURACY,
        status=DimensionEvaluationStatus.SCORED,
        score=0.95,
        dimension_semantics_version="knowledge-rubric-v2.1",
    )

    assert res.dimension_semantics_version == "knowledge-rubric-v2.1"


def test_11_result_is_append_only_and_immutable():
    """Invariant 11: DimensionEvaluationResult cannot be mutated in place."""
    res = DimensionEvaluationResult(
        evaluation_target_id="target-1",
        dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
        status=DimensionEvaluationStatus.SCORED,
        score=0.8,
    )

    with pytest.raises((ValidationError, TypeError)):
        res.score = 0.9  # type: ignore[misc]

    assert res.score == 0.8


def test_12_snapshot_requires_exactly_four_dimensions():
    """Invariant 12: A complete EvaluationSnapshot requires exactly four dimensions."""
    target = EvaluationTarget(
        shot_id="s1", shot_revision_id="r1", shot_asset_version_id="a1"
    )
    dim_results = _build_dummy_dimension_results(target.evaluation_target_id)

    snapshot = create_evaluation_snapshot(
        target=target,
        dimension_results=dim_results,
        decision=EvaluationDecision.PASS,
    )

    assert len(snapshot.dimension_result_ids) == 4
    assert snapshot.evaluation_target_id == target.evaluation_target_id


def test_13_snapshot_rejects_duplicate_dimension():
    """Invariant 13: Duplicate dimension result in one Snapshot is rejected."""
    target = EvaluationTarget(
        shot_id="s1", shot_revision_id="r1", shot_asset_version_id="a1"
    )
    dim_results = _build_dummy_dimension_results(target.evaluation_target_id)

    # Replace composition suitability with a second semantic alignment
    duplicate_res = DimensionEvaluationResult(
        evaluation_target_id=target.evaluation_target_id,
        dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
        status=DimensionEvaluationStatus.SCORED,
        score=0.7,
    )
    dim_results[3] = duplicate_res

    with pytest.raises(DuplicateEvaluationDimensionError, match="Duplicate dimension result"):
        create_evaluation_snapshot(
            target=target,
            dimension_results=dim_results,
            decision=EvaluationDecision.PASS,
        )


def test_14_snapshot_rejects_missing_dimension():
    """Invariant 14: Missing any required dimension is rejected."""
    target = EvaluationTarget(
        shot_id="s1", shot_revision_id="r1", shot_asset_version_id="a1"
    )
    dim_results = _build_dummy_dimension_results(target.evaluation_target_id)
    # Provide only 3 dimensions
    partial_results = dim_results[:3]

    with pytest.raises(EvaluationDimensionMissingError, match="Missing: COMPOSITION_SUITABILITY"):
        create_evaluation_snapshot(
            target=target,
            dimension_results=partial_results,
            decision=EvaluationDecision.FAIL,
        )


def test_15_snapshot_rejects_results_belonging_to_different_targets():
    """Invariant 15: Results belonging to different EvaluationTargets cannot be mixed."""
    target = EvaluationTarget(
        shot_id="s1", shot_revision_id="r1", shot_asset_version_id="a1"
    )
    dim_results = _build_dummy_dimension_results(target.evaluation_target_id)
    # Tamper with one result to point to another target
    dim_results[0] = DimensionEvaluationResult(
        evaluation_target_id="other-target-uuid",
        dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
        status=DimensionEvaluationStatus.SCORED,
        score=0.9,
    )

    with pytest.raises(EvaluationTargetMismatchError, match="belongs to target 'other-target-uuid'"):
        create_evaluation_snapshot(
            target=target,
            dimension_results=dim_results,
            decision=EvaluationDecision.PASS,
        )


def test_16_snapshot_freezes_exact_dimension_result_ids():
    """Invariant 16: EvaluationSnapshot freezes exact DimensionEvaluationResult IDs."""
    target = EvaluationTarget(
        shot_id="s1", shot_revision_id="r1", shot_asset_version_id="a1"
    )
    dim_results = _build_dummy_dimension_results(target.evaluation_target_id)
    expected_ids = tuple(r.dimension_result_id for r in dim_results)

    snapshot = create_evaluation_snapshot(
        target=target,
        dimension_results=dim_results,
        decision=EvaluationDecision.PASS,
    )

    # Deterministic dimension order: SEMANTIC_ALIGNMENT, VISUAL_QUALITY, KNOWLEDGE_ACCURACY, COMPOSITION_SUITABILITY
    assert set(snapshot.dimension_result_ids) == set(expected_ids)
    for r in dim_results:
        assert r.dimension_result_id in snapshot.dimension_result_ids


def test_17_creating_newer_result_does_not_alter_older_snapshot():
    """Invariant 17: Creating a newer DimensionEvaluationResult does NOT alter an older Snapshot."""
    target = EvaluationTarget(
        shot_id="s1", shot_revision_id="r1", shot_asset_version_id="a1"
    )
    dim_results_1 = _build_dummy_dimension_results(target.evaluation_target_id)
    snapshot_1 = create_evaluation_snapshot(
        target=target,
        dimension_results=dim_results_1,
        decision=EvaluationDecision.PASS,
    )

    # Later: re-evaluate semantic alignment
    new_semantic_res = DimensionEvaluationResult(
        evaluation_target_id=target.evaluation_target_id,
        dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
        status=DimensionEvaluationStatus.SCORED,
        score=0.99,
    )

    # Snapshot 1 remains unchanged
    assert new_semantic_res.dimension_result_id not in snapshot_1.dimension_result_ids
    assert dim_results_1[0].dimension_result_id in snapshot_1.dimension_result_ids


def test_18_pass_rejected_when_semantic_alignment_is_indeterminate():
    """Invariant 18: PASS is rejected when SEMANTIC_ALIGNMENT is INDETERMINATE."""
    target = EvaluationTarget(
        shot_id="s1", shot_revision_id="r1", shot_asset_version_id="a1"
    )
    dim_results = _build_dummy_dimension_results(
        target.evaluation_target_id,
        override_statuses={EvaluationDimension.SEMANTIC_ALIGNMENT: (DimensionEvaluationStatus.INDETERMINATE, None)},
    )

    with pytest.raises(InvalidEvaluationDecisionError, match="SEMANTIC_ALIGNMENT has status INDETERMINATE"):
        create_evaluation_snapshot(
            target=target,
            dimension_results=dim_results,
            decision=EvaluationDecision.PASS,
        )


def test_19_pass_rejected_when_semantic_alignment_is_error():
    """Invariant 19: PASS is rejected when SEMANTIC_ALIGNMENT is ERROR."""
    target = EvaluationTarget(
        shot_id="s1", shot_revision_id="r1", shot_asset_version_id="a1"
    )
    dim_results = _build_dummy_dimension_results(
        target.evaluation_target_id,
        override_statuses={EvaluationDimension.SEMANTIC_ALIGNMENT: (DimensionEvaluationStatus.ERROR, None)},
    )

    with pytest.raises(InvalidEvaluationDecisionError, match="SEMANTIC_ALIGNMENT has status ERROR"):
        create_evaluation_snapshot(
            target=target,
            dimension_results=dim_results,
            decision=EvaluationDecision.PASS,
        )


def test_20_pass_rejected_when_knowledge_accuracy_is_indeterminate():
    """Invariant 20: PASS is rejected when KNOWLEDGE_ACCURACY is INDETERMINATE."""
    target = EvaluationTarget(
        shot_id="s1", shot_revision_id="r1", shot_asset_version_id="a1"
    )
    dim_results = _build_dummy_dimension_results(
        target.evaluation_target_id,
        override_statuses={EvaluationDimension.KNOWLEDGE_ACCURACY: (DimensionEvaluationStatus.INDETERMINATE, None)},
    )

    with pytest.raises(InvalidEvaluationDecisionError, match="KNOWLEDGE_ACCURACY has status INDETERMINATE"):
        create_evaluation_snapshot(
            target=target,
            dimension_results=dim_results,
            decision=EvaluationDecision.PASS,
        )


def test_21_pass_rejected_when_knowledge_accuracy_is_error():
    """Invariant 21: PASS is rejected when KNOWLEDGE_ACCURACY is ERROR."""
    target = EvaluationTarget(
        shot_id="s1", shot_revision_id="r1", shot_asset_version_id="a1"
    )
    dim_results = _build_dummy_dimension_results(
        target.evaluation_target_id,
        override_statuses={EvaluationDimension.KNOWLEDGE_ACCURACY: (DimensionEvaluationStatus.ERROR, None)},
    )

    with pytest.raises(InvalidEvaluationDecisionError, match="KNOWLEDGE_ACCURACY has status ERROR"):
        create_evaluation_snapshot(
            target=target,
            dimension_results=dim_results,
            decision=EvaluationDecision.PASS,
        )


def test_22_snapshot_preserves_policy_version():
    """Invariant 22: EvaluationSnapshot preserves policy_version."""
    target = EvaluationTarget(
        shot_id="s1", shot_revision_id="r1", shot_asset_version_id="a1"
    )
    dim_results = _build_dummy_dimension_results(target.evaluation_target_id)

    snapshot = create_evaluation_snapshot(
        target=target,
        dimension_results=dim_results,
        decision=EvaluationDecision.PASS,
        policy_version="policy-strict-v2.0",
    )

    assert snapshot.policy_version == "policy-strict-v2.0"


def test_23_re_evaluation_creates_new_snapshot_rather_than_mutating():
    """Invariant 23: Re-evaluation creates a NEW EvaluationSnapshot rather than mutating the old one."""
    target = EvaluationTarget(
        shot_id="s1", shot_revision_id="r1", shot_asset_version_id="a1"
    )
    dim_results_1 = _build_dummy_dimension_results(target.evaluation_target_id)
    snapshot_1 = create_evaluation_snapshot(
        target=target,
        dimension_results=dim_results_1,
        decision=EvaluationDecision.FAIL,
    )

    # New evaluation on the same target with improved results
    dim_results_2 = _build_dummy_dimension_results(target.evaluation_target_id)
    snapshot_2 = create_evaluation_snapshot(
        target=target,
        dimension_results=dim_results_2,
        decision=EvaluationDecision.PASS,
    )

    assert snapshot_1.evaluation_snapshot_id != snapshot_2.evaluation_snapshot_id
    assert snapshot_1.decision == EvaluationDecision.FAIL
    assert snapshot_2.decision == EvaluationDecision.PASS


def test_24_no_vlm_or_llm_provider_invoked_in_phase_6_1():
    """Invariant 24: No VLM or LLM network provider modules are imported or called."""
    # Verify no multimodal or provider client modules are imported for evaluation
    forbidden_modules = [
        "google.genai",
        "openai",
        "anthropic",
        "qwen_vl",
    ]
    for mod in forbidden_modules:
        assert mod not in sys.modules, f"Forbidden provider module {mod} was imported in Phase 6.1"


# ==============================================================================
# FUTURE COMPATIBILITY & PARTIAL REUSE TESTS
# ==============================================================================


def test_25_future_compatibility_exact_target_identity_persists_across_revision_updates():
    """
    Shows why exact target identity matters:
    Shot A has Revision A2 and Asset V1 forming EvaluationTarget T1.
    Later Revision A3 is created.
    Verify: T1 still references A2 + V1, NOT A3.
    """
    rev_a2 = _build_dummy_shot_revision(shot_id="shot-A", shot_revision_id="rev-A2")
    asset_v1 = _build_dummy_asset_version(shot_id="shot-A", shot_revision_id="rev-A2", version_id="asset-V1")

    target_t1 = create_evaluation_target_from_shot(rev_a2, asset_v1)

    # Later: Shot A is edited to Revision A3
    rev_a3 = _build_dummy_shot_revision(shot_id="shot-A", shot_revision_id="rev-A3")

    # Target T1 must remain strictly frozen to A2 + V1
    assert target_t1.shot_id == "shot-A"
    assert target_t1.shot_revision_id == "rev-A2"
    assert target_t1.shot_asset_version_id == "asset-V1"
    assert target_t1.shot_revision_id != rev_a3.shot_revision_id


def test_26_partial_result_history_snapshot_preserves_exact_references_when_dimension_recalculated():
    """
    Shows architectural support for future partial dimension reuse:
    Target T1 has S1, V1, K1, C1 referenced by Snapshot E1.
    Later S2 is generated for T1.
    Verify: Snapshot E1 still references S1 (exact historical ID), while S2 is available for future snapshots.
    """
    target_t1 = EvaluationTarget(
        shot_id="shot-B", shot_revision_id="rev-B1", shot_asset_version_id="asset-B1"
    )
    dim_results_e1 = _build_dummy_dimension_results(target_t1.evaluation_target_id)
    s1 = next(r for r in dim_results_e1 if r.dimension == EvaluationDimension.SEMANTIC_ALIGNMENT)
    v1 = next(r for r in dim_results_e1 if r.dimension == EvaluationDimension.VISUAL_QUALITY)
    k1 = next(r for r in dim_results_e1 if r.dimension == EvaluationDimension.KNOWLEDGE_ACCURACY)
    c1 = next(r for r in dim_results_e1 if r.dimension == EvaluationDimension.COMPOSITION_SUITABILITY)

    snapshot_e1 = create_evaluation_snapshot(
        target=target_t1,
        dimension_results=(s1, v1, k1, c1),
        decision=EvaluationDecision.PASS,
    )

    # Later S2 is created
    s2 = DimensionEvaluationResult(
        evaluation_target_id=target_t1.evaluation_target_id,
        dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
        status=DimensionEvaluationStatus.SCORED,
        score=0.98,
        reason_codes=("RE_EVALUATED",),
    )

    # Snapshot E1 still points to S1
    assert s1.dimension_result_id in snapshot_e1.dimension_result_ids
    assert s2.dimension_result_id not in snapshot_e1.dimension_result_ids

    # Future snapshot E2 can combine reused V1, K1, C1 with new S2!
    snapshot_e2 = create_evaluation_snapshot(
        target=target_t1,
        dimension_results=(s2, v1, k1, c1),
        decision=EvaluationDecision.PASS,
    )
    assert s2.dimension_result_id in snapshot_e2.dimension_result_ids
    assert v1.dimension_result_id in snapshot_e2.dimension_result_ids
    assert snapshot_e1.evaluation_snapshot_id != snapshot_e2.evaluation_snapshot_id
