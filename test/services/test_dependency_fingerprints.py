from __future__ import annotations

from app.services.evaluation.dependency_fingerprint import (
    compute_composition_suitability_fingerprint,
    compute_knowledge_accuracy_fingerprint,
    compute_semantic_alignment_fingerprint,
    compute_visual_quality_fingerprint,
)


def test_semantic_alignment_fingerprint_invalidation():
    fp_base = compute_semantic_alignment_fingerprint(
        asset_file_hash="hash_abc",
        narration="Hello world",
        visual_goal="Show sunny sky",
        scene_description="Bright clouds",
        generation_prompt="sunny sky clouds",
        visual_type="IMAGE",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )
    # Same inputs yield identical fingerprint
    fp_same = compute_semantic_alignment_fingerprint(
        asset_file_hash="hash_abc",
        narration="Hello world",
        visual_goal="Show sunny sky",
        scene_description="Bright clouds",
        generation_prompt="sunny sky clouds",
        visual_type="IMAGE",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )
    assert fp_base == fp_same

    # Narration change invalidates
    fp_narr_change = compute_semantic_alignment_fingerprint(
        asset_file_hash="hash_abc",
        narration="Hello changed world",
        visual_goal="Show sunny sky",
        scene_description="Bright clouds",
        generation_prompt="sunny sky clouds",
        visual_type="IMAGE",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )
    assert fp_base != fp_narr_change

    # Prompt change invalidates
    fp_prompt_change = compute_semantic_alignment_fingerprint(
        asset_file_hash="hash_abc",
        narration="Hello world",
        visual_goal="Show sunny sky",
        scene_description="Bright clouds",
        generation_prompt="cinematic blue sky",
        visual_type="IMAGE",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )
    assert fp_base != fp_prompt_change


def test_visual_quality_fingerprint_narration_independence():
    fp_1 = compute_visual_quality_fingerprint(
        asset_file_hash="asset_hash_999",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )
    # Notice: there is NO narration or prompt parameter!
    fp_2 = compute_visual_quality_fingerprint(
        asset_file_hash="asset_hash_999",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )
    assert fp_1 == fp_2

    # Different asset hash invalidates
    fp_diff = compute_visual_quality_fingerprint(
        asset_file_hash="asset_hash_888",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )
    assert fp_1 != fp_diff


def test_knowledge_accuracy_fingerprint_sorting_and_invalidation():
    # Order of evidence_refs does not affect fingerprint because it is canonically sorted
    fp_a = compute_knowledge_accuracy_fingerprint(
        asset_file_hash="asset_hash_1",
        narration="Photosynthesis produces glucose",
        evidence_refs=["ref_2", "ref_1"],
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )
    fp_b = compute_knowledge_accuracy_fingerprint(
        asset_file_hash="asset_hash_1",
        narration="Photosynthesis produces glucose",
        evidence_refs=["ref_1", "ref_2"],
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )
    assert fp_a == fp_b

    # Evidence change invalidates
    fp_c = compute_knowledge_accuracy_fingerprint(
        asset_file_hash="asset_hash_1",
        narration="Photosynthesis produces glucose",
        evidence_refs=["ref_1", "ref_3"],
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )
    assert fp_a != fp_c


def test_composition_suitability_fingerprint():
    fp_comp_1 = compute_composition_suitability_fingerprint(
        preview_file_hash="prev_hash_1",
        render_context_fingerprint="ctx_fp_1",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )
    fp_comp_2 = compute_composition_suitability_fingerprint(
        preview_file_hash="prev_hash_1",
        render_context_fingerprint="ctx_fp_1",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )
    assert fp_comp_1 == fp_comp_2

    # Different preview file hash invalidates
    fp_comp_3 = compute_composition_suitability_fingerprint(
        preview_file_hash="prev_hash_2",
        render_context_fingerprint="ctx_fp_1",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )
    assert fp_comp_1 != fp_comp_3
