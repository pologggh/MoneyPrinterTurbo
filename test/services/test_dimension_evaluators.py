from __future__ import annotations

from unittest.mock import MagicMock

from app.domain.asset_execution import AssetMediaType
from app.domain.evaluation import (
    DimensionEvaluationStatus,
    EvaluationDimension,
    EvaluationReasonCode,
)
from app.services.evaluation.composition_preview import (
    CompositionPreview,
    RenderContext,
)
from app.services.evaluation.dimension_evaluators import (
    CompositionSuitabilityEvaluator,
    KnowledgeAccuracyEvaluator,
    SemanticAlignmentEvaluator,
    VisualQualityEvaluator,
)
from app.services.evaluation.media_preparation import PreparedMediaObservation
from app.services.evaluation.mpt_multimodal_adapter import (
    MultimodalEvaluationObserverResult,
    MultimodalEvaluatorAdapter,
)


def test_knowledge_accuracy_evaluator_missing_evidence_yields_indeterminate():
    adapter = MagicMock(spec=MultimodalEvaluatorAdapter)
    adapter.evaluator_version = "openai/gpt-4o:evaluator-v1"

    obs = PreparedMediaObservation(
        media_type=AssetMediaType.IMAGE,
        frame_paths=("dummy.png",),
        timestamps=(0.0,),
        asset_file_hash="asset_hash_1",
    )

    # Empty evidence refs
    result = KnowledgeAccuracyEvaluator.evaluate(
        evaluation_target_id="target_1",
        media_observation=obs,
        narration="Photosynthesis is the process by which green plants...",
        evidence_refs=(),
        adapter=adapter,
    )

    assert result.status == DimensionEvaluationStatus.INDETERMINATE
    assert result.score is None
    assert EvaluationReasonCode.MISSING_EVIDENCE_CONTEXT.value in result.reason_codes
    assert result.dimension == EvaluationDimension.KNOWLEDGE_ACCURACY
    # Crucial: Adapter was NOT invoked
    adapter.evaluate_observation.assert_not_called()


def test_knowledge_accuracy_evaluator_with_evidence():
    adapter = MagicMock(spec=MultimodalEvaluatorAdapter)
    adapter.evaluator_version = "openai/gpt-4o:evaluator-v1"
    adapter.evaluate_observation.return_value = MultimodalEvaluationObserverResult(
        status=DimensionEvaluationStatus.SCORED,
        score=0.92,
        reason_codes=("FACTUALLY_CORRECT",),
        concise_summary="Visual formula matches chemical equation.",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )

    obs = PreparedMediaObservation(
        media_type=AssetMediaType.IMAGE,
        frame_paths=("dummy.png",),
        timestamps=(0.0,),
        asset_file_hash="asset_hash_1",
    )

    result = KnowledgeAccuracyEvaluator.evaluate(
        evaluation_target_id="target_1",
        media_observation=obs,
        narration="Photosynthesis formula",
        evidence_refs=["6CO2 + 6H2O -> C6H12O6 + 6O2"],
        adapter=adapter,
    )

    assert result.status == DimensionEvaluationStatus.SCORED
    assert result.score == 0.92
    assert adapter.evaluate_observation.call_count == 1


def test_visual_quality_evaluator():
    adapter = MagicMock(spec=MultimodalEvaluatorAdapter)
    adapter.evaluator_version = "openai/gpt-4o:evaluator-v1"
    adapter.evaluate_observation.return_value = MultimodalEvaluationObserverResult(
        status=DimensionEvaluationStatus.SCORED,
        score=0.85,
        reason_codes=("CRISP_DETAILS",),
        concise_summary="Sharp image with no visible artifacts.",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )

    obs = PreparedMediaObservation(
        media_type=AssetMediaType.IMAGE,
        frame_paths=("dummy.png",),
        timestamps=(0.0,),
        asset_file_hash="asset_hash_1",
    )

    result = VisualQualityEvaluator.evaluate(
        evaluation_target_id="target_1",
        media_observation=obs,
        adapter=adapter,
    )

    assert result.status == DimensionEvaluationStatus.SCORED
    assert result.score == 0.85
    assert result.dimension == EvaluationDimension.VISUAL_QUALITY
    # Verify prompt does NOT contain script text
    prompt_used = adapter.evaluate_observation.call_args[1]["prompt"]
    assert "script" not in prompt_used.lower() or "do not judge" in prompt_used.lower()


def test_composition_suitability_evaluator():
    adapter = MagicMock(spec=MultimodalEvaluatorAdapter)
    adapter.evaluator_version = "openai/gpt-4o:evaluator-v1"
    adapter.evaluate_observation.return_value = MultimodalEvaluationObserverResult(
        status=DimensionEvaluationStatus.SCORED,
        score=0.78,
        reason_codes=("SAFE_MARGIN_RESPECTED",),
        concise_summary="Subject centered in canvas.",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )

    preview = CompositionPreview(
        shot_asset_version_id="asset_1",
        render_context_fingerprint="ctx_fp_1",
        preview_path="preview_canvas.png",
        file_hash="prev_hash_1",
        width=1080,
        height=1920,
    )
    ctx = RenderContext(target_width=1080, target_height=1920, aspect_ratio="9:16", fit_mode="contain")

    result = CompositionSuitabilityEvaluator.evaluate(
        evaluation_target_id="target_1",
        composition_preview=preview,
        render_context=ctx,
        adapter=adapter,
    )

    assert result.status == DimensionEvaluationStatus.SCORED
    assert result.score == 0.78
    assert result.dimension == EvaluationDimension.COMPOSITION_SUITABILITY
    assert adapter.evaluate_observation.call_args[1]["image_paths"] == ["preview_canvas.png"]


def test_semantic_alignment_evaluator():
    adapter = MagicMock(spec=MultimodalEvaluatorAdapter)
    adapter.evaluator_version = "openai/gpt-4o:evaluator-v1"
    adapter.evaluate_observation.return_value = MultimodalEvaluationObserverResult(
        status=DimensionEvaluationStatus.SCORED,
        score=0.91,
        reason_codes=("ALIGNED_SUBJECT",),
        concise_summary="Visual shows green leaves as narrated.",
        evaluator_version="openai/gpt-4o:evaluator-v1",
    )

    obs = PreparedMediaObservation(
        media_type=AssetMediaType.IMAGE,
        frame_paths=("dummy.png",),
        timestamps=(0.0,),
        asset_file_hash="asset_hash_1",
    )

    result = SemanticAlignmentEvaluator.evaluate(
        evaluation_target_id="target_1",
        media_observation=obs,
        narration="Plant leaves absorb sunlight",
        visual_goal="Close-up of green leaf cell",
        scene_description="Macro photography of vibrant leaf",
        generation_prompt="macro shot of plant leaf cells",
        visual_type="IMAGE",
        adapter=adapter,
    )

    assert result.status == DimensionEvaluationStatus.SCORED
    assert result.score == 0.91
    assert result.dimension == EvaluationDimension.SEMANTIC_ALIGNMENT
