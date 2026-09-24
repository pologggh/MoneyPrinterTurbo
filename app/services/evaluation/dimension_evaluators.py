from __future__ import annotations

from collections.abc import Sequence

from app.domain.evaluation import (
    DimensionEvaluationResult,
    DimensionEvaluationStatus,
    EvaluationDimension,
    EvaluationReasonCode,
)
from app.services.evaluation.composition_preview import (
    CompositionPreview,
    RenderContext,
)
from app.services.evaluation.dependency_fingerprint import (
    COMPOSITION_SUITABILITY_SEMANTICS_VERSION,
    KNOWLEDGE_ACCURACY_SEMANTICS_VERSION,
    SEMANTIC_ALIGNMENT_SEMANTICS_VERSION,
    VISUAL_QUALITY_SEMANTICS_VERSION,
    compute_composition_suitability_fingerprint,
    compute_knowledge_accuracy_fingerprint,
    compute_semantic_alignment_fingerprint,
    compute_visual_quality_fingerprint,
)
from app.services.evaluation.media_preparation import PreparedMediaObservation
from app.services.evaluation.mpt_multimodal_adapter import MultimodalEvaluatorAdapter


class SemanticAlignmentEvaluator:
    """
    Evaluator for SEMANTIC_ALIGNMENT.
    Measures how faithfully the visual media reflects the shot's narration, visual goal, and scene.
    """

    @classmethod
    def evaluate(
        cls,
        evaluation_target_id: str,
        media_observation: PreparedMediaObservation,
        narration: str,
        visual_goal: str,
        scene_description: str,
        generation_prompt: str,
        visual_type: str,
        adapter: MultimodalEvaluatorAdapter,
    ) -> DimensionEvaluationResult:
        fingerprint = compute_semantic_alignment_fingerprint(
            asset_file_hash=media_observation.asset_file_hash,
            narration=narration,
            visual_goal=visual_goal,
            scene_description=scene_description,
            generation_prompt=generation_prompt,
            visual_type=visual_type,
            evaluator_version=adapter.evaluator_version,
            semantics_version=SEMANTIC_ALIGNMENT_SEMANTICS_VERSION,
            sampling_policy_version=media_observation.sampling_policy_version,
        )

        prompt = f"""
Evaluate the SEMANTIC ALIGNMENT of the provided visual asset against the following target context:
- Narration: {narration}
- Visual Goal: {visual_goal}
- Scene Description: {scene_description}
- Generation Prompt: {generation_prompt}
- Visual Type: {visual_type}

Assess whether the media depicts the key subjects, actions, and concepts described in the narration and visual goal.
Return a score between 0.0 and 1.0 reflecting alignment strength.
""".strip()

        obs_res = adapter.evaluate_observation(
            prompt=prompt,
            image_paths=media_observation.frame_paths,
        )

        return DimensionEvaluationResult(
            evaluation_target_id=evaluation_target_id,
            dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
            status=obs_res.status,
            score=obs_res.score,
            reason_codes=obs_res.reason_codes,
            concise_summary=obs_res.concise_summary,
            evaluator_version=adapter.evaluator_version,
            dimension_semantics_version=SEMANTIC_ALIGNMENT_SEMANTICS_VERSION,
            dependency_fingerprint=fingerprint,
            error_code=obs_res.error_code,
        )


class VisualQualityEvaluator:
    """
    Evaluator for VISUAL_QUALITY.
    Measures perceptual quality, resolution, lighting, and absence of AI distortions or noise.
    Invariant: Independent of script/narration.
    """

    @classmethod
    def evaluate(
        cls,
        evaluation_target_id: str,
        media_observation: PreparedMediaObservation,
        adapter: MultimodalEvaluatorAdapter,
    ) -> DimensionEvaluationResult:
        fingerprint = compute_visual_quality_fingerprint(
            asset_file_hash=media_observation.asset_file_hash,
            evaluator_version=adapter.evaluator_version,
            semantics_version=VISUAL_QUALITY_SEMANTICS_VERSION,
            sampling_policy_version=media_observation.sampling_policy_version,
        )

        prompt = """
Evaluate the VISUAL QUALITY of the provided visual asset purely on aesthetic and technical craftsmanship:
- Clarity, sharpness, and resolution
- Lighting, color grading, and contrast
- Absence of unnatural AI distortions, weird artifacts, or heavy blur
Do NOT judge semantic relevance to any external script.
Return a score between 0.0 and 1.0 reflecting perceptual craftsmanship.
""".strip()

        obs_res = adapter.evaluate_observation(
            prompt=prompt,
            image_paths=media_observation.frame_paths,
        )

        return DimensionEvaluationResult(
            evaluation_target_id=evaluation_target_id,
            dimension=EvaluationDimension.VISUAL_QUALITY,
            status=obs_res.status,
            score=obs_res.score,
            reason_codes=obs_res.reason_codes,
            concise_summary=obs_res.concise_summary,
            evaluator_version=adapter.evaluator_version,
            dimension_semantics_version=VISUAL_QUALITY_SEMANTICS_VERSION,
            dependency_fingerprint=fingerprint,
            error_code=obs_res.error_code,
        )


class KnowledgeAccuracyEvaluator:
    """
    Evaluator for KNOWLEDGE_ACCURACY.
    Evidence-first factual verification. If frozen evidence references are missing,
    cleanly returns INDETERMINATE with MISSING_EVIDENCE_CONTEXT without calling models or browsing.
    """

    @classmethod
    def evaluate(
        cls,
        evaluation_target_id: str,
        media_observation: PreparedMediaObservation,
        narration: str,
        evidence_refs: Sequence[str],
        adapter: MultimodalEvaluatorAdapter,
    ) -> DimensionEvaluationResult:
        clean_refs = tuple(r.strip() for r in evidence_refs if r and r.strip())
        fingerprint = compute_knowledge_accuracy_fingerprint(
            asset_file_hash=media_observation.asset_file_hash,
            narration=narration,
            evidence_refs=clean_refs,
            evaluator_version=adapter.evaluator_version,
            semantics_version=KNOWLEDGE_ACCURACY_SEMANTICS_VERSION,
            sampling_policy_version=media_observation.sampling_policy_version,
        )

        # Evidence-first rule: missing evidence context yields INDETERMINATE
        if not clean_refs:
            return DimensionEvaluationResult(
                evaluation_target_id=evaluation_target_id,
                dimension=EvaluationDimension.KNOWLEDGE_ACCURACY,
                status=DimensionEvaluationStatus.INDETERMINATE,
                score=None,
                reason_codes=(EvaluationReasonCode.MISSING_EVIDENCE_CONTEXT.value,),
                concise_summary="Missing evidence context for factual verification; knowledge accuracy cannot be verified without reference evidence.",
                evaluator_version=adapter.evaluator_version,
                dimension_semantics_version=KNOWLEDGE_ACCURACY_SEMANTICS_VERSION,
                dependency_fingerprint=fingerprint,
                error_code=None,
            )

        refs_formatted = "\n".join(f"- {ref}" for ref in sorted(clean_refs))
        prompt = f"""
Evaluate the KNOWLEDGE ACCURACY of the visual asset and narration strictly against the following frozen evidence items:
Evidence Context:
{refs_formatted}

Narration:
{narration}

Verify whether the depicted visual concepts and narration are factually consistent with the evidence context.
Do NOT use external unverified knowledge.
Return a score between 0.0 and 1.0 reflecting factual accuracy against the evidence.
""".strip()

        obs_res = adapter.evaluate_observation(
            prompt=prompt,
            image_paths=media_observation.frame_paths,
        )

        return DimensionEvaluationResult(
            evaluation_target_id=evaluation_target_id,
            dimension=EvaluationDimension.KNOWLEDGE_ACCURACY,
            status=obs_res.status,
            score=obs_res.score,
            reason_codes=obs_res.reason_codes,
            concise_summary=obs_res.concise_summary,
            evaluator_version=adapter.evaluator_version,
            dimension_semantics_version=KNOWLEDGE_ACCURACY_SEMANTICS_VERSION,
            dependency_fingerprint=fingerprint,
            error_code=obs_res.error_code,
        )


class CompositionSuitabilityEvaluator:
    """
    Evaluator for COMPOSITION_SUITABILITY.
    Evaluates the asset placed inside a derived CompositionPreview canvas (framing, margins, safe zones).
    """

    @classmethod
    def evaluate(
        cls,
        evaluation_target_id: str,
        composition_preview: CompositionPreview,
        render_context: RenderContext,
        adapter: MultimodalEvaluatorAdapter,
    ) -> DimensionEvaluationResult:
        fingerprint = compute_composition_suitability_fingerprint(
            preview_file_hash=composition_preview.file_hash,
            render_context_fingerprint=composition_preview.render_context_fingerprint,
            evaluator_version=adapter.evaluator_version,
            semantics_version=COMPOSITION_SUITABILITY_SEMANTICS_VERSION,
            preview_policy_version=composition_preview.preview_policy_version,
        )

        prompt = f"""
Evaluate the COMPOSITION SUITABILITY of the asset within the framed canvas preview:
- Target Dimensions: {render_context.target_width}x{render_context.target_height} ({render_context.aspect_ratio})
- Fit Mode: {render_context.fit_mode}
- Safe Margin: The preview image has a subtle white guideline showing the {int(render_context.safe_margin_percent * 100)}% text-safe boundary.

Assess:
1. Is the subject properly framed and centered?
2. Are important visual elements clipped awkwardly by crop or canvas borders?
3. Does the visual respect safe margins without critical content shoved against the outer edges?

Return a score between 0.0 and 1.0 reflecting composition and layout suitability.
""".strip()

        obs_res = adapter.evaluate_observation(
            prompt=prompt,
            image_paths=[composition_preview.preview_path],
        )

        return DimensionEvaluationResult(
            evaluation_target_id=evaluation_target_id,
            dimension=EvaluationDimension.COMPOSITION_SUITABILITY,
            status=obs_res.status,
            score=obs_res.score,
            reason_codes=obs_res.reason_codes,
            concise_summary=obs_res.concise_summary,
            evaluator_version=adapter.evaluator_version,
            dimension_semantics_version=COMPOSITION_SUITABILITY_SEMANTICS_VERSION,
            dependency_fingerprint=fingerprint,
            error_code=obs_res.error_code,
        )
