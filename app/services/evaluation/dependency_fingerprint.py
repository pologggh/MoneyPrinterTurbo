from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

SEMANTIC_ALIGNMENT_SEMANTICS_VERSION: str = "semantic-alignment-v1"
VISUAL_QUALITY_SEMANTICS_VERSION: str = "visual-quality-v1"
KNOWLEDGE_ACCURACY_SEMANTICS_VERSION: str = "knowledge-accuracy-v1"
COMPOSITION_SUITABILITY_SEMANTICS_VERSION: str = "composition-suitability-v1"

DEFAULT_SAMPLING_POLICY_VERSION: str = "media-sampling-v1"
DEFAULT_PREVIEW_POLICY_VERSION: str = "composition-preview-v1"


def _canonical_sha256(data: dict) -> str:
    """Serializes dictionary to canonical sorted JSON and computes SHA-256 hex digest."""
    canon = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def compute_semantic_alignment_fingerprint(
    asset_file_hash: str,
    narration: str,
    visual_goal: str,
    scene_description: str,
    generation_prompt: str,
    visual_type: str,
    evaluator_version: str,
    semantics_version: str = SEMANTIC_ALIGNMENT_SEMANTICS_VERSION,
    sampling_policy_version: str = DEFAULT_SAMPLING_POLICY_VERSION,
) -> str:
    """
    Computes dependency fingerprint for SEMANTIC_ALIGNMENT.
    Invalidated by: changes to asset, narration, visual goal, scene description, prompt, or visual type.
    """
    payload = {
        "asset_file_hash": asset_file_hash,
        "evaluator_version": evaluator_version,
        "generation_prompt": generation_prompt,
        "narration": narration,
        "sampling_policy_version": sampling_policy_version,
        "scene_description": scene_description,
        "semantics_version": semantics_version,
        "visual_goal": visual_goal,
        "visual_type": visual_type,
    }
    return _canonical_sha256(payload)


def compute_visual_quality_fingerprint(
    asset_file_hash: str,
    evaluator_version: str,
    semantics_version: str = VISUAL_QUALITY_SEMANTICS_VERSION,
    sampling_policy_version: str = DEFAULT_SAMPLING_POLICY_VERSION,
) -> str:
    """
    Computes dependency fingerprint for VISUAL_QUALITY.
    Invariant: Independent of script/narration/prompt. Reused when only text content changes.
    """
    payload = {
        "asset_file_hash": asset_file_hash,
        "evaluator_version": evaluator_version,
        "sampling_policy_version": sampling_policy_version,
        "semantics_version": semantics_version,
    }
    return _canonical_sha256(payload)


def compute_knowledge_accuracy_fingerprint(
    asset_file_hash: str,
    narration: str,
    evidence_refs: Sequence[str],
    evaluator_version: str,
    semantics_version: str = KNOWLEDGE_ACCURACY_SEMANTICS_VERSION,
    sampling_policy_version: str = DEFAULT_SAMPLING_POLICY_VERSION,
) -> str:
    """
    Computes dependency fingerprint for KNOWLEDGE_ACCURACY.
    Invalidated by: changes to asset, narration, or evidence references.
    Sorts evidence references deterministically.
    """
    sorted_refs = sorted(str(ref) for ref in evidence_refs)
    payload = {
        "asset_file_hash": asset_file_hash,
        "evaluator_version": evaluator_version,
        "evidence_refs": sorted_refs,
        "narration": narration,
        "sampling_policy_version": sampling_policy_version,
        "semantics_version": semantics_version,
    }
    return _canonical_sha256(payload)


def compute_composition_suitability_fingerprint(
    preview_file_hash: str,
    render_context_fingerprint: str,
    evaluator_version: str,
    semantics_version: str = COMPOSITION_SUITABILITY_SEMANTICS_VERSION,
    preview_policy_version: str = DEFAULT_PREVIEW_POLICY_VERSION,
) -> str:
    """
    Computes dependency fingerprint for COMPOSITION_SUITABILITY.
    Invalidated by: changes to derived preview canvas or render context.
    """
    payload = {
        "evaluator_version": evaluator_version,
        "preview_file_hash": preview_file_hash,
        "preview_policy_version": preview_policy_version,
        "render_context_fingerprint": render_context_fingerprint,
        "semantics_version": semantics_version,
    }
    return _canonical_sha256(payload)
