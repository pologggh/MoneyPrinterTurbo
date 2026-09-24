from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.domain.asset_execution import (
    AssetMediaType,
    AssetReuseDecision,
    AssetReuseDecisionType,
    AssetReuseMode,
    AssetReuseReasonCode,
    ShotAssetVersion,
)
from app.domain.asset_router import AssetRoutingRequest, GenerationMode
from app.domain.enums import VisualType
from app.services.asset_probe_service import calculate_sha256


class AssetReusePolicy:
    """
    Pure deterministic policy evaluating whether historical ShotAssetVersions
    can be technically reused for a given routing request.

    STRICT CONSERVATIVE RULES:
    1. FORCE_REGENERATE mode immediately forces fresh execution.
    2. No candidate versions -> EXECUTE (NO_PREVIOUS_ASSET).
    3. Exact shot_revision_id match required (no fuzzy/semantic matching).
    4. Requested visual type must match candidate media type (VIDEO vs IMAGE).
    5. Generation mode must be compatible if specified.
    6. Aspect ratio must match within tolerance.
    7. Duration must match within tolerance for video assets.
    8. Physical file must exist on disk and be non-empty.
    9. File size and SHA-256 hash must verify.

    NOTE: Technical reusability does NOT imply semantic quality approval (Phase 6 boundary).
    """

    def __init__(
        self,
        duration_tolerance_seconds: float = 2.0,
        aspect_ratio_tolerance: float = 0.05,
        verify_hash: bool = True,
    ):
        self.duration_tolerance_seconds = duration_tolerance_seconds
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.verify_hash = verify_hash

    def evaluate(
        self,
        request: AssetRoutingRequest,
        candidate_versions: Sequence[ShotAssetVersion],
        reuse_mode: AssetReuseMode = AssetReuseMode.REUSE_COMPATIBLE,
        expected_generation_mode: GenerationMode | None = None,
    ) -> AssetReuseDecision:
        # Rule 1: Force regenerate bypasses all reuse checks
        if reuse_mode == AssetReuseMode.FORCE_REGENERATE:
            return AssetReuseDecision(
                shot_id=request.shot_id,
                shot_revision_id=request.shot_revision_id,
                candidate_asset_version_id=None,
                decision=AssetReuseDecisionType.EXECUTE,
                reason_codes=(AssetReuseReasonCode.FORCE_REGENERATE,),
            )

        # Rule 2: No previous assets
        if not candidate_versions:
            return AssetReuseDecision(
                shot_id=request.shot_id,
                shot_revision_id=request.shot_revision_id,
                candidate_asset_version_id=None,
                decision=AssetReuseDecisionType.EXECUTE,
                reason_codes=(AssetReuseReasonCode.NO_PREVIOUS_ASSET,),
            )

        # Sort candidate versions newest first
        sorted_candidates = sorted(
            candidate_versions,
            key=lambda v: v.created_at,
            reverse=True,
        )

        all_candidate_reasons: list[AssetReuseReasonCode] = []

        for candidate in sorted_candidates:
            reasons: list[AssetReuseReasonCode] = []

            # Check 1: Exact shot revision ID
            if candidate.shot_revision_id != request.shot_revision_id:
                reasons.append(AssetReuseReasonCode.SHOT_REVISION_CHANGED)

            # Check 2: Visual Type / Media Type Compatibility
            if not self._is_visual_type_compatible(request.requested_visual_type, candidate.media_type):
                reasons.append(AssetReuseReasonCode.VISUAL_TYPE_CHANGED)

            # Check 3: Generation Mode Compatibility
            if expected_generation_mode is not None and candidate.generation_mode != expected_generation_mode:
                reasons.append(AssetReuseReasonCode.GENERATION_MODE_CHANGED)

            # Check 4: Aspect Ratio Compatibility
            if request.aspect_ratio and not self._is_aspect_ratio_compatible(request.aspect_ratio, candidate.width, candidate.height):
                reasons.append(AssetReuseReasonCode.ASPECT_RATIO_INCOMPATIBLE)

            # Check 5: Duration Compatibility (for VIDEO)
            if (
                candidate.media_type == AssetMediaType.VIDEO
                and request.target_duration > 0
                and not self._is_duration_compatible(
                    request.target_duration, candidate.duration_seconds
                )
            ):
                reasons.append(AssetReuseReasonCode.DURATION_INCOMPATIBLE)

            # Check 6 & 7: Physical File Existence, Size, and Hash Integrity
            file_path = Path(candidate.file_path)
            if not file_path.is_file():
                reasons.append(AssetReuseReasonCode.ASSET_MISSING)
            else:
                try:
                    file_size = file_path.stat().st_size
                    if file_size <= 0:
                        reasons.append(AssetReuseReasonCode.ASSET_MISSING)
                        reasons.append(AssetReuseReasonCode.ASSET_INTEGRITY_FAILED)
                    elif file_size != candidate.file_size_bytes:
                        reasons.append(AssetReuseReasonCode.ASSET_INTEGRITY_FAILED)
                    elif self.verify_hash:
                        actual_hash = calculate_sha256(file_path)
                        if actual_hash.lower() != candidate.file_hash.lower():
                            reasons.append(AssetReuseReasonCode.ASSET_INTEGRITY_FAILED)
                except OSError:
                    reasons.append(AssetReuseReasonCode.ASSET_MISSING)

            # If this candidate satisfies all criteria, reuse it!
            if not reasons:
                return AssetReuseDecision(
                    shot_id=request.shot_id,
                    shot_revision_id=request.shot_revision_id,
                    candidate_asset_version_id=candidate.shot_asset_version_id,
                    decision=AssetReuseDecisionType.REUSE,
                    reason_codes=(AssetReuseReasonCode.EXACT_COMPATIBLE_ASSET,),
                )

            for r in reasons:
                if r not in all_candidate_reasons:
                    all_candidate_reasons.append(r)

        # No candidate passed all checks
        return AssetReuseDecision(
            shot_id=request.shot_id,
            shot_revision_id=request.shot_revision_id,
            candidate_asset_version_id=None,
            decision=AssetReuseDecisionType.EXECUTE,
            reason_codes=tuple(all_candidate_reasons) if all_candidate_reasons else (AssetReuseReasonCode.NO_PREVIOUS_ASSET,),
        )

    def _is_visual_type_compatible(self, requested_type: VisualType, media_type: AssetMediaType) -> bool:
        if requested_type in (VisualType.STOCK_VIDEO, VisualType.AI_VIDEO):
            return media_type == AssetMediaType.VIDEO
        if requested_type in (VisualType.AI_IMAGE, VisualType.DIAGRAM):
            return media_type == AssetMediaType.IMAGE
        if requested_type in (VisualType.SOURCE_ASSET, VisualType.USER_ASSET):
            return media_type in (AssetMediaType.VIDEO, AssetMediaType.IMAGE)
        return True

    def _is_aspect_ratio_compatible(self, aspect_str: str, width: int, height: int) -> bool:
        if width <= 0 or height <= 0:
            return False
        candidate_ratio = width / height
        expected_ratio = self._parse_aspect_ratio(aspect_str)
        if expected_ratio is None:
            return True
        diff = abs(candidate_ratio - expected_ratio) / expected_ratio
        return diff <= self.aspect_ratio_tolerance

    def _is_duration_compatible(self, target_duration: float, candidate_duration: float) -> bool:
        diff = abs(candidate_duration - target_duration)
        tolerance = max(self.duration_tolerance_seconds, target_duration * 0.25)
        return diff <= tolerance

    @staticmethod
    def _parse_aspect_ratio(aspect_str: str) -> float | None:
        cleaned = aspect_str.strip()
        if ":" in cleaned:
            parts = cleaned.split(":")
            if len(parts) == 2:
                try:
                    w = float(parts[0])
                    h = float(parts[1])
                    if w > 0 and h > 0:
                        return w / h
                except ValueError:
                    pass
        return None
