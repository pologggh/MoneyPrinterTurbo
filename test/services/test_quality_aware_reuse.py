from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from app.domain.asset_execution import (
    AssetMediaType,
    AssetReuseDecisionType,
    ShotAssetVersion,
)
from app.domain.asset_router import AssetRoutingRequest, GenerationMode
from app.domain.enums import VisualType
from app.domain.evaluation import (
    DimensionEvaluationResult,
    DimensionEvaluationStatus,
    EvaluationDecision,
    EvaluationDimension,
    EvaluationSnapshot,
)
from app.services.asset_probe_service import calculate_sha256
from app.services.quality_aware_reuse_policy import (
    QualityAwareReusePolicy,
    QualityReuseStatus,
)


def _create_temp_file(tmp_dir: Path, content: bytes = b"test video binary content") -> tuple[str, str, int]:
    fpath = tmp_dir / f"test_{uuid4().hex[:6]}.mp4"
    fpath.write_bytes(content)
    return str(fpath), calculate_sha256(fpath), len(content)


def _make_request() -> AssetRoutingRequest:
    return AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="srev-1",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=5.0,
        aspect_ratio="16:9",
        visual_goal="Show quantum computer chip",
        scene_description="Close-up of quantum processor with circuits",
        generation_prompt="Quantum computer chip with glowing circuits",
        camera_movement="zoom in",
    )


class TestQualityAwareReuse:

    def test_technical_reuse_without_quality_pass_requires_evaluation(self):
        """Invariant 39: Technical reuse can be TECHNICALLY_REUSABLE without implying quality PASS."""
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fpath, fhash, fsize = _create_temp_file(tmp_path)

            version = ShotAssetVersion(
                shot_asset_version_id="v-1",
                shot_id="shot-1",
                shot_revision_id="srev-1",
                execution_attempt_id="att-1",
                file_path=fpath,
                file_hash=fhash,
                file_size_bytes=fsize,
                media_type=AssetMediaType.VIDEO,
                mime_type="video/mp4",
                duration_seconds=5.0,
                width=1920,
                height=1080,
                provider="mock_p",
                model="mock_m",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
            )

            req = _make_request()

            policy = QualityAwareReusePolicy()
            # No historical evaluation snapshot provided
            decision = policy.evaluate(req, candidate_versions=[version], historical_snapshots=[])

            # Technical reuse is satisfied, but quality status is REUSE_TECHNICAL_ONLY_NEEDS_EVALUATION
            assert decision.technical_decision.decision == AssetReuseDecisionType.REUSE
            assert decision.status == QualityReuseStatus.REUSE_TECHNICAL_ONLY_NEEDS_EVALUATION
            assert decision.selected_asset_version_id == "v-1"

    def test_quality_aware_reuse_with_valid_pass_snapshot(self):
        """Invariant 38: Quality-aware reuse accepts asset when current policy has valid PASS."""
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fpath, fhash, fsize = _create_temp_file(tmp_path)

            version = ShotAssetVersion(
                shot_asset_version_id="v-1",
                shot_id="shot-1",
                shot_revision_id="srev-1",
                execution_attempt_id="att-1",
                file_path=fpath,
                file_hash=fhash,
                file_size_bytes=fsize,
                media_type=AssetMediaType.VIDEO,
                mime_type="video/mp4",
                duration_seconds=5.0,
                width=1920,
                height=1080,
                provider="mock_p",
                model="mock_m",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
            )

            req = _make_request()

            pass_snapshot = EvaluationSnapshot(
                evaluation_snapshot_id="snap-pass-1",
                evaluation_target_id="target-1",
                dimension_result_ids=("d1", "d2", "d3", "d4"),
                decision=EvaluationDecision.PASS,
                policy_version="evaluation-policy-v1",
                evaluator_version="mock-v1",
                overall_score=0.90,
            )

            policy = QualityAwareReusePolicy()
            decision = policy.evaluate(
                req,
                candidate_versions=[version],
                historical_snapshots=[pass_snapshot],
            )

            assert decision.status == QualityReuseStatus.REUSE_QUALITY_ACCEPTED
            assert decision.selected_asset_version_id == "v-1"
            assert decision.evaluation_snapshot_id == "snap-pass-1"

    def test_stale_pass_under_stricter_policy_is_not_blindly_reused(self):
        """Invariant 37: Old PASS result under stale policy is re-evaluated and rejected if below new threshold."""
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fpath, fhash, fsize = _create_temp_file(tmp_path)

            version = ShotAssetVersion(
                shot_asset_version_id="v-1",
                shot_id="shot-1",
                shot_revision_id="srev-1",
                execution_attempt_id="att-1",
                file_path=fpath,
                file_hash=fhash,
                file_size_bytes=fsize,
                media_type=AssetMediaType.VIDEO,
                mime_type="video/mp4",
                duration_seconds=5.0,
                width=1920,
                height=1080,
                provider="mock_p",
                model="mock_m",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
            )

            req = _make_request()

            # Snapshot was marked PASS under old "v0-lenient" policy
            old_snapshot = EvaluationSnapshot(
                evaluation_snapshot_id="snap-old-pass",
                evaluation_target_id="target-1",
                dimension_result_ids=("d1", "d2", "d3", "d4"),
                decision=EvaluationDecision.PASS,
                policy_version="v0-lenient",
                evaluator_version="mock-v1",
                overall_score=0.76,
            )

            # Historical dimension scores: visual quality is 0.60
            # Under new policy, visual quality threshold is 0.65 (so 0.60 fails!)
            dim_results = [
                DimensionEvaluationResult(
                    dimension_result_id="d1",
                    evaluation_target_id="target-1",
                    dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
                    status=DimensionEvaluationStatus.SCORED,
                    score=0.90,
                ),
                DimensionEvaluationResult(
                    dimension_result_id="d2",
                    evaluation_target_id="target-1",
                    dimension=EvaluationDimension.VISUAL_QUALITY,
                    status=DimensionEvaluationStatus.SCORED,
                    score=0.60,  # Below 0.65 threshold of evaluation-policy-v1!
                ),
                DimensionEvaluationResult(
                    dimension_result_id="d3",
                    evaluation_target_id="target-1",
                    dimension=EvaluationDimension.KNOWLEDGE_ACCURACY,
                    status=DimensionEvaluationStatus.SCORED,
                    score=0.95,
                ),
                DimensionEvaluationResult(
                    dimension_result_id="d4",
                    evaluation_target_id="target-1",
                    dimension=EvaluationDimension.COMPOSITION_SUITABILITY,
                    status=DimensionEvaluationStatus.SCORED,
                    score=0.85,
                ),
            ]

            policy = QualityAwareReusePolicy()  # Uses evaluation-policy-v1
            decision = policy.evaluate(
                req,
                candidate_versions=[version],
                historical_snapshots=[old_snapshot],
                historical_dimension_results=dim_results,
            )

            # Old PASS is rejected because under new policy, visual quality 0.60 < 0.65
            assert decision.status == QualityReuseStatus.POLICY_CHANGED_FAILED
            assert decision.selected_asset_version_id is None
            assert "STALE_PASS_FAILED_UNDER_NEW_POLICY" in decision.reason_codes
