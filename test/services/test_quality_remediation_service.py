from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from PIL import Image

from app.domain.asset_execution import (
    AdapterExecutionResult,
    AssetMediaType,
    ProviderOutcomeType,
    ShotAssetVersion,
)
from app.domain.asset_router import (
    AssetCapability,
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutingRequest,
    CandidateScore,
    GenerationMode,
    RoutingStrategy,
    StaticCapabilityMetadata,
    TierLevel,
)
from app.domain.content_plan import ContentBeat
from app.domain.enums import BeatType, VisualType
from app.domain.evaluation import (
    DimensionEvaluationResult,
    DimensionEvaluationStatus,
    EvaluationDecision,
    EvaluationDimension,
    EvaluationSnapshot,
)
from app.domain.quality_remediation import (
    QualityRemediationAction,
    QualityRemediationPolicy,
)
from app.domain.shot import ShotRevision
from app.services.asset_adapters.base import AssetExecutionAdapter
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.asset_capability_registry import AssetCapabilityRegistry
from app.services.evaluation.quality_remediation_service import (
    QualityRemediationService,
)
from app.services.hybrid_asset_router import HybridAssetRouter
from app.services.shot_execution_service import ShotExecutionService


class StaticCapabilityProvider:

    def __init__(self, caps: tuple[AssetCapability, ...]):
        self._caps = caps

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        return self._caps


class ConfigurableMockAdapter(AssetExecutionAdapter):

    def __init__(self, outcomes: list[ProviderOutcomeType] | None = None):
        self.outcomes = outcomes or [ProviderOutcomeType.SUCCESS]
        self.call_count = 0

    def execute(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        idx = min(self.call_count, len(self.outcomes) - 1)
        outcome = self.outcomes[idx]
        self.call_count += 1

        if outcome == ProviderOutcomeType.SUCCESS:
            target_dir.mkdir(parents=True, exist_ok=True)
            fpath = target_dir / f"asset_{uuid4().hex[:6]}.png"
            img = Image.new("RGB", (1920, 1080), color="blue")
            img.save(str(fpath))
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUCCESS,
                file_path=str(fpath),
                status="SUCCEEDED",
            )
        else:
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                error_code="ERROR",
                error_message="Provider technical error",
                status="FAILED",
            )


def _make_candidate(
    cap_id: str,
    provider: str,
    model: str,
    vt: VisualType = VisualType.AI_VIDEO,
    gen_mode: GenerationMode = GenerationMode.TEXT_TO_VIDEO,
) -> AssetRouteCandidate:
    return AssetRouteCandidate(
        capability_id=cap_id,
        provider=provider,
        model=model,
        generation_mode=gen_mode,
        requested_visual_type=vt,
        is_eligible=True,
        rejection_reasons=(),
        static_metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.HIGH,
            cost_tier=TierLevel.MEDIUM,
            latency_tier=TierLevel.LOW,
        ),
        score=CandidateScore(
            capability_id=cap_id,
            quality_score=0.9,
            cost_score=0.6,
            latency_score=0.9,
            final_score=0.8,
            ranking_position=1,
        ),
    )


def _make_dimension_results(
    sem_score: float = 0.9,
    vis_score: float = 0.8,
    know_score: float = 0.95,
    comp_score: float = 0.85,
) -> list[DimensionEvaluationResult]:
    return [
        DimensionEvaluationResult(
            evaluation_target_id="target-1",
            dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
            status=DimensionEvaluationStatus.SCORED,
            score=sem_score,
        ),
        DimensionEvaluationResult(
            evaluation_target_id="target-1",
            dimension=EvaluationDimension.VISUAL_QUALITY,
            status=DimensionEvaluationStatus.SCORED,
            score=vis_score,
        ),
        DimensionEvaluationResult(
            evaluation_target_id="target-1",
            dimension=EvaluationDimension.KNOWLEDGE_ACCURACY,
            status=DimensionEvaluationStatus.SCORED,
            score=know_score,
        ),
        DimensionEvaluationResult(
            evaluation_target_id="target-1",
            dimension=EvaluationDimension.COMPOSITION_SUITABILITY,
            status=DimensionEvaluationStatus.SCORED,
            score=comp_score,
        ),
    ]


def _make_snapshot(decision: EvaluationDecision, dim_results: list[DimensionEvaluationResult]) -> EvaluationSnapshot:
    return EvaluationSnapshot(
        evaluation_snapshot_id=f"snap_{uuid4().hex[:6]}",
        evaluation_target_id="target-1",
        dimension_result_ids=tuple(r.dimension_result_id for r in dim_results),
        decision=decision,
        policy_version="evaluation-policy-v1",
        evaluator_version="mock-v1",
        overall_score=0.88 if decision == EvaluationDecision.PASS else 0.50,
    )


class TestQualityRemediationService:

    def test_same_route_regeneration_creates_new_asset_and_accepts_on_pass(self):
        """Invariant 7, 8, 36: Same-route quality regen preserves candidate, creates NEW asset, and PASS freezes both."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Setup adapters
            adapter = ConfigurableMockAdapter([ProviderOutcomeType.SUCCESS, ProviderOutcomeType.SUCCESS])
            reg = AdapterRegistry()
            reg.register_adapter("mock_p", adapter)
            exec_service = ShotExecutionService(adapter_registry=reg)

            shot_rev = ShotRevision(
                shot_revision_id="srev-1",
                shot_id="shot-1",
                revision_number=1,
                beat_lineage_id="lineage-1",
                created_from_beat_instance_id="beat-1",
                narration="Superposition enables exponential speedup in quantum algorithms.",
                target_duration=5.0,
                visual_goal="3D Bloch sphere representation",
                visual_type=VisualType.AI_VIDEO,
                scene_description="Spinning sphere",
                generation_prompt="Spinning quantum sphere, high quality",
                camera_movement="pan",
                evidence_refs=("ev-1",),
            )

            beat = ContentBeat(
                beat_id="beat-1",
                beat_lineage_id="lineage-1",
                beat_type=BeatType.HOOK,
                order=1,
                intent="Intro",
                target_duration=5.0,
                importance=0.9,
                evidence_refs=("ev-1",),
            )

            initial_asset = ShotAssetVersion(
                shot_asset_version_id="asset-initial",
                shot_id="shot-1",
                shot_revision_id="srev-1",
                execution_attempt_id="att-0",
                file_path=str(tmp_path / "initial.png"),
                file_hash="a" * 64,
                file_size_bytes=100,
                media_type=AssetMediaType.IMAGE,
                mime_type="image/png",
                width=1920,
                height=1080,
                duration_seconds=0.0,
                provider="mock_p",
                model="model-1",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
            )

            cand = _make_candidate("cap-1", "mock_p", "model-1")
            route_decision = AssetRouteDecision(
                shot_id="shot-1",
                shot_revision_id="srev-1",
                routing_strategy=RoutingStrategy.BALANCED,
                selected_candidate=cand,
                eligible_candidates=(cand,),
            )

            # Initial evaluation failed visual quality
            initial_dims = _make_dimension_results(sem_score=0.9, vis_score=0.45)
            initial_snap = _make_snapshot(EvaluationDecision.FAIL, initial_dims)

            # Evaluator function: returns PASS on second evaluation (after same-route regen)
            def mock_eval_fn(rev: ShotRevision, asset: ShotAssetVersion):
                dims = _make_dimension_results(sem_score=0.92, vis_score=0.88)
                snap = _make_snapshot(EvaluationDecision.PASS, dims)
                return snap, dims

            service = QualityRemediationService(
                execution_service=exec_service,
                storage_base_dir=tmp_path,
            )

            result = service.run_remediation_chain(
                shot_revision=shot_rev,
                initial_asset_version=initial_asset,
                initial_snapshot=initial_snap,
                initial_dimension_results=initial_dims,
                route_decision=route_decision,
                beat=beat,
                evaluation_fn=mock_eval_fn,
            )

            assert result.status == "ACCEPTED"
            assert result.final_action == QualityRemediationAction.ACCEPT_ASSET
            # Invariant 8: NEW asset version was created
            assert result.accepted_asset_version is not None
            assert result.accepted_asset_version.shot_asset_version_id != initial_asset.shot_asset_version_id
            # Invariant 9: Old failed asset remains in history
            assert len(result.all_generated_asset_versions) == 2
            assert result.all_generated_asset_versions[0] == initial_asset
            assert result.all_generated_asset_versions[1] == result.accepted_asset_version
            # Invariant 36: Selection freezes exact asset + snapshot
            assert result.quality_selection is not None
            assert result.quality_selection.shot_asset_version_id == result.accepted_asset_version.shot_asset_version_id
            assert result.quality_selection.evaluation_snapshot_id == result.accepted_evaluation_snapshot.evaluation_snapshot_id

    def test_critical_loop_termination_on_continuous_failure(self):
        """Invariant 40 & Critical Loop Termination: Continual failure terminates in NEEDS_USER_ACTION."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            adapter = ConfigurableMockAdapter([ProviderOutcomeType.SUCCESS] * 10)
            reg = AdapterRegistry()
            reg.register_adapter("mock_p1", adapter)
            reg.register_adapter("mock_p2", adapter)
            exec_service = ShotExecutionService(adapter_registry=reg)

            # Build capability registry and mock router for controlled replan
            cap1 = AssetCapability(
                capability_id="cap-1",
                provider="mock_p1",
                model="model-1",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
                supported_visual_types=(VisualType.AI_VIDEO, VisualType.AI_IMAGE),
            )
            cap2 = AssetCapability(
                capability_id="cap-2",
                provider="mock_p2",
                model="model-2",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
                supported_visual_types=(VisualType.AI_VIDEO, VisualType.AI_IMAGE),
            )
            cap_reg = AssetCapabilityRegistry(
                providers=[StaticCapabilityProvider((cap1, cap2))]
            )
            router = HybridAssetRouter(registry=cap_reg)

            shot_rev = ShotRevision(
                shot_revision_id="srev-1",
                shot_id="shot-1",
                revision_number=1,
                beat_lineage_id="lineage-1",
                created_from_beat_instance_id="beat-1",
                narration="Quantum computing narrative.",
                target_duration=5.0,
                visual_goal="Complex prompt",
                visual_type=VisualType.AI_VIDEO,
                scene_description="Complex description",
                generation_prompt="Complex prompt",
                camera_movement="pan",
                evidence_refs=("ev-1",),
            )

            beat = ContentBeat(
                beat_id="beat-1",
                beat_lineage_id="lineage-1",
                beat_type=BeatType.HOOK,
                order=1,
                intent="Intro",
                target_duration=5.0,
                importance=0.9,
                evidence_refs=("ev-1",),
            )

            initial_asset = ShotAssetVersion(
                shot_asset_version_id="asset-0",
                shot_id="shot-1",
                shot_revision_id="srev-1",
                execution_attempt_id="att-0",
                file_path=str(tmp_path / "initial.png"),
                file_hash="b" * 64,
                file_size_bytes=100,
                media_type=AssetMediaType.IMAGE,
                mime_type="image/png",
                width=1920,
                height=1080,
                duration_seconds=0.0,
                provider="mock_p1",
                model="model-1",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
            )

            cands = [
                _make_candidate("cap-1", "mock_p1", "model-1"),
                _make_candidate("cap-2", "mock_p2", "model-2"),
            ]
            route_decision = AssetRouteDecision(
                shot_id="shot-1",
                shot_revision_id="srev-1",
                routing_strategy=RoutingStrategy.BALANCED,
                selected_candidate=cands[0],
                eligible_candidates=tuple(cands),
                model_selection_mode="AUTO",
            )

            initial_dims = _make_dimension_results(sem_score=0.50, vis_score=0.50)
            initial_snap = _make_snapshot(EvaluationDecision.FAIL, initial_dims)

            # Evaluator function: ALWAYS returns FAIL
            def always_fail_eval(rev: ShotRevision, asset: ShotAssetVersion):
                dims = _make_dimension_results(sem_score=0.50, vis_score=0.50)
                snap = _make_snapshot(EvaluationDecision.FAIL, dims)
                return snap, dims

            policy = QualityRemediationPolicy(
                max_same_route_quality_regenerations=1,
                max_route_fallbacks=1,
                max_controlled_auto_replans=1,
            )

            service = QualityRemediationService(
                execution_service=exec_service,
                router=router,
                policy=policy,
                storage_base_dir=tmp_path,
            )

            result = service.run_remediation_chain(
                shot_revision=shot_rev,
                initial_asset_version=initial_asset,
                initial_snapshot=initial_snap,
                initial_dimension_results=initial_dims,
                route_decision=route_decision,
                beat=beat,
                evaluation_fn=always_fail_eval,
            )

            # Loop MUST terminate deterministically at NEEDS_USER_ACTION
            assert result.status == "NEEDS_USER_ACTION"
            assert result.final_action == QualityRemediationAction.NEEDS_USER_ACTION
            assert result.accepted_asset_version is None

            actions = [d.action for d in result.remediation_decisions]
            # Expected sequence:
            # 1. REGENERATE_SAME_ROUTE (same-route budget consumed)
            # 2. FALLBACK_NEXT_ROUTE_CANDIDATE (fallback budget consumed)
            # 3. CONTROLLED_VISUAL_REPLAN (replan budget consumed)
            # 4. NEEDS_USER_ACTION (all budgets exhausted)
            assert QualityRemediationAction.REGENERATE_SAME_ROUTE in actions
            assert QualityRemediationAction.FALLBACK_NEXT_ROUTE_CANDIDATE in actions
            assert QualityRemediationAction.CONTROLLED_VISUAL_REPLAN in actions
            assert actions[-1] == QualityRemediationAction.NEEDS_USER_ACTION

    def test_technical_failure_during_quality_regen_does_not_pollute_quality_remediation(self):
        """Technical Failure During Quality Regen: Technical errors are handled via Phase 5 without creating bogus quality decisions."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Adapter fails with DEFINITIVE_TECHNICAL_FAILURE
            adapter = ConfigurableMockAdapter([ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE])
            reg = AdapterRegistry()
            reg.register_adapter("mock_p", adapter)
            exec_service = ShotExecutionService(adapter_registry=reg, max_retries_per_candidate=1)

            shot_rev = ShotRevision(
                shot_revision_id="srev-1",
                shot_id="shot-1",
                revision_number=1,
                beat_lineage_id="lineage-1",
                created_from_beat_instance_id="beat-1",
                narration="Narrative",
                target_duration=5.0,
                visual_goal="Goal",
                visual_type=VisualType.AI_VIDEO,
                scene_description="Desc",
                generation_prompt="Prompt",
                camera_movement="pan",
            )

            beat = ContentBeat(
                beat_id="beat-1",
                beat_lineage_id="lineage-1",
                beat_type=BeatType.HOOK,
                order=1,
                intent="Intro",
                target_duration=5.0,
                importance=0.9,
            )

            cand = _make_candidate("cap-1", "mock_p", "model-1")
            route_decision = AssetRouteDecision(
                shot_id="shot-1",
                shot_revision_id="srev-1",
                routing_strategy=RoutingStrategy.BALANCED,
                selected_candidate=cand,
                eligible_candidates=(cand,),
            )

            initial_asset = ShotAssetVersion(
                shot_asset_version_id="asset-0",
                shot_id="shot-1",
                shot_revision_id="srev-1",
                execution_attempt_id="att-0",
                file_path=str(tmp_path / "initial.png"),
                file_hash="c" * 64,
                file_size_bytes=100,
                media_type=AssetMediaType.IMAGE,
                mime_type="image/png",
                width=1920,
                height=1080,
                duration_seconds=0.0,
                provider="mock_p",
                model="model-1",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
            )

            initial_dims = _make_dimension_results(vis_score=0.4)
            initial_snap = _make_snapshot(EvaluationDecision.FAIL, initial_dims)

            eval_called = False

            def mock_eval(rev, asset):
                nonlocal eval_called
                eval_called = True
                return initial_snap, initial_dims

            service = QualityRemediationService(
                execution_service=exec_service,
                storage_base_dir=tmp_path,
            )

            result = service.run_remediation_chain(
                shot_revision=shot_rev,
                initial_asset_version=initial_asset,
                initial_snapshot=initial_snap,
                initial_dimension_results=initial_dims,
                route_decision=route_decision,
                beat=beat,
                evaluation_fn=mock_eval,
            )

            # Because execution failed technically, the evaluation was NOT called on a non-existent asset
            assert result.status == "NEEDS_USER_ACTION"
            assert not eval_called

    def test_router_called_only_for_controlled_visual_replan_never_for_same_route_or_fallback(self):
        """Invariant 12, 13, 29, 30: Router is NEVER called for same-route or fallback; ONLY called for replan."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            adapter = ConfigurableMockAdapter([ProviderOutcomeType.SUCCESS] * 10)
            reg = AdapterRegistry()
            reg.register_adapter("mock_p1", adapter)
            reg.register_adapter("mock_p2", adapter)
            exec_service = ShotExecutionService(adapter_registry=reg)

            cap1 = AssetCapability(
                capability_id="cap-1",
                provider="mock_p1",
                model="model-1",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
                supported_visual_types=(VisualType.AI_VIDEO,),
            )
            cap2 = AssetCapability(
                capability_id="cap-2",
                provider="mock_p2",
                model="model-2",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
                supported_visual_types=(VisualType.AI_VIDEO,),
            )
            cap_reg = AssetCapabilityRegistry(
                providers=[StaticCapabilityProvider((cap1, cap2))]
            )

            router_calls = 0

            class TrackingRouter(HybridAssetRouter):
                def route(self, req, capabilities=None, policy=None, raise_if_unavailable=False):
                    nonlocal router_calls
                    router_calls += 1
                    return super().route(req, capabilities, policy, raise_if_unavailable=raise_if_unavailable)

            tracking_router = TrackingRouter(registry=cap_reg)

            shot_rev = ShotRevision(
                shot_revision_id="srev-1",
                shot_id="shot-1",
                revision_number=1,
                beat_lineage_id="lineage-1",
                created_from_beat_instance_id="beat-1",
                narration="Quantum computing narrative.",
                target_duration=5.0,
                visual_goal="Complex prompt",
                visual_type=VisualType.AI_VIDEO,
                scene_description="Complex description",
                generation_prompt="Complex prompt",
                camera_movement="pan",
                evidence_refs=("ev-1",),
            )

            beat = ContentBeat(
                beat_id="beat-1",
                beat_lineage_id="lineage-1",
                beat_type=BeatType.HOOK,
                order=1,
                intent="Intro",
                target_duration=5.0,
                importance=0.9,
                evidence_refs=("ev-1",),
            )

            initial_asset = ShotAssetVersion(
                shot_asset_version_id="asset-0",
                shot_id="shot-1",
                shot_revision_id="srev-1",
                execution_attempt_id="att-0",
                file_path=str(tmp_path / "initial.png"),
                file_hash="d" * 64,
                file_size_bytes=100,
                media_type=AssetMediaType.IMAGE,
                mime_type="image/png",
                width=1920,
                height=1080,
                duration_seconds=0.0,
                provider="mock_p1",
                model="model-1",
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
            )

            cands = [
                _make_candidate("cap-1", "mock_p1", "model-1"),
                _make_candidate("cap-2", "mock_p2", "model-2"),
            ]
            route_decision = AssetRouteDecision(
                shot_id="shot-1",
                shot_revision_id="srev-1",
                routing_strategy=RoutingStrategy.BALANCED,
                selected_candidate=cands[0],
                eligible_candidates=tuple(cands),
                model_selection_mode="AUTO",
            )

            initial_dims = _make_dimension_results(sem_score=0.50, vis_score=0.50)
            initial_snap = _make_snapshot(EvaluationDecision.FAIL, initial_dims)

            eval_count = 0

            def stepwise_eval(rev: ShotRevision, asset: ShotAssetVersion):
                nonlocal eval_count
                eval_count += 1
                # 1: after same-route regen -> fail
                # 2: after fallback to next candidate -> fail
                # 3: after controlled visual replan -> PASS
                if eval_count == 3:
                    dims = _make_dimension_results(sem_score=0.95, vis_score=0.95)
                    snap = _make_snapshot(EvaluationDecision.PASS, dims)
                    return snap, dims
                dims = _make_dimension_results(sem_score=0.50, vis_score=0.50)
                snap = _make_snapshot(EvaluationDecision.FAIL, dims)
                return snap, dims

            policy = QualityRemediationPolicy(
                max_same_route_quality_regenerations=1,
                max_route_fallbacks=1,
                max_controlled_auto_replans=1,
            )

            service = QualityRemediationService(
                execution_service=exec_service,
                router=tracking_router,
                policy=policy,
                storage_base_dir=tmp_path,
            )

            result = service.run_remediation_chain(
                shot_revision=shot_rev,
                initial_asset_version=initial_asset,
                initial_snapshot=initial_snap,
                initial_dimension_results=initial_dims,
                route_decision=route_decision,
                beat=beat,
                evaluation_fn=stepwise_eval,
            )

            assert result.status == "ACCEPTED"
            # Router should have been called EXACTLY ONCE for the controlled visual replan
            # (NEVER for same-route regen, NEVER for route candidate fallback)
            assert router_calls == 1

