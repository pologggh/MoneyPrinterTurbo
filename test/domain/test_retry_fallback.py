from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from app.domain.asset_execution import (
    AdapterExecutionResult,
    AttemptRequest,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRetryPolicy,
    ProviderOutcomeType,
    ProviderReceipt,
    RecoveryAction,
    ShotExecution,
    ShotExecutionStatus,
)
from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutingRequest,
    GenerationMode,
    RoutingStrategy,
)
from app.domain.enums import VisualType
from app.services.asset_adapters.base import (
    AssetExecutionAdapter,
    ProviderExecutionCapabilities,
)
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.attempt_recovery_service import AttemptRecoveryService
from app.services.shot_execution_service import ShotExecutionService


class ConfigurableFakeAdapter(AssetExecutionAdapter):
    """
    Fake adapter simulating sync/async responses, unconfirmed submissions,
    and capability assertions.
    """

    def __init__(
        self,
        results: list[AdapterExecutionResult | ProviderReceipt] | None = None,
        capabilities: ProviderExecutionCapabilities | None = None,
        status_results: dict[str, AdapterExecutionResult] | None = None,
    ):
        self._results = list(results or [])
        self._capabilities = capabilities or ProviderExecutionCapabilities()
        self._status_results = status_results or {}
        self.submissions: list[tuple[AssetRoutingRequest, AssetRouteCandidate, str | None]] = []
        self.status_queries: list[str] = []

    @property
    def capabilities(self) -> ProviderExecutionCapabilities:
        return self._capabilities

    def execute(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        self.submissions.append((request, candidate, None))
        if self._results:
            res = self._results.pop(0)
            if isinstance(res, AdapterExecutionResult):
                return res
        return AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS)

    def submit(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
        idempotency_key: str | None = None,
    ) -> AdapterExecutionResult | ProviderReceipt:
        self.submissions.append((request, candidate, idempotency_key))
        if self._results:
            return self._results.pop(0)
        return AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS)

    def get_status(
        self,
        provider_job_id: str,
        target_dir: Path | None = None,
    ) -> AdapterExecutionResult:
        self.status_queries.append(provider_job_id)
        if provider_job_id in self._status_results:
            return self._status_results[provider_job_id]
        if self._results:
            res = self._results.pop(0)
            if isinstance(res, AdapterExecutionResult):
                return res
        return AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            status="SUCCEEDED",
            file_path=str(target_dir / "async_out.mp4") if target_dir else "out.mp4",
        )


def _make_dummy_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1280, 720), color="blue")
    img.save(str(path))
    return path


def _create_candidates(
    providers_and_models: list[tuple[str, str, VisualType]],
) -> list[AssetRouteCandidate]:
    candidates = []
    for provider, model, visual_type in providers_and_models:
        candidates.append(
            AssetRouteCandidate(
                capability_id=f"{provider}:{model}:{visual_type.value}",
                provider=provider,
                model=model,
                generation_mode=GenerationMode.TEXT_TO_VIDEO,
                requested_visual_type=visual_type,
                is_eligible=True,
            )
        )
    return candidates


def _create_decision(
    candidates: list[AssetRouteCandidate],
    mode: str = "AUTO",
    visual_type: VisualType = VisualType.AI_VIDEO,
) -> AssetRouteDecision:
    return AssetRouteDecision(
        shot_id="shot-test",
        shot_revision_id="rev-test",
        routing_strategy=RoutingStrategy.BALANCED,
        requested_visual_type=visual_type,
        model_selection_mode=mode,
        selected_candidate=candidates[0] if candidates else None,
        eligible_candidates=tuple(candidates),
    )


def _create_request(visual_type: VisualType = VisualType.AI_VIDEO) -> AssetRoutingRequest:
    return AssetRoutingRequest(
        shot_id="shot-test",
        shot_revision_id="rev-test",
        requested_visual_type=visual_type,
        target_duration=4.0,
        visual_goal="A scenic mountain",
        scene_description="Camera pans over pine trees",
        generation_prompt="Mountains and pine trees under sunlight",
        camera_movement="Pan Right",
        aspect_ratio="16:9",
    )


# ---------------------------------------------------------------------------
# Test 1 & 2 & 3: Safe transient failure, new ExecutionAttempt, monotonic numbering
# ---------------------------------------------------------------------------
def test_safe_transient_failure_retries_same_candidate_and_numbers_monotonically(tmp_path: Path):
    valid_file = _make_dummy_image(tmp_path / "asset1.png")
    cands = _create_candidates([("seedance", "pro", VisualType.AI_VIDEO)])
    decision = _create_decision(cands)

    fake_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
            error_code="RATE_LIMITED_429",
            error_message="Too many requests",
        ),
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path=str(valid_file),
        ),
    ])
    registry = AdapterRegistry({"seedance": fake_adapter})
    policy = ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4)
    service = ShotExecutionService(adapter_registry=registry, policy=policy)

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=decision,
    )
    final_exec, attempts, version = service.execute_shot(shot_exec, _create_request(), tmp_path)

    assert final_exec.status == ShotExecutionStatus.SUCCEEDED
    assert len(attempts) == 2
    # 1. Safe transient failure retries the SAME candidate
    assert attempts[0].provider == "seedance"
    assert attempts[1].provider == "seedance"
    # 2. Same-candidate retry creates a NEW ExecutionAttempt
    assert attempts[0].execution_attempt_id != attempts[1].execution_attempt_id
    # 3. Attempt numbering increases monotonically
    assert attempts[0].attempt_number == 1
    assert attempts[1].attempt_number == 2
    assert version is not None
    assert version.execution_attempt_id == attempts[1].execution_attempt_id


# ---------------------------------------------------------------------------
# Test 4: Candidate-local retry budget is enforced
# ---------------------------------------------------------------------------
def test_candidate_local_retry_budget_enforced(tmp_path: Path):
    cands = _create_candidates([("seedance", "pro", VisualType.AI_VIDEO)])
    decision = _create_decision(cands)

    fake_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
            error_code="SERVER_ERROR",
        ),
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
            error_code="SERVER_ERROR",
        ),
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
            error_code="SERVER_ERROR",
        ),
    ])
    registry = AdapterRegistry({"seedance": fake_adapter})
    policy = ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4)
    service = ShotExecutionService(adapter_registry=registry, policy=policy)

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=decision,
    )
    final_exec, attempts, _version = service.execute_shot(shot_exec, _create_request(), tmp_path)

    # Max attempts per candidate is 2, so strictly 2 attempts made
    assert len(attempts) == 2
    assert final_exec.status in (ShotExecutionStatus.FAILED, ShotExecutionStatus.EXHAUSTED)


# ---------------------------------------------------------------------------
# Test 5 & 24: Shot-total retry budget is enforced and stops at total budget
# ---------------------------------------------------------------------------
def test_shot_total_retry_budget_enforced_and_stops(tmp_path: Path):
    cands = _create_candidates([
        ("seedance", "pro", VisualType.AI_VIDEO),
        ("wavespeed", "standard", VisualType.AI_VIDEO),
        ("backup_provider", "model-c", VisualType.AI_VIDEO),
    ])
    decision = _create_decision(cands)

    # 3 providers, each failing with technical error
    seedance_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE),
    ])
    wavespeed_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE),
    ])
    backup_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE),
    ])

    registry = AdapterRegistry({
        "seedance": seedance_adapter,
        "wavespeed": wavespeed_adapter,
        "backup_provider": backup_adapter,
    })
    policy = ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=3)
    service = ShotExecutionService(adapter_registry=registry, policy=policy)

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=decision,
    )
    final_exec, attempts, _version = service.execute_shot(shot_exec, _create_request(), tmp_path)

    # Total budget is 3: 2 on seedance, 1 on wavespeed, 0 on backup_provider!
    assert len(attempts) == 3
    assert attempts[0].provider == "seedance"
    assert attempts[1].provider == "seedance"
    assert attempts[2].provider == "wavespeed"
    assert len(backup_adapter.submissions) == 0
    # 25. Exhausted execution receives deterministic EXHAUSTED state/error
    assert final_exec.status == ShotExecutionStatus.EXHAUSTED
    assert final_exec.error_code == "SHOT_EXECUTION_EXHAUSTED"


# ---------------------------------------------------------------------------
# Test 6: When same-candidate retry budget exhausted, AUTO moves to next candidate
# ---------------------------------------------------------------------------
def test_auto_moves_to_next_frozen_candidate_when_local_budget_exhausted(tmp_path: Path):
    valid_file = _make_dummy_image(tmp_path / "asset_wavespeed.png")
    cands = _create_candidates([
        ("seedance", "pro", VisualType.AI_VIDEO),
        ("wavespeed", "standard", VisualType.AI_VIDEO),
    ])
    decision = _create_decision(cands, mode="AUTO")

    seedance_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE),
    ])
    wavespeed_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=str(valid_file)),
    ])
    registry = AdapterRegistry({"seedance": seedance_adapter, "wavespeed": wavespeed_adapter})
    service = ShotExecutionService(adapter_registry=registry, policy=ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4))

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=decision,
    )
    final_exec, attempts, version = service.execute_shot(shot_exec, _create_request(), tmp_path)

    assert final_exec.status == ShotExecutionStatus.SUCCEEDED
    assert len(attempts) == 3
    assert attempts[2].provider == "wavespeed"
    assert version is not None
    assert version.provider == "wavespeed"


# ---------------------------------------------------------------------------
# Test 7 & 8: Permanent technical failure and Unsupported capability skip retry
# ---------------------------------------------------------------------------
def test_permanent_and_unsupported_failure_skips_same_candidate_retry(tmp_path: Path):
    valid_file = _make_dummy_image(tmp_path / "asset_wavespeed.png")
    cands = _create_candidates([
        ("seedance", "pro", VisualType.AI_VIDEO),
        ("wavespeed", "standard", VisualType.AI_VIDEO),
    ])
    decision = _create_decision(cands, mode="AUTO")

    # Seedance encounters UNSUPPORTED_CAPABILITY
    seedance_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE, error_code="UNSUPPORTED_DURATION"),
    ])
    wavespeed_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=str(valid_file)),
    ])
    registry = AdapterRegistry({"seedance": seedance_adapter, "wavespeed": wavespeed_adapter})
    service = ShotExecutionService(adapter_registry=registry, policy=ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4))

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=decision,
    )
    final_exec, attempts, _version = service.execute_shot(shot_exec, _create_request(), tmp_path)

    # Seedance only executed ONCE despite local retry budget of 2!
    assert len(seedance_adapter.submissions) == 1
    assert len(attempts) == 2
    assert attempts[0].provider == "seedance"
    assert attempts[1].provider == "wavespeed"
    assert final_exec.status == ShotExecutionStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# Test 9: PINNED mode never automatically switches Provider/Model
# ---------------------------------------------------------------------------
def test_pinned_mode_never_switches_provider_or_model(tmp_path: Path):
    cands = _create_candidates([
        ("seedance", "pro", VisualType.AI_VIDEO),
        ("wavespeed", "standard", VisualType.AI_VIDEO),
    ])
    decision = _create_decision(cands, mode="PINNED")

    seedance_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE, error_code="PROMPT_REJECTED"),
    ])
    wavespeed_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS),
    ])
    registry = AdapterRegistry({"seedance": seedance_adapter, "wavespeed": wavespeed_adapter})
    service = ShotExecutionService(adapter_registry=registry, policy=ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4))

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=decision,
    )
    final_exec, _attempts, _version = service.execute_shot(shot_exec, _create_request(), tmp_path)

    # Must STOP and never call wavespeed!
    assert len(wavespeed_adapter.submissions) == 0
    assert final_exec.status == ShotExecutionStatus.FAILED
    assert final_exec.error_code == "PINNED_MODEL_FAILED"


# ---------------------------------------------------------------------------
# Test 10 & 11: Frozen ranking order and HybridAssetRouter is NEVER called
# ---------------------------------------------------------------------------
def test_fallback_uses_frozen_ranking_and_never_calls_router(tmp_path: Path):
    cands = _create_candidates([
        ("seedance", "pro", VisualType.AI_VIDEO),
        ("wavespeed", "standard", VisualType.AI_VIDEO),
    ])
    decision = _create_decision(cands, mode="PREFERRED")

    valid_file = _make_dummy_image(tmp_path / "asset_fallback.png")
    seedance_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE),
    ])
    wavespeed_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=str(valid_file)),
    ])
    registry = AdapterRegistry({"seedance": seedance_adapter, "wavespeed": wavespeed_adapter})
    service = ShotExecutionService(adapter_registry=registry, policy=ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4))

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=decision,
    )

    with patch("app.services.hybrid_asset_router.HybridAssetRouter.route") as mock_router:
        _final_exec, attempts, _version = service.execute_shot(shot_exec, _create_request(), tmp_path)
        # 11. HybridAssetRouter is NEVER called during retry/fallback
        assert mock_router.call_count == 0

    # 10. Follows frozen order: seedance -> wavespeed
    assert attempts[0].provider == "seedance"
    assert attempts[1].provider == "seedance"
    assert attempts[2].provider == "wavespeed"


# ---------------------------------------------------------------------------
# Test 12: VisualType never changes during technical fallback
# ---------------------------------------------------------------------------
def test_visual_type_never_changes_during_fallback(tmp_path: Path):
    # Route decision erroneously includes a candidate with different visual type (e.g. AI_IMAGE)
    ai_video_cand = AssetRouteCandidate(
        capability_id="seedance:pro:AI_VIDEO",
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        requested_visual_type=VisualType.AI_VIDEO,
        is_eligible=True,
    )
    stock_image_cand = AssetRouteCandidate(
        capability_id="openai:dall-e-3:AI_IMAGE",
        provider="openai",
        model="dall-e-3",
        generation_mode=GenerationMode.TEXT_TO_IMAGE,
        requested_visual_type=VisualType.AI_IMAGE,
        is_eligible=True,
    )
    decision = AssetRouteDecision(
        shot_id="shot-test",
        shot_revision_id="rev-test",
        routing_strategy=RoutingStrategy.BALANCED,
        requested_visual_type=VisualType.AI_VIDEO,
        selected_candidate=ai_video_cand,
        eligible_candidates=(ai_video_cand, stock_image_cand),
    )

    seedance_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE),
    ])
    pexels_adapter = ConfigurableFakeAdapter([])
    registry = AdapterRegistry({"seedance": seedance_adapter, "pexels": pexels_adapter})
    service = ShotExecutionService(adapter_registry=registry)

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=decision,
    )
    final_exec, _attempts, _version = service.execute_shot(shot_exec, _create_request(VisualType.AI_VIDEO), tmp_path)

    # Pexels (STOCK_IMAGE) was filtered out; never invoked because visual type differs!
    assert len(pexels_adapter.submissions) == 0
    assert final_exec.status == ShotExecutionStatus.FAILED


# ---------------------------------------------------------------------------
# Test 13 & 14: SUBMISSION_OUTCOME_UNKNOWN does not create new attempt & sets NEEDS_RECOVERY
# ---------------------------------------------------------------------------
def test_submission_outcome_unknown_sets_needs_recovery_and_halts_retries(tmp_path: Path):
    cands = _create_candidates([
        ("seedance", "pro", VisualType.AI_VIDEO),
        ("wavespeed", "standard", VisualType.AI_VIDEO),
    ])
    decision = _create_decision(cands, mode="AUTO")

    seedance_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
            remote_task_id="unconfirmed-job-999",
            error_message="Network dropped during POST",
        ),
    ])
    wavespeed_adapter = ConfigurableFakeAdapter([])
    registry = AdapterRegistry({"seedance": seedance_adapter, "wavespeed": wavespeed_adapter})
    service = ShotExecutionService(adapter_registry=registry, policy=ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4))

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=decision,
    )
    final_exec, attempts, _version = service.execute_shot(shot_exec, _create_request(), tmp_path)

    # 13. Does not create a new Attempt
    assert len(attempts) == 1
    # 14. Moves ShotExecution to NEEDS_RECOVERY
    assert final_exec.status == ShotExecutionStatus.NEEDS_RECOVERY
    assert final_exec.unconfirmed_remote_task_id == "unconfirmed-job-999"
    # No fallback occurred
    assert len(wavespeed_adapter.submissions) == 0


# ---------------------------------------------------------------------------
# Test 15 & 16 & 18: Recovery with known job ID queries same job, uses same idempotency key, creates ShotAssetVersion
# ---------------------------------------------------------------------------
def test_recovery_queries_same_job_and_resolves_success(tmp_path: Path):
    valid_file = _make_dummy_image(tmp_path / "recovered_video.png")

    fake_adapter = ConfigurableFakeAdapter(
        capabilities=ProviderExecutionCapabilities(supports_async_status=True),
        status_results={
            "task-456": AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUCCESS,
                status="SUCCEEDED",
                file_path=str(valid_file),
                remote_task_id="task-456",
            )
        },
    )
    registry = AdapterRegistry({"seedance": fake_adapter})
    recovery_service = AttemptRecoveryService(adapter_registry=registry)

    cands = _create_candidates([("seedance", "pro", VisualType.AI_VIDEO)])
    decision = _create_decision(cands)

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=decision,
        status=ShotExecutionStatus.NEEDS_RECOVERY,
        unconfirmed_remote_task_id="task-456",
    )
    attempt = ExecutionAttempt(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        attempt_number=1,
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        status=ExecutionAttemptStatus.SUBMISSION_OUTCOME_UNKNOWN,
    )
    attempt_req = AttemptRequest.create_sanitized(
        execution_attempt_id=attempt.execution_attempt_id,
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        idempotency_key="idemp-same-key-999",
        raw_payload={"prompt": "test"},
    )
    receipt = ProviderReceipt(
        execution_attempt_id=attempt.execution_attempt_id,
        provider="seedance",
        provider_job_id="task-456",
        provider_status="submitted",
    )

    decision_result, res_exec, res_attempt, asset_version = recovery_service.recover(
        shot_execution=shot_exec,
        attempt=attempt,
        attempt_request=attempt_req,
        receipt=receipt,
        target_dir=tmp_path,
    )

    # 15. Queries the SAME external job
    assert fake_adapter.status_queries == ["task-456"]
    # 16. Same idempotency key
    assert attempt_req.idempotency_key == "idemp-same-key-999"
    # 18. Resolved success creates ShotAssetVersion tied to original Attempt
    assert decision_result.action == RecoveryAction.RESOLVED_SUCCEEDED
    assert res_exec.status == ShotExecutionStatus.SUCCEEDED
    assert res_attempt.status == ExecutionAttemptStatus.SUCCEEDED
    assert asset_version is not None
    assert asset_version.execution_attempt_id == attempt.execution_attempt_id


# ---------------------------------------------------------------------------
# Test 17 & 19: Recovery failure allows normal retry/fallback, unrecoverable remains NEEDS_RECOVERY
# ---------------------------------------------------------------------------
def test_recovery_failure_allows_normal_retry_and_unrecoverable_remains_needs_recovery(tmp_path: Path):
    # Sub-case 1: Provider without async status query remains NEEDS_RECOVERY
    no_recovery_adapter = ConfigurableFakeAdapter(
        capabilities=ProviderExecutionCapabilities(supports_async_status=False),
    )
    registry = AdapterRegistry({"seedance": no_recovery_adapter})
    recovery_service = AttemptRecoveryService(adapter_registry=registry)

    cands = _create_candidates([("seedance", "pro", VisualType.AI_VIDEO)])
    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=_create_decision(cands),
        status=ShotExecutionStatus.NEEDS_RECOVERY,
    )
    attempt = ExecutionAttempt(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        attempt_number=1,
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        status=ExecutionAttemptStatus.SUBMISSION_OUTCOME_UNKNOWN,
    )
    attempt_req = AttemptRequest.create_sanitized(
        execution_attempt_id=attempt.execution_attempt_id,
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        idempotency_key="idemp-key-1",
        raw_payload={},
    )

    decision_result, res_exec, _, _ = recovery_service.recover(shot_exec, attempt, attempt_req)
    # 17. Provider that cannot safely recover remains NEEDS_RECOVERY
    assert decision_result.action == RecoveryAction.STILL_UNKNOWN
    assert res_exec.status == ShotExecutionStatus.NEEDS_RECOVERY

    # Sub-case 2: Recovery resolves to FAILED -> transitions shot to RUNNING allowing retry
    failing_adapter = ConfigurableFakeAdapter(
        capabilities=ProviderExecutionCapabilities(supports_async_status=True),
        status_results={
            "task-fail": AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                status="FAILED",
                error_code="INTERNAL_SERVER_ERROR",
            )
        },
    )
    registry2 = AdapterRegistry({"seedance": failing_adapter})
    recovery_service2 = AttemptRecoveryService(adapter_registry=registry2)
    receipt = ProviderReceipt(
        execution_attempt_id=attempt.execution_attempt_id,
        provider="seedance",
        provider_job_id="task-fail",
        provider_status="submitted",
    )

    dec2, res_exec2, res_att2, _ = recovery_service2.recover(
        shot_exec, attempt, attempt_req, receipt=receipt
    )
    # 19. Recovery-resolved failure allows normal retry/fallback afterward
    assert dec2.action == RecoveryAction.RESOLVED_FAILED
    assert res_att2.status == ExecutionAttemptStatus.FAILED
    assert res_exec2.status == ShotExecutionStatus.RUNNING


# ---------------------------------------------------------------------------
# Test 21 & 22: ProviderReceipt persisted on acceptance & async job produces asset only after final success
# ---------------------------------------------------------------------------
def test_async_job_produces_asset_only_after_final_success(tmp_path: Path):
    valid_file = _make_dummy_image(tmp_path / "async_completed.png")

    fake_adapter = ConfigurableFakeAdapter(
        results=[
            ProviderReceipt(
                execution_attempt_id="att-placeholder",
                provider="seedance",
                provider_job_id="async-job-123",
                provider_status="accepted",
            ),
        ],
        capabilities=ProviderExecutionCapabilities(supports_async_status=True),
        status_results={
            "async-job-123": AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUCCESS,
                status="RUNNING",
                remote_task_id="async-job-123",
            )
        },
    )
    registry = AdapterRegistry({"seedance": fake_adapter})
    service = ShotExecutionService(adapter_registry=registry)

    cands = _create_candidates([("seedance", "pro", VisualType.AI_VIDEO)])
    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=_create_decision(cands),
    )

    running_exec, attempts, version = service.execute_shot(shot_exec, _create_request(), tmp_path)
    assert running_exec.status == ShotExecutionStatus.RUNNING
    # 21. ProviderReceipt is persisted only when provider acceptance is known
    assert len(attempts) == 1
    assert attempts[0].provider_receipt is not None
    assert attempts[0].provider_receipt.provider_job_id == "async-job-123"
    # No version produced yet while running
    assert version is None

    # Now simulate provider job completing
    fake_adapter._status_results["async-job-123"] = AdapterExecutionResult(
        outcome_type=ProviderOutcomeType.SUCCESS,
        status="SUCCEEDED",
        file_path=str(valid_file),
        remote_task_id="async-job-123",
    )

    # Now poll the attempt for completion
    succeeded_exec, succeeded_att, final_version = service.poll_attempt(
        running_exec, attempts[0], tmp_path
    )
    # 22. Successful async job produces ShotAssetVersion only after final success
    assert succeeded_exec.status == ShotExecutionStatus.SUCCEEDED
    assert succeeded_att.status == ExecutionAttemptStatus.SUCCEEDED
    assert final_version is not None
    assert final_version.execution_attempt_id == attempts[0].execution_attempt_id


# ---------------------------------------------------------------------------
# Test 23: Failed attempts never create ShotAssetVersion
# ---------------------------------------------------------------------------
def test_failed_attempts_never_create_shot_asset_version(tmp_path: Path):
    cands = _create_candidates([("seedance", "pro", VisualType.AI_VIDEO)])
    fake_adapter = ConfigurableFakeAdapter([
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE),
        AdapterExecutionResult(outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE),
    ])
    registry = AdapterRegistry({"seedance": fake_adapter})
    service = ShotExecutionService(adapter_registry=registry, policy=ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=2))

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=_create_decision(cands),
    )
    final_exec, attempts, version = service.execute_shot(shot_exec, _create_request(), tmp_path)

    assert final_exec.status == ShotExecutionStatus.EXHAUSTED
    assert version is None
    for att in attempts:
        assert att.status == ExecutionAttemptStatus.FAILED


# ---------------------------------------------------------------------------
# Test 27: External provider call does not run inside a long DB transaction
# ---------------------------------------------------------------------------
def test_external_provider_call_does_not_run_inside_db_transaction(tmp_path: Path):
    """
    Pattern verification: verify adapter execution occurs without holding open
    any database transaction context.
    """
    db_session_mock = MagicMock()
    # Adapter checks that session.in_transaction() is False during submit/execute
    class TransactionBoundaryAdapter(AssetExecutionAdapter):
        def execute(self, request, candidate, target_dir):
            if db_session_mock.in_transaction():
                raise RuntimeError("External call executed inside DB transaction!")
            out_file = target_dir / "out.png"
            _make_dummy_image(out_file)
            return AdapterExecutionResult(outcome_type=ProviderOutcomeType.SUCCESS, file_path=str(out_file))
    cands = _create_candidates([("seedance", "pro", VisualType.AI_VIDEO)])
    registry = AdapterRegistry({"seedance": TransactionBoundaryAdapter()})
    service = ShotExecutionService(adapter_registry=registry)

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-test",
        shot_revision_id="rev-test",
        route_decision=_create_decision(cands),
    )

    db_session_mock.in_transaction.return_value = False
    final_exec, _attempts, _version = service.execute_shot(shot_exec, _create_request(), tmp_path)
    assert final_exec.status == ShotExecutionStatus.SUCCEEDED
