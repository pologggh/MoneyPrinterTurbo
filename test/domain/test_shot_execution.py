from pathlib import Path

import pytest
from PIL import Image

from app.domain.asset_execution import (
    AdapterExecutionResult,
    AssetMediaType,
    BlindSubmissionForbiddenError,
    ExecutionAttemptStatus,
    ProviderOutcomeType,
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
from app.services.asset_adapters.base import AssetExecutionAdapter
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.shot_execution_service import ShotExecutionService


class MockAdapter(AssetExecutionAdapter):
    """Configurable mock adapter for testing provider outcomes."""

    def __init__(self, responses: list[AdapterExecutionResult]):
        self._responses = responses
        self.call_count = 0

    def execute(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        idx = min(self.call_count, len(self._responses) - 1)
        resp = self._responses[idx]
        self.call_count += 1
        return resp


def _create_test_file(target_dir: Path, filename: str = "asset.png") -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / filename
    img = Image.new("RGB", (1280, 720), color="green")
    img.save(str(file_path))
    return file_path


def _build_test_decision() -> AssetRouteDecision:
    primary = AssetRouteCandidate(
        capability_id="seedance:pro:TEXT_TO_VIDEO",
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        requested_visual_type=VisualType.AI_VIDEO,
        is_eligible=True,
    )
    fallback = AssetRouteCandidate(
        capability_id="wavespeed:standard:TEXT_TO_VIDEO",
        provider="wavespeed",
        model="standard",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        requested_visual_type=VisualType.AI_VIDEO,
        is_eligible=True,
    )
    return AssetRouteDecision(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        routing_strategy=RoutingStrategy.BALANCED,
        requested_visual_type=VisualType.AI_VIDEO,
        selected_candidate=primary,
        eligible_candidates=(primary, fallback),
    )


def test_shot_execution_success_creates_shot_asset_version(tmp_path: Path):
    decision = _build_test_decision()
    valid_file = _create_test_file(tmp_path)

    adapter = MockAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path=str(valid_file),
            raw_response={"task_id": "seedance-1"},
        )
    ])
    registry = AdapterRegistry({"seedance": adapter})
    service = ShotExecutionService(adapter_registry=registry)

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        route_decision=decision,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=3.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Pan",
    )

    final_exec, attempts, version = service.execute_shot(shot_exec, req, tmp_path)

    assert final_exec.status == ShotExecutionStatus.SUCCEEDED
    assert version is not None
    assert version.media_type == AssetMediaType.IMAGE
    assert version.provider == "seedance"
    assert len(attempts) == 1
    assert attempts[0].status == ExecutionAttemptStatus.SUCCEEDED
    assert adapter.call_count == 1


def test_definitive_technical_failure_retries_same_candidate(tmp_path: Path):
    decision = _build_test_decision()
    valid_file = _create_test_file(tmp_path)

    # First attempt: 500 error; Second attempt: Success
    adapter = MockAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
            error_code="INTERNAL_SERVER_ERROR",
            error_message="500 Internal Error",
        ),
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path=str(valid_file),
            raw_response={"task_id": "seedance-2"},
        ),
    ])
    registry = AdapterRegistry({"seedance": adapter})
    service = ShotExecutionService(adapter_registry=registry, max_retries_per_candidate=2)

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        route_decision=decision,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=3.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Pan",
    )

    final_exec, attempts, version = service.execute_shot(shot_exec, req, tmp_path)

    assert final_exec.status == ShotExecutionStatus.SUCCEEDED
    assert version is not None
    assert len(attempts) == 2
    assert attempts[0].status == ExecutionAttemptStatus.FAILED
    assert attempts[0].error_code == "INTERNAL_SERVER_ERROR"
    assert attempts[1].status == ExecutionAttemptStatus.SUCCEEDED
    assert adapter.call_count == 2
    assert final_exec.current_candidate_index == 0  # Succeeded on candidate 0


def test_definitive_technical_failure_exhausts_retries_and_falls_back(tmp_path: Path):
    decision = _build_test_decision()
    valid_file = _create_test_file(tmp_path)

    # Candidate 0 (seedance): Fails twice
    seedance_adapter = MockAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
            error_code="SERVICE_UNAVAILABLE",
            error_message="503 Service Unavailable",
        )
    ])
    # Candidate 1 (wavespeed): Succeeds
    wavespeed_adapter = MockAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path=str(valid_file),
            raw_response={"task_id": "wavespeed-1"},
        )
    ])
    registry = AdapterRegistry({
        "seedance": seedance_adapter,
        "wavespeed": wavespeed_adapter,
    })
    service = ShotExecutionService(adapter_registry=registry, max_retries_per_candidate=2)

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        route_decision=decision,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=3.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Pan",
    )

    final_exec, attempts, version = service.execute_shot(shot_exec, req, tmp_path)

    assert final_exec.status == ShotExecutionStatus.SUCCEEDED
    assert len(attempts) == 3  # 2 on seedance, 1 on wavespeed
    assert attempts[0].provider == "seedance"
    assert attempts[1].provider == "seedance"
    assert attempts[2].provider == "wavespeed"
    assert attempts[2].status == ExecutionAttemptStatus.SUCCEEDED
    assert final_exec.current_candidate_index == 1
    assert version is not None
    assert version.provider == "wavespeed"


def test_capability_incompatible_immediately_falls_back_without_retry(tmp_path: Path):
    decision = _build_test_decision()
    valid_file = _create_test_file(tmp_path)

    # Candidate 0 (seedance): Rejected due to content policy / 400
    seedance_adapter = MockAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
            error_code="CONTENT_POLICY_VIOLATION",
            error_message="Prompt blocked by content safety",
        )
    ])
    # Candidate 1 (wavespeed): Succeeds
    wavespeed_adapter = MockAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            file_path=str(valid_file),
        )
    ])
    registry = AdapterRegistry({
        "seedance": seedance_adapter,
        "wavespeed": wavespeed_adapter,
    })
    service = ShotExecutionService(adapter_registry=registry, max_retries_per_candidate=3)

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        route_decision=decision,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=3.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Pan",
    )

    final_exec, attempts, version = service.execute_shot(shot_exec, req, tmp_path)

    assert final_exec.status == ShotExecutionStatus.SUCCEEDED
    assert version is not None
    # Candidate 0 was called strictly ONCE despite max_retries_per_candidate=3!
    assert seedance_adapter.call_count == 1
    assert len(attempts) == 2
    assert attempts[0].provider == "seedance"
    assert attempts[0].error_code == "CONTENT_POLICY_VIOLATION"
    assert attempts[1].provider == "wavespeed"
    assert attempts[1].status == ExecutionAttemptStatus.SUCCEEDED
    assert final_exec.current_candidate_index == 1


def test_unconfirmed_task_transitions_to_submission_outcome_unknown(tmp_path: Path):
    decision = _build_test_decision()

    # Candidate 0 returns SUBMISSION_OUTCOME_UNKNOWN (e.g. VolcEngineSeedanceUnconfirmedTaskError)
    seedance_adapter = MockAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
            remote_task_id="seedance-task-999",
            error_code="SUBMISSION_OUTCOME_UNKNOWN",
            error_message="Task creation unconfirmed on remote server",
        )
    ])
    # Wavespeed should NEVER be called
    wavespeed_adapter = MockAdapter([])

    registry = AdapterRegistry({
        "seedance": seedance_adapter,
        "wavespeed": wavespeed_adapter,
    })
    service = ShotExecutionService(adapter_registry=registry, max_retries_per_candidate=3)

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        route_decision=decision,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=3.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Pan",
    )

    final_exec, attempts, version = service.execute_shot(shot_exec, req, tmp_path)

    # Status must be SUBMISSION_OUTCOME_UNKNOWN
    assert final_exec.status == ShotExecutionStatus.SUBMISSION_OUTCOME_UNKNOWN
    assert final_exec.unconfirmed_remote_task_id == "seedance-task-999"
    assert version is None
    # Strictly only 1 attempt was made! No retry, no fallback to wavespeed!
    assert len(attempts) == 1
    assert seedance_adapter.call_count == 1
    assert wavespeed_adapter.call_count == 0


def test_blind_resubmission_strictly_forbidden_on_unknown_outcome(tmp_path: Path):
    decision = _build_test_decision()
    adapter = MockAdapter([])
    registry = AdapterRegistry({"seedance": adapter})
    service = ShotExecutionService(adapter_registry=registry)

    unknown_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        route_decision=decision,
        status=ShotExecutionStatus.SUBMISSION_OUTCOME_UNKNOWN,
        unconfirmed_remote_task_id="task-xyz",
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=3.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Pan",
    )

    # Calling execute on a shot with SUBMISSION_OUTCOME_UNKNOWN must raise BlindSubmissionForbiddenError
    with pytest.raises(BlindSubmissionForbiddenError, match="Blind re-submission is strictly forbidden"):
        service.execute_shot(unknown_exec, req, tmp_path)

    assert adapter.call_count == 0


def test_all_candidates_exhausted_marks_shot_execution_failed(tmp_path: Path):
    decision = _build_test_decision()

    seedance_adapter = MockAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
            error_code="REJECTED_A",
        )
    ])
    wavespeed_adapter = MockAdapter([
        AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
            error_code="REJECTED_B",
        )
    ])
    registry = AdapterRegistry({
        "seedance": seedance_adapter,
        "wavespeed": wavespeed_adapter,
    })
    service = ShotExecutionService(adapter_registry=registry)

    shot_exec = ShotExecution(
        execution_run_id="run-1",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        route_decision=decision,
    )
    req = AssetRoutingRequest(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=3.0,
        visual_goal="Goal",
        scene_description="Scene",
        generation_prompt="Prompt",
        camera_movement="Pan",
    )

    final_exec, attempts, version = service.execute_shot(shot_exec, req, tmp_path)

    assert final_exec.status == ShotExecutionStatus.FAILED
    assert final_exec.error_code in ("ALL_CANDIDATES_EXHAUSTED", "NO_FALLBACK_CANDIDATE")
    assert version is None
    assert len(attempts) == 2
