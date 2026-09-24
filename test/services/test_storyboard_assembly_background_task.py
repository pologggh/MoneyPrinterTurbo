from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.controllers.v1.storyboard import (
    ApiSynthesizeVideoRequest,
    ApiExecuteAssetRoutePlanRequest,
    _run_storyboard_assembly_worker,
    execute_asset_route_plan,
    synthesize_storyboard_video,
)
from app.models import const
from app.services import state as sm
from app.services.storyboard_video_assembly_service import StoryboardAssemblyError


def test_synthesize_video_returns_immediately_with_processing_state():
    """
    Area 4: Verify synthesize_storyboard_video returns HTTP 200 immediately
    with task_id and state=PROCESSING without blocking on actual assembly.
    """
    with patch("app.controllers.v1.video.task_manager.add_task") as mock_add_task:
        res = synthesize_storyboard_video(
            storyboard_snapshot_id="snap_bg_01",
            body=ApiSynthesizeVideoRequest(execution_run_id="run_bg_01"),
        )
        assert res["status"] == 200
        assert "task_id" in res["data"]
        assert res["data"]["state"] == const.TASK_STATE_PROCESSING
        assert res["data"]["execution_run_id"] == "run_bg_01"
        mock_add_task.assert_called_once()


def test_synthesize_video_rejects_duplicate_busy_task():
    """
    Area 4: Verify 409 Conflict is raised when task_id already exists and is busy.
    """
    busy_task_id = "task_busy_123"
    sm.state.update_task(busy_task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        synthesize_storyboard_video(
            storyboard_snapshot_id="snap_bg_02",
            body=ApiSynthesizeVideoRequest(task_id=busy_task_id),
        )

    assert exc_info.value.status_code == 409
    assert "already running/busy" in exc_info.value.detail


def test_background_worker_catches_error_and_sets_task_failed():
    """
    Area 4: Verify background worker catches exceptions and marks sm.state as FAILED with error message.
    """
    task_id = "task_worker_fail_01"
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    with patch(
        "app.controllers.v1.storyboard.StoryboardVideoAssemblyService.assemble_video",
        side_effect=StoryboardAssemblyError("Missing required shot assets"),
    ):
        _run_storyboard_assembly_worker(
            task_id=task_id,
            storyboard_snapshot_id="snap_fail",
            execution_run_id="run_fail",
            video_params=None,
        )

    task_state = sm.state.get_task(task_id)
    assert task_state is not None
    assert task_state["state"] == const.TASK_STATE_FAILED
    assert task_state["error_code"] == "STORYBOARD_ASSEMBLY_FAILED"
    assert "Check server logs" in task_state["error"]


def test_execute_asset_route_plan_queues_first_execution() -> None:
    @contextmanager
    def fake_session_scope():
        yield MagicMock()

    ready_plan = SimpleNamespace(is_ready=True)
    with (
        patch(
            "app.controllers.v1.storyboard.get_session",
            side_effect=fake_session_scope,
        ),
        patch(
            "app.controllers.v1.storyboard.AssetRoutePlanRepository.get_route_plan",
            return_value=ready_plan,
        ),
        patch("app.controllers.v1.video.task_manager.add_task") as add_task,
    ):
        response = execute_asset_route_plan(
            asset_route_plan_id="route_plan_first_01",
            body=ApiExecuteAssetRoutePlanRequest(task_id="asset_task_first_01"),
        )

    assert response["status"] == 200
    assert response["data"]["task_id"] == "asset_task_first_01"
    add_task.assert_called_once()


def test_direct_assembly_exception_marks_task_failed() -> None:
    from app.services.storyboard_video_assembly_service import (
        StoryboardVideoAssemblyService,
    )

    task_id = "direct_assembly_failure_01"
    service = StoryboardVideoAssemblyService()
    with (
        patch.object(
            service,
            "_execute_assembly",
            side_effect=RuntimeError("unexpected encoder failure"),
        ),
        pytest.raises(RuntimeError, match="encoder failure"),
    ):
        service.assemble_video(
            storyboard_snapshot_id="snapshot_direct_failure",
            task_id=task_id,
            session=MagicMock(),
        )

    task_state = sm.state.get_task(task_id)
    assert task_state is not None
    assert task_state["state"] == const.TASK_STATE_FAILED
    assert task_state["error_code"] == "STORYBOARD_ASSEMBLY_FAILED"
