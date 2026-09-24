from __future__ import annotations

from loguru import logger

from app.domain.asset_execution import AssetReuseMode, ExecutionStatus
from app.models import const
from app.persistence.repositories import AssetRoutePlanRepository
from app.persistence.session import get_session
from app.services import state as sm
from app.services.asset_route_plan_execution_service import (
    AssetRoutePlanExecutionService,
)
from app.services.storyboard_video_assembly_service import (
    StoryboardVideoAssemblyService,
)
from app.utils import utils


def run_storyboard_assembly_task(
    task_id: str,
    storyboard_snapshot_id: str,
    execution_run_id: str | None,
    video_params: dict | None,
) -> None:
    try:
        StoryboardVideoAssemblyService().assemble_video(
            storyboard_snapshot_id=storyboard_snapshot_id,
            execution_run_id=execution_run_id,
            video_params=video_params,
            task_id=task_id,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            f"Background storyboard video assembly failed for task {task_id}"
        )
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_FAILED,
            progress=100,
            error_code="STORYBOARD_ASSEMBLY_FAILED",
            error="Storyboard video assembly failed. Check server logs for details.",
        )


def run_asset_route_plan_task(
    task_id: str,
    asset_route_plan_id: str,
    reuse_mode: str,
) -> None:
    try:
        with get_session() as session:
            plan = AssetRoutePlanRepository(session).get_route_plan(
                asset_route_plan_id
            )
        if plan is None:
            raise ValueError(f"AssetRoutePlan '{asset_route_plan_id}' not found")

        result = AssetRoutePlanExecutionService().execute_route_plan(
            plan=plan,
            storage_base_dir=utils.storage_dir("asset_executions", create=True),
            reuse_mode=AssetReuseMode(reuse_mode),
        )
        result_data = result.model_dump(mode="json")
        if result.state == ExecutionStatus.COMPLETED:
            sm.state.update_task(
                task_id,
                state=const.TASK_STATE_COMPLETE,
                progress=100,
                execution_run_id=result.execution_run_id,
                asset_execution_result=result_data,
            )
        else:
            sm.state.update_task(
                task_id,
                state=const.TASK_STATE_FAILED,
                progress=100,
                execution_run_id=result.execution_run_id,
                asset_execution_result=result_data,
                error_code=result.state.value,
                error="Asset route plan execution did not complete successfully.",
            )
    except Exception:  # noqa: BLE001
        logger.exception(f"Background asset route execution failed for task {task_id}")
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_FAILED,
            progress=100,
            error_code="ASSET_ROUTE_EXECUTION_FAILED",
            error="Asset route plan execution failed. Check server logs for details.",
        )
