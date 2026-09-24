from fastapi import Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.controllers import base
from app.controllers.manager.base_manager import TaskQueueFullError
from app.controllers.v1.base import new_router
from app.domain.asset_execution import AssetReuseMode
from app.domain.asset_router import (
    ConfiguredModelNotFoundError,
    ModelSelectionMode,
    RoutingStrategy,
    StoryboardNotApprovedError,
)
from app.domain.enums import VisualType
from app.domain.planner import ContentPlanner, PlannerError, PlannerInput
from app.domain.storyboard_agent import StoryboardAgent
from app.domain.storyboard_approval import (
    ApprovalValidationError,
    ContentReplanRequiredError,
    InvalidSnapshotStateForApprovalError,
    StaleApprovalSnapshotError,
    StoryboardApprovalError,
    StoryboardApprovalService,
    StoryboardBeatReplanError,
    StoryboardBeatReplanInput,
)
from app.domain.storyboard_editing import (
    CrossBeatMoveNotAllowedError,
    EditShotInput,
    InvalidShotEditError,
    InvalidShotOrderError,
    ReorderShotsInput,
    ShotNotFoundError,
    ShotNotInStoryboardError,
    StaleShotRevisionError,
    StoryboardEditingService,
    StoryboardNotFoundError,
)
from app.domain.storyboard_orchestrator import (
    StoryboardBuildInput,
    StoryboardOrchestrationError,
    StoryboardOrchestrator,
)
from app.models import const
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    ContentPlanRepository,
    ExecutionRepository,
    ShotRepository,
    StoryboardApprovalRepository,
    StoryboardRepository,
)
from app.persistence.session import get_session
from app.services import state as sm
from app.services.asset_route_plan_execution_service import (
    AssetRoutePlanExecutionService,
)
from app.services.asset_route_planning_service import (
    AssetRoutePlanningService,
    CreateAssetRoutePlanInput,
)
from app.services.attempt_recovery_service import AttemptRecoveryService
from app.services.storyboard_generation_pipeline import (
    EvidenceInputItem,
    StoryboardGenerationInput,
    StoryboardGenerationPipeline,
)
from app.services.storyboard_video_assembly_service import (
    StoryboardVideoAssemblyService,
)
from app.services.storyboard_background_tasks import (
    run_asset_route_plan_task,
    run_storyboard_assembly_task,
)
from app.utils import utils

router = new_router(dependencies=[Depends(base.verify_token)])


class ApiEditShotRequest(BaseModel):
    base_shot_revision_id: str
    narration: str | None = None
    target_duration: float | None = None
    visual_goal: str | None = None
    visual_type: VisualType | None = None
    scene_description: str | None = None
    generation_prompt: str | None = None
    camera_movement: str | None = None


class ApiReorderShotsRequest(BaseModel):
    ordered_shot_ids: list[str]


class ApiApproveStoryboardRequest(BaseModel):
    approved_by: str = "local_user"
    user_note: str | None = None


class ApiReplanBeatRequest(BaseModel):
    user_instruction: str | None = None
    video_title: str | None = None
    target_aspect_ratio: str | None = None
    language: str | None = None
    source_grounded: bool = True


class ApiCreateAssetRoutePlanRequest(BaseModel):
    routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED
    model_selection_mode: ModelSelectionMode = ModelSelectionMode.AUTO
    selected_provider: str | None = None
    selected_model: str | None = None
    target_aspect_ratio: str | None = None


class ApiEvidenceItem(BaseModel):
    evidence_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    content: str = Field(min_length=1)
    title: str | None = None
    url: str | None = None


class ApiCreateStoryboardFromTopicRequest(BaseModel):
    topic: str
    target_video_duration: float = 60.0
    title: str | None = None
    user_instruction: str | None = None
    target_aspect_ratio: str = "16:9"
    language: str = "zh"
    source_grounded: bool = False
    knowledge_context: list[str] = Field(default_factory=list)
    evidence_items: list[ApiEvidenceItem] = Field(default_factory=list)


class ApiCreateContentPlanRequest(BaseModel):
    topic: str
    target_video_duration: float = 60.0
    title: str | None = None
    user_instruction: str | None = None
    source_grounded: bool = False
    knowledge_context: list[str] = Field(default_factory=list)
    evidence_items: list[ApiEvidenceItem] = Field(default_factory=list)


class ApiBuildStoryboardFromPlanRequest(BaseModel):
    content_plan_revision_id: str
    user_instruction: str | None = None
    target_aspect_ratio: str = "16:9"
    language: str = "zh"
    source_grounded: bool = False


class ApiSynthesizeVideoRequest(BaseModel):
    execution_run_id: str | None = None
    video_params: dict | None = None
    task_id: str | None = None


class ApiRetryExecutionRunRequest(BaseModel):
    reuse_mode: AssetReuseMode = AssetReuseMode.REUSE_COMPATIBLE


class ApiExecuteAssetRoutePlanRequest(BaseModel):
    reuse_mode: AssetReuseMode = AssetReuseMode.REUSE_COMPATIBLE
    task_id: str | None = None


class ApiRecoverExecutionRunRequest(BaseModel):
    force_timeout: bool = False


@router.get(
    "/storyboards/{storyboard_snapshot_id}",
    summary="Get storyboard editing state",
)
def get_storyboard_state(storyboard_snapshot_id: str):
    with get_session() as session:
        service = StoryboardEditingService(
            plan_repository=ContentPlanRepository(session),
            shot_repository=ShotRepository(session),
            storyboard_repository=StoryboardRepository(session),
            execution_repository=ExecutionRepository(session),
        )
        try:
            view = service.get_storyboard_for_editing(storyboard_snapshot_id)
            return utils.get_response(200, view.model_dump())
        except StoryboardNotFoundError as exc:
            raise HTTPException(status_code=404, detail=exc.message) from exc


@router.post(
    "/storyboards/{storyboard_snapshot_id}/shots/{shot_id}",
    summary="Save shot edit",
)
def edit_shot(
    storyboard_snapshot_id: str,
    shot_id: str,
    body: ApiEditShotRequest,
):
    with get_session() as session:
        service = StoryboardEditingService(
            plan_repository=ContentPlanRepository(session),
            shot_repository=ShotRepository(session),
            storyboard_repository=StoryboardRepository(session),
        )
        input_data = EditShotInput(
            storyboard_snapshot_id=storyboard_snapshot_id,
            shot_id=shot_id,
            base_shot_revision_id=body.base_shot_revision_id,
            narration=body.narration,
            target_duration=body.target_duration,
            visual_goal=body.visual_goal,
            visual_type=body.visual_type,
            scene_description=body.scene_description,
            generation_prompt=body.generation_prompt,
            camera_movement=body.camera_movement,
        )
        try:
            new_snap = service.edit_shot(input_data)
            return utils.get_response(
                200,
                {
                    "storyboard_snapshot_id": new_snap.storyboard_snapshot_id,
                    "content_plan_revision_id": new_snap.content_plan_revision_id,
                    "snapshot_state": new_snap.snapshot_state.value,
                    "shot_revision_ids": list(new_snap.shot_revision_ids),
                },
            )
        except StoryboardNotFoundError as exc:
            raise HTTPException(status_code=404, detail=exc.message) from exc
        except (ShotNotFoundError, ShotNotInStoryboardError) as exc:
            raise HTTPException(status_code=404, detail=exc.message) from exc
        except StaleShotRevisionError as exc:
            raise HTTPException(status_code=409, detail=exc.message) from exc
        except InvalidShotEditError as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc


@router.post(
    "/storyboards/{storyboard_snapshot_id}/beats/{beat_lineage_id}/reorder",
    summary="Reorder shots within one beat",
)
def reorder_shots(
    storyboard_snapshot_id: str,
    beat_lineage_id: str,
    body: ApiReorderShotsRequest,
):
    with get_session() as session:
        service = StoryboardEditingService(
            plan_repository=ContentPlanRepository(session),
            shot_repository=ShotRepository(session),
            storyboard_repository=StoryboardRepository(session),
        )
        input_data = ReorderShotsInput(
            storyboard_snapshot_id=storyboard_snapshot_id,
            beat_lineage_id=beat_lineage_id,
            ordered_shot_ids=tuple(body.ordered_shot_ids),
        )
        try:
            new_snap = service.reorder_shots_within_beat(input_data)
            return utils.get_response(
                200,
                {
                    "storyboard_snapshot_id": new_snap.storyboard_snapshot_id,
                    "snapshot_state": new_snap.snapshot_state.value,
                    "shot_revision_ids": list(new_snap.shot_revision_ids),
                },
            )
        except StoryboardNotFoundError as exc:
            raise HTTPException(status_code=404, detail=exc.message) from exc
        except CrossBeatMoveNotAllowedError as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc
        except InvalidShotOrderError as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc


@router.post(
    "/storyboards/{storyboard_snapshot_id}/approve",
    summary="Approve storyboard snapshot",
)
def approve_storyboard(
    storyboard_snapshot_id: str,
    body: ApiApproveStoryboardRequest | None = None,
):
    req_body = body or ApiApproveStoryboardRequest()
    with get_session() as session:
        service = StoryboardApprovalService(
            plan_repository=ContentPlanRepository(session),
            shot_repository=ShotRepository(session),
            storyboard_repository=StoryboardRepository(session),
            approval_repository=StoryboardApprovalRepository(session),
        )
        try:
            record = service.approve_storyboard(
                storyboard_snapshot_id=storyboard_snapshot_id,
                approved_by=req_body.approved_by,
                user_note=req_body.user_note,
            )
            return utils.get_response(
                200,
                {
                    "storyboard_approval_id": record.storyboard_approval_id,
                    "source_draft_snapshot_id": record.source_draft_snapshot_id,
                    "approved_storyboard_snapshot_id": record.approved_storyboard_snapshot_id,
                    "content_plan_revision_id": record.content_plan_revision_id,
                    "exact_shot_revision_ids": list(record.exact_shot_revision_ids),
                    "approved_by": record.approved_by,
                    "approved_at": record.approved_at.isoformat(),
                    "user_note": record.user_note,
                },
            )
        except StoryboardNotFoundError as exc:
            raise HTTPException(status_code=404, detail=exc.message) from exc
        except StaleApprovalSnapshotError as exc:
            raise HTTPException(status_code=409, detail=exc.message) from exc
        except (
            InvalidSnapshotStateForApprovalError,
            ApprovalValidationError,
            StoryboardApprovalError,
        ) as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc


@router.post(
    "/storyboards/{storyboard_snapshot_id}/beats/{beat_lineage_id}/replan",
    summary="Controlled agent replan for a single beat",
)
def replan_beat(
    storyboard_snapshot_id: str,
    beat_lineage_id: str,
    body: ApiReplanBeatRequest | None = None,
):
    req_body = body or ApiReplanBeatRequest()
    with get_session() as session:
        service = StoryboardApprovalService(
            plan_repository=ContentPlanRepository(session),
            shot_repository=ShotRepository(session),
            storyboard_repository=StoryboardRepository(session),
            approval_repository=StoryboardApprovalRepository(session),
        )
        input_data = StoryboardBeatReplanInput(
            base_storyboard_snapshot_id=storyboard_snapshot_id,
            target_beat_lineage_id=beat_lineage_id,
            user_instruction=req_body.user_instruction,
            video_title=req_body.video_title,
            target_aspect_ratio=req_body.target_aspect_ratio,
            language=req_body.language,
            source_grounded=req_body.source_grounded,
        )
        try:
            result = service.replan_beat(input_data)
            return utils.get_response(
                200,
                {
                    "previous_storyboard_snapshot_id": result.previous_storyboard_snapshot_id,
                    "new_storyboard_snapshot_id": result.new_storyboard_snapshot_id,
                    "beat_lineage_id": result.beat_lineage_id,
                    "old_shot_revision_ids": list(result.old_shot_revision_ids),
                    "new_shot_revision_ids": list(result.new_shot_revision_ids),
                    "generated_shot_count": result.generated_shot_count,
                    "state": result.state.value,
                },
            )
        except StoryboardNotFoundError as exc:
            raise HTTPException(status_code=404, detail=exc.message) from exc
        except ContentReplanRequiredError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"CONTENT_REPLAN_REQUIRED: {exc.message}",
            ) from exc
        except (StoryboardBeatReplanError, StoryboardApprovalError) as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc


@router.get(
    "/storyboards/{storyboard_snapshot_id}/approval",
    summary="Get storyboard approval record",
)
def get_storyboard_approval(storyboard_snapshot_id: str):
    with get_session() as session:
        approval_repo = StoryboardApprovalRepository(session)
        rec = approval_repo.get_approval_by_approved_snapshot_id(storyboard_snapshot_id)
        if not rec:
            rec = approval_repo.get_approval_by_source_draft_id(storyboard_snapshot_id)
        if not rec:
            raise HTTPException(
                status_code=404,
                detail="No approval record found for this storyboard snapshot",
            )
        return utils.get_response(
            200,
            {
                "storyboard_approval_id": rec.storyboard_approval_id,
                "source_draft_snapshot_id": rec.source_draft_snapshot_id,
                "approved_storyboard_snapshot_id": rec.approved_storyboard_snapshot_id,
                "content_plan_revision_id": rec.content_plan_revision_id,
                "exact_shot_revision_ids": list(rec.exact_shot_revision_ids),
                "approved_by": rec.approved_by,
                "approved_at": rec.approved_at.isoformat(),
                "user_note": rec.user_note,
            },
        )


@router.post(
    "/storyboards/{storyboard_snapshot_id}/route-plan",
    summary="Create an immutable asset route plan for an approved storyboard snapshot",
)
def create_asset_route_plan(
    storyboard_snapshot_id: str,
    body: ApiCreateAssetRoutePlanRequest | None = None,
):
    req_body = body or ApiCreateAssetRoutePlanRequest()
    with get_session() as session:
        service = AssetRoutePlanningService(
            plan_repository=ContentPlanRepository(session),
            shot_repository=ShotRepository(session),
            storyboard_repository=StoryboardRepository(session),
            route_plan_repository=AssetRoutePlanRepository(session),
        )
        input_data = CreateAssetRoutePlanInput(
            approved_storyboard_snapshot_id=storyboard_snapshot_id,
            routing_strategy=req_body.routing_strategy,
            model_selection_mode=req_body.model_selection_mode,
            selected_provider=req_body.selected_provider,
            selected_model=req_body.selected_model,
            target_aspect_ratio=req_body.target_aspect_ratio,
        )
        try:
            plan = service.create_route_plan(input_data)
            session.commit()
            return utils.get_response(200, plan.model_dump(mode="json"))
        except StoryboardNotFoundError as exc:
            raise HTTPException(status_code=404, detail=exc.message) from exc
        except (StoryboardNotApprovedError, ConfiguredModelNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc


@router.get(
    "/storyboards/{storyboard_snapshot_id}/route-plan",
    summary="Get the latest asset route plan for an approved storyboard snapshot",
)
def get_latest_asset_route_plan(storyboard_snapshot_id: str):
    with get_session() as session:
        repo = AssetRoutePlanRepository(session)
        plan = repo.get_latest_route_plan_for_snapshot(storyboard_snapshot_id)
        if plan is None:
            raise HTTPException(
                status_code=404,
                detail=f"No asset route plan found for storyboard '{storyboard_snapshot_id}'",
            )
        return utils.get_response(200, plan.model_dump(mode="json"))


@router.post(
    "/storyboards/create-from-topic",
    summary="One-click generation of ContentPlan and Storyboard from user topic",
)
def create_storyboard_from_topic(body: ApiCreateStoryboardFromTopicRequest):
    if body.source_grounded and not body.evidence_items:
        raise HTTPException(
            status_code=400,
            detail="source_grounded=True requires non-empty evidence_items.",
        )

    pipeline = StoryboardGenerationPipeline()
    evidence_items = tuple(
        EvidenceInputItem(
            evidence_id=item.evidence_id,
            content=item.content,
            title=item.title,
            url=item.url,
        )
        for item in body.evidence_items
    )
    inp = StoryboardGenerationInput(
        topic=body.topic,
        target_video_duration=body.target_video_duration,
        title=body.title,
        user_instruction=body.user_instruction,
        target_aspect_ratio=body.target_aspect_ratio,
        language=body.language,
        source_grounded=body.source_grounded,
        knowledge_context=tuple(body.knowledge_context),
        evidence_items=evidence_items,
    )
    try:
        result = pipeline.generate(inp)
        return utils.get_response(200, result.model_dump())
    except (PlannerError, StoryboardOrchestrationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Storyboard generation failed")
        raise HTTPException(status_code=500, detail="Storyboard generation failed.")


@router.post(
    "/content-plans",
    summary="Generate a ContentPlanRevision from user topic",
)
def create_content_plan(body: ApiCreateContentPlanRequest):
    if body.source_grounded and not body.evidence_items:
        raise HTTPException(
            status_code=400,
            detail="source_grounded=True requires non-empty evidence_items.",
        )

    evidence_ids = tuple(e.evidence_id for e in body.evidence_items)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise HTTPException(status_code=400, detail="evidence_id values must be unique.")
    evidence_snippets = [f"[{e.evidence_id}] {e.content}" for e in body.evidence_items]
    combined_knowledge = tuple(body.knowledge_context + evidence_snippets)

    with get_session() as session:
        repo = ContentPlanRepository(session)
        planner = ContentPlanner(repository=repo)
        inp = PlannerInput(
            topic=body.topic,
            target_video_duration=body.target_video_duration,
            title=body.title,
            user_instruction=body.user_instruction,
            knowledge_context=combined_knowledge,
            available_evidence_ids=evidence_ids,
            source_grounded=body.source_grounded,
        )
        try:
            plan = planner.plan(inp)
            session.commit()
            return utils.get_response(200, plan.model_dump())
        except PlannerError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/content-plans/{content_plan_revision_id}",
    summary="Get a ContentPlanRevision and its beats",
)
def get_content_plan(content_plan_revision_id: str):
    with get_session() as session:
        repo = ContentPlanRepository(session)
        plan = repo.get_revision(content_plan_revision_id)
        if plan is None:
            raise HTTPException(
                status_code=404,
                detail=f"ContentPlanRevision '{content_plan_revision_id}' not found",
            )
        return utils.get_response(200, plan.model_dump())


@router.post(
    "/storyboards/build-from-plan",
    summary="Build a draft StoryboardSnapshot from an existing ContentPlanRevision",
)
def build_storyboard_from_plan(body: ApiBuildStoryboardFromPlanRequest):
    with get_session() as session:
        plan_repo = ContentPlanRepository(session)
        shot_repo = ShotRepository(session)
        storyboard_repo = StoryboardRepository(session)
        agent = StoryboardAgent(
            shot_repository=shot_repo,
            storyboard_repository=storyboard_repo,
        )
        orchestrator = StoryboardOrchestrator(
            plan_repository=plan_repo,
            shot_repository=shot_repo,
            storyboard_repository=storyboard_repo,
            storyboard_agent=agent,
        )
        build_input = StoryboardBuildInput(
            new_content_plan_revision_id=body.content_plan_revision_id,
            user_instruction=body.user_instruction,
            target_aspect_ratio=body.target_aspect_ratio,
            language=body.language,
            source_grounded=body.source_grounded,
        )
        try:
            res = orchestrator.build_storyboard(build_input)
            session.commit()
            return utils.get_response(
                200,
                {
                    "storyboard_snapshot_id": res.storyboard_snapshot.storyboard_snapshot_id,
                    "content_plan_revision_id": res.storyboard_snapshot.content_plan_revision_id,
                    "shot_revision_ids": list(res.storyboard_snapshot.shot_revision_ids),
                    "snapshot_state": res.storyboard_snapshot.snapshot_state.value,
                    "statistics": res.statistics.model_dump(),
                },
            )
        except StoryboardOrchestrationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


def _run_storyboard_assembly_worker(
    task_id: str,
    storyboard_snapshot_id: str,
    execution_run_id: str | None,
    video_params: dict | None,
):
    return run_storyboard_assembly_task(
        task_id=task_id,
        storyboard_snapshot_id=storyboard_snapshot_id,
        execution_run_id=execution_run_id,
        video_params=video_params,
    )


@router.post(
    "/storyboards/{storyboard_snapshot_id}/synthesize-video",
    summary="Assemble and synthesize final MP4 video from storyboard shot assets",
)
def synthesize_storyboard_video(
    storyboard_snapshot_id: str,
    body: ApiSynthesizeVideoRequest | None = None,
):
    from app.controllers.v1.video import task_manager
    from app.services import task as tm

    req_body = body or ApiSynthesizeVideoRequest()
    task_id = req_body.task_id or utils.get_uuid()

    existing_task = sm.state.get_task(task_id)
    if existing_task and tm.is_task_busy(existing_task):
        raise HTTPException(
            status_code=409,
            detail=f"Task '{task_id}' is already running/busy.",
        )

    # Pre-register initial task state
    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_PROCESSING,
        progress=0,
    )

    try:
        task_manager.add_task(
            run_storyboard_assembly_task,
            task_id=task_id,
            storyboard_snapshot_id=storyboard_snapshot_id,
            execution_run_id=req_body.execution_run_id,
            video_params=req_body.video_params,
        )
    except TaskQueueFullError as exc:
        sm.state.delete_task(task_id)
        raise HTTPException(
            status_code=429,
            detail=f"Task queue is full: {exc}",
        ) from exc
    except Exception:
        logger.exception("Failed to queue storyboard video assembly task")
        sm.state.delete_task(task_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to queue video assembly task.",
        )

    return utils.get_response(
        200,
        {
            "task_id": task_id,
            "storyboard_snapshot_id": storyboard_snapshot_id,
            "execution_run_id": req_body.execution_run_id,
            "state": const.TASK_STATE_PROCESSING,
            "progress": 0,
        },
    )


@router.post(
    "/storyboards/route-plans/{asset_route_plan_id}/execute",
    summary="Execute an asset route plan as a background task",
)
def execute_asset_route_plan(
    asset_route_plan_id: str,
    body: ApiExecuteAssetRoutePlanRequest | None = None,
):
    from app.controllers.v1.video import task_manager
    from app.services import task as tm

    req_body = body or ApiExecuteAssetRoutePlanRequest()
    with get_session() as session:
        plan = AssetRoutePlanRepository(session).get_route_plan(asset_route_plan_id)
    if plan is None:
        raise HTTPException(
            status_code=404,
            detail=f"AssetRoutePlan '{asset_route_plan_id}' not found",
        )
    if not plan.is_ready:
        raise HTTPException(
            status_code=400,
            detail=f"AssetRoutePlan '{asset_route_plan_id}' is not READY",
        )

    task_id = req_body.task_id or utils.get_uuid()
    existing_task = sm.state.get_task(task_id)
    if existing_task and tm.is_task_busy(existing_task):
        raise HTTPException(
            status_code=409,
            detail=f"Task '{task_id}' is already running/busy.",
        )

    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_PROCESSING,
        progress=0,
        asset_route_plan_id=asset_route_plan_id,
    )
    try:
        task_manager.add_task(
            run_asset_route_plan_task,
            task_id=task_id,
            asset_route_plan_id=asset_route_plan_id,
            reuse_mode=req_body.reuse_mode.value,
        )
    except TaskQueueFullError as exc:
        sm.state.delete_task(task_id)
        raise HTTPException(status_code=429, detail="Task queue is full.") from exc
    except Exception:
        logger.exception("Failed to queue asset route plan execution")
        sm.state.delete_task(task_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to queue asset route plan execution.",
        )

    return utils.get_response(
        200,
        {
            "task_id": task_id,
            "asset_route_plan_id": asset_route_plan_id,
            "state": const.TASK_STATE_PROCESSING,
            "progress": 0,
        },
    )


@router.post(
    "/storyboards/execution-runs/{run_id}/retry",
    summary="Incrementally retry failed shots of an execution run, reusing succeeded shots",
)
def retry_execution_run(run_id: str, body: ApiRetryExecutionRunRequest | None = None):
    req_body = body or ApiRetryExecutionRunRequest()
    with get_session() as session:
        exec_repo = ExecutionRepository(session)
        run = exec_repo.get_execution_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"ExecutionRun '{run_id}' not found")
        plan_repo = AssetRoutePlanRepository(session)
        plan = plan_repo.get_route_plan(run.asset_route_plan_id)
        if plan is None:
            raise HTTPException(
                status_code=404,
                detail=f"AssetRoutePlan '{run.asset_route_plan_id}' not found",
            )

    service = AssetRoutePlanExecutionService()
    storage_dir = utils.storage_dir("asset_executions", create=True)
    try:
        res = service.execute_route_plan(
            plan=plan,
            storage_base_dir=storage_dir,
            reuse_mode=req_body.reuse_mode,
        )
        return utils.get_response(200, res.model_dump(mode="json"))
    except Exception:
        logger.exception(f"Execution retry failed for run {run_id}")
        raise HTTPException(status_code=500, detail="Execution retry failed.")


@router.post(
    "/storyboards/execution-runs/{run_id}/recover",
    summary="Recover unconfirmed attempts of an execution run",
)
def recover_execution_run(run_id: str, body: ApiRecoverExecutionRunRequest | None = None):
    req_body = body or ApiRecoverExecutionRunRequest()
    with get_session() as session:
        exec_repo = ExecutionRepository(session)
        run = exec_repo.get_execution_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"ExecutionRun '{run_id}' not found")
        service = AttemptRecoveryService()
        storage_dir = utils.storage_dir("asset_executions", create=True)
        try:
            res = service.recover_run(
                run_id=run_id,
                session=session,
                storage_base_dir=storage_dir,
                force_timeout=req_body.force_timeout,
            )
            session.commit()
            return utils.get_response(200, res)
        except Exception:
            logger.exception(f"Execution recovery failed for run {run_id}")
            raise HTTPException(status_code=500, detail="Execution recovery failed.")
