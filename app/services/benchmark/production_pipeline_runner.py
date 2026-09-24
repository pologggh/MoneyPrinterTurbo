from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.domain.asset_execution import ExecutionStatus, ShotAssetVersion
from app.domain.asset_router import (
    AssetCapability,
    AssetRoutePlanStatus,
)
from app.domain.benchmark import (
    BenchmarkCase,
    BenchmarkCaseStatus,
    BenchmarkFailureStage,
    BenchmarkProductionResultRefs,
    BenchmarkVariant,
    evaluate_benchmark_structural_constraints,
)
from app.domain.evaluation import (
    EvaluationDecision,
    create_evaluation_target_from_shot,
)
from app.domain.evaluation_policy import EvaluationPolicy
from app.domain.planner import ContentPlanner, PlannerInput
from app.domain.quality_remediation import (
    QualityRemediationAction,
    QualityRemediationPolicy,
)
from app.domain.shot import ShotRevision
from app.domain.storyboard_agent import StoryboardAgent
from app.domain.storyboard_approval import StoryboardApprovalService
from app.domain.storyboard_orchestrator import (
    StoryboardBuildInput,
    StoryboardOrchestrator,
)
from app.domain.trace import TraceContext
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    ContentPlanRepository,
    EvaluationRepository,
    ExecutionRepository,
    ShotRepository,
    StoryboardApprovalRepository,
    StoryboardRepository,
)
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.asset_capability_registry import AssetCapabilityRegistry
from app.services.asset_route_plan_execution_service import (
    AssetRoutePlanExecutionService,
)
from app.services.asset_route_planning_service import (
    AssetRoutePlanningService,
    CreateAssetRoutePlanInput,
)
from app.services.evaluation.mpt_multimodal_adapter import MultimodalEvaluatorAdapter
from app.services.evaluation.quality_remediation_service import (
    QualityRemediationService,
)
from app.services.evaluation.shot_asset_evaluation_service import (
    ShotAssetEvaluationService,
)
from app.services.shot_execution_service import ShotExecutionService
from app.services.trace_service import TraceWriter

logger = logging.getLogger(__name__)


class StaticProvider:
    """Helper provider returning a static tuple of AssetCapabilities."""

    def __init__(self, capabilities: tuple[AssetCapability, ...]) -> None:
        self._capabilities = capabilities

    def get_capabilities(self) -> tuple[AssetCapability, ...]:
        return self._capabilities


@dataclass(frozen=True)
class ProductionPipelineResult:
    """Result from running a BenchmarkCase through the production pipeline."""
    status: BenchmarkCaseStatus
    production_result_refs: BenchmarkProductionResultRefs
    failure_stage: BenchmarkFailureStage | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    attributes: dict[str, Any] | None = None


class ProductionPipelineRunner:
    """
    Focused adapter over existing production services:
    ContentPlanner -> StoryboardOrchestrator -> Structural Validation ->
    Benchmark Auto-Approval -> HybridAssetRouter -> ExecutionService ->
    ShotAssetEvaluationService -> QualityRemediationService.

    Benchmark code observes production behavior; it does not duplicate it.
    """

    def __init__(
        self,
        session_factory: sessionmaker,
    ) -> None:
        self.session_factory = session_factory

    def run(
        self,
        case: BenchmarkCase,
        variant: BenchmarkVariant,
        storage_base_dir: Path,
        trace_context: TraceContext,
        trace_writer: TraceWriter | None,
        planner_llm_caller: Callable[[str], str],
        storyboard_agent: StoryboardAgent,
        adapter_registry: AdapterRegistry,
        evaluator_adapter: MultimodalEvaluatorAdapter,
        capabilities: tuple[AssetCapability, ...],
    ) -> ProductionPipelineResult:
        """Executes a single benchmark case through the end-to-end production pipeline."""
        session: Session = self.session_factory()
        try:
            plan_repo = ContentPlanRepository(session)
            shot_repo = ShotRepository(session)
            sb_repo = StoryboardRepository(session)
            appr_repo = StoryboardApprovalRepository(session)
            route_plan_repo = AssetRoutePlanRepository(session)
            exec_repo = ExecutionRepository(session)
            eval_repo = EvaluationRepository(session)

            ctx = trace_context
            result_refs = BenchmarkProductionResultRefs()

            # -----------------------------------------------------------------
            # 1. Content Planning
            # -----------------------------------------------------------------
            try:
                planner = ContentPlanner(
                    repository=plan_repo,
                    llm_caller=planner_llm_caller,
                )
                evidence_id = case.knowledge_fixture_ref.fixture_id
                plan_rev = planner.plan(
                    PlannerInput(
                        topic=case.topic,
                        target_video_duration=case.target_duration,
                        available_evidence_ids=(evidence_id,),
                        user_instruction=case.user_instruction,
                    ),
                    trace_context=ctx,
                    trace_writer=trace_writer,
                )
                session.commit()
                result_refs = BenchmarkProductionResultRefs(
                    content_plan_revision_id=plan_rev.content_plan_revision_id
                )
                ctx = ctx.child_context(
                    parent_event_id=ctx.parent_event_id,
                    content_plan_revision_id=plan_rev.content_plan_revision_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Planning failure on case {case.case_key}: {exc}")
                return ProductionPipelineResult(
                    status=BenchmarkCaseStatus.FAILED,
                    production_result_refs=result_refs,
                    failure_stage=BenchmarkFailureStage.PLANNING,
                    failure_code="PLANNING_FAILED",
                    failure_message=str(exc),
                )

            # -----------------------------------------------------------------
            # 2. Storyboard Generation
            # -----------------------------------------------------------------
            try:
                sb_orchestrator = StoryboardOrchestrator(
                    plan_repository=plan_repo,
                    shot_repository=shot_repo,
                    storyboard_repository=sb_repo,
                    storyboard_agent=storyboard_agent,
                )
                build_res = sb_orchestrator.build_storyboard(
                    StoryboardBuildInput(new_content_plan_revision_id=plan_rev.content_plan_revision_id),
                    trace_context=ctx,
                    trace_writer=trace_writer,
                )
                session.commit()
                draft_snapshot = build_res.storyboard_snapshot
                ctx = ctx.child_context(
                    parent_event_id=ctx.parent_event_id,
                    storyboard_snapshot_id=draft_snapshot.storyboard_snapshot_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Storyboard failure on case {case.case_key}: {exc}")
                return ProductionPipelineResult(
                    status=BenchmarkCaseStatus.FAILED,
                    production_result_refs=result_refs,
                    failure_stage=BenchmarkFailureStage.STORYBOARD,
                    failure_code="STORYBOARD_GENERATION_FAILED",
                    failure_message=str(exc),
                )

            # -----------------------------------------------------------------
            # 3. Deterministic Structural Constraint Validation
            # -----------------------------------------------------------------
            is_valid, validation_errors = evaluate_benchmark_structural_constraints(
                plan_rev=plan_rev,
                storyboard_snapshot=draft_snapshot,
                constraints=case.expected_constraints,
                target_case_duration=case.target_duration,
            )
            if not is_valid:
                err_msg = "; ".join(validation_errors)
                logger.warning(f"Structural validation failed on case {case.case_key}: {err_msg}")
                return ProductionPipelineResult(
                    status=BenchmarkCaseStatus.FAILED,
                    production_result_refs=result_refs,
                    failure_stage=BenchmarkFailureStage.STRUCTURAL_VALIDATION,
                    failure_code="STRUCTURAL_CONSTRAINTS_VIOLATED",
                    failure_message=err_msg,
                )

            # -----------------------------------------------------------------
            # 4. Benchmark Auto-Approval (calls existing StoryboardApprovalService)
            # -----------------------------------------------------------------
            try:
                appr_service = StoryboardApprovalService(
                    plan_repository=plan_repo,
                    shot_repository=shot_repo,
                    storyboard_repository=sb_repo,
                    approval_repository=appr_repo,
                )
                approval_res = appr_service.approve_storyboard(
                    storyboard_snapshot_id=draft_snapshot.storyboard_snapshot_id,
                    trace_context=ctx,
                    trace_writer=trace_writer,
                )
                session.commit()
                approved_snapshot = approval_res.approved_snapshot
                result_refs = BenchmarkProductionResultRefs(
                    content_plan_revision_id=plan_rev.content_plan_revision_id,
                    approved_storyboard_snapshot_id=approved_snapshot.storyboard_snapshot_id,
                )
                ctx = ctx.child_context(
                    parent_event_id=ctx.parent_event_id,
                    storyboard_snapshot_id=approved_snapshot.storyboard_snapshot_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Approval failure on case {case.case_key}: {exc}")
                return ProductionPipelineResult(
                    status=BenchmarkCaseStatus.FAILED,
                    production_result_refs=result_refs,
                    failure_stage=BenchmarkFailureStage.APPROVAL,
                    failure_code="APPROVAL_FAILED",
                    failure_message=str(exc),
                )

            # -----------------------------------------------------------------
            # 5. Asset Route Planning
            # -----------------------------------------------------------------
            try:
                cap_registry = AssetCapabilityRegistry(providers=[StaticProvider(capabilities)])
                router_svc = AssetRoutePlanningService(
                    plan_repository=plan_repo,
                    shot_repository=shot_repo,
                    storyboard_repository=sb_repo,
                    route_plan_repository=route_plan_repo,
                    capability_registry=cap_registry,
                )
                route_plan = router_svc.create_route_plan(
                    CreateAssetRoutePlanInput(
                        approved_storyboard_snapshot_id=approved_snapshot.storyboard_snapshot_id,
                        routing_strategy=variant.routing_strategy,
                    ),
                    capabilities=capabilities,
                    trace_context=ctx,
                    trace_writer=trace_writer,
                )
                session.commit()
                if route_plan.status != AssetRoutePlanStatus.READY:
                    return ProductionPipelineResult(
                        status=BenchmarkCaseStatus.FAILED,
                        production_result_refs=result_refs,
                        failure_stage=BenchmarkFailureStage.ROUTING,
                        failure_code="ROUTE_PLAN_NOT_READY",
                        failure_message=f"Route plan status is {route_plan.status.value}",
                    )
                result_refs = BenchmarkProductionResultRefs(
                    content_plan_revision_id=plan_rev.content_plan_revision_id,
                    approved_storyboard_snapshot_id=approved_snapshot.storyboard_snapshot_id,
                    asset_route_plan_id=route_plan.asset_route_plan_id,
                )
                ctx = ctx.child_context(
                    parent_event_id=ctx.parent_event_id,
                    asset_route_plan_id=route_plan.asset_route_plan_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Routing failure on case {case.case_key}: {exc}")
                return ProductionPipelineResult(
                    status=BenchmarkCaseStatus.FAILED,
                    production_result_refs=result_refs,
                    failure_stage=BenchmarkFailureStage.ROUTING,
                    failure_code="ROUTING_FAILED",
                    failure_message=str(exc),
                )

            # -----------------------------------------------------------------
            # 6. Asset Execution
            # -----------------------------------------------------------------
            try:
                shot_exec_service = ShotExecutionService(
                    adapter_registry=adapter_registry,
                    max_retries_per_candidate=1,
                )
                plan_exec_service = AssetRoutePlanExecutionService(
                    session_factory=self.session_factory,
                    shot_execution_service=shot_exec_service,
                )
                run_res = plan_exec_service.execute_route_plan(
                    plan=route_plan,
                    storage_base_dir=storage_base_dir,
                    trace_context=ctx,
                    trace_writer=trace_writer,
                )
                session.commit()
                if run_res.state != ExecutionStatus.COMPLETED:
                    return ProductionPipelineResult(
                        status=BenchmarkCaseStatus.FAILED,
                        production_result_refs=result_refs,
                        failure_stage=BenchmarkFailureStage.EXECUTION,
                        failure_code="EXECUTION_RUN_NOT_COMPLETED",
                        failure_message=f"Execution run state is {run_res.state.value}",
                    )
                result_refs = BenchmarkProductionResultRefs(
                    content_plan_revision_id=plan_rev.content_plan_revision_id,
                    approved_storyboard_snapshot_id=approved_snapshot.storyboard_snapshot_id,
                    asset_route_plan_id=route_plan.asset_route_plan_id,
                    execution_run_id=run_res.execution_run_id,
                )
                ctx = ctx.child_context(
                    parent_event_id=ctx.parent_event_id,
                    execution_run_id=run_res.execution_run_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Execution failure on case {case.case_key}: {exc}")
                return ProductionPipelineResult(
                    status=BenchmarkCaseStatus.FAILED,
                    production_result_refs=result_refs,
                    failure_stage=BenchmarkFailureStage.EXECUTION,
                    failure_code="EXECUTION_FAILED",
                    failure_message=str(exc),
                )

            # -----------------------------------------------------------------
            # 7. Evaluation & Quality Remediation
            # -----------------------------------------------------------------
            try:
                eval_service = ShotAssetEvaluationService(session)
                remed_service = QualityRemediationService(
                    execution_service=shot_exec_service,
                    policy=QualityRemediationPolicy(),
                    storage_base_dir=storage_base_dir,
                )

                accepted_asset_ids: list[str] = []
                final_eval_snap_ids: list[str] = []
                remed_decision_ids: list[str] = []
                overall_case_status = BenchmarkCaseStatus.SUCCEEDED

                for shot_res in run_res.shot_results:
                    if not shot_res.asset_version_id:
                        continue
                    asset_ver = exec_repo.get_shot_asset_version(shot_res.asset_version_id)
                    shot_rev = shot_repo.get_revision(shot_res.shot_revision_id)
                    if not asset_ver or not shot_rev:
                        continue

                    # Create evaluation target
                    target = create_evaluation_target_from_shot(
                        shot_revision=shot_rev,
                        shot_asset_version=asset_ver,
                    )
                    eval_repo.save_target(target)
                    session.commit()

                    # Evaluate target
                    eval_snap = eval_service.evaluate_target(
                        evaluation_target_id=target.evaluation_target_id,
                        policy=EvaluationPolicy(),
                        evaluator_adapter=evaluator_adapter,
                        trace_context=ctx,
                        trace_writer=trace_writer,
                    )
                    session.commit()
                    final_eval_snap_ids.append(eval_snap.evaluation_snapshot_id)

                    if eval_snap.decision == EvaluationDecision.PASS:
                        accepted_asset_ids.append(asset_ver.shot_asset_version_id)
                    elif eval_snap.decision == EvaluationDecision.FAIL:
                        # Find corresponding route decision and beat
                        route_dec = next(
                            (r.route_decision for r in route_plan.shot_routes if r.shot_id == shot_res.shot_id),
                            None,
                        )
                        beat = next(
                            (b for b in plan_rev.beats if b.beat_lineage_id == shot_rev.beat_lineage_id),
                            None,
                        )
                        if route_dec and beat:
                            dim_results = eval_repo.list_dimension_results_for_target(target.evaluation_target_id)

                            def _remed_eval_fn(srev: ShotRevision, aver: ShotAssetVersion):
                                if exec_repo.get_shot_asset_version(aver.shot_asset_version_id) is None:
                                    exec_repo.add_shot_asset_version(aver)
                                    session.commit()
                                t = create_evaluation_target_from_shot(srev, aver)
                                eval_repo.save_target(t)
                                session.commit()
                                s = eval_service.evaluate_target(
                                    evaluation_target_id=t.evaluation_target_id,
                                    policy=EvaluationPolicy(),
                                    evaluator_adapter=evaluator_adapter,
                                    trace_context=ctx,
                                    trace_writer=trace_writer,
                                )
                                session.commit()
                                dims = eval_repo.list_dimension_results_for_target(t.evaluation_target_id)
                                return s, list(dims)

                            remed_res = remed_service.run_remediation_chain(
                                shot_revision=shot_rev,
                                initial_asset_version=asset_ver,
                                initial_snapshot=eval_snap,
                                initial_dimension_results=dim_results,
                                route_decision=route_dec,
                                beat=beat,
                                evaluation_fn=_remed_eval_fn,
                                trace_context=ctx,
                                trace_writer=trace_writer,
                            )
                            session.commit()
                            if remed_res.remediation_decisions:
                                for dec in remed_res.remediation_decisions:
                                    remed_decision_ids.append(dec.remediation_decision_id)
                            if remed_res.final_action == QualityRemediationAction.ACCEPT_ASSET and remed_res.accepted_asset_version:
                                accepted_asset_ids.append(remed_res.accepted_asset_version.shot_asset_version_id)
                            elif remed_res.final_action == QualityRemediationAction.NEEDS_USER_ACTION:
                                overall_case_status = BenchmarkCaseStatus.NEEDS_USER_ACTION
                            else:
                                overall_case_status = BenchmarkCaseStatus.FAILED
                        else:
                            overall_case_status = BenchmarkCaseStatus.FAILED
                    else:
                        # INDETERMINATE -> NEEDS_USER_ACTION
                        overall_case_status = BenchmarkCaseStatus.NEEDS_USER_ACTION

                result_refs = BenchmarkProductionResultRefs(
                    content_plan_revision_id=plan_rev.content_plan_revision_id,
                    approved_storyboard_snapshot_id=approved_snapshot.storyboard_snapshot_id,
                    asset_route_plan_id=route_plan.asset_route_plan_id,
                    execution_run_id=run_res.execution_run_id,
                    accepted_shot_asset_version_ids=tuple(accepted_asset_ids),
                    final_evaluation_snapshot_ids=tuple(final_eval_snap_ids),
                    remediation_decision_ids=tuple(remed_decision_ids),
                )

                return ProductionPipelineResult(
                    status=overall_case_status,
                    production_result_refs=result_refs,
                )

            except Exception as exc:
                logger.exception(f"Evaluation/Remediation failure on case {case.case_key}")
                return ProductionPipelineResult(
                    status=BenchmarkCaseStatus.FAILED,
                    production_result_refs=result_refs,
                    failure_stage=BenchmarkFailureStage.EVALUATION,
                    failure_code="EVALUATION_FAILED",
                    failure_message=str(exc),
                )

        finally:
            session.close()
