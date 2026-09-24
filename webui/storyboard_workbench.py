"""Storyboard Workbench UI for MoneyPrinterTurbo.

Phase 3.2 Human-in-the-loop Storyboard Workbench implemented in Streamlit.
Enables inspecting Beats, viewing and selecting Shots in deterministic order,
editing Shot planning fields, reordering within the same Beat, and refreshing
seamlessly to newly created immutable DRAFT snapshots with stale edit protection.
"""

import os
from datetime import datetime
from typing import Any

import streamlit as st
from sqlalchemy import select

from app.domain.asset_execution import (
    AssetReuseMode,
    ExecutionRunResult,
    ExecutionStatus,
)
from app.domain.asset_router import (
    AssetRoutePlan,
    ConfiguredModelNotFoundError,
    ModelSelectionMode,
    RoutingStrategy,
    ShotRoutePlanStatus,
    StoryboardNotApprovedError,
)
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.storyboard_approval import (
    ApprovalValidationError,
    ContentReplanRequiredError,
    InvalidSnapshotStateForApprovalError,
    StaleApprovalSnapshotError,
    StoryboardApprovalError,
    StoryboardApprovalRecord,
    StoryboardApprovalService,
    StoryboardBeatReplanError,
    StoryboardBeatReplanInput,
    StoryboardBeatReplanResult,
)
from app.domain.storyboard_editing import (
    CrossBeatMoveNotAllowedError,
    EditShotInput,
    InvalidShotEditError,
    InvalidShotOrderError,
    ReorderShotsInput,
    ShotEditingView,
    StaleShotRevisionError,
    StoryboardEditingService,
    StoryboardEditingView,
    StoryboardNotFoundError,
)
from app.persistence.models import ContentPlanRevisionORM, StoryboardSnapshotORM
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    ContentPlanRepository,
    ExecutionRepository,
    ShotRepository,
    StoryboardApprovalRepository,
    StoryboardRepository,
)
from app.persistence.session import get_session
from app.services.asset_capability_registry import get_default_capability_registry
from app.services.asset_route_planning_service import (
    AssetRoutePlanningService,
    CreateAssetRoutePlanInput,
)

# VisualType friendly Chinese labels mapping to exact domain enum
VISUAL_TYPE_LABELS: dict[VisualType, str] = {
    VisualType.STOCK_VIDEO: "素材视频 (STOCK_VIDEO)",
    VisualType.AI_VIDEO: "AI 视频 (AI_VIDEO)",
    VisualType.AI_IMAGE: "AI 图片 (AI_IMAGE)",
    VisualType.DIAGRAM: "图解 (DIAGRAM)",
    VisualType.SOURCE_ASSET: "资料原图 (SOURCE_ASSET)",
    VisualType.USER_ASSET: "用户素材 (USER_ASSET)",
}

BEAT_TYPE_LABELS: dict[BeatType, str] = {
    BeatType.HOOK: "🪝 钩子 (HOOK)",
    BeatType.KNOWLEDGE: "💡 核心知识 (KNOWLEDGE)",
    BeatType.EXAMPLE: "🔍 案例解析 (EXAMPLE)",
    BeatType.TRANSITION: "🔄 承上启下 (TRANSITION)",
    BeatType.SUMMARY: "🏁 行动总结 (SUMMARY)",
}


def _get_editing_service(session: Any) -> StoryboardEditingService:
    return StoryboardEditingService(
        plan_repository=ContentPlanRepository(session),
        shot_repository=ShotRepository(session),
        storyboard_repository=StoryboardRepository(session),
        execution_repository=ExecutionRepository(session),
    )


def _get_approval_service(
    session: Any, agent: Any | None = None
) -> StoryboardApprovalService:
    return StoryboardApprovalService(
        plan_repository=ContentPlanRepository(session),
        shot_repository=ShotRepository(session),
        storyboard_repository=StoryboardRepository(session),
        approval_repository=StoryboardApprovalRepository(session),
        agent=agent,
    )


def _get_approval_record_for_snapshot(
    snapshot_id: str,
) -> StoryboardApprovalRecord | None:
    """Retrieve audit record for an approved storyboard snapshot or its source draft."""
    try:
        with get_session() as session:
            repo = StoryboardApprovalRepository(session)
            rec = repo.get_approval_by_approved_snapshot_id(snapshot_id)
            if not rec:
                rec = repo.get_approval_by_source_draft_id(snapshot_id)
            return rec
    except Exception:  # noqa: BLE001
        return None



def list_available_storyboard_snapshots(limit: int = 30) -> list[dict[str, Any]]:
    """Query recent Storyboard snapshots with plan topic for workbench selection."""
    try:
        with get_session() as session:
            stmt = (
                select(
                    StoryboardSnapshotORM.storyboard_snapshot_id,
                    StoryboardSnapshotORM.snapshot_state,
                    StoryboardSnapshotORM.created_at,
                    ContentPlanRevisionORM.topic,
                )
                .join(
                    ContentPlanRevisionORM,
                    StoryboardSnapshotORM.content_plan_revision_id
                    == ContentPlanRevisionORM.content_plan_revision_id,
                    isouter=True,
                )
                .order_by(StoryboardSnapshotORM.created_at.desc())
                .limit(limit)
            )
            rows = session.execute(stmt).all()
            results = []
            for row in rows:
                snap_id, state, created_at, topic = row
                topic_str = topic or "未命名主题"
                created_str = (
                    created_at.strftime("%Y-%m-%d %H:%M")
                    if isinstance(created_at, datetime)
                    else str(created_at or "-")
                )
                state_str = "草稿" if state == "DRAFT" else "已确认"
                label = f"[{state_str}] {topic_str} ({snap_id[:8]}...) - {created_str}"
                results.append(
                    {
                        "storyboard_snapshot_id": snap_id,
                        "label": label,
                        "state": state,
                        "topic": topic_str,
                    }
                )
            return results
    except Exception:  # noqa: BLE001
        return []


def load_storyboard_view(snapshot_id: str) -> StoryboardEditingView | None:
    """Load UI-ready StoryboardEditingView for the given snapshot ID."""
    if not snapshot_id:
        return None
    try:
        with get_session() as session:
            service = _get_editing_service(session)
            return service.get_storyboard_for_editing(snapshot_id)
    except StoryboardNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"加载分镜数据失败: {exc}")
        return None


def _handle_approve_storyboard(
    view: StoryboardEditingView,
    approved_by: str = "local_user",
    user_note: str | None = None,
) -> StoryboardApprovalRecord | None:
    """Commit human approval on a DRAFT storyboard snapshot."""
    try:
        with get_session() as session:
            service = _get_approval_service(session)
            record = service.approve_storyboard(
                storyboard_snapshot_id=view.storyboard_snapshot_id,
                approved_by=approved_by or "local_user",
                user_note=user_note or None,
            )
            st.session_state["sb_workbench_current_snapshot_id"] = (
                record.approved_storyboard_snapshot_id
            )
            st.session_state["sb_workbench_selected_shot_id"] = None
            st.toast("分镜方案已成功确认！已生成正式批准快照", icon="✅")
            st.rerun()
            return record
    except InvalidSnapshotStateForApprovalError as exc:
        st.error(f"❌ 无法批准: {exc.message}")
    except ApprovalValidationError as exc:
        st.error(f"❌ 校验失败: {exc.message}")
    except StaleApprovalSnapshotError as exc:
        st.error(f"⚠️ 状态冲突: {exc.message}")
    except StoryboardApprovalError as exc:
        st.error(f"❌ 批准失败: {exc.message}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ 操作异常: {exc}")
    return None


def _handle_replan_beat(
    view: StoryboardEditingView,
    beat_lineage_id: str,
    user_instruction: str | None = None,
    agent: Any | None = None,
) -> StoryboardBeatReplanResult | None:
    """Execute controlled agent replan for a single beat."""
    try:
        with get_session() as session:
            service = _get_approval_service(session, agent=agent)
            replan_input = StoryboardBeatReplanInput(
                base_storyboard_snapshot_id=view.storyboard_snapshot_id,
                target_beat_lineage_id=beat_lineage_id,
                user_instruction=user_instruction or None,
            )
            result = service.replan_beat(replan_input)
            st.session_state["sb_workbench_current_snapshot_id"] = (
                result.new_storyboard_snapshot_id
            )
            st.session_state["sb_workbench_selected_shot_id"] = None
            st.toast(
                f"本段分镜已重新规划完成，生成 {result.generated_shot_count} 个新镜头",
                icon="✅",
            )
            st.rerun()
            return result
    except ContentReplanRequiredError as exc:
        st.warning(
            f"⚠️ **需要重新规划内容方案 (CONTENT_REPLAN_REQUIRED)**: {exc.message}\n\n"
            "修改意图涉及整体主题或内容结构调整，不能仅在单个分镜段中重新规划。"
        )
    except StoryboardBeatReplanError as exc:
        st.error(f"❌ 重新规划失败: {exc.message}")
    except StoryboardApprovalError as exc:
        st.error(f"❌ 操作失败: {exc.message}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ 重新规划异常: {exc}")
    return None


def _get_route_plan_for_snapshot(snapshot_id: str) -> AssetRoutePlan | None:
    """Load latest asset route plan for a snapshot."""
    try:
        with get_session() as session:
            repo = AssetRoutePlanRepository(session)
            return repo.get_latest_route_plan_for_snapshot(snapshot_id)
    except Exception:  # noqa: BLE001
        return None


def _handle_create_route_plan(
    view: StoryboardEditingView,
    strategy: RoutingStrategy,
    mode: ModelSelectionMode,
    selected_provider: str | None,
    selected_model: str | None,
) -> AssetRoutePlan | None:
    """Execute asset route planning for an approved storyboard snapshot."""
    try:
        with get_session() as session:
            service = AssetRoutePlanningService(
                plan_repository=ContentPlanRepository(session),
                shot_repository=ShotRepository(session),
                storyboard_repository=StoryboardRepository(session),
                route_plan_repository=AssetRoutePlanRepository(session),
            )
            input_data = CreateAssetRoutePlanInput(
                approved_storyboard_snapshot_id=view.storyboard_snapshot_id,
                routing_strategy=strategy,
                model_selection_mode=mode,
                selected_provider=selected_provider or None,
                selected_model=selected_model or None,
            )
            plan = service.create_route_plan(input_data)
            session.commit()
            if plan.is_ready:
                st.toast("✅ 素材路线规划完成，所有分镜已就绪！", icon="🚀")
            else:
                st.toast(
                    f"⚠️ 素材路线规划受阻：{plan.blocked_shots} 个分镜无可用渠道",
                    icon="⚠️",
                )
            st.rerun()
            return plan
    except (StoryboardNotApprovedError, ConfiguredModelNotFoundError) as exc:
        st.error(f"❌ 路线规划错误: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ 路线规划异常: {exc}")
    return None


def _handle_execute_route_plan(
    route_plan: AssetRoutePlan,
    reuse_mode: AssetReuseMode = AssetReuseMode.REUSE_COMPATIBLE,
) -> ExecutionRunResult | None:
    """Execute asset route plan for the approved snapshot."""
    try:
        from app.services.asset_route_plan_execution_service import (
            AssetRoutePlanExecutionService,
        )

        service = AssetRoutePlanExecutionService()
        result = service.execute_route_plan(
            plan=route_plan,
            storage_base_dir="storage",
            reuse_mode=reuse_mode,
        )
        st.session_state[f"last_exec_result_{route_plan.asset_route_plan_id}"] = result
        st.toast(
            f"素材生成完成: {result.state.value} (复用: {result.reused_shots}, 生成: {result.generated_shots}, 失败: {result.failed_shots})",
            icon="🎉" if result.state == ExecutionStatus.COMPLETED else "⚠️",
        )
        st.rerun()
        return result
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ 素材生成异常: {exc}")
        return None


def _handle_recover_run(run_id: str) -> None:
    """Execute attempt recovery for uncertain submissions."""
    try:
        from app.services.attempt_recovery_service import AttemptRecoveryService

        with get_session() as session:
            service = AttemptRecoveryService()
            res = service.recover_run(
                run_id=run_id,
                session=session,
                storage_base_dir="storage",
            )
            session.commit()
            st.toast(
                f"故障恢复完成: 解决 {res['resolved_count']} 个镜头，状态: {res['overall_status']}",
                icon="🛠️",
            )
            st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ 恢复执行异常: {exc}")



def _render_top_header(
    view: StoryboardEditingView, available_snapshots: list[dict[str, Any]]
):
    """Render top header containing topic, status badges, approval actions, and snapshot switcher."""
    with st.container(border=True):
        col_meta, col_action, col_switcher = st.columns(
            [2.4, 1.2, 1.4], vertical_alignment="center"
        )

        with col_meta:
            state_badge = (
                "🟢 **已确认 (APPROVED)**"
                if view.state == StoryboardSnapshotState.APPROVED
                else "🔵 **草稿 (DRAFT)**"
            )
            st.markdown(
                f"### 🎬 主题：{view.topic}\n"
                f"{state_badge} &nbsp;|&nbsp; "
                f"**快照 ID**: `{view.storyboard_snapshot_id[:8]}...` &nbsp;|&nbsp; "
                f"**总分镜数**: `{view.total_shots}` 个 &nbsp;|&nbsp; "
                f"**规划时长**: `{view.total_duration:.1f}s` / 目标 `{view.overall_target_duration:.1f}s`"
            )

        with col_action:
            if view.state == StoryboardSnapshotState.DRAFT:
                with st.popover("✅ 确认分镜", use_container_width=True):
                    st.markdown("##### 确认分镜方案")
                    st.caption(
                        "确认后将生成正式的已批准版本快照，分镜序列将被严格冻结。"
                        "后续如需修改将自动创建新草稿。"
                    )
                    appr_by = st.text_input(
                        "确认人",
                        value="local_user",
                        key=f"sb_appr_by_{view.storyboard_snapshot_id}",
                    )
                    appr_note = st.text_input(
                        "批准备注（可选）",
                        value="",
                        key=f"sb_appr_note_{view.storyboard_snapshot_id}",
                    )
                    if st.button(
                        "确认批准",
                        type="primary",
                        key=f"sb_btn_confirm_appr_{view.storyboard_snapshot_id}",
                        use_container_width=True,
                    ):
                        _handle_approve_storyboard(view, appr_by, appr_note)
            else:
                with st.popover("🎯 规划素材路线", use_container_width=True):
                    st.markdown("##### 规划素材生成路线")
                    st.caption("为当前已批准分镜快照中的所有镜头计算最优生成/检索渠道（不发起真实扣费生成）。")
                    strat_choice = st.selectbox(
                        "路由策略",
                        options=[
                            RoutingStrategy.BALANCED,
                            RoutingStrategy.QUALITY_FIRST,
                            RoutingStrategy.COST_FIRST,
                        ],
                        format_func=lambda s: {
                            RoutingStrategy.BALANCED: "⚖️ 平衡模式 (BALANCED)",
                            RoutingStrategy.QUALITY_FIRST: "💎 画质优先 (QUALITY_FIRST)",
                            RoutingStrategy.COST_FIRST: "💰 成本优先 (COST_FIRST)",
                        }.get(s, s.value),
                        key=f"sb_route_strat_{view.storyboard_snapshot_id}",
                    )
                    mode_choice = st.selectbox(
                        "模型选择模式",
                        options=[
                            ModelSelectionMode.AUTO,
                            ModelSelectionMode.PINNED,
                            ModelSelectionMode.PREFERRED,
                        ],
                        format_func=lambda m: {
                            ModelSelectionMode.AUTO: "🔄 自动适配 (AUTO)",
                            ModelSelectionMode.PINNED: "🔒 锁定指定 (PINNED)",
                            ModelSelectionMode.PREFERRED: "⭐ 优先指定 (PREFERRED)",
                        }.get(m, m.value),
                        key=f"sb_route_mode_{view.storyboard_snapshot_id}",
                    )
                    chosen_p = None
                    chosen_m = None
                    if mode_choice in (
                        ModelSelectionMode.PINNED,
                        ModelSelectionMode.PREFERRED,
                    ):
                        reg = get_default_capability_registry()
                        caps = [c for c in reg.list_capabilities() if c.enabled]
                        opts = [f"{c.provider} :: {c.model}" for c in caps]
                        if opts:
                            ch = st.selectbox(
                                "选择目标模型",
                                options=opts,
                                key=f"sb_route_chosen_m_{view.storyboard_snapshot_id}",
                            )
                            parts = [x.strip() for x in ch.split("::", 1)]
                            chosen_p, chosen_m = parts[0], parts[1]
                    if st.button(
                        "开始路线规划",
                        type="primary",
                        use_container_width=True,
                        key=f"sb_btn_plan_route_{view.storyboard_snapshot_id}",
                    ):
                        _handle_create_route_plan(
                            view, strat_choice, mode_choice, chosen_p, chosen_m
                        )

        with col_switcher:
            options = [s["storyboard_snapshot_id"] for s in available_snapshots]
            labels_map = {
                s["storyboard_snapshot_id"]: s["label"] for s in available_snapshots
            }
            if view.storyboard_snapshot_id not in options:
                options.insert(0, view.storyboard_snapshot_id)
                labels_map[view.storyboard_snapshot_id] = (
                    f"当前快照: {view.storyboard_snapshot_id[:8]}... ({view.state.value})"
                )

            current_idx = (
                options.index(view.storyboard_snapshot_id)
                if view.storyboard_snapshot_id in options
                else 0
            )

            selected_id = st.selectbox(
                "切换分镜快照版本",
                options=options,
                index=current_idx,
                format_func=lambda sid: labels_map.get(sid, sid),
                key="sb_workbench_snapshot_selector",
                label_visibility="collapsed",
            )
            if selected_id and selected_id != view.storyboard_snapshot_id:
                st.session_state["sb_workbench_current_snapshot_id"] = selected_id
                st.session_state["sb_workbench_selected_shot_id"] = None
                st.rerun()

        # If approved, show persistent approval audit banner and route plan status
        if view.state == StoryboardSnapshotState.APPROVED:
            rec = _get_approval_record_for_snapshot(view.storyboard_snapshot_id)
            if rec:
                appr_time = (
                    rec.approved_at.strftime("%Y-%m-%d %H:%M:%S")
                    if isinstance(rec.approved_at, datetime)
                    else str(rec.approved_at)
                )
                note_suffix = (
                    f" &nbsp;|&nbsp; 📝 备注: {rec.user_note}" if rec.user_note else ""
                )
                st.info(
                    f"📋 **审批记录**: 确认人: `{rec.approved_by}` &nbsp;|&nbsp; "
                    f"时间: `{appr_time}` &nbsp;|&nbsp; "
                    f"源草稿: `{rec.source_draft_snapshot_id[:8]}...`{note_suffix}"
                )

            route_plan = _get_route_plan_for_snapshot(view.storyboard_snapshot_id)
            if route_plan:
                if route_plan.is_ready:
                    col_p_status, col_p_btn = st.columns([3, 1], vertical_alignment="center")
                    with col_p_status:
                        st.success(
                            f"🚀 **素材路线规划就绪 (READY)**: 全部 `{route_plan.total_shots}` 个分镜已规划候选渠道 &nbsp;|&nbsp; "
                            f"策略: `{route_plan.routing_strategy.value}` &nbsp;|&nbsp; "
                            f"模式: `{route_plan.model_selection_mode.value}`"
                        )
                    with col_p_btn, st.popover("🚀 执行素材生成", use_container_width=True):
                        st.markdown("##### 开始生成素材")
                        st.caption("按冻结路线执行素材生成。支持自动检测并复用已有兼容素材。")
                        reuse_mode_choice = st.radio(
                            "复用模式",
                            options=[
                                AssetReuseMode.REUSE_COMPATIBLE,
                                AssetReuseMode.FORCE_REGENERATE,
                            ],
                            format_func=lambda m: (
                                "♻️ 兼容素材优先复用 (增量模式)"
                                if m == AssetReuseMode.REUSE_COMPATIBLE
                                else "🔄 全部强制重新生成"
                            ),
                            key=f"sb_reuse_mode_{route_plan.asset_route_plan_id}",
                        )
                        if st.button(
                            "确认生成",
                            type="primary",
                            use_container_width=True,
                            key=f"sb_btn_exec_plan_{route_plan.asset_route_plan_id}",
                        ):
                            _handle_execute_route_plan(route_plan, reuse_mode_choice)

                    last_res: ExecutionRunResult | None = st.session_state.get(
                        f"last_exec_result_{route_plan.asset_route_plan_id}"
                    )
                    if last_res:
                        state_icon = "🟢" if last_res.state == ExecutionStatus.COMPLETED else ("🔴" if last_res.state == ExecutionStatus.FAILED else "🟠")
                        st.info(
                            f"{state_icon} **执行运行结果**: 状态 `{last_res.state.value}` &nbsp;|&nbsp; "
                            f"总镜头: `{last_res.total_shots}` &nbsp;|&nbsp; "
                            f"增量复用: `{last_res.reused_shots}` &nbsp;|&nbsp; "
                            f"全新生成: `{last_res.generated_shots}` &nbsp;|&nbsp; "
                            f"失败: `{last_res.failed_shots}` &nbsp;|&nbsp; "
                            f"待恢复: `{last_res.recovery_required_shots}`"
                        )
                        if last_res.failed_shots > 0 or last_res.state == ExecutionStatus.FAILED:
                            col_ret, col_rec = st.columns(2)
                            with col_ret:
                                if st.button(
                                    "🔄 增量重试失败分镜",
                                    key=f"sb_btn_retry_{last_res.execution_run_id}",
                                    type="primary",
                                    use_container_width=True,
                                ):
                                    _handle_execute_route_plan(route_plan, AssetReuseMode.REUSE_COMPATIBLE)
                            with col_rec:
                                if (
                                    last_res.recovery_required_shots > 0
                                    or last_res.state == ExecutionStatus.NEEDS_RECOVERY
                                ) and st.button(
                                    "🛠️ 执行故障恢复",
                                    key=f"sb_btn_rec_{last_res.execution_run_id}",
                                    use_container_width=True,
                                ):
                                    _handle_recover_run(last_res.execution_run_id)
                else:
                    blocked_info = [
                        f"镜头 `{e.shot_id[:6]}`: {', '.join(r.value for r in e.route_decision.decision_reason_codes)}"
                        for e in route_plan.shot_routes
                        if e.route_status == ShotRoutePlanStatus.BLOCKED
                    ]
                    st.warning(
                        f"⚠️ **素材路线规划受阻 (BLOCKED)**: `{route_plan.blocked_shots}` / `{route_plan.total_shots}` 个分镜暂无可用渠道 &nbsp;|&nbsp; "
                        + "；".join(blocked_info[:3])
                    )



def _render_beats_panel(view: StoryboardEditingView):
    """Left column: Display Content Plan Beats with single-beat replan trigger."""
    st.markdown("#### 📋 内容规划 (Beats)")
    st.caption("分镜所属内容大纲与叙事意图（只读，支持单段智能重新规划）")

    for beat in view.beats:
        with st.container(border=True):
            type_label = BEAT_TYPE_LABELS.get(beat.beat_type, beat.beat_type.value)
            col_b_head, col_b_action = st.columns(
                [2.2, 1.8], vertical_alignment="center"
            )
            with col_b_head:
                st.markdown(
                    f"**[{beat.order}] {type_label}**\n"
                    f"`{beat.target_duration:.1f}s` (权重: `{beat.importance:.1f}`)"
                )
            with col_b_action, st.popover("🔄 重新规划", use_container_width=True):
                st.markdown(f"##### 重新规划 Beat {beat.order}")
                st.caption(
                    "AI将仅为此段重新规划生成全新镜头，其余段落保持冻结不变。"
                    "完成后将自动生成新草稿快照。"
                )
                user_inst = st.text_area(
                    "规划引导指令 (可选)",
                    placeholder="例如：用图解方式清晰表达...",
                    key=f"sb_replan_inst_{beat.beat_lineage_id}_{view.storyboard_snapshot_id}",
                    height=70,
                )
                if st.button(
                    "🚀 开始重新规划",
                    type="primary",
                    key=f"sb_btn_replan_{beat.beat_lineage_id}_{view.storyboard_snapshot_id}",
                    use_container_width=True,
                ):
                    _handle_replan_beat(view, beat.beat_lineage_id, user_inst)

            st.markdown(f"**🎯 意图**: {beat.intent}")
            st.caption(f"包含 {len(beat.shots)} 个分镜镜头")



def _render_shots_panel(
    view: StoryboardEditingView, selected_shot_id: str | None
) -> str | None:
    """Center column: Display Shots grouped by Beat in deterministic order."""
    st.markdown("#### 🎞️ 分镜列表 (Shots)")
    st.caption("按 Beat 顺序与镜头局部序号排列，支持镜头同 Beat 顺序微调")

    new_selected_id = selected_shot_id

    for beat in view.beats:
        st.markdown(
            f"##### 📌 Beat {beat.order}: {beat.intent[:28]}{'...' if len(beat.intent) > 28 else ''}"
        )
        total_in_beat = len(beat.shots)

        for shot in beat.shots:
            is_selected = shot.shot_id == selected_shot_id
            # Render card with visual focus if selected
            with st.container(border=True):
                col_info, col_rev = st.columns([3.5, 1.5])
                with col_info:
                    visual_name = VISUAL_TYPE_LABELS.get(
                        shot.visual_type, shot.visual_type.value
                    ).split(" ")[0]
                    selection_marker = "👉 **[已选中]** " if is_selected else ""
                    st.markdown(
                        f"{selection_marker}**镜头 #{shot.local_order}** &nbsp;|&nbsp; "
                        f"🏷️ `{visual_name}` &nbsp;|&nbsp; "
                        f"⏱️ `{shot.target_duration:.1f}s`"
                    )
                with col_rev:
                    st.caption(
                        f"Rev {shot.revision_number} (`{shot.shot_revision_id[:6]}`)"
                    )

                st.markdown(f"🗣️ **旁白**: {shot.narration}")
                if shot.visual_goal:
                    st.caption(f"🎯 视觉目标: {shot.visual_goal}")
                if getattr(shot, "produced_asset_version_id", None):
                    dur_str = f"{shot.asset_duration:.1f}s" if shot.asset_duration else ""
                    dim_str = f"{shot.asset_width}x{shot.asset_height}" if shot.asset_width and shot.asset_height else ""
                    media_str = f"{shot.asset_media_type or '素材'}"
                    badge_parts = [p for p in [media_str, dur_str, dim_str] if p]
                    badge_detail = f" ({', '.join(badge_parts)})" if badge_parts else ""
                    st.success(f"🎨 **已就绪素材**{badge_detail}: `{shot.asset_file_path}`")

                # Actions row
                col_btn_edit, col_btn_up, col_btn_down = st.columns([1.5, 1.0, 1.0])
                with col_btn_edit:
                    if st.button(
                        "✏️ 编辑" if not is_selected else "✅ 正在编辑",
                        key=f"sb_btn_select_{shot.shot_id}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True,
                    ):
                        new_selected_id = shot.shot_id
                        st.session_state["sb_workbench_selected_shot_id"] = shot.shot_id
                        st.rerun()

                with col_btn_up:
                    is_first = shot.local_order <= 1
                    if st.button(
                        "↑ 上移",
                        key=f"sb_btn_up_{shot.shot_id}",
                        disabled=is_first,
                        use_container_width=True,
                    ):
                        _handle_reorder_shot(view, beat.beat_lineage_id, shot.shot_id, -1)

                with col_btn_down:
                    is_last = shot.local_order >= total_in_beat
                    if st.button(
                        "↓ 下移",
                        key=f"sb_btn_down_{shot.shot_id}",
                        disabled=is_last,
                        use_container_width=True,
                    ):
                        _handle_reorder_shot(view, beat.beat_lineage_id, shot.shot_id, 1)

    return new_selected_id


def _handle_reorder_shot(
    view: StoryboardEditingView,
    beat_lineage_id: str,
    shot_id: str,
    direction: int,
):
    """Reorder a shot within its beat by swapping with adjacent neighbor."""
    target_beat = next(
        (b for b in view.beats if b.beat_lineage_id == beat_lineage_id), None
    )
    if not target_beat:
        st.error("未找到对应 Beat")
        return

    shot_ids = [s.shot_id for s in target_beat.shots]
    if shot_id not in shot_ids:
        st.error("未找到对应 Shot")
        return

    idx = shot_ids.index(shot_id)
    new_idx = idx + direction
    if 0 <= new_idx < len(shot_ids):
        shot_ids[idx], shot_ids[new_idx] = shot_ids[new_idx], shot_ids[idx]
        try:
            with get_session() as session:
                service = _get_editing_service(session)
                reorder_input = ReorderShotsInput(
                    storyboard_snapshot_id=view.storyboard_snapshot_id,
                    beat_lineage_id=beat_lineage_id,
                    ordered_shot_ids=tuple(shot_ids),
                )
                new_snap = service.reorder_shots_within_beat(reorder_input)
                st.session_state["sb_workbench_current_snapshot_id"] = (
                    new_snap.storyboard_snapshot_id
                )
                st.toast("分镜顺序调整成功，已生成新草稿版本", icon="✅")
                st.rerun()
        except (CrossBeatMoveNotAllowedError, InvalidShotOrderError) as exc:
            st.error(f"排序失败: {exc.message}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"排序操作异常: {exc}")


def _render_shot_editor(
    view: StoryboardEditingView, selected_shot: ShotEditingView | None
):
    """Right column: Form for editing selected Shot's planning fields."""
    st.markdown("#### ✏️ 分镜详情与编辑")

    if not selected_shot:
        st.info("👈 请在中间分镜列表中点击“✏️ 编辑”选择一个分镜进行修改")
        return

    with st.container(border=True):
        st.markdown(
            f"**正在编辑**: 镜头 #{selected_shot.local_order} &nbsp;|&nbsp; "
            f"ID: `{selected_shot.shot_id[:8]}...`"
        )
        st.caption(
            f"基准版本: Rev {selected_shot.revision_number} (`{selected_shot.shot_revision_id[:8]}`)"
        )

        with st.form(
            key=f"sb_shot_edit_form_{selected_shot.shot_id}_{selected_shot.shot_revision_id}"
        ):
            narration = st.text_area(
                "旁白内容",
                value=selected_shot.narration,
                height=110,
                help="该镜头的画外音或讲授旁白内容",
            )

            col_dur, col_vis = st.columns([1.2, 1.8])
            with col_dur:
                duration = st.number_input(
                    "目标时长 (秒)",
                    value=float(selected_shot.target_duration),
                    min_value=0.5,
                    max_value=300.0,
                    step=0.5,
                    format="%.1f",
                    help="该分镜镜头的计划持续时间",
                )

            with col_vis:
                visual_type_keys = list(VISUAL_TYPE_LABELS.keys())
                current_vis_idx = (
                    visual_type_keys.index(selected_shot.visual_type)
                    if selected_shot.visual_type in visual_type_keys
                    else 0
                )
                visual_type = st.selectbox(
                    "视觉类型",
                    options=visual_type_keys,
                    index=current_vis_idx,
                    format_func=lambda vt: VISUAL_TYPE_LABELS[vt],
                    help="镜头所属视觉规划类型",
                )

            visual_goal = st.text_input(
                "视觉目标",
                value=selected_shot.visual_goal,
                help="该分镜的核心视觉呈现目的",
            )

            scene_description = st.text_area(
                "场景描述",
                value=selected_shot.scene_description,
                height=80,
                help="镜头画面构图、主体和视觉场景规划描述",
            )

            generation_prompt = st.text_area(
                "生成提示词",
                value=selected_shot.generation_prompt,
                height=80,
                help="用于下游素材生成或检索的提示词描述",
            )

            camera_movement = st.text_input(
                "镜头运动",
                value=selected_shot.camera_movement,
                help="镜头运镜方式，如 Static、Pan Left、Zoom In 等",
            )

            submitted = st.form_submit_button(
                "💾 保存修改",
                type="primary",
                use_container_width=True,
            )

            if submitted:
                _handle_save_shot_edit(
                    view=view,
                    shot_id=selected_shot.shot_id,
                    base_shot_revision_id=selected_shot.shot_revision_id,
                    narration=narration,
                    target_duration=duration,
                    visual_type=visual_type,
                    visual_goal=visual_goal,
                    scene_description=scene_description,
                    generation_prompt=generation_prompt,
                    camera_movement=camera_movement,
                )


def _handle_save_shot_edit(
    view: StoryboardEditingView,
    shot_id: str,
    base_shot_revision_id: str,
    narration: str,
    target_duration: float,
    visual_type: VisualType,
    visual_goal: str,
    scene_description: str,
    generation_prompt: str,
    camera_movement: str,
):
    """Validate and persist Shot edit, handling optimistic stale revision protection."""
    edit_input = EditShotInput(
        storyboard_snapshot_id=view.storyboard_snapshot_id,
        shot_id=shot_id,
        base_shot_revision_id=base_shot_revision_id,
        narration=narration,
        target_duration=target_duration,
        visual_type=visual_type,
        visual_goal=visual_goal,
        scene_description=scene_description,
        generation_prompt=generation_prompt,
        camera_movement=camera_movement,
    )

    try:
        with get_session() as session:
            service = _get_editing_service(session)
            new_snap = service.edit_shot(edit_input)
            st.session_state["sb_workbench_current_snapshot_id"] = (
                new_snap.storyboard_snapshot_id
            )
            st.session_state["sb_workbench_selected_shot_id"] = shot_id
            st.toast("分镜修改保存成功，已生成新草稿版本", icon="✅")
            st.rerun()

    except StaleShotRevisionError:
        st.error(
            "⚠️ **版本冲突 (STALE_SHOT_REVISION)**: 该分镜已被其他操作修改！\n\n"
            "为防止误覆盖，系统未应用此次提交。请确认最新分镜内容后再进行修改。"
        )
    except InvalidShotEditError as exc:
        st.error(f"❌ **参数校验失败**: {exc.message}")
    except StoryboardNotFoundError as exc:
        st.error(f"❌ **未找到对应分镜快照**: {exc.message}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ **保存失败**: {exc}")


def _render_creation_card(available_snapshots: list[dict[str, Any]]):
    """Render topic-to-storyboard generation panel."""
    with st.expander("✨ 新建分镜方案 (Create New Storyboard from Topic)", expanded=not bool(available_snapshots)):
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            new_topic = st.text_input(
                "视频主题 / 关键词 (Topic)",
                placeholder="例如：量子力学简史、AI 智能体工作原理、咖啡豆烘焙工艺...",
                key="sb_input_topic",
            )
        with col_t2:
            new_duration = st.number_input(
                "目标时长 (秒)",
                min_value=15,
                max_value=300,
                value=60,
                step=15,
                key="sb_input_dur",
            )

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            new_aspect = st.selectbox(
                "画面比例",
                options=["16:9 (横屏)", "9:16 (竖屏)"],
                index=0,
                key="sb_input_aspect",
            )
        with col_c2:
            new_lang = st.selectbox(
                "语言 (Language)",
                options=["zh", "en"],
                index=0,
                key="sb_input_lang",
            )

        new_instruction = st.text_area(
            "创意指导 / 补充说明 (可选)",
            placeholder="例如：注重底层科学原理，语言生动形象，适合科普受众...",
            key="sb_input_inst",
        )

        if st.button(
            "🚀 一键生成方案与分镜 (Generate Plan & Storyboard)",
            type="primary",
            use_container_width=True,
            key="sb_btn_create_pipeline",
        ):
            if not new_topic.strip():
                st.warning("请输入视频主题")
                return
            with st.spinner("AI 规划师正在构建内容大纲，分镜 Agent 正在逐段规划镜头..."):
                try:
                    from app.services.storyboard_generation_pipeline import (
                        StoryboardGenerationInput,
                        StoryboardGenerationPipeline,
                    )

                    pipeline = StoryboardGenerationPipeline()
                    aspect_val = "16:9" if "16:9" in new_aspect else "9:16"
                    gen_input = StoryboardGenerationInput(
                        topic=new_topic.strip(),
                        target_video_duration=float(new_duration),
                        target_aspect_ratio=aspect_val,
                        language=new_lang,
                        user_instruction=new_instruction.strip() if new_instruction.strip() else None,
                        source_grounded=False,
                    )
                    result = pipeline.generate(gen_input)
                    st.session_state["sb_workbench_current_snapshot_id"] = result.storyboard_snapshot_id
                    st.session_state["sb_workbench_selected_shot_id"] = None
                    st.toast(
                        f"🎉 方案与分镜生成成功！共 {result.beat_count} 个节奏，{result.shot_count} 个分镜镜头",
                        icon="🎉",
                    )
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ 生成失败: {exc}")


def _render_video_assembly_section(view: StoryboardEditingView):
    """Bottom section: Assemble full video from shot assets with TTS, subtitles, and BGM."""
    with st.expander("🎬 合成最终完整视频 (Assemble Final Video)", expanded=True):
        st.caption("将分镜素材与旁白解说按时间线无缝合成，生成最终 MP4 视频成片（支持 TTS 配音、智能字幕与背景音乐）。")
        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            voice_choice = st.selectbox(
                "配音声音 (TTS Voice)",
                options=[
                    "zh-CN-XiaoxiaoNeural (女生-晓晓-温暖)",
                    "zh-CN-YunxiNeural (男生-云希-活力)",
                    "zh-CN-YunjianNeural (男生-云健-沉稳)",
                    "zh-CN-XiaoyiNeural (女生-晓伊-知性)",
                    "en-US-JennyNeural (English - Female)",
                    "en-US-GuyNeural (English - Male)",
                ],
                index=0,
                key=f"sb_voice_{view.storyboard_snapshot_id}",
            )
            raw_voice_name = voice_choice.split(" ")[0]
        with col_v2:
            subtitle_enabled = st.checkbox(
                "生成并压制字幕 (Subtitles)",
                value=True,
                key=f"sb_sub_en_{view.storyboard_snapshot_id}",
            )
            voice_rate = st.slider(
                "语速 (Voice Rate)",
                min_value=0.8,
                max_value=1.5,
                value=1.0,
                step=0.1,
                key=f"sb_vrate_{view.storyboard_snapshot_id}",
            )
        with col_v3:
            bgm_type = st.selectbox(
                "背景音乐 (BGM)",
                options=["random (随机背景音)", "none (无背景音乐)"],
                index=0,
                key=f"sb_bgm_{view.storyboard_snapshot_id}",
            )
            raw_bgm_type = "random" if "random" in bgm_type else "none"
            bgm_volume = st.slider(
                "BGM 音量",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key=f"sb_bgm_vol_{view.storyboard_snapshot_id}",
            )

        if st.button(
            "🚀 开始合成最终完整视频 (Start Full Assembly)",
            type="primary",
            use_container_width=True,
            key=f"sb_btn_assemble_{view.storyboard_snapshot_id}",
        ):
            with st.spinner("正在合成视频：生成 TTS 配音、提取时间轴字幕、顺序拼接分镜画面并混音 BGM..."):
                try:
                    from app.services.storyboard_video_assembly_service import (
                        StoryboardVideoAssemblyService,
                    )

                    service = StoryboardVideoAssemblyService()
                    video_params = {
                        "video_subject": view.topic,
                        "voice_name": raw_voice_name,
                        "voice_rate": voice_rate,
                        "subtitle_enabled": subtitle_enabled,
                        "bgm_type": raw_bgm_type,
                        "bgm_volume": bgm_volume,
                    }
                    res = service.assemble_video(
                        storyboard_snapshot_id=view.storyboard_snapshot_id,
                        video_params=video_params,
                    )
                    st.session_state[f"assembly_result_{view.storyboard_snapshot_id}"] = res
                    st.toast("🎉 最终视频合成成功！", icon="🎉")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ 视频合成失败: {exc}")

        assembly_res = st.session_state.get(f"assembly_result_{view.storyboard_snapshot_id}")
        if assembly_res and getattr(assembly_res, "final_video_paths", None):
            final_mp4 = assembly_res.final_video_paths[0]
            st.success(f"✅ **成片渲染完成**: `{final_mp4}` (时长: {assembly_res.audio_duration:.1f}s)")
            if os.path.isfile(final_mp4):
                st.video(final_mp4)


def _render_storyboard_content():
    if not st.session_state.get("db_initialized", True):
        st.error(
            "❌ **数据库未就绪或连接失败**\n\n"
            "分镜工作台依赖数据库服务与表迁移。当前检测到数据库未正常初始化，已停止加载分镜数据。\n"
            "请确认数据库服务已启动，并检查 `DATABASE_URL` 配置。"
        )
        return

    available_snapshots = list_available_storyboard_snapshots()

    # 1. Topic-to-Storyboard Creation Card
    _render_creation_card(available_snapshots)

    if not available_snapshots:
        st.info("ℹ️ 暂无可用的分镜数据，请在上方卡片中输入主题生成首个方案。")
        return

    # Determine current snapshot ID from session state or latest available
    current_snapshot_id = st.session_state.get("sb_workbench_current_snapshot_id")
    if not current_snapshot_id or not any(
        s["storyboard_snapshot_id"] == current_snapshot_id for s in available_snapshots
    ):
        current_snapshot_id = available_snapshots[0]["storyboard_snapshot_id"]
        st.session_state["sb_workbench_current_snapshot_id"] = current_snapshot_id

    # Load editing read model
    view = load_storyboard_view(current_snapshot_id)
    if not view:
        st.warning(f"无法读取快照 `{current_snapshot_id}` 的数据，请重新选择。")
        return

    # Top status header
    _render_top_header(view, available_snapshots)

    # Locate currently selected shot
    selected_shot_id = st.session_state.get("sb_workbench_selected_shot_id")
    selected_shot: ShotEditingView | None = None
    if selected_shot_id:
        for beat in view.beats:
            for s in beat.shots:
                if s.shot_id == selected_shot_id:
                    selected_shot = s
                    break
            if selected_shot:
                break

    # If no shot selected yet, default to the first shot of the first beat
    if not selected_shot and view.beats and view.beats[0].shots:
        selected_shot = view.beats[0].shots[0]
        selected_shot_id = selected_shot.shot_id
        st.session_state["sb_workbench_selected_shot_id"] = selected_shot_id

    # 3-Panel Layout: Beats (3.0) | Shots (4.8) | Editor (4.2)
    col_beats, col_shots, col_editor = st.columns([3.0, 4.8, 4.2], gap="medium")

    with col_beats:
        _render_beats_panel(view)

    with col_shots:
        _render_shots_panel(view, selected_shot_id)

    with col_editor:
        _render_shot_editor(view, selected_shot)

    # 2. Final Video Assembly Section
    _render_video_assembly_section(view)


def render_storyboard_workbench():
    """Main entry point for rendering the Storyboard Workbench and Benchmark Reports in Streamlit."""
    tab_storyboard, tab_benchmark = st.tabs([
        "🎬 分镜工作台 (Storyboard Workbench)",
        "📊 基准评测报告与策略对比 (Benchmark & Strategy Comparison)",
    ])

    with tab_storyboard:
        _render_storyboard_content()

    with tab_benchmark:
        from webui.benchmark_workbench import render_benchmark_workbench

        render_benchmark_workbench()

