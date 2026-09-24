"""Knowledge Video Agent Unified UI Page.

Architecture Rule:
This module must NEVER import from app.persistence, app.domain,
app.application, app.services, or sqlalchemy.
All backend interactions MUST be mediated strictly via KnowledgeVideoApiClient.
"""

from __future__ import annotations

import json
import os
from typing import Any
import streamlit as st

from webui.api_client import ApiClientError, KnowledgeVideoApiClient

# 10 linear pipeline stages
PIPELINE_STAGES = [
    ("EVIDENCE", "1. 资料接入", "Evidence"),
    ("KNOWLEDGE_PLAN", "2. 知识方案", "Knowledge Plan"),
    ("SCRIPT", "3. 知识脚本", "Script"),
    ("STORYBOARD", "4. 分镜方案", "Storyboard"),
    ("PRODUCTION_PLAN", "5. 制片规划", "Production Plan"),
    ("ASSET", "6. 素材资产", "Asset"),
    ("AUDIO", "7. 音频旁白", "Audio"),
    ("COMPOSITION", "8. 合成渲染", "Composition"),
    ("QUALITY_REVIEW", "9. 质检把关", "Quality Review"),
    ("DELIVERY", "10. 交付封版", "Delivery"),
]

STAGE_KEYS = [s[0] for s in PIPELINE_STAGES]


def _get_api_client() -> KnowledgeVideoApiClient:
    """Returns or creates a cached API client in session state."""
    if "kva_api_client" not in st.session_state:
        st.session_state["kva_api_client"] = KnowledgeVideoApiClient()
    return st.session_state["kva_api_client"]


def render_agent_page() -> None:
    """Main render function for the Knowledge Video Agent page."""
    client = _get_api_client()

    # Header section
    _render_header()

    # View Mode Selector in sub-header: Create, Dashboard, or History
    view_col1, view_col2, view_col3 = st.columns([1.2, 1.2, 2.6])
    current_tab = st.session_state.get("kva_view_tab", "dashboard")

    with view_col1:
        if st.button(
            "➕ 新建任务 (New Task)",
            key="kva_btn_nav_create",
            type="primary" if current_tab == "create" else "secondary",
            width="stretch",
        ):
            st.session_state["kva_view_tab"] = "create"
            st.rerun()

    with view_col2:
        if st.button(
            "📊 任务看板 (Dashboard)",
            key="kva_btn_nav_dashboard",
            type="primary" if current_tab == "dashboard" else "secondary",
            width="stretch",
        ):
            st.session_state["kva_view_tab"] = "dashboard"
            st.rerun()

    with view_col3:
        if st.button(
            "📋 历史任务 (Task History)",
            key="kva_btn_nav_history",
            type="primary" if current_tab == "history" else "secondary",
            width="stretch",
        ):
            st.session_state["kva_view_tab"] = "history"
            st.rerun()

    st.divider()

    # Dispatch to current tab
    if current_tab == "create":
        _render_create_task_view(client)
    elif current_tab == "history":
        _render_task_history_view(client)
    else:
        _render_task_dashboard_view(client)


def _render_header() -> None:
    """Renders the agent page title and brief explanation."""
    header_col, badge_col = st.columns([3.5, 1.5], vertical_alignment="center")
    with header_col:
        st.markdown(
            "### 🚀 Knowledge Video Agent · 知识视频智能体\n"
            "全流程知识驱动视频生成系统：从资料证据、知识规划、脚本分镜、音画合成到自动化交付封版。"
        )
    with badge_col:
        active_task_id = st.session_state.get("kva_active_task_id")
        if active_task_id:
            st.info(f"📌 当前活动任务: `{active_task_id[:16]}...`")
        else:
            st.caption("⚪ 当前未选择活动任务")


def _render_create_task_view(client: KnowledgeVideoApiClient) -> None:
    """Form to create a new Knowledge Video Task."""
    st.markdown("#### 📝 新建知识视频任务 (Create Knowledge Video Task)")

    with st.form("kva_create_task_form"):
        col_topic, col_policy = st.columns([3, 1])
        with col_topic:
            topic = st.text_input(
                "任务主题 / 核心知识点 (Topic)",
                placeholder="例如：量子纠缠的原理与实验验证 / 工业革命对现代经济的影响",
                help="输入视频核心主题或知识讲解目标",
            )
        with col_policy:
            workflow_policy = st.selectbox(
                "工作流策略 (Policy)",
                options=["AUTO", "REVIEW"],
                format_func=lambda x: "全自动出片 (AUTO)" if x == "AUTO" else "人机审核把关 (REVIEW)",
                help="REVIEW 策略会在知识方案、脚本、分镜生成后进入暂停状态等待人工审核批准",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            target_duration = st.slider(
                "目标时长 (秒)",
                min_value=15.0,
                max_value=600.0,
                value=60.0,
                step=15.0,
                help="期望生成的最终成片时长（秒）",
            )
        with col2:
            aspect_ratio = st.selectbox(
                "视频画幅 (Aspect Ratio)",
                options=["16:9", "9:16", "1:1"],
                index=0,
                help="16:9 适合横屏（B站/YouTube/PC），9:16 适合竖屏（抖音/快手/Shorts）",
            )
        with col3:
            language = st.selectbox(
                "视频语言 (Language)",
                options=["zh", "en"],
                format_func=lambda x: "中文 (zh)" if x == "zh" else "English (en)",
                index=0,
            )

        allow_research = st.checkbox(
            "允许全网检索补充知识 (Allow Open-Web Research)",
            value=True,
            help="当输入证据资料不足时，允许 Agent 自动通过搜索引擎获取最新知识补充",
        )

        st.markdown("##### 📚 初始证据资料接入 (Optional Evidence)")
        ev_col1, ev_col2 = st.columns([1, 3])
        with ev_col1:
            source_type = st.selectbox(
                "证据类型",
                options=["TEXT", "URL", "KNOWLEDGE_BASE"],
                format_func=lambda x: {
                    "TEXT": "文本资料 (Text)",
                    "URL": "网络链接 (URL)",
                    "KNOWLEDGE_BASE": "知识库 (Knowledge Base)",
                }.get(x, x),
            )
        with ev_col2:
            source_title = st.text_input("资料标题 / 备注 (Title)", placeholder="例如：相关论文摘录或维基百科条目")

        source_content = st.text_area(
            "资料内容或 URL (Content / Locator)",
            placeholder="若选择文本请输入文本内容；若选择 URL 请输入完整 http/https 链接；若选择知识库请输入知识库 ID",
            height=100,
        )

        submitted = st.form_submit_button("🚀 启动 Agent 任务 (Start Task)", type="primary", use_container_width=True)

    if submitted:
        if not topic or not topic.strip():
            st.error("请输入任务主题 (Topic 不能为空)")
            return

        with st.spinner("正在初始化任务并建立流水线..."):
            try:
                task_res = client.create_task(
                    topic=topic.strip(),
                    target_duration=target_duration,
                    aspect_ratio=aspect_ratio,
                    language=language,
                    workflow_policy=workflow_policy,
                    allow_research=allow_research,
                )
                task_id = task_res.get("task_id")
                if not task_id:
                    st.error(f"创建任务失败，服务端未返回 task_id: {task_res}")
                    return

                # Register optional evidence if provided
                if source_content and source_content.strip():
                    clean_content = source_content.strip()
                    try:
                        if source_type == "TEXT":
                            client.register_evidence(
                                task_id=task_id,
                                source_type="TEXT",
                                text_content=clean_content,
                                title=source_title or topic,
                            )
                        elif source_type == "URL":
                            client.register_evidence(
                                task_id=task_id,
                                source_type="URL",
                                url=clean_content,
                                title=source_title or clean_content,
                            )
                        elif source_type == "KNOWLEDGE_BASE":
                            client.register_evidence(
                                task_id=task_id,
                                source_type="KNOWLEDGE_BASE",
                                kb_id=clean_content,
                                title=source_title or f"KB-{clean_content}",
                            )
                    except Exception as ev_err:
                        st.warning(f"任务创建成功，但附加证据注册出现提示: {ev_err}")

                st.session_state["kva_active_task_id"] = task_id
                st.session_state["kva_view_tab"] = "dashboard"
                st.success(f"任务创建成功！Task ID: {task_id}")
                st.rerun()

            except ApiClientError as err:
                st.error(f"创建任务失败: {err}")
            except Exception as exc:
                st.error(f"发生未知错误: {exc}")


def _render_task_dashboard_view(client: KnowledgeVideoApiClient) -> None:
    """Active task dashboard showing linear pipeline progress, human checkpoints, and reports."""
    active_task_id = st.session_state.get("kva_active_task_id")

    # Task selector row
    select_col1, select_col2 = st.columns([3.5, 1])
    with select_col1:
        # Load recent tasks for selector
        recent_tasks = []
        try:
            recent_tasks = client.list_tasks(limit=20)
        except Exception:
            pass

        task_options = {t["task_id"]: f"{t['topic']} ({t['task_status']}) [{t['task_id'][:8]}]" for t in recent_tasks}
        if active_task_id and active_task_id not in task_options:
            task_options[active_task_id] = f"当前任务 [{active_task_id[:8]}]"

        selected_id = st.selectbox(
            "选择查看的任务 (Select Task)",
            options=list(task_options.keys()),
            format_func=lambda x: task_options.get(x, x),
            index=list(task_options.keys()).index(active_task_id) if active_task_id in task_options else 0,
            key="kva_task_selector",
        )
        if selected_id and selected_id != active_task_id:
            st.session_state["kva_active_task_id"] = selected_id
            st.rerun()

    with select_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 刷新任务状态", use_container_width=True):
            st.rerun()

    current_task_id = st.session_state.get("kva_active_task_id")
    if not current_task_id:
        st.info("暂无活动任务，请先创建新任务或在上方下拉框中选择已有任务。")
        return

    # Fetch task detail from API
    try:
        task_data = client.get_task(current_task_id)
    except ApiClientError as err:
        st.error(f"获取任务状态失败: {err}")
        return
    except Exception as exc:
        st.error(f"连接 API 服务异常: {exc}")
        return

    # Render pipeline stepper
    _render_pipeline_stepper(task_data)

    # Render status bar & human checkpoint actions
    _render_checkpoint_and_actions(client, task_data)

    st.divider()

    # Inspection Tabs: Delivery, Artifacts, Events, Task Details
    tab_delivery, tab_artifacts, tab_events, tab_info = st.tabs([
        "🎬 交付成片与报告 (Delivery)",
        "📦 阶段制品快照 (Artifacts)",
        "📜 追踪审计日志 (Trace Events)",
        "⚙️ 任务诊断信息 (Diagnostics)",
    ])

    with tab_delivery:
        _render_delivery_tab(client, task_data)

    with tab_artifacts:
        _render_artifacts_tab(client, task_data)

    with tab_events:
        _render_events_tab(client, task_data)

    with tab_info:
        _render_diagnostics_tab(task_data)


def _render_pipeline_stepper(task_data: dict[str, Any]) -> None:
    """Renders the 10-stage linear pipeline stepper."""
    current_stage = task_data.get("current_stage", "EVIDENCE")
    task_status = task_data.get("task_status", "CREATED")

    try:
        curr_idx = STAGE_KEYS.index(current_stage)
    except ValueError:
        curr_idx = 0

    st.markdown("#### 📌 10 阶段流水线状态 (Pipeline Progress)")

    cols = st.columns(10)
    for idx, (stage_name, zh_name, en_name) in enumerate(PIPELINE_STAGES):
        with cols[idx]:
            if idx < curr_idx:
                icon = "✅"
                color = "#28a745"
            elif idx == curr_idx:
                if task_status == "COMPLETED" and stage_name == "DELIVERY":
                    icon = "✅"
                    color = "#28a745"
                elif task_status == "WAITING_USER":
                    icon = "⏸️"
                    color = "#ffc107"
                elif task_status in ("NEEDS_EVIDENCE", "NEEDS_RECOVERY"):
                    icon = "⚠️"
                    color = "#fd7e14"
                elif task_status == "FAILED":
                    icon = "❌"
                    color = "#dc3545"
                elif task_status == "CANCELLED":
                    icon = "🚫"
                    color = "#6c757d"
                else:
                    icon = "🔄"
                    color = "#007bff"
            else:
                icon = "⚪"
                color = "#6c757d"

            st.markdown(
                f"<div style='border: 1px solid {color}; border-radius: 6px; padding: 6px 4px; text-align: center;'>"
                f"<div style='font-size: 16px;'>{icon}</div>"
                f"<div style='font-weight: bold; font-size: 11px; margin-top: 4px; color: {color};'>{zh_name}</div>"
                f"<div style='font-size: 9px; color: #888;'>{stage_name}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_checkpoint_and_actions(client: KnowledgeVideoApiClient, task_data: dict[str, Any]) -> None:
    """Renders task status banner and human checkpoint approval / revision / retry controls."""
    task_id = task_data.get("task_id")
    task_status = task_data.get("task_status", "UNKNOWN")
    current_stage = task_data.get("current_stage", "UNKNOWN")
    workflow_policy = task_data.get("workflow_policy", "AUTO")
    waiting_reason = task_data.get("waiting_reason")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Status Alert Banner
    if task_status == "COMPLETED":
        st.success(f"🎉 **任务已完成并成功封版交付！** 当前阶段: `{current_stage}` | 策略: `{workflow_policy}`")
    elif task_status == "WAITING_USER":
        st.warning(
            f"⏸️ **人机审核检查点 (Human Review Checkpoint)**\n\n"
            f"任务暂停于 `{current_stage}` 阶段，原因: **{waiting_reason or '等待人工确认与批示'}**。\n\n"
            f"您可以查看下方制品，确认无误后点击「批准并推进」，或在下方提出修改建议后点击「重跑该阶段」。"
        )
    elif task_status == "NEEDS_EVIDENCE":
        st.warning(
            f"🔍 **资料证据不足 (Needs Evidence)**\n\n"
            f"知识检索未达到置信度阈值。您可以授权 Agent 进行全网公开搜索，或补充输入参考资料。"
        )
    elif task_status == "NEEDS_RECOVERY":
        err_msg = task_data.get("error_message") or "阶段执行发生可恢复异常"
        st.error(f"⚠️ **需要恢复重试 (Needs Recovery)**: {err_msg}")
    elif task_status == "FAILED":
        err_msg = task_data.get("error_message") or "任务执行遇到致命错误"
        st.error(f"❌ **任务执行失败 (Failed)**: {err_msg}")
    elif task_status == "CANCELLED":
        st.info("🚫 **任务已取消 (Cancelled)**")
    else:
        st.info(f"🔄 **任务正在运行中 (Running)** | 当前阶段: `{current_stage}` | 状态: `{task_status}`")

    # Action Controls Container
    action_box = st.container()
    with action_box:
        # Checkpoint: WAITING_USER
        if task_status == "WAITING_USER":
            st.markdown("##### 🛠️ 审核与流转决策 (Review Actions)")
            btn_col1, btn_col2 = st.columns([1.5, 2.5])
            with btn_col1:
                if st.button("✅ 确认批准并推进 (Approve & Advance)", type="primary", use_container_width=True):
                    with st.spinner("正在推进任务..."):
                        try:
                            client.approve_task(task_id)
                            st.success("已批准！任务已推进至下一阶段。")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"批准失败: {exc}")

            with btn_col2:
                with st.expander("✍️ 提出修改意见并重跑 (Revise & Rerun)", expanded=False):
                    feedback_text = st.text_area(
                        "修改指导意见 (Feedback)",
                        placeholder="例如：调整语言通俗度，分镜第2镜改为示意图展示",
                        height=80,
                    )
                    stage_options = STAGE_KEYS
                    target_stage = st.selectbox(
                        "重新执行的起始阶段",
                        options=stage_options,
                        index=stage_options.index(current_stage) if current_stage in stage_options else 0,
                    )
                    if st.button("🚀 提交修改并重跑 (Submit Revision)", use_container_width=True):
                        with st.spinner("正在提交修改意见..."):
                            try:
                                client.revise_task(
                                    task_id=task_id,
                                    target_stage=target_stage,
                                    feedback=feedback_text.strip() if feedback_text else None,
                                )
                                st.success("修改意见已接受，工作流已重新排队。")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"提交修改失败: {exc}")

        # Checkpoint: NEEDS_EVIDENCE
        elif task_status == "NEEDS_EVIDENCE":
            ev_col1, ev_col2 = st.columns([1.5, 2.5])
            with ev_col1:
                if st.button("🌐 授权全网检索并继续 (Authorize Web Research)", type="primary", use_container_width=True):
                    with st.spinner("正在授权网络检索..."):
                        try:
                            client.authorize_research(task_id, resume_if_waiting=True)
                            st.success("已授权网络检索，任务正在继续！")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"授权失败: {exc}")

            with ev_col2:
                with st.expander("📥 补充输入证据资料 (Add Evidence)", expanded=False):
                    add_type = st.selectbox("资料类型", options=["TEXT", "URL"])
                    add_content = st.text_area("资料内容或链接", height=80)
                    if st.button("提交补充资料并重试", use_container_width=True):
                        if not add_content or not add_content.strip():
                            st.warning("请输入资料内容")
                        else:
                            try:
                                if add_type == "TEXT":
                                    client.register_evidence(task_id, source_type="TEXT", text_content=add_content.strip())
                                else:
                                    client.register_evidence(task_id, source_type="URL", url=add_content.strip())
                                client.retry_task(task_id, reason="User added supplementary evidence")
                                st.success("资料已添加并触发重试！")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"添加资料失败: {exc}")

        # Recovery controls for recoverable / failed / running tasks
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
        if task_status in ("NEEDS_RECOVERY", "FAILED"):
            with ctrl_col1:
                if st.button("🔄 触发重试 (Retry)", type="primary", use_container_width=True):
                    try:
                        client.retry_task(task_id)
                        st.success("重试指令已发送！")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"重试失败: {exc}")

        if task_status in ("CREATED", "RUNNING", "WAITING_USER", "NEEDS_EVIDENCE", "NEEDS_RECOVERY"):
            with ctrl_col2:
                if st.button("🛑 取消任务 (Cancel)", type="secondary", use_container_width=True):
                    try:
                        client.cancel_task(task_id, reason="Cancelled from WebUI")
                        st.warning("任务已取消！")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"取消失败: {exc}")


def _render_delivery_tab(client: KnowledgeVideoApiClient, task_data: dict[str, Any]) -> None:
    """Renders the final delivery video player, download links, and reports."""
    task_id = task_data.get("task_id")
    task_status = task_data.get("task_status")

    if task_status != "COMPLETED":
        st.info("ℹ️ 视频尚未完成最终交付封版。当流程执行至 DELIVERY 阶段完成后，成片与交付报告将在此展示。")
        return

    # Fetch delivery manifest
    try:
        manifest = client.get_delivery_manifest(task_id)
    except ApiClientError as err:
        st.warning(f"未能获取交付清册 (DeliveryManifest): {err}")
        return
    except Exception as exc:
        st.error(f"获取交付清册网络异常: {exc}")
        return

    st.markdown("#### 🎬 最终交付成果 (Delivery Artifacts)")

    final_video_path = manifest.get("final_video_path")
    subtitle_path = manifest.get("subtitle_path")
    source_report_path = manifest.get("source_report_path")
    exec_report_path = manifest.get("execution_report_path")

    # Delivery Summary Badges
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    with sum_col1:
        st.metric("视频时长", f"{manifest.get('video_duration', 0.0):.1f} 秒")
    with sum_col2:
        size_mb = (manifest.get("file_size_bytes") or 0) / (1024 * 1024)
        st.metric("文件大小", f"{size_mb:.2f} MB")
    with sum_col3:
        st.metric("画幅比例", manifest.get("aspect_ratio", "16:9"))
    with sum_col4:
        st.metric("交付状态", manifest.get("delivery_status", "COMPLETED"))

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Video Player
    player_col, dl_col = st.columns([3, 2])
    with player_col:
        st.markdown("##### 📺 成片预览")
        if final_video_path and os.path.exists(final_video_path):
            st.video(final_video_path)
        else:
            try:
                vid_bytes = client.download_delivery_file(task_id, target="video")
                st.video(vid_bytes)
            except Exception as vid_err:
                st.info(f"无法在页面直接播放视频 ({vid_err})，请点击右侧按钮下载。")

    with dl_col:
        st.markdown("##### 📥 交付文件安全下载")
        # Download buttons
        try:
            if final_video_path and os.path.exists(final_video_path):
                with open(final_video_path, "rb") as vf:
                    st.download_button(
                        "🎬 下载最终成片 (MP4)",
                        data=vf.read(),
                        file_name=f"{task_id}_final.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
            else:
                vid_bytes = client.download_delivery_file(task_id, target="video")
                st.download_button(
                    "🎬 下载最终成片 (MP4)",
                    data=vid_bytes,
                    file_name=f"{task_id}_final.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
        except Exception as e:
            st.caption(f"下载视频接口提示: {e}")

        try:
            sub_bytes = client.download_delivery_file(task_id, target="subtitle")
            st.download_button(
                "📝 下载字幕文件 (SRT/TXT)",
                data=sub_bytes,
                file_name=f"{task_id}_subtitles.txt",
                mime="text/plain",
                use_container_width=True,
            )
        except Exception:
            pass

        try:
            src_rep_bytes = client.download_delivery_file(task_id, target="source_report")
            st.download_button(
                "📜 下载资料追溯报告 (source_report.md)",
                data=src_rep_bytes,
                file_name=f"{task_id}_source_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        except Exception:
            pass

        try:
            exec_rep_bytes = client.download_delivery_file(task_id, target="execution_report")
            st.download_button(
                "📊 下载执行概要报告 (execution_report.md)",
                data=exec_rep_bytes,
                file_name=f"{task_id}_execution_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        except Exception:
            pass

    st.divider()

    # Rendered Markdown Reports
    st.markdown("#### 📑 交付审计与追溯报告预览 (Audit & Traceability Reports)")

    report_tab1, report_tab2, report_tab3 = st.tabs([
        "📜 资料追溯报告 (Source Traceability Report)",
        "📊 执行过程审计报告 (Execution Summary Report)",
        "🔒 交付清册元数据 (Delivery Manifest JSON)",
    ])

    with report_tab1:
        try:
            src_rep_bytes = client.download_delivery_file(task_id, target="source_report")
            src_text = src_rep_bytes.decode("utf-8", errors="replace")
            st.markdown(src_text)
        except Exception as err:
            st.info(f"读取资料追溯报告提示: {err}")

    with report_tab2:
        try:
            exec_rep_bytes = client.download_delivery_file(task_id, target="execution_report")
            exec_text = exec_rep_bytes.decode("utf-8", errors="replace")
            st.markdown(exec_text)
        except Exception as err:
            st.info(f"读取执行报告提示: {err}")

    with report_tab3:
        st.json(manifest)


def _render_artifacts_tab(client: KnowledgeVideoApiClient, task_data: dict[str, Any]) -> None:
    """Renders all stage artifacts registered for the task."""
    task_id = task_data.get("task_id")
    st.markdown("#### 📦 各阶段制品快照清单 (Stage Artifacts)")

    try:
        artifacts = client.get_task_artifacts(task_id)
    except Exception as exc:
        st.error(f"获取制品失败: {exc}")
        return

    if not artifacts:
        st.info("当前任务尚未生成持久化阶段制品。")
        return

    st.caption(f"共记录 {len(artifacts)} 项阶段产物，均已冻结为不可变快照：")

    for art in artifacts:
        stage = art.get("stage", "UNKNOWN")
        art_id = art.get("task_artifact_ref_id", "")
        art_type = art.get("artifact_type", "")
        rev_id = art.get("artifact_revision_id", "")
        created_at = art.get("created_at", "")
        metadata = art.get("metadata_json") or {}

        with st.expander(f"📌 [{stage}] {art_type} · 版本 `{rev_id[:16]}...`", expanded=False):
            st.markdown(f"- **阶段 (Stage)**: `{stage}`")
            st.markdown(f"- **制品类型 (Type)**: `{art_type}`")
            st.markdown(f"- **制品 ID**: `{art_id}`")
            st.markdown(f"- **版本修订 (Revision)**: `{rev_id}`")
            st.markdown(f"- **生成时间 (Created)**: `{created_at}`")
            if metadata:
                st.markdown("**元数据信息 (Metadata):**")
                st.json(metadata)


def _render_events_tab(client: KnowledgeVideoApiClient, task_data: dict[str, Any]) -> None:
    """Renders chronological domain trace events."""
    task_id = task_data.get("task_id")
    st.markdown("#### 📜 任务执行事件流 (Trace Events Timeline)")

    try:
        events = client.get_task_events(task_id)
    except Exception as exc:
        st.error(f"获取事件流失败: {exc}")
        return

    if not events:
        st.info("暂无追踪事件记录。")
        return

    st.caption(f"共记录 {len(events)} 条可追溯的领域事件：")

    # Render event list
    for ev in events:
        ev_name = ev.get("event_name", "EVENT")
        created_at = ev.get("created_at", "")
        stage = ev.get("stage")
        payload = ev.get("payload") or {}

        badge_color = "blue"
        if "FAILED" in ev_name or "ERROR" in ev_name:
            badge_color = "red"
        elif "COMPLETED" in ev_name or "DELIVERY" in ev_name:
            badge_color = "green"
        elif "WAITING" in ev_name or "CHECKPOINT" in ev_name:
            badge_color = "orange"

        with st.expander(f"⏱️ `{created_at}` · **{ev_name}**" + (f" [{stage}]" if stage else ""), expanded=False):
            st.json(payload)


def _render_diagnostics_tab(task_data: dict[str, Any]) -> None:
    """Renders diagnostic information, executions history, and metadata."""
    st.markdown("#### ⚙️ 任务诊断与执行历史 (Diagnostics)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**核心属性 (Core Properties):**")
        st.markdown(f"- **Task ID**: `{task_data.get('task_id')}`")
        st.markdown(f"- **主题 (Topic)**: {task_data.get('topic')}")
        st.markdown(f"- **状态 (Status)**: `{task_data.get('task_status')}`")
        st.markdown(f"- **当前阶段 (Stage)**: `{task_data.get('current_stage')}`")
        st.markdown(f"- **策略 (Policy)**: `{task_data.get('workflow_policy')}`")
        st.markdown(f"- **画幅 (Aspect Ratio)**: {task_data.get('aspect_ratio')}")
        st.markdown(f"- **语言 (Language)**: {task_data.get('language')}")
        st.markdown(f"- **目标时长**: {task_data.get('target_duration')} 秒")

    with col2:
        st.markdown("**时间与异常 (Timestamps & Errors):**")
        st.markdown(f"- **创建时间**: {task_data.get('created_at')}")
        st.markdown(f"- **更新时间**: {task_data.get('updated_at')}")
        st.markdown(f"- **完成时间**: {task_data.get('finished_at')}")
        st.markdown(f"- **异常类型**: {task_data.get('error_type') or '无'}")
        st.markdown(f"- **异常详情**: {task_data.get('error_message') or '无'}")

    executions = task_data.get("stage_executions", [])
    if executions:
        st.markdown(f"##### 🔄 阶段执行历史记录 ({len(executions)} 次尝试)")
        for exc in executions:
            st.markdown(
                f"- **[{exc.get('stage')}]** 状态: `{exc.get('status')}` | "
                f"耗时: `{exc.get('duration_ms', 0)} ms` | "
                f"开始: `{exc.get('started_at')}`"
            )
            if exc.get("error_message"):
                st.caption(f"  ❌ 错误: {exc.get('error_message')}")

    metadata = task_data.get("task_metadata") or {}
    if metadata:
        st.markdown("##### 📋 任务元数据 (Task Metadata):")
        st.json(metadata)


def _render_task_history_view(client: KnowledgeVideoApiClient) -> None:
    """Lists recent tasks with quick switch actions."""
    st.markdown("#### 📋 历史任务列表 (Task History)")

    try:
        tasks = client.list_tasks(limit=50)
    except Exception as exc:
        st.error(f"获取任务列表失败: {exc}")
        return

    if not tasks:
        st.info("暂无历史任务记录。请点击顶部「新建任务」创建第一个视频生成任务。")
        return

    st.caption(f"最近 {len(tasks)} 条任务记录：")

    for t in tasks:
        t_id = t.get("task_id")
        topic = t.get("topic")
        status = t.get("task_status")
        stage = t.get("current_stage")
        policy = t.get("workflow_policy")
        created_at = t.get("created_at")

        row_col1, row_col2, row_col3, row_col4 = st.columns([3, 1.5, 1.5, 1])
        with row_col1:
            st.markdown(f"**{topic}**\n`{t_id}`")
        with row_col2:
            st.markdown(f"状态: `{status}`\n阶段: `{stage}`")
        with row_col3:
            st.caption(f"策略: {policy}\n时间: {created_at[:19] if created_at else ''}")
        with row_col4:
            if st.button("进入看板", key=f"kva_select_{t_id}", use_container_width=True):
                st.session_state["kva_active_task_id"] = t_id
                st.session_state["kva_view_tab"] = "dashboard"
                st.rerun()
        st.divider()
