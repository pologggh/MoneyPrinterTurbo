"""Benchmark Workbench UI for MoneyPrinterTurbo.

Phase 7.3 Streamlit view for inspecting versioned BenchmarkReports,
validating comparability hard gates, and exploring Strategy Comparisons
with transparent trade-offs and case-level drilldowns.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from sqlalchemy import select

from app.persistence.models import BenchmarkRunORM
from app.persistence.session import get_session
from app.services.benchmark.comparison_gate import validate_benchmark_comparability
from app.services.benchmark.comparison_service import BenchmarkComparisonService
from app.services.benchmark.report_service import BenchmarkReportService


def list_benchmark_runs() -> list[dict[str, Any]]:
    """Fetches list of all benchmark runs ordered by started_at descending."""
    with get_session() as session:
        stmt = select(BenchmarkRunORM).order_by(BenchmarkRunORM.started_at.desc())
        runs = session.scalars(stmt).all()
        return [
            {
                "benchmark_run_id": r.benchmark_run_id,
                "suite_key": r.suite_key,
                "suite_version": r.suite_version,
                "suite_fingerprint": r.suite_fingerprint,
                "variant_key": (r.variant_json or {}).get("variant_key", "unknown"),
                "execution_mode": r.execution_mode,
                "status": r.status,
                "started_at": r.started_at.strftime("%Y-%m-%d %H:%M:%S") if r.started_at else "",
                "total_cases": r.total_cases,
                "completed_cases": r.completed_cases,
                "failed_cases": r.failed_cases,
            }
            for r in runs
        ]


def render_benchmark_workbench() -> None:
    """Main entry point for rendering Benchmark Reports & Strategy Comparison."""
    st.markdown("## 📊 基准评测报告与策略对比 (Benchmark Reports & Strategy Comparison)")
    st.markdown(
        "基于 Phase 7.1 Trace 与 Phase 7.2 Benchmark 数据，完全由历史不可变记录聚合计算，"
        "**零二次调用生产管线、零重新请求模型**。"
    )

    runs = list_benchmark_runs()
    if not runs:
        st.info("ℹ️ 暂无基准评测运行记录。请先通过测试或 API 运行基准评测。")
        return

    tab_report, tab_compare = st.tabs([
        "📈 单次运行报告 (Run Report)",
        "⚖️ 策略对比分析 (Strategy Comparison)",
    ])

    with tab_report:
        _render_single_run_report(runs)

    with tab_compare:
        _render_strategy_comparison(runs)


def _render_single_run_report(runs: list[dict[str, Any]]) -> None:
    """Renders single benchmark run report inspection."""
    col1, col2 = st.columns([4, 2])
    with col1:
        run_options = {
            f"{r['benchmark_run_id']} [{r['suite_key']}:{r['variant_key']}] ({r['status']}, {r['started_at']})": r[
                "benchmark_run_id"
            ]
            for r in runs
        }
        selected_label = st.selectbox("选择基准运行 (Benchmark Run):", list(run_options.keys()))
        selected_run_id = run_options[selected_label]

    report_service = BenchmarkReportService(get_session)
    report = report_service.get_report_by_run(selected_run_id)

    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 重新生成 / 刷新报告", key="refresh_rep_btn"):
            report = report_service.generate_report(selected_run_id)
            st.success("报告已生成/刷新！")

    if not report:
        if st.button("🚀 生成此运行的评估报告", key="gen_rep_btn", type="primary"):
            report = report_service.generate_report(selected_run_id)
            st.rerun()
        else:
            st.info("此运行尚未生成汇总报告，请点击上方按钮生成。")
            return

    # Overview KPIs
    st.markdown("### 关键指标概览 (Key Performance Indicators)")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

    fp_val = report.metrics.get("first_asset_quality_pass_rate")
    with kpi_col1:
        st.metric(
            "首轮质检通过率",
            f"{fp_val.value * 100:.1f}%" if fp_val and fp_val.value is not None else "N/A",
            help="初次生成的资产即通过多模态评估的比例",
        )

    fn_val = report.metrics.get("final_shot_quality_pass_rate")
    with kpi_col2:
        st.metric(
            "最终分镜合格率",
            f"{fn_val.value * 100:.1f}%" if fn_val and fn_val.value is not None else "N/A",
            help="最终成功交付并被采纳的镜头资产比例",
        )

    att_val = report.metrics.get("mean_generation_attempts_per_shot")
    with kpi_col3:
        st.metric(
            "平均尝试次数/镜头",
            f"{att_val.value:.2f}" if att_val and att_val.value is not None else "N/A",
            help="平均每个镜头经历的生成尝试次数",
        )

    e2e_lat = report.stage_latencies.get("end_to_end")
    with kpi_col4:
        st.metric(
            "端到端平均延迟",
            f"{e2e_lat.mean_ms:.0f} ms" if e2e_lat and e2e_lat.mean_ms is not None else "N/A",
            help="每个 Case 的平均端到端生成总延迟",
        )

    with kpi_col5:
        total_obs = sum(report.cost_summary.observed_by_currency.values())
        synth_flag = " (离线模拟)" if report.cost_summary.is_synthetic else ""
        st.metric(
            "总观测成本" + synth_flag,
            f"${total_obs:.4f}" if total_obs > 0 else "$0.00",
            help="汇总各模型的调用开销",
        )

    # 4 Quality Dimensions
    st.markdown("### 多模态质检四维度得分 (Multimodal Dimensions)")
    dim_cols = st.columns(4)
    dimensions = [
        ("语义对齐 (Semantic)", "dimension_semantic_alignment_score"),
        ("视觉质量 (Visual)", "dimension_visual_quality_score"),
        ("知识准确 (Knowledge)", "dimension_knowledge_accuracy_score"),
        ("构图适配 (Composition)", "dimension_composition_suitability_score"),
    ]
    for idx, (label, mkey) in enumerate(dimensions):
        with dim_cols[idx]:
            mval = report.metrics.get(mkey)
            if mval and mval.value is not None:
                st.metric(label, f"{mval.value:.2f}")
                cov_str = f"覆盖率: {mval.coverage_ratio * 100:.0f}%" if mval.coverage_ratio is not None else ""
                st.caption(f"状态: `{mval.status.value}` {cov_str}")
            else:
                st.metric(label, "N/A")
                st.caption(f"状态: `{mval.status.value if mval else 'UNAVAILABLE'}`")

    # Complete Metrics Table
    st.markdown("### 完整评测指标明细 (Detailed Metrics)")
    metrics_data = []
    for k, v in sorted(report.metrics.items()):
        metrics_data.append({
            "指标名称 (Metric)": k,
            "数值 (Value)": f"{v.value:.4f}" if v.value is not None else "-",
            "状态 (Status)": v.status.value,
            "有效样本 (Samples)": f"{v.sample_count}/{v.eligible_count}",
            "覆盖率 (Coverage)": f"{v.coverage_ratio * 100:.1f}%" if v.coverage_ratio is not None else "-",
            "备注 (Notes)": v.notes or "",
        })
    st.dataframe(metrics_data, use_container_width=True)

    # Stage Latencies
    st.markdown("### 阶段延迟耗时分布 (Stage Latency Breakdown)")
    lat_data = []
    for stage, lat in sorted(report.stage_latencies.items()):
        lat_data.append({
            "执行阶段 (Stage)": stage,
            "平均耗时 (Mean ms)": lat.mean_ms or "-",
            "中位数 P50 (ms)": lat.p50_ms or "-",
            "最小耗时 (Min ms)": lat.min_ms or "-",
            "最大耗时 (Max ms)": lat.max_ms or "-",
            "样本数 (Count)": lat.sample_count,
        })
    st.dataframe(lat_data, use_container_width=True)

    # Export Report JSON
    st.download_button(
        label="📥 导出标准报告 JSON",
        data=report.to_json(),
        file_name=f"benchmark_report_{report.benchmark_run_id}.json",
        mime="application/json",
    )


def _render_strategy_comparison(runs: list[dict[str, Any]]) -> None:
    """Renders strategy comparison with strict comparability hard gates."""
    st.markdown("### 策略对比条件设置 (Select Variants to Compare)")
    col1, col2 = st.columns(2)

    run_options = {
        f"{r['benchmark_run_id']} [{r['suite_key']}:{r['variant_key']}] ({r['started_at']})": r["benchmark_run_id"]
        for r in runs
    }
    with col1:
        base_label = st.selectbox("基准策略 (Baseline Strategy):", list(run_options.keys()), index=0)
        base_run_id = run_options[base_label]

    with col2:
        cand_idx = 1 if len(run_options) > 1 else 0
        cand_label = st.selectbox("候选策略 (Candidate Strategy):", list(run_options.keys()), index=cand_idx)
        cand_run_id = run_options[cand_label]

    report_service = BenchmarkReportService(get_session)
    base_rep = report_service.get_report_by_run(base_run_id)
    cand_rep = report_service.get_report_by_run(cand_run_id)

    if not base_rep or not cand_rep:
        if st.button("🚀 为所选运行生成评估报告并开始对比", key="gen_and_compare_btn", type="primary"):
            if not base_rep:
                base_rep = report_service.generate_report(base_run_id)
            if not cand_rep:
                cand_rep = report_service.generate_report(cand_run_id)
            st.rerun()
        else:
            st.info("请先确保两个选中的评测运行都已生成报告。")
            return

    # Hard Gate Check
    is_valid, gate_reason = validate_benchmark_comparability(base_rep, cand_rep)
    if not is_valid:
        st.error(f"⛔ **无法执行策略对比（违反对比门禁）**: {gate_reason}")
        st.warning("根据基准系统科学性原则，不同测试集版本、不同案例指纹、或离线模式与真实外部调用之间严禁直接对比。")
        return

    st.success("✅ **对比硬门禁校验通过**: 测试集版本、指纹、案例集及执行模式严格一致。")

    # Generate or compute comparison
    comp_svc = BenchmarkComparisonService()
    comparison = comp_svc.compare_reports(base_rep, cand_rep)

    # Trade-off Highlights
    st.markdown("### 策略权衡结论 (Objective Trade-Off Summary)")
    findings = comparison.tradeoff_summary.get("findings", [])
    if findings:
        for f in findings:
            st.markdown(f"- 💡 {f}")
    else:
        st.markdown("- 两个策略运行表现一致，未发现显著差异。")

    # Comparative Metrics Table
    st.markdown("### 指标差异对比表 (Metric Deltas)")
    deltas_data = []
    for k, delta in sorted(comparison.metric_deltas.items()):
        imp_badge = "⚪"
        if delta.is_improvement is True:
            imp_badge = "🟢 改善"
        elif delta.is_improvement is False:
            imp_badge = "🔴 回退"

        diff_str = "-"
        if delta.absolute_delta is not None:
            sign = "+" if delta.absolute_delta > 0 else ""
            pct_str = f" ({sign}{delta.relative_delta * 100:.1f}%)" if delta.relative_delta is not None else ""
            diff_str = f"{sign}{delta.absolute_delta:.4f}{pct_str}"

        deltas_data.append({
            "评测指标 (Metric)": k,
            "基准值 (Baseline)": f"{delta.baseline_value:.4f}" if delta.baseline_value is not None else "-",
            "候选值 (Candidate)": f"{delta.candidate_value:.4f}" if delta.candidate_value is not None else "-",
            "差异 (Delta)": diff_str,
            "判定 (Evaluation)": imp_badge,
            "状态 (Status)": delta.status,
        })
    st.dataframe(deltas_data, use_container_width=True)

    # Latency Deltas
    st.markdown("### 各阶段耗时对比 (Latency Comparison)")
    lat_data = []
    for stage, ldelta in sorted(comparison.latency_deltas.items()):
        badge = "⚪"
        if ldelta.is_improvement is True:
            badge = "🟢 提速"
        elif ldelta.is_improvement is False:
            badge = "🔴 变慢"

        diff_str = "-"
        if ldelta.absolute_delta_ms is not None:
            sign = "+" if ldelta.absolute_delta_ms > 0 else ""
            diff_str = f"{sign}{ldelta.absolute_delta_ms:.1f} ms"

        lat_data.append({
            "阶段 (Stage)": stage,
            "基准耗时 (Baseline ms)": ldelta.baseline_mean_ms or "-",
            "候选耗时 (Candidate ms)": ldelta.candidate_mean_ms or "-",
            "耗时变化 (Delta)": diff_str,
            "判定 (Evaluation)": badge,
        })
    st.dataframe(lat_data, use_container_width=True)

    # Export Comparison JSON
    st.download_button(
        label="📥 导出策略对比报告 JSON",
        data=comparison.to_json(),
        file_name=f"benchmark_comparison_{comparison.baseline_variant_key}_vs_{comparison.candidate_variant_key}.json",
        mime="application/json",
    )
