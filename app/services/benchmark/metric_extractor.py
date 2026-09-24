from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from sqlalchemy.orm import Session

from app.domain.benchmark import (
    BenchmarkCaseResult,
    BenchmarkCaseStatus,
    BenchmarkExecutionMode,
    BenchmarkFailureStage,
    BenchmarkRun,
)
from app.domain.benchmark_metrics import (
    METRIC_SET_VERSION_V1,
    CostSummary,
    MetricValue,
    MetricValueStatus,
    StageLatencySummary,
)
from app.domain.trace import CostEstimateStatus, TraceEventType
from app.persistence.repositories import (
    BenchmarkRepository,
    EvaluationRepository,
    ExecutionRepository,
    ShotRepository,
    StoryboardRepository,
    TraceRepository,
)


class BenchmarkMetricExtractor:
    """
    Extracts versioned, deterministic benchmark metrics from existing immutable
    historical domain records and Trace events.

    Zero rerun of production pipelines.
    Zero new AI/LLM/VLM or media calls.
    Zero guessing/fabrication of missing metrics.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._trace_repo = TraceRepository(session)
        self._shot_repo = ShotRepository(session)
        self._sb_repo = StoryboardRepository(session)
        self._exec_repo = ExecutionRepository(session)
        self._eval_repo = EvaluationRepository(session)
        self._bench_repo = BenchmarkRepository(session)

    def extract_metrics_for_run(
        self,
        run: BenchmarkRun,
        case_results: list[BenchmarkCaseResult] | None = None,
    ) -> tuple[
        dict[str, MetricValue],
        dict[str, StageLatencySummary],
        CostSummary,
        dict[str, Any],
        int,  # total_cases
        int,  # total_shots
    ]:
        """
        Extracts all standard V1 metrics, stage latencies, cost summary,
        and provider/model usage for a BenchmarkRun.
        """
        if case_results is None:
            case_results = self._bench_repo.list_case_results_for_run(run.benchmark_run_id)

        total_cases = len(case_results)

        # Gather all trace events and domain records for these cases
        trace_events_by_case: dict[str, list[Any]] = {}
        all_trace_events: list[Any] = []
        for cres in case_results:
            events = self._trace_repo.list_events_by_trace(cres.trace_id)
            trace_events_by_case[cres.case_key] = events
            all_trace_events.extend(events)

        # ---------------------------------------------------------------------
        # 1. Inspect Storyboard Shots across all cases
        # ---------------------------------------------------------------------
        total_shots = 0
        evidence_supported_shots = 0
        shots_by_id: dict[str, Any] = {}

        for cres in case_results:
            snap_id = cres.production_result_refs.approved_storyboard_snapshot_id
            if not snap_id:
                # Check draft snapshot if not approved
                for ev in trace_events_by_case.get(cres.case_key, []):
                    if ev.event_type == TraceEventType.STORYBOARD_GENERATION_COMPLETED:
                        snap_id = ev.attributes.get("storyboard_snapshot_id")
                        break

            if snap_id:
                snap = self._sb_repo.get_snapshot(snap_id)
                if snap:
                    # Load shots
                    for srev_id in snap.shot_revision_ids:
                        srev = self._shot_repo.get_revision(srev_id)
                        if srev:
                            total_shots += 1
                            shots_by_id[srev.shot_id] = srev
                            # Check evidence support
                            if srev.evidence_refs:
                                evidence_supported_shots += 1

        # Fallback shot discovery via trace context if repository snapshots are absent
        if total_shots == 0:
            discovered_shot_ids = {
                ev.context.shot_id
                for ev in all_trace_events
                if ev.context and ev.context.shot_id
            }
            total_shots = len(discovered_shot_ids)
            for sid in discovered_shot_ids:
                shots_by_id[sid] = sid

        # ---------------------------------------------------------------------
        # 2. Planning Metrics
        # ---------------------------------------------------------------------
        # plan_validation_success_rate
        if total_cases > 0:
            valid_plan_cases = sum(
                1
                for c in case_results
                if c.failure_stage not in (
                    BenchmarkFailureStage.PLANNING,
                    BenchmarkFailureStage.STRUCTURAL_VALIDATION,
                    BenchmarkFailureStage.INPUT_VALIDATION,
                )
            )
            val_plan_rate = MetricValue(
                metric_key="plan_validation_success_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.AVAILABLE,
                value=round(valid_plan_cases / total_cases, 4),
                sample_count=valid_plan_cases,
                eligible_count=total_cases,
            )
        else:
            val_plan_rate = MetricValue(
                metric_key="plan_validation_success_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.UNAVAILABLE,
                value=None,
                sample_count=0,
                eligible_count=0,
            )

        # storyboard_generation_success_rate
        if total_cases > 0:
            storyboard_success_cases = sum(
                1
                for c in case_results
                if c.production_result_refs.approved_storyboard_snapshot_id is not None
                or c.failure_stage not in (
                    BenchmarkFailureStage.INPUT_VALIDATION,
                    BenchmarkFailureStage.PLANNING,
                    BenchmarkFailureStage.STORYBOARD,
                )
            )
            sb_gen_rate = MetricValue(
                metric_key="storyboard_generation_success_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.AVAILABLE,
                value=round(storyboard_success_cases / total_cases, 4),
                sample_count=storyboard_success_cases,
                eligible_count=total_cases,
            )
        else:
            sb_gen_rate = MetricValue(
                metric_key="storyboard_generation_success_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.UNAVAILABLE,
                value=None,
                sample_count=0,
                eligible_count=0,
            )

        # evidence_support_rate
        if total_shots > 0:
            ev_support_rate = MetricValue(
                metric_key="evidence_support_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.AVAILABLE,
                value=round(evidence_supported_shots / total_shots, 4),
                sample_count=evidence_supported_shots,
                eligible_count=total_shots,
            )
        else:
            ev_support_rate = MetricValue(
                metric_key="evidence_support_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.UNAVAILABLE,
                value=None,
                sample_count=0,
                eligible_count=0,
            )

        # ---------------------------------------------------------------------
        # 3. Execution & Generation Metrics
        # ---------------------------------------------------------------------
        # Identify FIRST evaluation for each shot
        eval_completed_events = [
            ev for ev in all_trace_events
            if ev.event_type == TraceEventType.EVALUATION_COMPLETED
        ]
        # Sort chronologically
        eval_completed_events.sort(key=lambda e: (e.started_at, e.trace_event_id))

        first_eval_by_shot: dict[str, Any] = {}
        for ev in eval_completed_events:
            sid = ev.context.shot_id if ev.context else None
            if sid and sid not in first_eval_by_shot:
                first_eval_by_shot[sid] = ev

        total_evaluated_first_assets = len(first_eval_by_shot)
        first_asset_passes = sum(
            1
            for ev in first_eval_by_shot.values()
            if ev.attributes.get("decision") == "PASS"
        )

        if total_evaluated_first_assets > 0:
            first_pass_rate = MetricValue(
                metric_key="first_asset_quality_pass_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.AVAILABLE,
                value=round(first_asset_passes / total_evaluated_first_assets, 4),
                sample_count=first_asset_passes,
                eligible_count=total_evaluated_first_assets,
            )
        else:
            first_pass_rate = MetricValue(
                metric_key="first_asset_quality_pass_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.UNAVAILABLE,
                value=None,
                sample_count=0,
                eligible_count=0,
            )

        # final_shot_quality_pass_rate
        # Collect accepted asset version IDs from case results or trace events
        accepted_shot_ids: set[str] = set()
        for cres in case_results:
            if cres.production_result_refs.accepted_shot_asset_version_ids:
                for aver_id in cres.production_result_refs.accepted_shot_asset_version_ids:
                    aver = self._exec_repo.get_shot_asset_version(aver_id)
                    if aver:
                        accepted_shot_ids.add(aver.shot_id)

        # Also inspect QUALITY_ASSET_ACCEPTED trace events
        for ev in all_trace_events:
            if ev.event_type == TraceEventType.QUALITY_ASSET_ACCEPTED:
                sid = ev.context.shot_id if ev.context else None
                if sid:
                    accepted_shot_ids.add(sid)

        final_pass_count = len(accepted_shot_ids)
        if total_shots > 0:
            final_pass_rate = MetricValue(
                metric_key="final_shot_quality_pass_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.AVAILABLE,
                value=round(min(1.0, final_pass_count / total_shots), 4),
                sample_count=final_pass_count,
                eligible_count=total_shots,
            )
        else:
            final_pass_rate = MetricValue(
                metric_key="final_shot_quality_pass_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.UNAVAILABLE,
                value=None,
                sample_count=0,
                eligible_count=0,
            )

        # Execution attempts, fallbacks, recoveries
        attempt_events = [
            ev for ev in all_trace_events
            if ev.event_type == TraceEventType.EXECUTION_ATTEMPT_COMPLETED
        ]
        total_attempts = len(attempt_events)

        fallback_events = [
            ev for ev in all_trace_events
            if ev.event_type == TraceEventType.EXECUTION_FALLBACK_SELECTED
            or (ev.event_type == TraceEventType.EXECUTION_ATTEMPT_COMPLETED and ev.attributes.get("retry_fallback_decision") == "FALLBACK")
        ]
        total_fallbacks = len(fallback_events)

        recovery_events = [
            ev for ev in all_trace_events
            if ev.event_type == TraceEventType.EXECUTION_RECOVERY_REQUIRED
            or (ev.event_type == TraceEventType.EXECUTION_ATTEMPT_COMPLETED and ev.attributes.get("submission_outcome_unknown") is True)
        ]
        total_recoveries = len(recovery_events)

        if total_shots > 0:
            mean_attempts = MetricValue(
                metric_key="mean_generation_attempts_per_shot",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.AVAILABLE,
                value=round(total_attempts / total_shots, 4),
                sample_count=total_attempts,
                eligible_count=total_shots,
            )
        else:
            mean_attempts = MetricValue(
                metric_key="mean_generation_attempts_per_shot",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.UNAVAILABLE,
                value=None,
                sample_count=0,
                eligible_count=0,
            )

        if total_attempts > 0:
            fb_rate = MetricValue(
                metric_key="fallback_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.AVAILABLE,
                value=round(total_fallbacks / total_attempts, 4),
                sample_count=total_fallbacks,
                eligible_count=total_attempts,
            )
            rec_rate = MetricValue(
                metric_key="recovery_required_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.AVAILABLE,
                value=round(total_recoveries / total_attempts, 4),
                sample_count=total_recoveries,
                eligible_count=total_attempts,
            )
        else:
            fb_rate = MetricValue(
                metric_key="fallback_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.UNAVAILABLE,
                value=None,
                sample_count=0,
                eligible_count=0,
            )
            rec_rate = MetricValue(
                metric_key="recovery_required_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.UNAVAILABLE,
                value=None,
                sample_count=0,
                eligible_count=0,
            )

        # ---------------------------------------------------------------------
        # 4. Quality & Remediation Metrics
        # ---------------------------------------------------------------------
        remed_events = [
            ev for ev in all_trace_events
            if ev.event_type == TraceEventType.QUALITY_REMEDIATION_DECIDED
        ]
        shots_triggering_remediation = {
            ev.context.shot_id for ev in remed_events
            if ev.context and ev.context.shot_id
        }
        total_remed_decisions = len(remed_events)
        for cres in case_results:
            total_remed_decisions = max(
                total_remed_decisions,
                len(cres.production_result_refs.remediation_decision_ids),
            )

        shots_with_visual_replan = {
            ev.context.shot_id for ev in remed_events
            if ev.attributes.get("action") == "CONTROLLED_VISUAL_REPLAN"
            and ev.context and ev.context.shot_id
        }
        for ev in all_trace_events:
            if ev.event_type == TraceEventType.CONTROLLED_VISUAL_REPLAN_CREATED and ev.context and ev.context.shot_id:
                shots_with_visual_replan.add(ev.context.shot_id)

        if total_evaluated_first_assets > 0:
            q_remed_rate = MetricValue(
                metric_key="quality_remediation_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.AVAILABLE,
                value=round(len(shots_triggering_remediation) / total_evaluated_first_assets, 4),
                sample_count=len(shots_triggering_remediation),
                eligible_count=total_evaluated_first_assets,
            )
            vis_replan_rate = MetricValue(
                metric_key="controlled_visual_replan_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.AVAILABLE,
                value=round(len(shots_with_visual_replan) / total_evaluated_first_assets, 4),
                sample_count=len(shots_with_visual_replan),
                eligible_count=total_evaluated_first_assets,
            )
        else:
            q_remed_rate = MetricValue(
                metric_key="quality_remediation_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.UNAVAILABLE,
                value=None,
                sample_count=0,
                eligible_count=0,
            )
            vis_replan_rate = MetricValue(
                metric_key="controlled_visual_replan_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.UNAVAILABLE,
                value=None,
                sample_count=0,
                eligible_count=0,
            )

        if total_shots > 0:
            mean_remed = MetricValue(
                metric_key="mean_quality_remediations_per_shot",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.AVAILABLE,
                value=round(total_remed_decisions / total_shots, 4),
                sample_count=total_remed_decisions,
                eligible_count=total_shots,
            )
        else:
            mean_remed = MetricValue(
                metric_key="mean_quality_remediations_per_shot",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.UNAVAILABLE,
                value=None,
                sample_count=0,
                eligible_count=0,
            )

        if total_cases > 0:
            needs_action_cases = sum(
                1 for c in case_results
                if c.status == BenchmarkCaseStatus.NEEDS_USER_ACTION
            )
            needs_user_rate = MetricValue(
                metric_key="needs_user_action_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.AVAILABLE,
                value=round(needs_action_cases / total_cases, 4),
                sample_count=needs_action_cases,
                eligible_count=total_cases,
            )
        else:
            needs_user_rate = MetricValue(
                metric_key="needs_user_action_rate",
                metric_version=METRIC_SET_VERSION_V1,
                status=MetricValueStatus.UNAVAILABLE,
                value=None,
                sample_count=0,
                eligible_count=0,
            )

        # ---------------------------------------------------------------------
        # 5. Multimodal Quality Dimensions
        # Strict Rule: mean calculated ONLY from status == SCORED.
        # INDETERMINATE/ERROR are never coerced to 0.0.
        # ---------------------------------------------------------------------
        dimension_keys = [
            ("SEMANTIC_ALIGNMENT", "dimension_semantic_alignment_score"),
            ("VISUAL_QUALITY", "dimension_visual_quality_score"),
            ("KNOWLEDGE_ACCURACY", "dimension_knowledge_accuracy_score"),
            ("COMPOSITION_SUITABILITY", "dimension_composition_suitability_score"),
        ]

        dim_events = [
            ev for ev in all_trace_events
            if ev.event_type == TraceEventType.DIMENSION_EVALUATION_COMPLETED
        ]

        # Also extract from EVALUATION_COMPLETED if DIMENSION events not individually emitted
        dimension_metrics: dict[str, MetricValue] = {}
        for dim_name, metric_key in dimension_keys:
            scored_scores: list[float] = []
            eligible_instances = 0

            # Inspect DIMENSION_EVALUATION_COMPLETED
            for dev in dim_events:
                if dev.attributes.get("dimension") == dim_name:
                    eligible_instances += 1
                    status = dev.attributes.get("status")
                    score = dev.attributes.get("score")
                    if status == "SCORED" and score is not None:
                        scored_scores.append(float(score))

            # If no individual DIMENSION events, inspect EVALUATION_COMPLETED
            if eligible_instances == 0:
                for eev in eval_completed_events:
                    statuses = eev.attributes.get("dimension_statuses", {})
                    scores = eev.attributes.get("dimension_scores", {})
                    if dim_name in statuses:
                        eligible_instances += 1
                        st = statuses[dim_name]
                        sc = scores.get(dim_name)
                        if st == "SCORED" and sc is not None:
                            scored_scores.append(float(sc))

            if eligible_instances == 0:
                dim_val = MetricValue(
                    metric_key=metric_key,
                    metric_version=METRIC_SET_VERSION_V1,
                    status=MetricValueStatus.UNAVAILABLE,
                    value=None,
                    sample_count=0,
                    eligible_count=0,
                )
            elif len(scored_scores) == 0:
                dim_val = MetricValue(
                    metric_key=metric_key,
                    metric_version=METRIC_SET_VERSION_V1,
                    status=MetricValueStatus.UNAVAILABLE,
                    value=None,
                    sample_count=0,
                    eligible_count=eligible_instances,
                )
            else:
                mean_score = sum(scored_scores) / len(scored_scores)
                val_status = (
                    MetricValueStatus.AVAILABLE
                    if len(scored_scores) == eligible_instances
                    else MetricValueStatus.PARTIAL
                )
                dim_val = MetricValue(
                    metric_key=metric_key,
                    metric_version=METRIC_SET_VERSION_V1,
                    status=val_status,
                    value=round(mean_score, 4),
                    sample_count=len(scored_scores),
                    eligible_count=eligible_instances,
                )
            dimension_metrics[metric_key] = dim_val

        # ---------------------------------------------------------------------
        # 6. Human-in-the-Loop Metrics (NOT_APPLICABLE for automated benchmarks)
        # ---------------------------------------------------------------------
        human_edit_val = MetricValue(
            metric_key="human_edit_rate",
            metric_version=METRIC_SET_VERSION_V1,
            status=MetricValueStatus.NOT_APPLICABLE,
            value=None,
            sample_count=0,
            eligible_count=total_shots,
            notes="Unattended automated benchmark run",
        )
        beat_replan_val = MetricValue(
            metric_key="beat_replan_rate",
            metric_version=METRIC_SET_VERSION_V1,
            status=MetricValueStatus.NOT_APPLICABLE,
            value=None,
            sample_count=0,
            eligible_count=total_cases,
            notes="Unattended automated benchmark run",
        )

        # ---------------------------------------------------------------------
        # 7. Stage Latencies
        # ---------------------------------------------------------------------
        stage_durations: dict[str, list[float]] = defaultdict(list)

        # End-to-end duration from case results
        for c in case_results:
            if c.duration_ms is not None:
                stage_durations["end_to_end"].append(c.duration_ms)

        # Stage durations from trace events
        for ev in all_trace_events:
            dur = ev.duration_ms
            if dur is None:
                continue

            if ev.event_type == TraceEventType.CONTENT_PLANNER_COMPLETED:
                stage_durations["planning"].append(dur)
            elif ev.event_type == TraceEventType.STORYBOARD_GENERATION_COMPLETED:
                stage_durations["storyboard"].append(dur)
            elif ev.event_type == TraceEventType.ASSET_ROUTE_PLAN_CREATED:
                stage_durations["routing"].append(dur)
            elif ev.event_type == TraceEventType.EXECUTION_ATTEMPT_COMPLETED:
                stage_durations["generation"].append(dur)
            elif ev.event_type == TraceEventType.EVALUATION_COMPLETED:
                stage_durations["evaluation"].append(dur)
            elif ev.event_type == TraceEventType.QUALITY_REMEDIATION_DECIDED:
                stage_durations["remediation"].append(dur)

        stage_latency_summaries: dict[str, StageLatencySummary] = {}
        for stage_name in ["end_to_end", "planning", "storyboard", "routing", "generation", "evaluation", "remediation"]:
            durs = stage_durations.get(stage_name, [])
            if durs:
                stage_latency_summaries[stage_name] = StageLatencySummary(
                    stage=stage_name,
                    mean_ms=round(sum(durs) / len(durs), 2),
                    p50_ms=round(float(statistics.median(durs)), 2),
                    min_ms=round(min(durs), 2),
                    max_ms=round(max(durs), 2),
                    sample_count=len(durs),
                )
            else:
                stage_latency_summaries[stage_name] = StageLatencySummary(
                    stage=stage_name,
                    mean_ms=None,
                    p50_ms=None,
                    min_ms=None,
                    max_ms=None,
                    sample_count=0,
                )

        # Build latency MetricValues
        latency_metric_values: dict[str, MetricValue] = {}
        for stage_name in ["end_to_end", "planning", "storyboard", "routing", "generation", "evaluation", "remediation"]:
            lat_sum = stage_latency_summaries[stage_name]
            metric_mean_key = f"latency_{stage_name}_mean_ms"
            if lat_sum.sample_count > 0 and lat_sum.mean_ms is not None:
                latency_metric_values[metric_mean_key] = MetricValue(
                    metric_key=metric_mean_key,
                    metric_version=METRIC_SET_VERSION_V1,
                    status=MetricValueStatus.AVAILABLE,
                    value=lat_sum.mean_ms,
                    sample_count=lat_sum.sample_count,
                    eligible_count=lat_sum.sample_count,
                )
            else:
                latency_metric_values[metric_mean_key] = MetricValue(
                    metric_key=metric_mean_key,
                    metric_version=METRIC_SET_VERSION_V1,
                    status=MetricValueStatus.UNAVAILABLE,
                    value=None,
                    sample_count=0,
                    eligible_count=0,
                )

            if stage_name == "end_to_end":
                metric_p50_key = "latency_end_to_end_p50_ms"
                if lat_sum.sample_count > 0 and lat_sum.p50_ms is not None:
                    latency_metric_values[metric_p50_key] = MetricValue(
                        metric_key=metric_p50_key,
                        metric_version=METRIC_SET_VERSION_V1,
                        status=MetricValueStatus.AVAILABLE,
                        value=lat_sum.p50_ms,
                        sample_count=lat_sum.sample_count,
                        eligible_count=lat_sum.sample_count,
                    )
                else:
                    latency_metric_values[metric_p50_key] = MetricValue(
                        metric_key=metric_p50_key,
                        metric_version=METRIC_SET_VERSION_V1,
                        status=MetricValueStatus.UNAVAILABLE,
                        value=None,
                        sample_count=0,
                        eligible_count=0,
                    )

        # ---------------------------------------------------------------------
        # 8. Cost Aggregation
        # ---------------------------------------------------------------------
        observed_by_curr: dict[str, float] = defaultdict(float)
        estimated_by_curr: dict[str, float] = defaultdict(float)
        observed_ops = 0
        estimated_ops = 0
        unknown_ops = 0

        for ev in all_trace_events:
            usage = ev.attributes.get("usage")
            if isinstance(usage, dict):
                cost_status = usage.get("cost_status")
                cost_amount = usage.get("cost_amount")
                curr = usage.get("currency") or "USD"

                if cost_status == CostEstimateStatus.OBSERVED.value:
                    if cost_amount is not None:
                        observed_by_curr[curr] += float(cost_amount)
                        observed_ops += 1
                    else:
                        unknown_ops += 1
                elif cost_status == CostEstimateStatus.ESTIMATED.value:
                    if cost_amount is not None:
                        estimated_by_curr[curr] += float(cost_amount)
                        estimated_ops += 1
                    else:
                        unknown_ops += 1
                elif cost_status == CostEstimateStatus.UNKNOWN.value:
                    unknown_ops += 1

        is_offline = (run.execution_mode == BenchmarkExecutionMode.OFFLINE)
        if unknown_ops == 0 and (observed_ops > 0 or estimated_ops > 0):
            cost_comp = "COMPLETE"
        elif unknown_ops > 0 and (observed_ops > 0 or estimated_ops > 0):
            cost_comp = "PARTIAL"
        else:
            cost_comp = "UNKNOWN"

        cost_summary = CostSummary(
            observed_by_currency=dict(observed_by_curr),
            estimated_by_currency=dict(estimated_by_curr),
            observed_ops=observed_ops,
            estimated_ops=estimated_ops,
            unknown_ops=unknown_ops,
            completeness=cost_comp,
            is_synthetic=is_offline,
        )

        # ---------------------------------------------------------------------
        # 9. Provider / Model Usage
        # ---------------------------------------------------------------------
        usage_by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"attempts": 0, "successes": 0, "failures": 0})
        for ev in attempt_events:
            provider = ev.attributes.get("provider", "unknown")
            model = ev.attributes.get("model", "unknown")
            key = f"{provider}/{model}"
            usage_by_model[key]["attempts"] += 1
            if ev.attributes.get("resulting_shot_asset_version_id"):
                usage_by_model[key]["successes"] += 1
            else:
                usage_by_model[key]["failures"] += 1

        # ---------------------------------------------------------------------
        # Assemble All Metrics
        # ---------------------------------------------------------------------
        all_metrics: dict[str, MetricValue] = {
            "plan_validation_success_rate": val_plan_rate,
            "storyboard_generation_success_rate": sb_gen_rate,
            "evidence_support_rate": ev_support_rate,
            "first_asset_quality_pass_rate": first_pass_rate,
            "final_shot_quality_pass_rate": final_pass_rate,
            "mean_generation_attempts_per_shot": mean_attempts,
            "fallback_rate": fb_rate,
            "recovery_required_rate": rec_rate,
            "quality_remediation_rate": q_remed_rate,
            "mean_quality_remediations_per_shot": mean_remed,
            "controlled_visual_replan_rate": vis_replan_rate,
            "needs_user_action_rate": needs_user_rate,
            "human_edit_rate": human_edit_val,
            "beat_replan_rate": beat_replan_val,
        }
        all_metrics.update(dimension_metrics)
        all_metrics.update(latency_metric_values)

        return (
            all_metrics,
            stage_latency_summaries,
            cost_summary,
            dict(usage_by_model),
            total_cases,
            total_shots,
        )
