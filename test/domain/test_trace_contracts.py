import unittest
from datetime import UTC, datetime

from app.domain.trace import (
    ChainOfThoughtPersistenceError,
    CostEstimateStatus,
    CostUsageTraceData,
    EvaluationCompletedTraceData,
    ExecutionFallbackTraceData,
    PlannerCompletedTraceData,
    PlannerStartedTraceData,
    RemediationTraceData,
    RouterDecisionTraceData,
    TraceContext,
    TraceEvent,
    TraceEventStatus,
    TraceEventType,
    TraceRoot,
    sanitize_attributes,
)


class TestTraceContracts(unittest.TestCase):

    def test_trace_root_initialization(self):
        """Invariant 1: One production workflow creates one stable trace_id."""
        root = TraceRoot()
        self.assertTrue(root.trace_id)
        self.assertEqual(root.root_type, "PRODUCTION_WORKFLOW")
        self.assertEqual(root.metadata_version, "v1")

    def test_trace_context_preserves_exact_domain_ids(self):
        """Invariant 3: TraceContext preserves explicit domain IDs without generic map."""
        ctx = TraceContext(
            trace_id="tr-100",
            parent_event_id="ev-parent",
            content_plan_revision_id="cpr-1",
            storyboard_snapshot_id="sb-1",
            asset_route_plan_id="arp-1",
            execution_run_id="run-1",
            shot_id="shot-1",
            shot_revision_id="srev-1",
            execution_attempt_id="att-1",
            shot_asset_version_id="asset-1",
            evaluation_snapshot_id="eval-1",
        )
        self.assertEqual(ctx.trace_id, "tr-100")
        self.assertEqual(ctx.parent_event_id, "ev-parent")
        self.assertEqual(ctx.content_plan_revision_id, "cpr-1")
        self.assertEqual(ctx.shot_asset_version_id, "asset-1")
        self.assertEqual(ctx.evaluation_snapshot_id, "eval-1")

        # Child context derivation propagates trace_id and updates parent_event_id
        child_ctx = ctx.child_context(parent_event_id="ev-child-parent", shot_id="shot-2")
        self.assertEqual(child_ctx.trace_id, "tr-100")
        self.assertEqual(child_ctx.parent_event_id, "ev-child-parent")
        self.assertEqual(child_ctx.shot_id, "shot-2")
        self.assertEqual(child_ctx.content_plan_revision_id, "cpr-1")

    def test_trace_event_lifecycle_and_duration(self):
        """Invariant 5 & 27: Event completion records deterministic duration and prevents mutation."""
        ctx = TraceContext(trace_id="tr-100")
        t0 = datetime(2026, 9, 11, 10, 0, 0, tzinfo=UTC)
        t1 = datetime(2026, 9, 11, 10, 0, 5, tzinfo=UTC)

        event = TraceEvent(
            trace_id="tr-100",
            event_type=TraceEventType.CONTENT_PLANNER_STARTED,
            status=TraceEventStatus.STARTED,
            started_at=t0,
            context=ctx,
        )
        self.assertEqual(event.status, TraceEventStatus.STARTED)
        self.assertIsNone(event.completed_at)
        self.assertIsNone(event.duration_ms)

        # Complete the event
        completed = event.complete(
            status=TraceEventStatus.SUCCEEDED,
            completed_at=t1,
            duration_ms=5000.0,
            attributes_update={"plan_id": "cpr-1"},
        )
        self.assertEqual(completed.trace_event_id, event.trace_event_id)
        self.assertEqual(completed.status, TraceEventStatus.SUCCEEDED)
        self.assertEqual(completed.completed_at, t1)
        self.assertEqual(completed.duration_ms, 5000.0)
        self.assertEqual(completed.attributes["plan_id"], "cpr-1")

        # Cannot complete already completed event
        with self.assertRaises(ValueError):
            completed.complete(status=TraceEventStatus.SUCCEEDED)

    def test_prohibit_chain_of_thought(self):
        """Invariant 19: No Chain-of-Thought or hidden reasoning is allowed in Trace attributes."""
        ctx = TraceContext(trace_id="tr-100")

        with self.assertRaises((ChainOfThoughtPersistenceError, ValueError)):
            TraceEvent(
                trace_id="tr-100",
                event_type=TraceEventType.CONTENT_PLANNER_COMPLETED,
                context=ctx,
                attributes={"chain_of_thought": "Thinking step by step..."},
            )

        with self.assertRaises(ChainOfThoughtPersistenceError):
            sanitize_attributes({"internal_monologue": "Secret thought"})

    def test_secrets_redaction(self):
        """Invariant 20: API keys, tokens, and secrets are redacted."""
        ctx = TraceContext(trace_id="tr-100")
        event = TraceEvent(
            trace_id="tr-100",
            event_type=TraceEventType.EXECUTION_ATTEMPT_STARTED,
            context=ctx,
            attributes={
                "provider": "openai",
                "api_key": "sk-1234567890abcdef",
                "bearer_token": "Bearer xyz",
                "nested": {"password": "admin"},
            },
        )
        self.assertEqual(event.attributes["provider"], "openai")
        self.assertEqual(event.attributes["api_key"], "[REDACTED]")
        self.assertEqual(event.attributes["bearer_token"], "[REDACTED]")
        self.assertEqual(event.attributes["nested"]["password"], "[REDACTED]")

    def test_large_content_and_binary_safety(self):
        """Invariant 21: Large media binary is not placed into TraceEvent payload."""
        ctx = TraceContext(trace_id="tr-100")
        binary_data = b"\x00\x01\x02" * 5000
        event = TraceEvent(
            trace_id="tr-100",
            event_type=TraceEventType.SHOT_ASSET_CREATED,
            context=ctx,
            attributes={"media_bytes": binary_data, "file_hash": "a" * 64},
        )
        self.assertIn("[BINARY CONTENT", event.attributes["media_bytes"])
        self.assertEqual(event.attributes["file_hash"], "a" * 64)

    def test_cost_usage_semantics(self):
        """Invariant 22 & 23: Cost status distinguishes OBSERVED/ESTIMATED/UNKNOWN, UNKNOWN is not 0."""
        # UNKNOWN with None is valid
        c1 = CostUsageTraceData(cost_status=CostEstimateStatus.UNKNOWN, cost_amount=None)
        self.assertEqual(c1.cost_status, CostEstimateStatus.UNKNOWN)
        self.assertIsNone(c1.cost_amount)

        # UNKNOWN cannot be converted to 0.0
        with self.assertRaises(ValueError):
            CostUsageTraceData(cost_status=CostEstimateStatus.UNKNOWN, cost_amount=0.0)

        # OBSERVED cost
        c2 = CostUsageTraceData(
            cost_status=CostEstimateStatus.OBSERVED,
            cost_amount=0.015,
            currency="USD",
            input_tokens=150,
            output_tokens=300,
        )
        self.assertEqual(c2.cost_status, CostEstimateStatus.OBSERVED)
        self.assertEqual(c2.cost_amount, 0.015)

        # ESTIMATED cost
        c3 = CostUsageTraceData(
            cost_status=CostEstimateStatus.ESTIMATED,
            cost_amount=0.02,
        )
        self.assertEqual(c3.cost_status, CostEstimateStatus.ESTIMATED)

    def test_typed_attribute_schemas(self):
        """Validate typed schemas for all major trace event families."""
        # Planner
        p_start = PlannerStartedTraceData(topic="AI revolution", target_duration=60.0)
        self.assertEqual(p_start.topic, "AI revolution")

        p_comp = PlannerCompletedTraceData(
            content_plan_revision_id="cpr-1",
            beat_count=4,
            evidence_ref_count=2,
            repair_retry_count=1,
            model_reference="gpt-4o",
        )
        self.assertEqual(p_comp.beat_count, 4)

        # Router
        r_data = RouterDecisionTraceData(
            asset_route_plan_id="arp-1",
            shot_id="shot-1",
            shot_revision_id="srev-1",
            requested_visual_type="AI_VIDEO",
            strategy="QUALITY_FIRST",
            selection_mode="AUTO",
            selected_provider="seedance",
            selected_model="pro",
            generation_mode="TEXT_TO_VIDEO",
            selected_score=0.92,
            eligible_candidate_count=3,
            rejected_candidate_count=1,
            key_reason_codes=("HIGH_QUALITY",),
            routing_policy_version="v1",
        )
        self.assertEqual(r_data.selected_provider, "seedance")

        # Execution Fallback
        f_data = ExecutionFallbackTraceData(
            execution_run_id="run-1",
            shot_id="shot-1",
            from_provider="kling",
            from_model="standard",
            to_provider="seedance",
            to_model="pro",
            reason_code="PROVIDER_TIMEOUT",
        )
        self.assertEqual(f_data.reason_code, "PROVIDER_TIMEOUT")

        # Evaluation
        e_data = EvaluationCompletedTraceData(
            evaluation_target_id="tgt-1",
            evaluation_snapshot_id="snap-1",
            shot_asset_version_id="asset-1",
            evaluation_policy_version="v1",
            dimension_statuses={
                "SEMANTIC_ALIGNMENT": "SCORED",
                "VISUAL_QUALITY": "SCORED",
                "KNOWLEDGE_ACCURACY": "SCORED",
                "COMPOSITION_SUITABILITY": "SCORED",
            },
            dimension_scores={
                "SEMANTIC_ALIGNMENT": 0.85,
                "VISUAL_QUALITY": 0.90,
                "KNOWLEDGE_ACCURACY": 0.95,
                "COMPOSITION_SUITABILITY": 0.88,
            },
            decision="PASS",
        )
        self.assertEqual(e_data.decision, "PASS")
        self.assertEqual(e_data.dimension_scores["VISUAL_QUALITY"], 0.90)

        # Remediation
        rem_data = RemediationTraceData(
            remediation_decision_id="rem-1",
            quality_chain_id="qchain-1",
            source_evaluation_snapshot_id="snap-1",
            action="REGENERATE_SAME_ROUTE",
            reason_codes=("VISUAL_QUALITY_BELOW_THRESHOLD",),
            quality_attempt_index=1,
        )
        self.assertEqual(rem_data.action, "REGENERATE_SAME_ROUTE")
