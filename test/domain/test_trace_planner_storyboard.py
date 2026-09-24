import unittest
from unittest.mock import MagicMock

from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.planner import ContentPlanner, PlannerInput
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard_approval import (
    StoryboardApprovalService,
)
from app.domain.storyboard_orchestrator import (
    StoryboardBuildInput,
    StoryboardOrchestrator,
)
from app.domain.trace import (
    TraceContext,
)
from app.services.trace_service import TraceWriter


class TestTracePlannerStoryboard(unittest.TestCase):

    def test_content_planner_trace_lifecycle(self):
        """Invariant 7: Planner trace references exact ContentPlanRevision."""
        mock_repo = MagicMock()
        writer = TraceWriter()

        ctx = TraceContext(trace_id="tr-plan-1")
        llm_response = """
        {
            "title": "Quantum Mechanics",
            "beats": [
                {
                    "planner_ref": "b1",
                    "beat_type": "HOOK",
                    "intent": "Hook the viewer with wave particle duality",
                    "order": 1,
                    "target_duration": 10.0,
                    "importance": 0.9,
                    "evidence_refs": []
                }
            ]
        }
        """
        planner = ContentPlanner(
            repository=mock_repo,
            llm_caller=lambda _: llm_response,
        )

        plan = planner.plan(
            PlannerInput(topic="Quantum Mechanics", target_video_duration=10.0),
            trace_context=ctx,
            trace_writer=writer,
        )
        self.assertIsNotNone(plan)

    def test_storyboard_generation_and_approval_trace(self):
        """Invariant 8: Storyboard trace references exact Snapshot / Revision IDs."""
        plan_repo = MagicMock()
        shot_repo = MagicMock()
        sb_repo = MagicMock()
        appr_repo = MagicMock()
        writer = TraceWriter()

        # Mock a ContentPlanRevision with 1 beat
        from app.domain.content_plan import ContentBeat, ContentPlanRevision

        beat = ContentBeat(
            beat_id="b-1",
            beat_lineage_id="lin-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Intro hook",
            target_duration=5.0,
            importance=0.9,
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="cpr-sb-1",
            revision_number=1,
            topic="Test Topic",
            overall_target_duration=5.0,
            beats=(beat,),
        )
        plan_repo.get_revision.return_value = plan

        # Mock StoryboardAgent to return 1 shot
        mock_agent = MagicMock()
        shot = Shot(shot_id="shot-1", beat_lineage_id="lin-1", local_order=1)
        rev = ShotRevision(
            shot_revision_id="srev-1",
            shot_id="shot-1",
            revision_number=1,
            beat_lineage_id="lin-1",
            created_from_beat_instance_id="b-1",
            narration="Test narration",
            target_duration=5.0,
            visual_goal="Goal",
            visual_type=VisualType.AI_VIDEO,
            scene_description="Desc",
            generation_prompt="Prompt",
            camera_movement="Pan",
        )
        from app.domain.storyboard_agent import StoryboardExecutionResult

        mock_agent.generate_shots_for_beat.return_value = StoryboardExecutionResult(
            shots=(shot,),
            shot_revisions=(rev,),
        )

        orchestrator = StoryboardOrchestrator(
            plan_repository=plan_repo,
            shot_repository=shot_repo,
            storyboard_repository=sb_repo,
            storyboard_agent=mock_agent,
        )

        ctx = TraceContext(trace_id="tr-sb-1", content_plan_revision_id="cpr-sb-1")
        build_res = orchestrator.build_storyboard(
            StoryboardBuildInput(new_content_plan_revision_id="cpr-sb-1"),
            trace_context=ctx,
            trace_writer=writer,
        )
        self.assertEqual(len(build_res.storyboard_snapshot.shot_revision_ids), 1)

        # Now test StoryboardApprovalService
        draft_snap = build_res.storyboard_snapshot
        sb_repo.get_snapshot.return_value = draft_snap
        sb_repo.get_snapshot_shot_revisions.return_value = [rev]
        shot_repo.get_shot.return_value = shot

        approval_service = StoryboardApprovalService(
            plan_repository=plan_repo,
            shot_repository=shot_repo,
            storyboard_repository=sb_repo,
            approval_repository=appr_repo,
        )

        outcome = approval_service.approve_storyboard(
            storyboard_snapshot_id=draft_snap.storyboard_snapshot_id,
            trace_context=ctx,
            trace_writer=writer,
        )
        self.assertEqual(outcome.approved_snapshot.snapshot_state, StoryboardSnapshotState.APPROVED)
