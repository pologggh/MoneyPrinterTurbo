from __future__ import annotations

from pydantic import BaseModel

from app.domain.workflow_state import Stage, WorkflowPolicyType


class WorkflowPolicy(BaseModel):
    """
    Controls milestone pause points and operational boundaries.
    Both AUTO and REVIEW execution modes use the exact same workflow state machine.
    """
    policy_type: WorkflowPolicyType = WorkflowPolicyType.AUTO

    def should_pause_for_review(self, completed_stage: Stage) -> bool:
        """
        Returns True if the workflow should transition to WAITING_USER
        after completing the specified stage.
        """
        if self.policy_type == WorkflowPolicyType.AUTO:
            return False

        # Under REVIEW policy, pauses occur at:
        # 1. After KNOWLEDGE_PLAN (Evidence & Outline ready for user review)
        # 2. After STORYBOARD (Script & Visual Storyboard ready for user approval)
        # 3. After QUALITY_REVIEW (Full composition preview ready before delivery)
        review_checkpoints = {
            Stage.KNOWLEDGE_PLAN,
            Stage.STORYBOARD,
            Stage.QUALITY_REVIEW,
        }
        return completed_stage in review_checkpoints
