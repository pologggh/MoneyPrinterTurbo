from __future__ import annotations

import pytest

from app.domain.workflow_state import (
    InvalidStageTransitionError,
    InvalidStateTransitionError,
    JobErrorType,
    Stage,
    TaskStatus,
    TerminalStateImmutableError,
    WorkflowPolicyType,
)
from app.domain.knowledge_video_task import KnowledgeVideoTask


def test_task_status_and_current_stage_are_distinct_types():
    """Verify task_status and current_stage use strictly separated enums."""
    task = KnowledgeVideoTask.create(
        topic="Quantum Computing",
        target_duration=60.0,
        aspect_ratio="16:9",
        language="zh",
        workflow_policy=WorkflowPolicyType.REVIEW,
    )
    assert isinstance(task.task_status, TaskStatus)
    assert isinstance(task.current_stage, Stage)
    assert task.task_status == TaskStatus.CREATED
    assert task.current_stage == Stage.EVIDENCE


def test_legal_state_transitions():
    """Verify all required legal state transitions succeed."""
    task = KnowledgeVideoTask.create(topic="Physics")

    # CREATED -> RUNNING
    task.transition_to(TaskStatus.RUNNING)
    assert task.task_status == TaskStatus.RUNNING

    # RUNNING -> WAITING_USER
    task.transition_to(TaskStatus.WAITING_USER, reason="Review outline")
    assert task.task_status == TaskStatus.WAITING_USER
    assert task.waiting_reason == "Review outline"

    # WAITING_USER -> RUNNING
    task.transition_to(TaskStatus.RUNNING)
    assert task.task_status == TaskStatus.RUNNING
    assert task.waiting_reason is None

    # RUNNING -> NEEDS_EVIDENCE
    task.transition_to(TaskStatus.NEEDS_EVIDENCE, reason="Lack of primary sources")
    assert task.task_status == TaskStatus.NEEDS_EVIDENCE

    # NEEDS_EVIDENCE -> RUNNING
    task.transition_to(TaskStatus.RUNNING)
    assert task.task_status == TaskStatus.RUNNING

    # RUNNING -> NEEDS_RECOVERY
    task.transition_to(TaskStatus.NEEDS_RECOVERY, reason="Max retries reached in asset generation")
    assert task.task_status == TaskStatus.NEEDS_RECOVERY

    # NEEDS_RECOVERY -> RUNNING
    task.transition_to(TaskStatus.RUNNING)
    assert task.task_status == TaskStatus.RUNNING

    # RUNNING -> COMPLETED
    task.transition_to(TaskStatus.COMPLETED)
    assert task.task_status == TaskStatus.COMPLETED
    assert task.finished_at is not None


def test_non_terminal_to_failed_or_cancelled():
    """Verify any non-terminal state can transition to FAILED or CANCELLED."""
    # Test FAILED
    task1 = KnowledgeVideoTask.create(topic="Math")
    task1.transition_to(TaskStatus.RUNNING)
    task1.transition_to(TaskStatus.WAITING_USER)
    task1.transition_to(TaskStatus.FAILED, error_type=JobErrorType.FATAL, error_message="Fatal crash")
    assert task1.task_status == TaskStatus.FAILED
    assert task1.error_type == JobErrorType.FATAL
    assert task1.error_message == "Fatal crash"

    # Test CANCELLED
    task2 = KnowledgeVideoTask.create(topic="Chemistry")
    task2.transition_to(TaskStatus.RUNNING)
    task2.transition_to(TaskStatus.NEEDS_EVIDENCE)
    task2.transition_to(TaskStatus.CANCELLED, reason="Cancelled by user")
    assert task2.task_status == TaskStatus.CANCELLED


def test_terminal_state_immutable():
    """Verify completed, failed, and cancelled tasks reject any further transitions."""
    # Completed task
    task_comp = KnowledgeVideoTask.create(topic="Topic A")
    task_comp.transition_to(TaskStatus.RUNNING)
    task_comp.transition_to(TaskStatus.COMPLETED)

    with pytest.raises(TerminalStateImmutableError):
        task_comp.transition_to(TaskStatus.RUNNING)

    with pytest.raises(TerminalStateImmutableError):
        task_comp.transition_to(TaskStatus.FAILED)

    with pytest.raises(TerminalStateImmutableError):
        task_comp.transition_to(TaskStatus.CANCELLED)

    # Cancelled task
    task_canc = KnowledgeVideoTask.create(topic="Topic B")
    task_canc.transition_to(TaskStatus.CANCELLED)

    with pytest.raises(TerminalStateImmutableError):
        task_canc.transition_to(TaskStatus.RUNNING)

    with pytest.raises(TerminalStateImmutableError):
        task_canc.transition_to(TaskStatus.CANCELLED)

    # Failed task
    task_fail = KnowledgeVideoTask.create(topic="Topic C")
    task_fail.transition_to(TaskStatus.FAILED)

    with pytest.raises(TerminalStateImmutableError):
        task_fail.transition_to(TaskStatus.COMPLETED)


def test_illegal_state_transitions():
    """Verify illegal transitions throw InvalidStateTransitionError."""
    task = KnowledgeVideoTask.create(topic="Topic D")

    # CREATED cannot jump directly to COMPLETED or WAITING_USER
    with pytest.raises(InvalidStateTransitionError):
        task.transition_to(TaskStatus.COMPLETED)

    with pytest.raises(InvalidStateTransitionError):
        task.transition_to(TaskStatus.WAITING_USER)


def test_stage_progression_order():
    """Verify stages progress sequentially through the defined knowledge video lifecycle."""
    task = KnowledgeVideoTask.create(topic="History")
    assert task.current_stage == Stage.EVIDENCE

    task.transition_to(TaskStatus.RUNNING)
    task.advance_stage(Stage.KNOWLEDGE_PLAN)
    assert task.current_stage == Stage.KNOWLEDGE_PLAN

    task.advance_stage(Stage.SCRIPT)
    assert task.current_stage == Stage.SCRIPT

    task.advance_stage(Stage.STORYBOARD)
    assert task.current_stage == Stage.STORYBOARD

    # Disallow skipping stages (e.g., directly from STORYBOARD to DELIVERY)
    with pytest.raises(InvalidStageTransitionError):
        task.advance_stage(Stage.DELIVERY)
