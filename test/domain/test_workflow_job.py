from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain.workflow_state import JobStatus, Stage, WorkflowConflictError
from app.domain.workflow_job import WorkflowJob


def test_create_workflow_job_defaults():
    now = datetime.now(UTC)
    job = WorkflowJob.create(
        task_id="task_123",
        stage=Stage.EVIDENCE,
        idempotency_key="idemp_task_123_evidence_1",
        now=now,
    )
    assert job.task_id == "task_123"
    assert job.stage == Stage.EVIDENCE
    assert job.status == JobStatus.QUEUED
    assert job.attempt_number == 1
    assert job.max_attempts == 3
    assert job.lease_owner is None
    assert job.lease_expires_at is None


def test_acquire_lease_and_heartbeat():
    now = datetime.now(UTC)
    job = WorkflowJob.create(
        task_id="task_123",
        stage=Stage.EVIDENCE,
        idempotency_key="idemp_1",
        now=now,
    )

    # Worker A acquires lease for 300s
    job.acquire_lease(owner="worker_a", lease_duration_seconds=300, now=now)
    assert job.status == JobStatus.LEASED
    assert job.lease_owner == "worker_a"
    assert job.lease_expires_at == now + timedelta(seconds=300)
    assert job.heartbeat_at == now

    # Worker B cannot acquire active lease
    with pytest.raises(WorkflowConflictError):
        job.acquire_lease(owner="worker_b", lease_duration_seconds=300, now=now + timedelta(seconds=10))

    # Worker A can renew lease
    job.renew_lease(owner="worker_a", extend_seconds=180, now=now + timedelta(seconds=50))
    assert job.lease_expires_at == now + timedelta(seconds=50 + 180)


def test_lease_expiration_and_reacquisition():
    now = datetime.now(UTC)
    job = WorkflowJob.create(
        task_id="task_123",
        stage=Stage.EVIDENCE,
        idempotency_key="idemp_1",
        now=now,
    )
    job.acquire_lease(owner="worker_a", lease_duration_seconds=60, now=now)

    # At t=70s, lease has expired
    t_after_expiry = now + timedelta(seconds=70)
    assert job.is_lease_expired(now=t_after_expiry)

    # Worker B can acquire the expired lease
    job.acquire_lease(owner="worker_b", lease_duration_seconds=300, now=t_after_expiry)
    assert job.lease_owner == "worker_b"
    assert job.status == JobStatus.LEASED


def test_job_mark_succeeded_and_mark_failed():
    now = datetime.now(UTC)
    job = WorkflowJob.create(
        task_id="task_123",
        stage=Stage.KNOWLEDGE_PLAN,
        idempotency_key="idemp_kp_1",
        now=now,
    )
    job.acquire_lease(owner="worker_a", lease_duration_seconds=300, now=now)
    job.start(now=now)
    assert job.status == JobStatus.RUNNING

    job.mark_succeeded(output_artifact_revision_id="art_kp_rev_1", now=now + timedelta(seconds=10))
    assert job.status == JobStatus.SUCCEEDED
    assert job.output_artifact_revision_id == "art_kp_rev_1"
    assert job.finished_at is not None
