from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.application.knowledge_base_service import (
    KnowledgeBaseNotFoundError,
    KnowledgeBaseServiceError,
)
from app.application.knowledge_video_workflow import KnowledgeVideoWorkflow
from app.domain.knowledge_base import KnowledgeBaseStatus
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    JobStatus,
    Stage,
    TaskStatus,
    TerminalStateImmutableError,
    WorkflowConflictError,
    WorkflowPolicyType,
)
from app.persistence.repositories import (
    KnowledgeBaseRepository,
    KnowledgeVideoTaskRepository,
    StageExecutionRepository,
    WorkflowJobRepository,
)


class TaskCommandService:
    """
    Application command service handling state-changing lifecycle operations for KnowledgeVideoTask.
    Ensures transactional consistency across tasks, jobs, and state transitions.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self.task_repo = KnowledgeVideoTaskRepository(session)
        self.job_repo = WorkflowJobRepository(session)
        self.exec_repo = StageExecutionRepository(session)
        self.kb_repo = KnowledgeBaseRepository(session)
        self.workflow = KnowledgeVideoWorkflow(session)

    def create_task(
        self,
        topic: str,
        target_duration: float = 60.0,
        aspect_ratio: str = "16:9",
        language: str = "zh",
        workflow_policy: WorkflowPolicyType = WorkflowPolicyType.AUTO,
        task_metadata: dict[str, Any] | None = None,
        allow_research: bool = False,
        knowledge_base_ids: Sequence[str] | None = None,
        initial_evidence: Sequence[Any] | None = None,
        task_id: str | None = None,
        now: datetime | None = None,
    ) -> KnowledgeVideoTask:
        """
        Atomically creates a new KnowledgeVideoTask, attaches any initial knowledge bases
        and initial evidence sources (TEXT/URL), and enqueues its initial EVIDENCE job.
        Returns the created domain task.
        """
        from app.application.evidence_service import TaskEvidenceCommandService

        ts = now or datetime.now(UTC)

        # 1. Validate knowledge bases before modifying state
        kb_ids_to_attach: list[str] = []
        if knowledge_base_ids:
            for kb_id in knowledge_base_ids:
                clean_id = kb_id.strip() if isinstance(kb_id, str) else str(kb_id)
                if not clean_id:
                    continue
                kb = self.kb_repo.get_knowledge_base(clean_id)
                if kb is None:
                    raise KnowledgeBaseNotFoundError(f"Knowledge Base '{clean_id}' not found.")
                if kb.status != KnowledgeBaseStatus.ACTIVE:
                    raise KnowledgeBaseServiceError(
                        f"Cannot attach archived Knowledge Base '{clean_id}' (status: {kb.status.value})."
                    )
                if clean_id not in kb_ids_to_attach:
                    kb_ids_to_attach.append(clean_id)

        # 2. Validate initial evidence before modifying state
        validated_evidence: list[dict[str, Any]] = []
        if initial_evidence:
            for idx, item in enumerate(initial_evidence):
                if isinstance(item, dict):
                    st = item.get("source_type")
                    tc = item.get("text_content")
                    u = item.get("url")
                    tit = item.get("title")
                    auth = item.get("author")
                    meta = item.get("metadata") or {}
                else:
                    st = getattr(item, "source_type", None)
                    tc = getattr(item, "text_content", None)
                    u = getattr(item, "url", None)
                    tit = getattr(item, "title", None)
                    auth = getattr(item, "author", None)
                    meta = getattr(item, "metadata", None) or {}

                st_val = st.value if hasattr(st, "value") else str(st or "").upper()
                if st_val == "TEXT":
                    if not tc or not str(tc).strip():
                        raise ValueError(f"Initial evidence #{idx + 1} (TEXT) must have non-empty text_content.")
                    validated_evidence.append({
                        "source_type": "TEXT",
                        "text_content": str(tc).strip(),
                        "title": tit,
                        "author": auth,
                        "metadata": meta,
                    })
                elif st_val == "URL":
                    if not u or not str(u).strip():
                        raise ValueError(f"Initial evidence #{idx + 1} (URL) must have non-empty url.")
                    clean_u = str(u).strip()
                    if not (clean_u.startswith("http://") or clean_u.startswith("https://")):
                        raise ValueError(f"Initial evidence #{idx + 1} (URL) must start with http:// or https://: {clean_u}")
                    validated_evidence.append({
                        "source_type": "URL",
                        "url": clean_u,
                        "title": tit,
                        "author": auth,
                        "metadata": meta,
                    })
                else:
                    raise ValueError(f"Unsupported initial evidence source type: '{st_val}'. Only TEXT and URL are supported.")

        # 3. Create and persist the Task aggregate root
        task = KnowledgeVideoTask.create(
            topic=topic,
            target_duration=target_duration,
            aspect_ratio=aspect_ratio,
            language=language,
            workflow_policy=workflow_policy,
            task_id=task_id,
            task_metadata=task_metadata,
            allow_research=allow_research,
            now=ts,
        )
        persisted_task = self.task_repo.save_task(task)

        # 4. Attach knowledge bases
        for kb_id in kb_ids_to_attach:
            self.kb_repo.attach_to_task(persisted_task.task_id, kb_id)

        # 5. Attach initial evidence sources
        if validated_evidence:
            evidence_cmd = TaskEvidenceCommandService(self._session)
            for ev in validated_evidence:
                if ev["source_type"] == "TEXT":
                    evidence_cmd.add_text_source(
                        task_id=persisted_task.task_id,
                        text=ev["text_content"],
                        title=ev["title"],
                        author=ev["author"],
                        metadata=ev["metadata"],
                        now=ts,
                    )
                elif ev["source_type"] == "URL":
                    evidence_cmd.register_url_source(
                        task_id=persisted_task.task_id,
                        url=ev["url"],
                        title=ev["title"],
                        author=ev["author"],
                        metadata=ev["metadata"],
                        now=ts,
                    )

        # 6. Create and persist the initial EVIDENCE job
        idempotency_key = f"idemp_{persisted_task.task_id}_evidence_1"
        initial_job = WorkflowJob.create(
            task_id=persisted_task.task_id,
            stage=Stage.EVIDENCE,
            idempotency_key=idempotency_key,
            attempt_number=1,
            now=ts,
        )
        self.job_repo.create_job(initial_job)
        self._session.flush()

        return persisted_task

    def approve_task(
        self,
        task_id: str,
        now: datetime | None = None,
    ) -> KnowledgeVideoTask:
        """
        Resumes a task paused in WAITING_USER, advancing to next stage and scheduling its job.
        """
        ts = now or datetime.now(UTC)
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")

        self.workflow.resume_after_approval(task, now=ts)
        return self.task_repo.get_task(task_id) or task

    def authorize_research(
        self,
        task_id: str,
        resume_if_waiting: bool = True,
        now: datetime | None = None,
    ) -> KnowledgeVideoTask:
        """
        Authorizes web research for a task. If the task is currently waiting
        in NEEDS_EVIDENCE or WAITING_USER, automatically re-enqueues an EVIDENCE job.
        """
        ts = now or datetime.now(UTC)
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")

        if task.task_status.is_terminal:
            raise TerminalStateImmutableError(
                f"Cannot authorize research: task '{task_id}' is in terminal state '{task.task_status.value}'."
            )

        task.authorize_research(now=ts)
        self.task_repo.save_task(task)

        if resume_if_waiting and task.task_status in (
            TaskStatus.NEEDS_EVIDENCE,
            TaskStatus.WAITING_USER,
        ):
            return self.retry_task(task_id, reason="Web research authorized", now=ts)

        return task

    def retry_task(
        self,
        task_id: str,
        reason: str | None = None,
        now: datetime | None = None,
    ) -> KnowledgeVideoTask:
        """
        Retries a failed or paused task by re-enqueuing a job for its current stage.
        """
        ts = now or datetime.now(UTC)
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")

        if task.task_status.is_terminal:
            raise TerminalStateImmutableError(
                f"Cannot retry task '{task_id}' in terminal state '{task.task_status.value}'."
            )

        if task.task_status not in (
            TaskStatus.NEEDS_RECOVERY,
            TaskStatus.NEEDS_EVIDENCE,
        ):
            raise WorkflowConflictError(
                f"Task '{task_id}' is in status '{task.task_status.value}' and cannot be retried."
            )

        cur_job = self.job_repo.get_current_job_for_task(task_id)
        next_attempt = (cur_job.attempt_number + 1) if cur_job else 1

        task.transition_to(TaskStatus.RUNNING, reason=reason, now=ts)
        self.task_repo.save_task(task)

        idempotency_key = f"idemp_{task.task_id}_{task.current_stage.value.lower()}_{next_attempt}"
        new_job = WorkflowJob.create(
            task_id=task.task_id,
            stage=task.current_stage,
            idempotency_key=idempotency_key,
            attempt_number=next_attempt,
            now=ts,
        )
        self.job_repo.create_job(new_job)

        return task

    def cancel_task(
        self,
        task_id: str,
        reason: str | None = None,
        now: datetime | None = None,
    ) -> KnowledgeVideoTask:
        """
        Authoritatively cancels a task and aborts any active or queued job.
        """
        ts = now or datetime.now(UTC)
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")

        task.transition_to(TaskStatus.CANCELLED, reason=reason, now=ts)
        self.task_repo.save_task(task)

        cur_job = self.job_repo.get_current_job_for_task(task_id)
        if cur_job and cur_job.status in (
            JobStatus.QUEUED,
            JobStatus.LEASED,
            JobStatus.RUNNING,
        ):
            cur_job.status = JobStatus.CANCELLED
            cur_job.finished_at = ts
            self.job_repo.update_job(cur_job)

        return task

    def revise_task(
        self,
        task_id: str,
        target_stage: Stage | None = None,
        feedback: str | None = None,
        topic: str | None = None,
        target_duration: float | None = None,
        aspect_ratio: str | None = None,
        now: datetime | None = None,
    ) -> KnowledgeVideoTask:
        """
        Revises a task with feedback, updated parameters, or requests a rerun from an earlier stage.
        """
        from app.domain.workflow_state import STAGE_ORDER
        from app.persistence.repositories import TaskArtifactRepository

        ts = now or datetime.now(UTC)
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")

        if task.task_status.is_terminal:
            raise TerminalStateImmutableError(
                f"Cannot revise task '{task_id}' in terminal state '{task.task_status.value}'."
            )

        if topic:
            task.topic = topic
        if target_duration:
            task.target_duration = target_duration
        if aspect_ratio:
            task.aspect_ratio = aspect_ratio
        if feedback:
            meta = dict(task.task_metadata or {})
            meta["revision_feedback"] = feedback
            meta["is_partial_rerun"] = True
            task.task_metadata = meta

        stage_to_run = target_stage or task.current_stage
        task.set_stage_for_rerun(stage_to_run, now=ts)

        if task.task_status != TaskStatus.RUNNING:
            task.transition_to(TaskStatus.RUNNING, reason=feedback or "Revision requested", now=ts)
        self.task_repo.save_task(task)

        # Cancel active job if any
        cur_job = self.job_repo.get_current_job_for_task(task_id)
        if cur_job and cur_job.status in (
            JobStatus.QUEUED,
            JobStatus.LEASED,
            JobStatus.RUNNING,
        ):
            cur_job.status = JobStatus.CANCELLED
            cur_job.finished_at = ts
            self.job_repo.update_job(cur_job)

        # Mark downstream artifacts stale
        art_repo = TaskArtifactRepository(self._session)
        idx = STAGE_ORDER.index(stage_to_run)
        for s in STAGE_ORDER[idx:]:
            for ref in art_repo.list_artifact_refs_for_task(task_id, stage=s):
                art_repo.mark_artifact_stale(
                    ref.task_artifact_ref_id, reason="revision_rerun", now=ts
                )

        input_ref_id = None
        if idx > 0:
            prior_stage = STAGE_ORDER[idx - 1]
            prior_ref = art_repo.get_latest_artifact_ref(task_id, stage=prior_stage)
            if prior_ref:
                input_ref_id = prior_ref.task_artifact_ref_id

        seq = 1
        idempotency_key = f"idemp_{task.task_id}_{stage_to_run.value.lower()}_rev_{seq}"
        while self.job_repo.get_job_by_idempotency_key(idempotency_key) is not None:
            seq += 1
            idempotency_key = f"idemp_{task.task_id}_{stage_to_run.value.lower()}_rev_{seq}"

        new_job = WorkflowJob.create(
            task_id=task.task_id,
            stage=stage_to_run,
            idempotency_key=idempotency_key,
            attempt_number=1,
            input_task_artifact_ref_id=input_ref_id,
            now=ts,
        )
        self.job_repo.create_job(new_job)

        return task
