"""
Delivery Report Service.

Generates frozen, authoritative Markdown reports for the Delivery stage:
1. source_report.md: Complete source and evidence traceability report.
2. execution_report.md: Workflow execution, quality evaluation, and composition report.

Guarantees:
- Zero external network calls.
- Sensitive credentials / tokens / keys in URLs or metadata are redacted.
- Generates SHA-256 content hashes.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path
import re
from typing import Any

from sqlalchemy.orm import Session

from app.domain.workflow_state import Stage
from app.persistence.repositories import (
    CompositionOutputRepository,
    ContentPlanRepository,
    EvaluationRepository,
    EvidenceRepository,
    ExecutionRepository,
    KnowledgeVideoTaskRepository,
    ScriptRepository,
    StoryboardRepository,
    WorkflowJobRepository,
)


_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|access[_-]?token|secret|password|auth)=([^&\s\"'>]+)"),
    re.compile(r"(?i)(bearer\s+)([a-zA-Z0-9_\-\.]{12,})"),
]


def redact_secrets(text: str) -> str:
    """Redacts known API tokens, passwords, and secret query parameters."""
    if not text:
        return ""
    sanitized = text
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub(r"\1=[REDACTED]", sanitized)
    return sanitized


class DeliveryReportService:
    """Service for generating delivery reports for KnowledgeVideo tasks."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._task_repo = KnowledgeVideoTaskRepository(session)
        self._evidence_repo = EvidenceRepository(session)
        self._script_repo = ScriptRepository(session)
        self._job_repo = WorkflowJobRepository(session)
        self._comp_repo = CompositionOutputRepository(session)
        self._eval_repo = EvaluationRepository(session)
        self._exec_repo = ExecutionRepository(session)

    def generate_source_report(
        self,
        task_id: str,
        output_file_path: str,
    ) -> tuple[str, str]:
        """
        Generates source_report.md documenting all sources, evidence, claims,
        and their linkage to the produced video script.

        Returns (output_file_path, sha256_hash).
        """
        task = self._task_repo.get_task(task_id)
        topic = task.topic if task else "Unknown"
        target_dur = task.target_duration if task else 0.0

        # Gather sources and evidence snapshot
        sources = self._evidence_repo.list_sources_for_task(task_id)
        latest_snapshot = self._evidence_repo.get_latest_snapshot_for_task(task_id)
        latest_retrieval = self._evidence_repo.get_latest_retrieval_snapshot_for_task(task_id)

        evidence_items = []
        knowledge_claims = []
        if latest_snapshot:
            for eid in latest_snapshot.evidence_ids:
                item = self._evidence_repo.get_evidence_item(eid)
                if item:
                    evidence_items.append(item)
            for cid in latest_snapshot.knowledge_claim_ids:
                claim = self._evidence_repo.get_knowledge_claim(cid)
                if claim:
                    knowledge_claims.append(claim)
        elif sources:
            for src in sources:
                evidence_items.extend(
                    self._evidence_repo.list_evidence_items_for_source(src.source_document_id)
                )

        # Gather script segments for linkage
        script_rev = self._script_repo.get_latest_revision_for_task(task_id)
        segments = script_rev.segments if script_rev else ()

        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines: list[str] = [
            "# Source & Evidence Traceability Report",
            "",
            "## Task Summary",
            f"- **Task ID**: `{task_id}`",
            f"- **Topic**: {topic}",
            f"- **Target Duration**: {target_dur}s",
            f"- **Report Generated**: {now_str}",
            "",
            "## Evidence Snapshot",
        ]

        if latest_snapshot:
            lines.extend([
                f"- **Snapshot ID**: `{latest_snapshot.evidence_snapshot_id}`",
                f"- **Snapshot Version**: `{latest_snapshot.snapshot_version}`",
                f"- **Content Fingerprint**: `{latest_snapshot.content_fingerprint}`",
                f"- **Total Sources**: {len(latest_snapshot.source_document_ids)}",
                f"- **Total Evidence Items**: {len(latest_snapshot.evidence_ids)}",
                f"- **Total Knowledge Claims**: {len(latest_snapshot.knowledge_claim_ids)}",
            ])
        else:
            lines.append("_No frozen evidence snapshot recorded for this task._")

        lines.extend([
            "",
            "## Retrieval & Hybrid RAG Summary",
        ])
        if latest_retrieval:
            eff_mode = getattr(latest_retrieval, "effective_retrieval_mode", "BM25_ONLY")
            lines.extend([
                f"- **Retrieval Snapshot ID**: `{latest_retrieval.retrieval_snapshot_id}`",
                f"- **Effective Retrieval Mode**: `{eff_mode}`",
                f"- **Retrieval Policy Version**: `{latest_retrieval.retrieval_policy_version}`",
                f"- **Query**: {latest_retrieval.query}",
                f"- **Ranked Candidates**: {len(latest_retrieval.candidates)}",
            ])
        else:
            lines.append("_No retrieval execution snapshot recorded for this task._")

        lines.extend([
            "",
            "## Source Documents",
        ])

        if sources:
            lines.extend([
                "| Document ID | Title | Locator / URL | Media Type | Captured At |",
                "| --- | --- | --- | --- | --- |",
            ])
            for src in sources:
                safe_locator = redact_secrets(src.source_locator or "")
                cap_at = src.captured_at.strftime("%Y-%m-%d %H:%M:%S") if src.captured_at else "N/A"
                lines.append(
                    f"| `{src.source_document_id}` | {src.title or 'N/A'} | {safe_locator} | {src.media_type or 'text'} | {cap_at} |"
                )
        else:
            lines.append("_No source documents associated with this task._")

        lines.extend([
            "",
            "## Extracted Evidence Items",
        ])

        if evidence_items:
            lines.extend([
                "| Evidence ID | Source Doc ID | Normalized Fact / Claim | Role | Confidence |",
                "| --- | --- | --- | --- | --- |",
            ])
            for itm in evidence_items:
                fact_text = redact_secrets(itm.normalized_fact or itm.original_excerpt or "").replace("\n", " ")
                if len(fact_text) > 100:
                    fact_text = fact_text[:97] + "..."
                role_val = itm.evidence_role.value if hasattr(itm.evidence_role, "value") else str(itm.evidence_role)
                lines.append(
                    f"| `{itm.evidence_id}` | `{itm.source_document_id}` | {fact_text} | {role_val} | {itm.confidence:.2f} |"
                )
        else:
            lines.append("_No evidence items recorded._")

        lines.extend([
            "",
            "## Knowledge Claims & Verification",
        ])

        if knowledge_claims:
            lines.extend([
                "| Claim ID | Claim Statement | Status | Evidence Refs |",
                "| --- | --- | --- | --- |",
            ])
            for clm in knowledge_claims:
                stmt_text = redact_secrets(clm.claim_text or "").replace("\n", " ")
                v_stat = clm.verification_status.value if hasattr(clm.verification_status, "value") else str(clm.verification_status)
                refs = ", ".join(f"`{r}`" for r in clm.evidence_refs) if clm.evidence_refs else "None"
                lines.append(f"| `{clm.knowledge_claim_id}` | {stmt_text} | {v_stat} | {refs} |")
        else:
            lines.append("_No structured knowledge claims recorded._")

        lines.extend([
            "",
            "## Script & Segment Traceability",
        ])

        if segments:
            lines.extend([
                "| Segment # | Narration Excerpt | Target Duration | Linked Evidence |",
                "| --- | --- | --- | --- |",
            ])
            for seg in segments:
                narr = seg.narration_text.replace("\n", " ") if seg.narration_text else ""
                if len(narr) > 80:
                    narr = narr[:77] + "..."
                linked_ev = ", ".join(f"`{e}`" for e in seg.evidence_refs) if seg.evidence_refs else "N/A"
                lines.append(f"| {seg.order} | {narr} | {seg.target_duration:.1f}s | {linked_ev} |")
        else:
            lines.append("_No script segments available._")

        lines.append("")

        content = "\n".join(lines)
        content_bytes = content.encode("utf-8")
        out_path = Path(output_file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(content_bytes)

        sha256_hash = hashlib.sha256(content_bytes).hexdigest()
        return str(out_path), sha256_hash

    def generate_execution_report(
        self,
        task_id: str,
        output_file_path: str,
        evaluation_snapshot_id: str | None = None,
        composition_output_id: str | None = None,
    ) -> tuple[str, str]:
        """
        Generates execution_report.md documenting workflow execution history,
        quality evaluations, remediations, and composition details.

        Returns (output_file_path, sha256_hash).
        """
        task = self._task_repo.get_task(task_id)
        topic = task.topic if task else "Unknown"
        policy_val = task.workflow_policy.value if (task and hasattr(task.workflow_policy, "value")) else "AUTO"
        task_status = task.task_status.value if (task and hasattr(task.task_status, "value")) else "RUNNING"
        created_at_str = task.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if (task and task.created_at) else "N/A"

        jobs = self._job_repo.list_jobs_for_task(task_id)
        sorted_jobs = sorted(jobs, key=lambda j: j.created_at)

        # Composition details
        comp_output = None
        if composition_output_id:
            comp_output = self._comp_repo.get_composition_output(composition_output_id)
        if not comp_output:
            comp_output = self._comp_repo.get_latest_composition_output_for_task(task_id)

        # Evaluation details
        eval_snap = None
        dim_results = ()
        if evaluation_snapshot_id:
            eval_snap = self._eval_repo.get_snapshot(evaluation_snapshot_id)
        if eval_snap:
            dim_results = self._eval_repo.get_snapshot_dimension_results(eval_snap.evaluation_snapshot_id)

        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines: list[str] = [
            "# Workflow Execution & Quality Delivery Report",
            "",
            "## Task Execution Overview",
            f"- **Task ID**: `{task_id}`",
            f"- **Topic**: {topic}",
            f"- **Workflow Policy**: `{policy_val}`",
            f"- **Status**: `{task_status}`",
            f"- **Started At**: {created_at_str}",
            f"- **Completed / Delivered At**: {now_str}",
            "",
            "## Workflow Stages History",
            "| Stage | Job ID | Status | Attempt # | Duration (s) | Error / Info |",
            "| --- | --- | --- | --- | --- | --- |",
        ]

        if sorted_jobs:
            for j in sorted_jobs:
                dur_str = "N/A"
                job_end = getattr(j, "finished_at", None) or getattr(j, "completed_at", None)
                if j.started_at and job_end:
                    dur_sec = (job_end - j.started_at).total_seconds()
                    dur_str = f"{dur_sec:.2f}"
                elif j.started_at:
                    dur_str = "running"

                err_info = redact_secrets(j.error_message or "-")
                if len(err_info) > 60:
                    err_info = err_info[:57] + "..."

                lines.append(
                    f"| `{j.stage.value}` | `{j.job_id}` | `{j.status.value}` | {j.attempt_number} | {dur_str} | {err_info} |"
                )
        else:
            lines.append("| _None_ | _None_ | _None_ | _None_ | _None_ | _No jobs recorded_ |")

        lines.extend([
            "",
            "## Quality Review & Evaluation Summary",
        ])

        if eval_snap:
            lines.extend([
                f"- **Evaluation Snapshot ID**: `{eval_snap.evaluation_snapshot_id}`",
                f"- **Overall Decision**: **{eval_snap.decision.value if hasattr(eval_snap.decision, 'value') else eval_snap.decision}**",
                f"- **Overall Quality Score**: {eval_snap.overall_score:.2f}" if eval_snap.overall_score is not None else "- **Overall Quality Score**: N/A",
                f"- **Policy Version**: `{eval_snap.policy_version}`",
                f"- **Evaluator Version**: `{eval_snap.evaluator_version}`",
                f"- **Reason Codes**: {', '.join(eval_snap.summary_reason_codes) if eval_snap.summary_reason_codes else 'None'}",
                "",
                "### Multimodal Dimension Results",
            ])
            if dim_results:
                lines.extend([
                    "| Dimension | Status | Score | Reason Codes |",
                    "| --- | --- | --- | --- |",
                ])
                for dr in dim_results:
                    dim_name = dr.dimension.value if hasattr(dr.dimension, "value") else str(dr.dimension)
                    stat_val = dr.status.value if hasattr(dr.status, "value") else str(dr.status)
                    score_val = f"{dr.score:.2f}" if dr.score is not None else "N/A"
                    r_codes = ", ".join(dr.reason_codes) if dr.reason_codes else "None"
                    lines.append(f"| {dim_name} | {stat_val} | {score_val} | {r_codes} |")
            else:
                lines.append("_No individual dimension results recorded in snapshot._")
        else:
            lines.append("_No evaluation snapshot associated with this delivery._")

        lines.extend([
            "",
            "## Final Media Composition Artifacts",
        ])

        if comp_output:
            fps_str = f"{comp_output.fps:.2f} fps" if comp_output.fps is not None else "30.00 fps"
            vcodec_str = comp_output.video_codec or "h264"
            acodec_str = comp_output.audio_codec or "aac"
            fsize_str = f"{comp_output.file_size:,}" if comp_output.file_size is not None else "0"
            lines.extend([
                f"- **Composition Output ID**: `{comp_output.composition_output_id}`",
                f"- **Video Path**: `{comp_output.video_path}`",
                f"- **Video SHA-256**: `{comp_output.video_hash}`",
                f"- **Duration**: {comp_output.duration:.2f}s",
                f"- **Resolution**: {comp_output.width}x{comp_output.height}",
                f"- **Frame Rate**: {fps_str}",
                f"- **Video Codec**: {vcodec_str}",
                f"- **Audio Codec**: {acodec_str}",
                f"- **File Size**: {fsize_str} bytes",
            ])
        else:
            lines.append("_No composition output record found._")

        lines.append("")

        content = "\n".join(lines)
        content_bytes = content.encode("utf-8")
        out_path = Path(output_file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(content_bytes)

        sha256_hash = hashlib.sha256(content_bytes).hexdigest()
        return str(out_path), sha256_hash
