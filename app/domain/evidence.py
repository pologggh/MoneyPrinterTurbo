from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# Domain Enums
# =============================================================================


class SourceType(str, Enum):
    """Authoritative source category.

    Strictly factual / external provenance only. No synthetic LLM hallucination
    or model memory is permitted as a source document.
    """
    TEXT = "TEXT"
    FILE = "FILE"
    URL = "URL"
    KNOWLEDGE_BASE = "KNOWLEDGE_BASE"


class SourceStatus(str, Enum):
    """Lifecycle status representing actual physical/ingestion reality."""
    REGISTERED = "REGISTERED"  # Captured locator / metadata / raw bytes; parsing pending
    READY = "READY"            # Content ingested and available for evidence extraction
    FAILED = "FAILED"          # Download, read, or validation failure


class EvidenceRole(str, Enum):
    """Semantic role that an evidence item fulfills."""
    FACTUAL_SUPPORT = "FACTUAL_SUPPORT"
    SOURCE_VISUAL = "SOURCE_VISUAL"
    CONTEXT = "CONTEXT"


class ClaimType(str, Enum):
    """Epistemic classification of a knowledge statement."""
    FACT = "FACT"
    INTERPRETATION = "INTERPRETATION"
    OPINION = "OPINION"


class VerificationStatus(str, Enum):
    """Grounding verification state of a knowledge claim."""
    GROUNDED = "GROUNDED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTED = "CONFLICTED"
    NOT_REQUIRED = "NOT_REQUIRED"


# =============================================================================
# Domain Exceptions
# =============================================================================


class EvidenceDomainError(Exception):
    """Base exception for evidence domain rule violations."""


class EvidenceNotFoundError(EvidenceDomainError):
    """Raised when an evidence ID or reference cannot be resolved."""


class InvalidClaimGroundingError(EvidenceDomainError):
    """Raised when a claim violates grounding invariants (e.g. ungrounded FACT marked GROUNDED)."""


class UnsupportedSourceTypeError(EvidenceDomainError):
    """Raised when an unsupported or synthetic source type is supplied."""


class DocumentTextNotExtractableError(EvidenceDomainError):
    """Raised when a document (such as a scanned or image-only PDF) has no extractable text."""


class SourceContentUnavailableError(EvidenceDomainError):
    """Raised when source content, file bytes, or storage locator is unavailable or missing."""


class UrlFetchError(EvidenceDomainError):
    """Raised when an explicit URL cannot be fetched or violates security / size constraints."""


# =============================================================================
# Deterministic Hashing & Identification Helpers
# =============================================================================


def compute_sha256(content: str | bytes) -> str:
    """Compute deterministic SHA-256 hash. Never uses Python's randomized hash()."""
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    else:
        content_bytes = content
    return hashlib.sha256(content_bytes).hexdigest()


def compute_source_fingerprint(
    source_type: SourceType,
    canonical_locator: str,
    content_hash: str,
) -> str:
    """Compute deterministic fingerprint for a SourceDocument."""
    payload = f"{source_type.value}:{canonical_locator.strip()}:{content_hash}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_evidence_id(
    source_fingerprint: str,
    locator: dict[str, Any] | str,
    excerpt_hash: str,
    evidence_role: EvidenceRole,
) -> str:
    """Compute deterministic, stable evidence ID in the format `ev_<sha256[:24]>`."""
    if isinstance(locator, dict):
        loc_str = json.dumps(locator, sort_keys=True, separators=(",", ":"))
    else:
        loc_str = str(locator).strip()
    payload = f"{source_fingerprint}:{loc_str}:{excerpt_hash}:{evidence_role.value}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"ev_{digest[:24]}"


def compute_snapshot_fingerprint(
    source_document_ids: Sequence[str],
    evidence_ids: Sequence[str],
    knowledge_claim_ids: Sequence[str],
) -> str:
    """Compute deterministic content fingerprint across all entities in an EvidenceSnapshot."""
    srcs = ",".join(sorted(source_document_ids))
    evs = ",".join(sorted(evidence_ids))
    claims = ",".join(sorted(knowledge_claim_ids))
    payload = f"{srcs}|{evs}|{claims}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_chunk_id(
    source_fingerprint: str,
    processing_version: str,
    locator: dict[str, Any] | str,
    normalized_text: str,
) -> str:
    """Compute deterministic, stable chunk ID in the format `chk_<sha256[:24]>`."""
    if isinstance(locator, dict):
        loc_str = json.dumps(locator, sort_keys=True, separators=(",", ":"))
    else:
        loc_str = str(locator).strip()
    text_hash = compute_sha256(normalized_text)
    payload = f"{source_fingerprint}:{processing_version}:{loc_str}:{text_hash}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"chk_{digest[:24]}"


def compute_retrieval_snapshot_fingerprint(
    task_id: str,
    query: str,
    source_scope_ids: Sequence[str],
    retrieval_policy_version: str,
    processing_version: str,
    candidates: Sequence[Any],
    selected_evidence_ids: Sequence[str],
) -> str:
    """Compute deterministic content fingerprint for a RetrievalSnapshot."""
    scopes_str = ",".join(sorted(source_scope_ids))
    ev_str = ",".join(sorted(selected_evidence_ids))
    cand_items = []
    for c in candidates:
        cid = getattr(c, "chunk_id", str(c))
        score = getattr(c, "score", 0.0)
        method = getattr(c, "retrieval_method", "")
        cand_items.append(f"{cid}:{score:.4f}:{method}")
    cand_str = "|".join(cand_items)
    payload = f"{task_id}:{query.strip()}:{scopes_str}:{retrieval_policy_version}:{processing_version}:{cand_str}:{ev_str}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# =============================================================================
# Domain Models
# =============================================================================


class EvidenceLocator(BaseModel):
    """Fine-grained locator within a source document."""
    model_config = ConfigDict(frozen=True)

    page: int | None = None
    section: str | None = None
    paragraph: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    time_start_ms: int | None = None
    time_end_ms: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        return {k: v for k, v in d.items() if v is not None and v != {}}


class SourceDocument(BaseModel):
    """Original source material registered or ingested for a task."""
    model_config = ConfigDict(frozen=True)

    source_document_id: str
    source_type: SourceType
    title: str | None = None
    source_locator: str
    content_snapshot: str | None = None
    content_hash: str
    source_fingerprint: str
    author: str | None = None
    published_at: datetime | None = None
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    media_type: str = "text/plain"
    status: SourceStatus = SourceStatus.REGISTERED
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create_text(
        cls,
        text: str,
        title: str | None = None,
        author: str | None = None,
        published_at: datetime | None = None,
        metadata_json: dict[str, Any] | None = None,
        source_document_id: str | None = None,
        now: datetime | None = None,
    ) -> SourceDocument:
        ts = now or datetime.now(UTC)
        c_hash = compute_sha256(text)
        locator = f"inline:sha256:{c_hash[:16]}"
        fp = compute_source_fingerprint(SourceType.TEXT, locator, c_hash)
        return cls(
            source_document_id=source_document_id or f"src_{uuid4().hex[:24]}",
            source_type=SourceType.TEXT,
            title=title or "Inline Text Source",
            source_locator=locator,
            content_snapshot=text,
            content_hash=c_hash,
            source_fingerprint=fp,
            author=author,
            published_at=published_at,
            captured_at=ts,
            media_type="text/plain",
            status=SourceStatus.READY,
            metadata_json=metadata_json or {},
            created_at=ts,
        )

    @classmethod
    def register_file(
        cls,
        filename: str,
        file_bytes: bytes,
        media_type: str,
        storage_path: str,
        title: str | None = None,
        author: str | None = None,
        published_at: datetime | None = None,
        metadata_json: dict[str, Any] | None = None,
        source_document_id: str | None = None,
        now: datetime | None = None,
    ) -> SourceDocument:
        ts = now or datetime.now(UTC)
        c_hash = compute_sha256(file_bytes)
        locator = storage_path
        fp = compute_source_fingerprint(SourceType.FILE, locator, c_hash)
        meta = dict(metadata_json or {})
        meta["original_filename"] = filename
        meta["file_size_bytes"] = len(file_bytes)
        return cls(
            source_document_id=source_document_id or f"src_{uuid4().hex[:24]}",
            source_type=SourceType.FILE,
            title=title or filename,
            source_locator=locator,
            content_snapshot=None,  # Unparsed raw file in E1
            content_hash=c_hash,
            source_fingerprint=fp,
            author=author,
            published_at=published_at,
            captured_at=ts,
            media_type=media_type,
            status=SourceStatus.REGISTERED,
            metadata_json=meta,
            created_at=ts,
        )

    @classmethod
    def register_url(
        cls,
        url: str,
        title: str | None = None,
        author: str | None = None,
        published_at: datetime | None = None,
        metadata_json: dict[str, Any] | None = None,
        source_document_id: str | None = None,
        now: datetime | None = None,
    ) -> SourceDocument:
        ts = now or datetime.now(UTC)
        norm_url = url.strip()
        c_hash = compute_sha256(norm_url)
        locator = norm_url
        fp = compute_source_fingerprint(SourceType.URL, locator, c_hash)
        return cls(
            source_document_id=source_document_id or f"src_{uuid4().hex[:24]}",
            source_type=SourceType.URL,
            title=title or norm_url,
            source_locator=locator,
            content_snapshot=None,  # Unfetched URL in E1
            content_hash=c_hash,
            source_fingerprint=fp,
            author=author,
            published_at=published_at,
            captured_at=ts,
            media_type="text/html",
            status=SourceStatus.REGISTERED,
            metadata_json=metadata_json or {},
            created_at=ts,
        )


class EvidenceItem(BaseModel):
    """A granular piece of cited evidence extracted from a source document.

    Maintains a deterministic evidence_id compatible with Phase 1-7 evidence_refs.
    """
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    source_document_id: str
    locator: dict[str, Any] = Field(default_factory=dict)
    original_excerpt: str
    normalized_fact: str | None = None
    evidence_role: EvidenceRole = EvidenceRole.FACTUAL_SUPPORT
    confidence: float = 1.0
    extraction_method: str = "DIRECT_EXTRACT"
    content_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        source_document: SourceDocument,
        original_excerpt: str,
        locator: dict[str, Any] | EvidenceLocator | None = None,
        normalized_fact: str | None = None,
        evidence_role: EvidenceRole = EvidenceRole.FACTUAL_SUPPORT,
        confidence: float = 1.0,
        extraction_method: str = "DIRECT_EXTRACT",
        now: datetime | None = None,
    ) -> EvidenceItem:
        ts = now or datetime.now(UTC)
        loc_dict: dict[str, Any]
        if isinstance(locator, EvidenceLocator):
            loc_dict = locator.to_dict()
        elif isinstance(locator, dict):
            loc_dict = locator
        else:
            loc_dict = {}

        excerpt_hash = compute_sha256(original_excerpt)
        ev_id = compute_evidence_id(
            source_fingerprint=source_document.source_fingerprint,
            locator=loc_dict,
            excerpt_hash=excerpt_hash,
            evidence_role=evidence_role,
        )

        return cls(
            evidence_id=ev_id,
            source_document_id=source_document.source_document_id,
            locator=loc_dict,
            original_excerpt=original_excerpt,
            normalized_fact=normalized_fact,
            evidence_role=evidence_role,
            confidence=max(0.0, min(1.0, float(confidence))),
            extraction_method=extraction_method,
            content_hash=excerpt_hash,
            created_at=ts,
        )


class KnowledgeClaim(BaseModel):
    """An asserted knowledge statement linked to supporting or conflicting evidence."""
    model_config = ConfigDict(frozen=True)

    knowledge_claim_id: str
    claim_type: ClaimType
    claim_text: str
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    verification_status: VerificationStatus = VerificationStatus.NOT_REQUIRED
    conflict_evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_claim_invariants(self) -> KnowledgeClaim:
        # Invariant 1: A FACT claim cannot be GROUNDED without evidence references
        if self.claim_type == ClaimType.FACT and self.verification_status == VerificationStatus.GROUNDED:
            if not self.evidence_refs:
                raise InvalidClaimGroundingError(
                    f"KnowledgeClaim '{self.knowledge_claim_id}' is a FACT marked GROUNDED "
                    f"but has empty evidence_refs."
                )
        return self

    @classmethod
    def create(
        cls,
        claim_type: ClaimType,
        claim_text: str,
        evidence_refs: Sequence[str] = (),
        verification_status: VerificationStatus | None = None,
        conflict_evidence_refs: Sequence[str] = (),
        knowledge_claim_id: str | None = None,
        now: datetime | None = None,
    ) -> KnowledgeClaim:
        ts = now or datetime.now(UTC)
        ev_tuple = tuple(evidence_refs)
        conflict_tuple = tuple(conflict_evidence_refs)

        # Default verification status based on claim type and evidence refs if not explicitly set
        if verification_status is None:
            if conflict_tuple:
                v_status = VerificationStatus.CONFLICTED
            elif claim_type == ClaimType.FACT:
                v_status = VerificationStatus.GROUNDED if ev_tuple else VerificationStatus.INSUFFICIENT_EVIDENCE
            elif claim_type == ClaimType.INTERPRETATION:
                v_status = VerificationStatus.GROUNDED if ev_tuple else VerificationStatus.INSUFFICIENT_EVIDENCE
            else:  # OPINION
                v_status = VerificationStatus.NOT_REQUIRED
        else:
            v_status = verification_status

        return cls(
            knowledge_claim_id=knowledge_claim_id or f"clm_{uuid4().hex[:24]}",
            claim_type=claim_type,
            claim_text=claim_text,
            evidence_refs=ev_tuple,
            verification_status=v_status,
            conflict_evidence_refs=conflict_tuple,
            created_at=ts,
        )


class EvidenceSnapshot(BaseModel):
    """An immutable, fingerprinted snapshot of evidence collected for a task."""
    model_config = ConfigDict(frozen=True)

    evidence_snapshot_id: str
    task_id: str
    snapshot_version: int = 1
    source_document_ids: tuple[str, ...] = Field(default_factory=tuple)
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    knowledge_claim_ids: tuple[str, ...] = Field(default_factory=tuple)
    content_fingerprint: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        task_id: str,
        source_document_ids: Iterable[str],
        evidence_ids: Iterable[str],
        knowledge_claim_ids: Iterable[str] = (),
        snapshot_version: int = 1,
        evidence_snapshot_id: str | None = None,
        now: datetime | None = None,
    ) -> EvidenceSnapshot:
        ts = now or datetime.now(UTC)
        src_tuple = tuple(sorted(set(source_document_ids)))
        ev_tuple = tuple(sorted(set(evidence_ids)))
        claim_tuple = tuple(sorted(set(knowledge_claim_ids)))

        fp = compute_snapshot_fingerprint(src_tuple, ev_tuple, claim_tuple)
        return cls(
            evidence_snapshot_id=evidence_snapshot_id or f"es_{uuid4().hex[:24]}",
            task_id=task_id,
            snapshot_version=snapshot_version,
            source_document_ids=src_tuple,
            evidence_ids=ev_tuple,
            knowledge_claim_ids=claim_tuple,
            content_fingerprint=fp,
            created_at=ts,
        )


class KnowledgeChunk(BaseModel):
    """An immutable, deterministic chunk of parsed source knowledge."""
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    source_document_id: str
    processing_version: str = "knowledge_processing_v1"
    chunk_index: int
    normalized_text: str
    text_hash: str
    locator: dict[str, Any] = Field(default_factory=dict)
    content_fingerprint: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        source_document: SourceDocument,
        normalized_text: str,
        chunk_index: int,
        locator: dict[str, Any] | EvidenceLocator | None = None,
        processing_version: str = "knowledge_processing_v1",
        chunk_id: str | None = None,
        now: datetime | None = None,
    ) -> KnowledgeChunk:
        ts = now or datetime.now(UTC)
        loc_dict: dict[str, Any]
        if isinstance(locator, EvidenceLocator):
            loc_dict = locator.to_dict()
        elif isinstance(locator, dict):
            loc_dict = locator
        else:
            loc_dict = {}

        t_hash = compute_sha256(normalized_text)
        cid = chunk_id or compute_chunk_id(
            source_fingerprint=source_document.source_fingerprint,
            processing_version=processing_version,
            locator=loc_dict,
            normalized_text=normalized_text,
        )
        fp = compute_sha256(f"{cid}:{source_document.source_document_id}:{processing_version}:{chunk_index}:{t_hash}")
        return cls(
            chunk_id=cid,
            source_document_id=source_document.source_document_id,
            processing_version=processing_version,
            chunk_index=chunk_index,
            normalized_text=normalized_text,
            text_hash=t_hash,
            locator=loc_dict,
            content_fingerprint=fp,
            created_at=ts,
        )


class RetrievalCandidate(BaseModel):
    """A scored and ranked candidate chunk returned during retrieval."""
    model_config = ConfigDict(frozen=True)

    rank: int
    chunk_id: str
    source_document_id: str
    score: float
    retrieval_method: str = "LEXICAL_BM25"
    locator: dict[str, Any] = Field(default_factory=dict)
    excerpt: str


class RetrievalSnapshot(BaseModel):
    """An immutable, frozen record of a retrieval execution and candidate ranking."""
    model_config = ConfigDict(frozen=True)

    retrieval_snapshot_id: str
    task_id: str
    query: str
    source_scope_ids: tuple[str, ...] = Field(default_factory=tuple)
    retrieval_policy_version: str = "lexical_bm25_v1"
    processing_version: str = "knowledge_processing_v1"
    candidates: tuple[RetrievalCandidate, ...] = Field(default_factory=tuple)
    selected_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    content_fingerprint: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        task_id: str,
        query: str,
        source_scope_ids: Sequence[str],
        candidates: Sequence[RetrievalCandidate],
        selected_evidence_ids: Sequence[str] = (),
        retrieval_policy_version: str = "lexical_bm25_v1",
        processing_version: str = "knowledge_processing_v1",
        retrieval_snapshot_id: str | None = None,
        now: datetime | None = None,
    ) -> RetrievalSnapshot:
        ts = now or datetime.now(UTC)
        sorted_scopes = tuple(sorted(set(source_scope_ids)))
        cands_tuple = tuple(candidates)
        sel_ev_tuple = tuple(selected_evidence_ids)

        fp = compute_retrieval_snapshot_fingerprint(
            task_id=task_id,
            query=query,
            source_scope_ids=sorted_scopes,
            retrieval_policy_version=retrieval_policy_version,
            processing_version=processing_version,
            candidates=cands_tuple,
            selected_evidence_ids=sel_ev_tuple,
        )
        return cls(
            retrieval_snapshot_id=retrieval_snapshot_id or f"rs_{uuid4().hex[:24]}",
            task_id=task_id,
            query=query,
            source_scope_ids=sorted_scopes,
            retrieval_policy_version=retrieval_policy_version,
            processing_version=processing_version,
            candidates=cands_tuple,
            selected_evidence_ids=sel_ev_tuple,
            content_fingerprint=fp,
            created_at=ts,
        )

