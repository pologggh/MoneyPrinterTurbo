from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import re
from uuid import uuid4

import pytest

from app.domain.evidence import (
    ClaimType,
    EvidenceDomainError,
    EvidenceItem,
    EvidenceLocator,
    EvidenceRole,
    EvidenceSnapshot,
    InvalidClaimGroundingError,
    KnowledgeClaim,
    SourceDocument,
    SourceStatus,
    SourceType,
    UnsupportedSourceTypeError,
    VerificationStatus,
    compute_evidence_id,
    compute_sha256,
    compute_snapshot_fingerprint,
    compute_source_fingerprint,
)


def test_evidence_enums():
    """Verify all evidence and provenance enums have required values and strict boundaries."""
    assert SourceType.TEXT == "TEXT"
    assert SourceType.FILE == "FILE"
    assert SourceType.URL == "URL"
    assert SourceType.KNOWLEDGE_BASE == "KNOWLEDGE_BASE"

    # Strictly no synthetic or LLM memory types in source categories
    source_values = {s.value for s in SourceType}
    assert "MODEL_MEMORY" not in source_values
    assert "LLM_KNOWLEDGE" not in source_values
    assert "SYNTHETIC_FACT" not in source_values

    assert SourceStatus.REGISTERED == "REGISTERED"
    assert SourceStatus.READY == "READY"
    assert SourceStatus.FAILED == "FAILED"

    assert EvidenceRole.FACTUAL_SUPPORT == "FACTUAL_SUPPORT"
    assert EvidenceRole.SOURCE_VISUAL == "SOURCE_VISUAL"
    assert EvidenceRole.CONTEXT == "CONTEXT"

    assert ClaimType.FACT == "FACT"
    assert ClaimType.INTERPRETATION == "INTERPRETATION"
    assert ClaimType.OPINION == "OPINION"

    assert VerificationStatus.GROUNDED == "GROUNDED"
    assert VerificationStatus.INSUFFICIENT_EVIDENCE == "INSUFFICIENT_EVIDENCE"
    assert VerificationStatus.CONFLICTED == "CONFLICTED"
    assert VerificationStatus.NOT_REQUIRED == "NOT_REQUIRED"


def test_sha256_computation_and_independence_from_python_hash():
    """Ensure hashing is deterministic SHA-256 and never uses randomized Python hash()."""
    text = "Quantum computing leverages superposition and entanglement."
    digest = compute_sha256(text)
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert digest == expected
    assert len(digest) == 64

    # Binary input
    bytes_data = b"\x00\x01\x02\x03\x04\x05"
    assert compute_sha256(bytes_data) == hashlib.sha256(bytes_data).hexdigest()


def test_source_document_text_creation():
    """Verify creating an inline text source results in READY status and deterministic fingerprint."""
    text = "Photosynthesis converts light energy into chemical energy."
    now = datetime(2026, 9, 12, 12, 0, 0, tzinfo=UTC)

    doc = SourceDocument.create_text(
        text=text,
        title="Photosynthesis Overview",
        author="Dr. Botanist",
        now=now,
    )

    assert doc.source_type == SourceType.TEXT
    assert doc.status == SourceStatus.READY
    assert doc.content_snapshot == text
    assert doc.title == "Photosynthesis Overview"
    assert doc.author == "Dr. Botanist"
    assert doc.media_type == "text/plain"
    assert doc.created_at == now

    # Fingerprint determinism
    expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert doc.content_hash == expected_hash
    expected_fp = compute_source_fingerprint(SourceType.TEXT, doc.source_locator, expected_hash)
    assert doc.source_fingerprint == expected_fp


def test_source_document_file_registration():
    """Verify registered file has REGISTERED status and None content_snapshot (unparsed in E1)."""
    file_bytes = b"PDF-1.4 raw binary stream representing document"
    filename = "report_2026.pdf"
    now = datetime(2026, 9, 12, 12, 0, 0, tzinfo=UTC)

    doc = SourceDocument.register_file(
        filename=filename,
        file_bytes=file_bytes,
        media_type="application/pdf",
        storage_path="storage/evidence_sources/report_2026.pdf",
        title="Annual Report 2026",
        now=now,
    )

    assert doc.source_type == SourceType.FILE
    assert doc.status == SourceStatus.REGISTERED
    assert doc.content_snapshot is None  # Unparsed raw file in E1
    assert doc.media_type == "application/pdf"
    assert doc.source_locator == "storage/evidence_sources/report_2026.pdf"
    assert doc.metadata_json["original_filename"] == filename
    assert doc.metadata_json["file_size_bytes"] == len(file_bytes)


def test_source_document_url_registration():
    """Verify registered URL has REGISTERED status and None content_snapshot (unfetched in E1)."""
    url = "https://en.wikipedia.org/wiki/General_relativity"
    now = datetime(2026, 9, 12, 12, 0, 0, tzinfo=UTC)

    doc = SourceDocument.register_url(
        url=url,
        title="General Relativity - Wikipedia",
        now=now,
    )

    assert doc.source_type == SourceType.URL
    assert doc.status == SourceStatus.REGISTERED
    assert doc.content_snapshot is None  # Unfetched URL in E1
    assert doc.source_locator == url


def test_evidence_item_deterministic_id():
    """Verify evidence_id is deterministic and follows the format ev_<sha256[:24]>."""
    doc = SourceDocument.create_text(
        text="Water boils at 100 degrees Celsius at 1 atm pressure.",
        title="Thermodynamics Basics",
    )

    locator = EvidenceLocator(paragraph=1, line_start=1, line_end=2)
    excerpt = "Water boils at 100 degrees Celsius at 1 atm pressure."

    item1 = EvidenceItem.create(
        source_document=doc,
        original_excerpt=excerpt,
        locator=locator,
        evidence_role=EvidenceRole.FACTUAL_SUPPORT,
    )

    item2 = EvidenceItem.create(
        source_document=doc,
        original_excerpt=excerpt,
        locator=locator,
        evidence_role=EvidenceRole.FACTUAL_SUPPORT,
    )

    # Determinism: same input yields identical evidence_id
    assert item1.evidence_id == item2.evidence_id
    assert item1.evidence_id.startswith("ev_")
    assert len(item1.evidence_id) == 27  # "ev_" + 24 chars

    # Complies with Phase 1-7 regex ^[A-Za-z0-9_-]+$
    pattern = re.compile(r"^[A-Za-z0-9_-]+$")
    assert pattern.match(item1.evidence_id) is not None

    # Different excerpt yields different evidence_id
    item3 = EvidenceItem.create(
        source_document=doc,
        original_excerpt="Different excerpt text.",
        locator=locator,
        evidence_role=EvidenceRole.FACTUAL_SUPPORT,
    )
    assert item3.evidence_id != item1.evidence_id


def test_knowledge_claim_invariants():
    """Verify knowledge claim constraints: FACT cannot be GROUNDED without evidence."""
    # Valid: FACT with evidence marked GROUNDED
    valid_claim = KnowledgeClaim.create(
        claim_type=ClaimType.FACT,
        claim_text="The speed of light in vacuum is approximately 299,792 km/s.",
        evidence_refs=["ev_123456789012345678901234"],
        verification_status=VerificationStatus.GROUNDED,
    )
    assert valid_claim.verification_status == VerificationStatus.GROUNDED

    # Invalid: FACT marked GROUNDED with no evidence refs
    with pytest.raises(InvalidClaimGroundingError):
        KnowledgeClaim(
            knowledge_claim_id="clm_test",
            claim_type=ClaimType.FACT,
            claim_text="An unproven fact claim.",
            evidence_refs=(),
            verification_status=VerificationStatus.GROUNDED,
        )

    # Valid: FACT with no evidence marked INSUFFICIENT_EVIDENCE
    ungrounded_fact = KnowledgeClaim.create(
        claim_type=ClaimType.FACT,
        claim_text="An unverified assertion.",
        evidence_refs=(),
    )
    assert ungrounded_fact.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE

    # Valid: OPINION does not require evidence
    opinion_claim = KnowledgeClaim.create(
        claim_type=ClaimType.OPINION,
        claim_text="Quantum mechanics is the most elegant theory in physics.",
    )
    assert opinion_claim.verification_status == VerificationStatus.NOT_REQUIRED

    # Valid: Claim with conflicting evidence marked CONFLICTED
    conflicted_claim = KnowledgeClaim.create(
        claim_type=ClaimType.FACT,
        claim_text="Controversial historical date.",
        evidence_refs=["ev_source_a"],
        conflict_evidence_refs=["ev_source_b"],
    )
    assert conflicted_claim.verification_status == VerificationStatus.CONFLICTED


def test_evidence_snapshot_immutability_and_fingerprint():
    """Verify EvidenceSnapshot immutability and deterministic fingerprint computation."""
    task_id = f"task_{uuid4().hex[:12]}"
    src_ids = ["src_b", "src_a", "src_c"]
    ev_ids = ["ev_2", "ev_1"]
    claim_ids = ["clm_z", "clm_x"]

    snap1 = EvidenceSnapshot.create(
        task_id=task_id,
        source_document_ids=src_ids,
        evidence_ids=ev_ids,
        knowledge_claim_ids=claim_ids,
        snapshot_version=1,
    )

    snap2 = EvidenceSnapshot.create(
        task_id=task_id,
        source_document_ids=["src_c", "src_a", "src_b"],  # different input ordering
        evidence_ids=["ev_1", "ev_2"],
        knowledge_claim_ids=["clm_x", "clm_z"],
        snapshot_version=1,
    )

    # Sorted order guarantees identical content fingerprint
    assert snap1.content_fingerprint == snap2.content_fingerprint
    assert snap1.source_document_ids == ("src_a", "src_b", "src_c")
    assert snap1.evidence_ids == ("ev_1", "ev_2")
    assert snap1.knowledge_claim_ids == ("clm_x", "clm_z")

    # Immutability check
    with pytest.raises(Exception):
        snap1.snapshot_version = 2


def test_evidence_locator_serialization():
    """Verify EvidenceLocator serializes cleanly to dict omitting None values."""
    loc = EvidenceLocator(
        page=3,
        paragraph=2,
        line_start=15,
        line_end=18,
    )
    d = loc.to_dict()
    assert d == {
        "page": 3,
        "paragraph": 2,
        "line_start": 15,
        "line_end": 18,
    }
    assert "section" not in d
    assert "time_start_ms" not in d


def test_different_evidence_roles_produce_distinct_deterministic_ids():
    """Verify that different EvidenceRoles with identical excerpt and locator yield distinct IDs."""
    doc = SourceDocument.create_text("A key diagram showing orbital trajectories.", title="Orbital")
    loc = {"figure": 1}
    excerpt = "orbital trajectories"

    item_fact = EvidenceItem.create(doc, excerpt, locator=loc, evidence_role=EvidenceRole.FACTUAL_SUPPORT)
    item_visual = EvidenceItem.create(doc, excerpt, locator=loc, evidence_role=EvidenceRole.SOURCE_VISUAL)
    item_context = EvidenceItem.create(doc, excerpt, locator=loc, evidence_role=EvidenceRole.CONTEXT)

    assert item_fact.evidence_id != item_visual.evidence_id
    assert item_fact.evidence_id != item_context.evidence_id
    assert item_visual.evidence_id != item_context.evidence_id


def test_interpretation_claim_variants():
    """Verify INTERPRETATION claims: requires evidence for GROUNDED, falls back to INSUFFICIENT_EVIDENCE."""
    grounded = KnowledgeClaim.create(
        claim_type=ClaimType.INTERPRETATION,
        claim_text="The author implies that technology advances exponentially.",
        evidence_refs=["ev_123456789012345678901234"],
    )
    assert grounded.verification_status == VerificationStatus.GROUNDED

    ungrounded = KnowledgeClaim.create(
        claim_type=ClaimType.INTERPRETATION,
        claim_text="The author may believe AI will surpass humans.",
        evidence_refs=(),
    )
    assert ungrounded.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE

