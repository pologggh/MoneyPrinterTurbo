from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pypdfium2 as pdfium
import pytest
from PIL import Image

from app.domain.asset_execution import ProviderOutcomeType
from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRoutingRequest,
    GenerationMode,
    StaticCapabilityMetadata,
    TierLevel,
)
from app.domain.enums import VisualType
from app.domain.evidence import (
    EvidenceItem,
    SourceDocument,
    SourceStatus,
    SourceType,
)
from app.services.asset_adapters.local_asset_adapter import LocalAssetAdapter
from app.services.asset_probe_service import probe_media_file
from app.services.source_asset_resolver import (
    EvidenceResolutionStatus,
    ResolvedEvidenceResult,
    SourceAssetResolverProtocol,
)


def _make_candidate(
    provider: str = "system_source_asset",
    model: str = "knowledge_figure_extractor",
    generation_mode: GenerationMode = GenerationMode.SOURCE_ASSET_USE,
    requested_visual_type: VisualType = VisualType.SOURCE_ASSET,
) -> AssetRouteCandidate:
    return AssetRouteCandidate(
        capability_id=f"{provider}:{model}:{generation_mode.value}",
        provider=provider,
        model=model,
        generation_mode=generation_mode,
        requested_visual_type=requested_visual_type,
        is_eligible=True,
        rejection_reasons=(),
        static_metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.HIGH,
            cost_tier=TierLevel.LOW,
            latency_tier=TierLevel.LOW,
        ),
    )


def _make_request(
    source_asset_refs: tuple[str, ...] = (),
    aspect_ratio: str = "16:9",
    shot_id: str = "shot-001",
) -> AssetRoutingRequest:
    return AssetRoutingRequest(
        shot_id=shot_id,
        shot_revision_id=f"rev-{shot_id}",
        requested_visual_type=VisualType.SOURCE_ASSET,
        target_duration=4.0,
        visual_goal="Explain architecture",
        scene_description="System architecture breakdown",
        generation_prompt="",
        camera_movement="static",
        aspect_ratio=aspect_ratio,
        source_asset_refs=source_asset_refs,
    )


def _create_test_pdf(path: Path) -> Path:
    pdf = pdfium.PdfDocument.new()
    # page 1: portrait 300x400
    pdf.new_page(300, 400)
    # page 2: landscape 500x300
    pdf.new_page(500, 300)
    pdf.save(str(path))
    return path


# 1. Direct file regression
def test_direct_file_regression(tmp_path: Path):
    real_file = tmp_path / "valid_image.png"
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    img.save(real_file)

    adapter = LocalAssetAdapter()
    req = _make_request(source_asset_refs=(str(real_file),))
    cand = _make_candidate()
    target_dir = tmp_path / "target"

    res = adapter.execute(req, cand, target_dir)
    assert res.outcome_type == ProviderOutcomeType.SUCCESS
    assert res.file_path is not None
    assert Path(res.file_path).is_file()
    assert res.raw_response.get("resolver_type") == "direct_file"
    assert res.raw_response.get("fallback_used") is False


# 2. PDF evidence resolution
def test_pdf_evidence_resolution(tmp_path: Path):
    pdf_file = tmp_path / "test_doc.pdf"
    _create_test_pdf(pdf_file)

    source_doc = SourceDocument(
        source_document_id="src_pdf_1",
        source_type=SourceType.FILE,
        source_locator=str(pdf_file),
        content_hash="hash123",
        source_fingerprint="fp123",
        media_type="application/pdf",
        status=SourceStatus.READY,
    )
    ev_item = EvidenceItem(
        evidence_id="ev_valid_pdf_1",
        source_document_id="src_pdf_1",
        locator={"page": 1, "paragraph": 1},
        original_excerpt="Architecture overview diagram",
        content_hash="hash_ev_1",
    )

    mock_resolver = MagicMock(spec=SourceAssetResolverProtocol)
    mock_resolver.resolve_evidence.return_value = ResolvedEvidenceResult(
        evidence_id="ev_valid_pdf_1",
        status=EvidenceResolutionStatus.FOUND,
        evidence_item=ev_item,
        source_document=source_doc,
    )

    adapter = LocalAssetAdapter(source_resolver=mock_resolver)
    req = _make_request(source_asset_refs=("ev_valid_pdf_1",), aspect_ratio="16:9")
    cand = _make_candidate()
    target_dir = tmp_path / "target"

    res = adapter.execute(req, cand, target_dir)
    assert res.outcome_type == ProviderOutcomeType.SUCCESS
    assert res.file_path is not None
    out_path = Path(res.file_path)
    assert out_path.is_file()
    assert out_path.stat().st_size > 0

    # Probe resulting media file
    probe = probe_media_file(out_path)
    assert probe.width == 1920
    assert probe.height == 1080

    assert res.raw_response.get("resolver_type") == "evidence_pdf_page"
    assert res.raw_response.get("evidence_id") == "ev_valid_pdf_1"
    assert res.raw_response.get("source_document_id") == "src_pdf_1"
    assert res.raw_response.get("pdf_page_number") == 1
    assert res.raw_response.get("fallback_used") is False


# 3. Correct page selection (1-based page)
def test_pdf_correct_page_selection(tmp_path: Path):
    pdf_file = tmp_path / "multi_page.pdf"
    _create_test_pdf(pdf_file)

    source_doc = SourceDocument(
        source_document_id="src_pdf_multi",
        source_type=SourceType.FILE,
        source_locator=str(pdf_file),
        content_hash="hash_multi",
        source_fingerprint="fp_multi",
        media_type="application/pdf",
        status=SourceStatus.READY,
    )

    # Resolve Page 1 vs Page 2
    ev_p1 = EvidenceItem(
        evidence_id="ev_p1",
        source_document_id="src_pdf_multi",
        locator={"page": 1},
        original_excerpt="Page 1",
        content_hash="h1",
    )
    ev_p2 = EvidenceItem(
        evidence_id="ev_p2",
        source_document_id="src_pdf_multi",
        locator={"page": 2},
        original_excerpt="Page 2",
        content_hash="h2",
    )

    def mock_resolve(ev_id: str) -> ResolvedEvidenceResult:
        if ev_id == "ev_p1":
            return ResolvedEvidenceResult(
                evidence_id="ev_p1",
                status=EvidenceResolutionStatus.FOUND,
                evidence_item=ev_p1,
                source_document=source_doc,
            )
        elif ev_id == "ev_p2":
            return ResolvedEvidenceResult(
                evidence_id="ev_p2",
                status=EvidenceResolutionStatus.FOUND,
                evidence_item=ev_p2,
                source_document=source_doc,
            )
        return ResolvedEvidenceResult(
            evidence_id=ev_id,
            status=EvidenceResolutionStatus.EVIDENCE_NOT_FOUND,
        )

    mock_resolver = MagicMock(spec=SourceAssetResolverProtocol)
    mock_resolver.resolve_evidence.side_effect = mock_resolve

    adapter = LocalAssetAdapter(source_resolver=mock_resolver)
    cand = _make_candidate()

    res1 = adapter.execute(_make_request(source_asset_refs=("ev_p1",), shot_id="shot_1"), cand, tmp_path / "t1")
    res2 = adapter.execute(_make_request(source_asset_refs=("ev_p2",), shot_id="shot_2"), cand, tmp_path / "t2")

    assert res1.outcome_type == ProviderOutcomeType.SUCCESS
    assert res2.outcome_type == ProviderOutcomeType.SUCCESS
    assert res1.raw_response.get("pdf_page_number") == 1
    assert res2.raw_response.get("pdf_page_number") == 2

    # Files must differ because page 1 is portrait (300x400) and page 2 is landscape (500x300)
    data1 = Path(res1.file_path).read_bytes()
    data2 = Path(res2.file_path).read_bytes()
    assert data1 != data2


# 4. Aspect ratios (16:9, 9:16, 1:1)
@pytest.mark.parametrize(
    "aspect_ratio,expected_dims",
    [
        ("16:9", (1920, 1080)),
        ("9:16", (1080, 1920)),
        ("1:1", (1080, 1080)),
    ],
)
def test_pdf_aspect_ratio_dimensions(tmp_path: Path, aspect_ratio: str, expected_dims: tuple[int, int]):
    pdf_file = tmp_path / "aspect_test.pdf"
    _create_test_pdf(pdf_file)

    source_doc = SourceDocument(
        source_document_id="src_ar",
        source_type=SourceType.FILE,
        source_locator=str(pdf_file),
        content_hash="hash_ar",
        source_fingerprint="fp_ar",
        media_type="application/pdf",
        status=SourceStatus.READY,
    )
    ev_item = EvidenceItem(
        evidence_id="ev_ar",
        source_document_id="src_ar",
        locator={"page": 1},
        original_excerpt="Aspect test",
        content_hash="h_ar",
    )
    mock_resolver = MagicMock(spec=SourceAssetResolverProtocol)
    mock_resolver.resolve_evidence.return_value = ResolvedEvidenceResult(
        evidence_id="ev_ar",
        status=EvidenceResolutionStatus.FOUND,
        evidence_item=ev_item,
        source_document=source_doc,
    )

    adapter = LocalAssetAdapter(source_resolver=mock_resolver)
    req = _make_request(source_asset_refs=("ev_ar",), aspect_ratio=aspect_ratio)
    cand = _make_candidate()
    target_dir = tmp_path / f"t_{aspect_ratio.replace(':', '_')}"

    res = adapter.execute(req, cand, target_dir)
    assert res.outcome_type == ProviderOutcomeType.SUCCESS
    probe = probe_media_file(res.file_path)
    assert (probe.width, probe.height) == expected_dims


# 5. Multiple references: First invalid/unusable + second valid succeeds
def test_multiple_references_failover_succeeds(tmp_path: Path):
    pdf_file = tmp_path / "valid.pdf"
    _create_test_pdf(pdf_file)

    source_doc = SourceDocument(
        source_document_id="src_valid",
        source_type=SourceType.FILE,
        source_locator=str(pdf_file),
        content_hash="h_valid",
        source_fingerprint="fp_valid",
        media_type="application/pdf",
        status=SourceStatus.READY,
    )
    ev_valid = EvidenceItem(
        evidence_id="ev_valid",
        source_document_id="src_valid",
        locator={"page": 1},
        original_excerpt="Valid page",
        content_hash="h_ev_valid",
    )

    def mock_resolve(ev_id: str) -> ResolvedEvidenceResult:
        if ev_id == "ev_invalid":
            return ResolvedEvidenceResult(
                evidence_id="ev_invalid",
                status=EvidenceResolutionStatus.EVIDENCE_NOT_FOUND,
                error_message="Evidence ev_invalid not found",
            )
        elif ev_id == "ev_valid":
            return ResolvedEvidenceResult(
                evidence_id="ev_valid",
                status=EvidenceResolutionStatus.FOUND,
                evidence_item=ev_valid,
                source_document=source_doc,
            )
        return ResolvedEvidenceResult(
            evidence_id=ev_id,
            status=EvidenceResolutionStatus.EVIDENCE_NOT_FOUND,
        )

    mock_resolver = MagicMock(spec=SourceAssetResolverProtocol)
    mock_resolver.resolve_evidence.side_effect = mock_resolve

    adapter = LocalAssetAdapter(source_resolver=mock_resolver)
    req = _make_request(source_asset_refs=("ev_invalid", "ev_valid"))
    cand = _make_candidate()
    target_dir = tmp_path / "target_multi"

    res = adapter.execute(req, cand, target_dir)
    assert res.outcome_type == ProviderOutcomeType.SUCCESS
    assert res.raw_response.get("evidence_id") == "ev_valid"
    assert res.raw_response.get("resolver_type") == "evidence_pdf_page"


# 6. Valid non-visual evidence renders deterministic knowledge card fallback
def test_valid_non_visual_evidence_fallback(tmp_path: Path):
    source_doc = SourceDocument(
        source_document_id="src_text_1",
        source_type=SourceType.TEXT,
        source_locator="inline:sha256:12345",
        content_hash="hash_txt",
        source_fingerprint="fp_txt",
        media_type="text/plain",
        status=SourceStatus.READY,
    )
    ev_item = EvidenceItem(
        evidence_id="ev_text_1",
        source_document_id="src_text_1",
        locator={"paragraph": 1},
        original_excerpt="Pure text knowledge fact with no visual page",
        content_hash="h_txt",
    )

    mock_resolver = MagicMock(spec=SourceAssetResolverProtocol)
    mock_resolver.resolve_evidence.return_value = ResolvedEvidenceResult(
        evidence_id="ev_text_1",
        status=EvidenceResolutionStatus.FOUND,
        evidence_item=ev_item,
        source_document=source_doc,
    )

    adapter = LocalAssetAdapter(source_resolver=mock_resolver)
    req = _make_request(source_asset_refs=("ev_text_1",), aspect_ratio="16:9")
    cand = _make_candidate()
    target_dir = tmp_path / "target_text"

    res = adapter.execute(req, cand, target_dir)
    assert res.outcome_type == ProviderOutcomeType.SUCCESS
    assert Path(res.file_path).is_file()
    assert res.raw_response.get("resolver_type") == "evidence_knowledge_card_fallback"
    assert res.raw_response.get("evidence_id") == "ev_text_1"
    assert res.raw_response.get("fallback_used") is True


# 7. Missing evidence returns exact SOURCE_EVIDENCE_NOT_FOUND error
def test_missing_evidence_structured_failure(tmp_path: Path):
    mock_resolver = MagicMock(spec=SourceAssetResolverProtocol)
    mock_resolver.resolve_evidence.return_value = ResolvedEvidenceResult(
        evidence_id="ev_nonexistent",
        status=EvidenceResolutionStatus.EVIDENCE_NOT_FOUND,
        error_message="Evidence item 'ev_nonexistent' not found in repository.",
    )

    adapter = LocalAssetAdapter(source_resolver=mock_resolver)
    req = _make_request(source_asset_refs=("ev_nonexistent",))
    cand = _make_candidate()
    target_dir = tmp_path / "target_err"

    res = adapter.execute(req, cand, target_dir)
    assert res.outcome_type == ProviderOutcomeType.CAPABILITY_INCOMPATIBLE
    assert res.error_code == "SOURCE_EVIDENCE_NOT_FOUND"
    assert res.error_code != "LOCAL_ASSET_FILE_NOT_FOUND"


# 8. Missing source document returns exact SOURCE_DOCUMENT_NOT_FOUND error
def test_missing_source_document_structured_failure(tmp_path: Path):
    ev_item = EvidenceItem(
        evidence_id="ev_orphan",
        source_document_id="src_missing",
        locator={"page": 1},
        original_excerpt="Orphan evidence",
        content_hash="h_orphan",
    )
    mock_resolver = MagicMock(spec=SourceAssetResolverProtocol)
    mock_resolver.resolve_evidence.return_value = ResolvedEvidenceResult(
        evidence_id="ev_orphan",
        status=EvidenceResolutionStatus.DOCUMENT_NOT_FOUND,
        evidence_item=ev_item,
        error_message="Source document 'src_missing' not found in repository.",
    )

    adapter = LocalAssetAdapter(source_resolver=mock_resolver)
    req = _make_request(source_asset_refs=("ev_orphan",))
    cand = _make_candidate()
    target_dir = tmp_path / "target_orphan"

    res = adapter.execute(req, cand, target_dir)
    assert res.outcome_type == ProviderOutcomeType.CAPABILITY_INCOMPATIBLE
    assert res.error_code == "SOURCE_DOCUMENT_NOT_FOUND"
    assert res.error_code != "LOCAL_ASSET_FILE_NOT_FOUND"


# 9. No-reference behavior preserves NO_LOCAL_ASSET_REFERENCES
def test_no_reference_behavior(tmp_path: Path):
    adapter = LocalAssetAdapter()
    req = _make_request(source_asset_refs=())
    cand = _make_candidate()
    target_dir = tmp_path / "target_empty"

    res = adapter.execute(req, cand, target_dir)
    assert res.outcome_type == ProviderOutcomeType.CAPABILITY_INCOMPATIBLE
    assert res.error_code == "NO_LOCAL_ASSET_REFERENCES"


# 10. PDF page invalid returns exact SOURCE_PDF_PAGE_INVALID
def test_pdf_page_invalid_returns_error(tmp_path: Path):
    pdf_file = tmp_path / "doc.pdf"
    _create_test_pdf(pdf_file)

    source_doc = SourceDocument(
        source_document_id="src_p_inv",
        source_type=SourceType.FILE,
        source_locator=str(pdf_file),
        content_hash="h_p_inv",
        source_fingerprint="fp_p_inv",
        media_type="application/pdf",
        status=SourceStatus.READY,
    )
    # Page 99 on a 2-page document
    ev_item = EvidenceItem(
        evidence_id="ev_p_99",
        source_document_id="src_p_inv",
        locator={"page": 99},
        original_excerpt="Invalid page number",
        content_hash="h_p99",
    )
    mock_resolver = MagicMock(spec=SourceAssetResolverProtocol)
    mock_resolver.resolve_evidence.return_value = ResolvedEvidenceResult(
        evidence_id="ev_p_99",
        status=EvidenceResolutionStatus.FOUND,
        evidence_item=ev_item,
        source_document=source_doc,
    )

    adapter = LocalAssetAdapter(source_resolver=mock_resolver)
    req = _make_request(source_asset_refs=("ev_p_99",))
    cand = _make_candidate()
    target_dir = tmp_path / "target_inv_page"

    res = adapter.execute(req, cand, target_dir)
    assert res.outcome_type == ProviderOutcomeType.CAPABILITY_INCOMPATIBLE
    assert res.error_code == "SOURCE_PDF_PAGE_INVALID"


# 11. Image evidence resolution
def test_image_evidence_resolution(tmp_path: Path):
    img_file = tmp_path / "diagram_source.png"
    img = Image.new("RGB", (800, 600), color=(50, 100, 150))
    img.save(img_file)

    source_doc = SourceDocument(
        source_document_id="src_img_1",
        source_type=SourceType.FILE,
        source_locator=str(img_file),
        content_hash="h_img",
        source_fingerprint="fp_img",
        media_type="image/png",
        status=SourceStatus.READY,
    )
    ev_item = EvidenceItem(
        evidence_id="ev_img_1",
        source_document_id="src_img_1",
        locator={"paragraph": 1},
        original_excerpt="Image figure from article",
        content_hash="h_ev_img",
    )

    mock_resolver = MagicMock(spec=SourceAssetResolverProtocol)
    mock_resolver.resolve_evidence.return_value = ResolvedEvidenceResult(
        evidence_id="ev_img_1",
        status=EvidenceResolutionStatus.FOUND,
        evidence_item=ev_item,
        source_document=source_doc,
    )

    adapter = LocalAssetAdapter(source_resolver=mock_resolver)
    req = _make_request(source_asset_refs=("ev_img_1",))
    cand = _make_candidate()
    target_dir = tmp_path / "target_img"

    res = adapter.execute(req, cand, target_dir)
    assert res.outcome_type == ProviderOutcomeType.SUCCESS
    assert Path(res.file_path).is_file()
    assert res.raw_response.get("resolver_type") == "evidence_image"
    assert res.raw_response.get("evidence_id") == "ev_img_1"
    assert res.raw_response.get("source_document_id") == "src_img_1"
    assert res.raw_response.get("fallback_used") is False


# 12. DefaultSourceAssetResolver integration with SQLite session
def test_default_source_asset_resolver_integration(tmp_path: Path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.persistence.models import Base
    from app.persistence.repositories import EvidenceRepository
    from app.services.source_asset_resolver import DefaultSourceAssetResolver

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    pdf_file = tmp_path / "real_doc.pdf"
    _create_test_pdf(pdf_file)

    with session_factory() as session:
        repo = EvidenceRepository(session)
        doc = SourceDocument(
            source_document_id="src_db_test",
            source_type=SourceType.FILE,
            source_locator=str(pdf_file),
            content_hash="h_db_test",
            source_fingerprint="fp_db_test",
            media_type="application/pdf",
            status=SourceStatus.READY,
        )
        repo.save_source_document(doc)

        item = EvidenceItem(
            evidence_id="ev_db_test_1",
            source_document_id="src_db_test",
            locator={"page": 1},
            original_excerpt="Database backed evidence test",
            content_hash="h_db_ev_1",
        )
        repo.save_evidence_item(item)
        session.commit()

    resolver = DefaultSourceAssetResolver(session_factory=session_factory)

    # Test found
    res_found = resolver.resolve_evidence("ev_db_test_1")
    assert res_found.status == EvidenceResolutionStatus.FOUND
    assert res_found.evidence_item.evidence_id == "ev_db_test_1"
    assert res_found.source_document.source_document_id == "src_db_test"

    # Test evidence not found
    res_not_found = resolver.resolve_evidence("ev_nonexistent_in_db")
    assert res_not_found.status == EvidenceResolutionStatus.EVIDENCE_NOT_FOUND

    # Test end-to-end LocalAssetAdapter using this DefaultSourceAssetResolver
    adapter = LocalAssetAdapter(source_resolver=resolver)
    req = _make_request(source_asset_refs=("ev_db_test_1",))
    cand = _make_candidate()
    target_dir = tmp_path / "target_db"

    res_exec = adapter.execute(req, cand, target_dir)
    assert res_exec.outcome_type == ProviderOutcomeType.SUCCESS
    assert Path(res_exec.file_path).is_file()
    assert res_exec.raw_response.get("resolver_type") == "evidence_pdf_page"


# 13. End-to-end ShotExecutionService execution with system_source_asset and PDF evidence
def test_shot_execution_service_with_pdf_evidence(tmp_path: Path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.domain.asset_execution import ShotExecution, ShotExecutionStatus
    from app.domain.asset_router import AssetRouteDecision, RoutingStrategy
    from app.persistence.models import Base
    from app.persistence.repositories import EvidenceRepository
    from app.services.asset_adapters.registry import AdapterRegistry
    from app.services.shot_execution_service import ShotExecutionService

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    pdf_file = tmp_path / "article.pdf"
    _create_test_pdf(pdf_file)

    with session_factory() as session:
        repo = EvidenceRepository(session)
        doc = SourceDocument(
            source_document_id="src_e2e_pdf",
            source_type=SourceType.FILE,
            source_locator=str(pdf_file),
            content_hash="h_e2e_pdf",
            source_fingerprint="fp_e2e_pdf",
            media_type="application/pdf",
            status=SourceStatus.READY,
        )
        repo.save_source_document(doc)

        item = EvidenceItem(
            evidence_id="ev_e2e_pdf_1",
            source_document_id="src_e2e_pdf",
            locator={"page": 2},
            original_excerpt="Architecture block diagram on page 2",
            content_hash="h_e2e_ev_1",
        )
        repo.save_evidence_item(item)
        session.commit()

    cand = _make_candidate(
        provider="system_source_asset",
        model="knowledge_figure_extractor",
        generation_mode=GenerationMode.SOURCE_ASSET_USE,
        requested_visual_type=VisualType.SOURCE_ASSET,
    )
    req = _make_request(
        source_asset_refs=("ev_e2e_pdf_1",),
        aspect_ratio="16:9",
        shot_id="shot_e2e_source",
    )

    decision = AssetRouteDecision(
        shot_id="shot_e2e_source",
        shot_revision_id="rev-shot_e2e_source",
        routing_strategy=RoutingStrategy.BALANCED,
        requested_visual_type=VisualType.SOURCE_ASSET,
        selected_candidate=cand,
        eligible_candidates=(cand,),
        rejected_candidates=(),
    )
    shot_exec = ShotExecution(
        execution_run_id="run_e2e_01",
        shot_id="shot_e2e_source",
        shot_revision_id="rev-shot_e2e_source",
        route_decision=decision,
        status=ShotExecutionStatus.PENDING,
    )

    registry = AdapterRegistry(session_factory=session_factory)
    service = ShotExecutionService(
        adapter_registry=registry,
        session_factory=session_factory,
    )

    final_exec, attempts, asset_ver = service.execute_shot(
        shot_execution=shot_exec,
        request=req,
        storage_base_dir=tmp_path / "storage",
    )

    assert final_exec.status == ShotExecutionStatus.SUCCEEDED
    assert len(attempts) == 1
    assert attempts[0].status.value == "SUCCEEDED"
    assert asset_ver is not None
    assert Path(asset_ver.file_path).is_file()
    assert asset_ver.width == 1920
    assert asset_ver.height == 1080
    assert attempts[0].raw_provider_response.get("resolver_type") == "evidence_pdf_page"
    assert attempts[0].raw_provider_response.get("pdf_page_number") == 2
