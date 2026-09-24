from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger
from PIL import Image
from sqlalchemy.orm import Session

from app.persistence.repositories import EvidenceRepository
from app.persistence.session import get_session

_CANVAS_SIZES = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
}


class EvidenceResolutionStatus(str, Enum):
    """Status of resolving an evidence identifier."""

    FOUND = "FOUND"
    EVIDENCE_NOT_FOUND = "EVIDENCE_NOT_FOUND"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"


@dataclass(frozen=True)
class ResolvedEvidenceResult:
    """Outcome of resolving an evidence reference."""

    evidence_id: str
    status: EvidenceResolutionStatus
    evidence_item: Any | None = None
    source_document: Any | None = None
    error_message: str | None = None


@runtime_checkable
class SourceAssetResolverProtocol(Protocol):
    """Protocol for resolving evidence identifiers to persistent evidence and documents."""

    def resolve_evidence(self, evidence_id: str) -> ResolvedEvidenceResult:
        """Resolves an evidence reference to its EvidenceItem and SourceDocument."""
        ...


class DefaultSourceAssetResolver:
    """
    Default repository-backed resolver for evidence items and source documents.
    """

    def __init__(self, session_factory: Any | None = None) -> None:
        self._session_factory = session_factory

    @contextmanager
    def _session_scope(self) -> Generator[Session, None, None]:
        if self._session_factory is not None:
            with get_session(self._session_factory) as session:
                yield session
        else:
            with get_session() as session:
                yield session

    def resolve_evidence(self, evidence_id: str) -> ResolvedEvidenceResult:
        """
        Looks up EvidenceItem -> source_document_id -> SourceDocument in the database.
        """
        try:
            with self._session_scope() as session:
                repo = EvidenceRepository(session)
                ev_item = repo.get_evidence_item(evidence_id)
                if ev_item is None:
                    return ResolvedEvidenceResult(
                        evidence_id=evidence_id,
                        status=EvidenceResolutionStatus.EVIDENCE_NOT_FOUND,
                        error_message=f"Evidence item '{evidence_id}' was not found in repository.",
                    )
                doc = repo.get_source_document(ev_item.source_document_id)
                if doc is None:
                    return ResolvedEvidenceResult(
                        evidence_id=evidence_id,
                        status=EvidenceResolutionStatus.DOCUMENT_NOT_FOUND,
                        evidence_item=ev_item,
                        error_message=(
                            f"SourceDocument '{ev_item.source_document_id}' for evidence "
                            f"'{evidence_id}' was not found in repository."
                        ),
                    )
                return ResolvedEvidenceResult(
                    evidence_id=evidence_id,
                    status=EvidenceResolutionStatus.FOUND,
                    evidence_item=ev_item,
                    source_document=doc,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error resolving evidence '{evidence_id}': {exc}")
            return ResolvedEvidenceResult(
                evidence_id=evidence_id,
                status=EvidenceResolutionStatus.EVIDENCE_NOT_FOUND,
                error_message=f"Failed to query evidence repository for '{evidence_id}': {exc}",
            )


def resolve_source_file_path(source_locator: str | None) -> Path | None:
    """
    Resolves a source document locator string into a valid local filesystem path if available.
    """
    if not source_locator:
        return None
    loc = source_locator.strip()
    if loc.startswith(("http://", "https://", "inline:")):
        return None
    p = Path(loc)
    if p.is_file():
        return p.resolve()
    cwd_p = (Path.cwd() / loc).resolve()
    if cwd_p.is_file():
        return cwd_p
    return None


def extract_page_number(locator: dict[str, Any] | Any) -> int | None:
    """
    Extracts 1-based page number from evidence locator dictionary.
    """
    if not isinstance(locator, dict):
        return None
    val = locator.get("page")
    if val is None:
        val = locator.get("page_number")
    if val is None:
        val = locator.get("page_num")
    if val is None and isinstance(locator.get("extra"), dict):
        val = locator["extra"].get("page") or locator["extra"].get("page_number")

    if val is not None:
        try:
            val_int = int(val)
            if val_int > 0:
                return val_int
        except (ValueError, TypeError):
            pass
    return None


def render_pdf_page_to_png(
    pdf_path: Path,
    page_number: int,
    target_path: Path,
    aspect_ratio: str = "16:9",
) -> Path:
    """
    Renders a 1-based PDF page to PNG fitted/letterboxed to the requested aspect ratio.
    Preserves the entire page without cropping.
    """
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        if page_number < 1 or page_number > total_pages:
            raise ValueError(
                f"Page number {page_number} is out of bounds (document has {total_pages} pages)"
            )
        page = pdf.pages[page_number - 1]
        page_img = page.to_image(resolution=150)
        pil_img = page_img.original.convert("RGB")

    target_width, target_height = _CANVAS_SIZES.get(aspect_ratio, (1920, 1080))
    sw, sh = pil_img.size

    # Fit / letterbox without cropping
    scale = min(target_width / sw, target_height / sh)
    nw = max(1, round(sw * scale))
    nh = max(1, round(sh * scale))

    resized = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    offset_x = (target_width - nw) // 2
    offset_y = (target_height - nh) // 2
    canvas.paste(resized, (offset_x, offset_y))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(target_path), format="PNG")
    return target_path
