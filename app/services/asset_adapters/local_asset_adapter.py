from __future__ import annotations

import re
import shutil
from pathlib import Path

from app.domain.asset_execution import AdapterExecutionResult, ProviderOutcomeType
from app.domain.asset_router import AssetRouteCandidate, AssetRoutingRequest
from app.services.asset_adapters.base import AssetExecutionAdapter
from app.services.diagram_renderer import render_diagram_card
from app.services.source_asset_resolver import (
    EvidenceResolutionStatus,
    SourceAssetResolverProtocol,
    extract_page_number,
    render_pdf_page_to_png,
    resolve_source_file_path,
)


class LocalAssetAdapter(AssetExecutionAdapter):
    """
    Adapter for resolving local system assets:
    - system_source_asset: original knowledge source figures/images
    - system_user_asset: user-supplied uploaded media
    - system_diagram: conceptual diagram renderer (disabled by default)
    - system_knowledge_card: local knowledge card fallback renderer for AI_IMAGE
    """

    def __init__(
        self,
        source_resolver: SourceAssetResolverProtocol | None = None,
    ) -> None:
        self._source_resolver = source_resolver

    def _get_resolver(self) -> SourceAssetResolverProtocol:
        if self._source_resolver is not None:
            return self._source_resolver
        from app.services.source_asset_resolver import DefaultSourceAssetResolver

        return DefaultSourceAssetResolver()

    def execute(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        target_dir.mkdir(parents=True, exist_ok=True)
        provider = candidate.provider.lower()

        if provider in ("system_diagram", "system_knowledge_card"):
            try:
                diagram_path = render_diagram_card(request, target_dir)
                renderer_name = (
                    "knowledge_card_v1"
                    if provider == "system_knowledge_card"
                    else "diagram_card_v1"
                )
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.SUCCESS,
                    file_path=str(diagram_path),
                    raw_response={
                        "provider": candidate.provider,
                        "renderer": renderer_name,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                err_code = (
                    "KNOWLEDGE_CARD_RENDER_FAILED"
                    if provider == "system_knowledge_card"
                    else "DIAGRAM_RENDER_FAILED"
                )
                msg_type = (
                    "knowledge card"
                    if provider == "system_knowledge_card"
                    else "diagram"
                )
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                    error_code=err_code,
                    error_message=f"Failed to render local {msg_type}: {exc}",
                )

        refs: tuple[str, ...] = ()
        if provider in ("system_source_asset", "source_asset"):
            refs = request.source_asset_refs
        elif provider in ("system_user_asset", "user_asset"):
            refs = request.user_asset_refs
        else:
            refs = request.user_asset_refs or request.source_asset_refs

        if not refs:
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                error_code="NO_LOCAL_ASSET_REFERENCES",
                error_message=f"No asset references supplied for {candidate.provider}",
            )

        failures: list[tuple[str, str, ProviderOutcomeType]] = []
        safe_shot_id = re.sub(r"[^A-Za-z0-9_-]+", "_", request.shot_id).strip("_")

        for ref in refs:
            # Case A: Direct file reference
            if not ref.startswith("ev_"):
                candidate_path = Path(ref)
                if candidate_path.is_file() and candidate_path.stat().st_size > 0:
                    dest_filename = f"{candidate.provider}_{candidate_path.name}"
                    dest_path = target_dir / dest_filename
                    try:
                        if candidate_path.resolve() != dest_path.resolve():
                            shutil.copy2(candidate_path, dest_path)
                        else:
                            dest_path = candidate_path

                        return AdapterExecutionResult(
                            outcome_type=ProviderOutcomeType.SUCCESS,
                            file_path=str(dest_path),
                            raw_response={
                                "provider": candidate.provider,
                                "resolver_type": "direct_file",
                                "original_ref": str(candidate_path),
                                "fallback_used": False,
                            },
                        )
                    except Exception as exc:  # noqa: BLE001
                        failures.append((
                            "LOCAL_ASSET_COPY_FAILED",
                            f"Failed to copy local asset {candidate_path} to {dest_path}: {exc}",
                            ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                        ))
                        continue
                else:
                    failures.append((
                        "LOCAL_ASSET_FILE_NOT_FOUND",
                        f"Direct file reference does not exist or is empty: {ref}",
                        ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                    ))
                    continue

            # Case B: Evidence reference (starting with ev_)
            resolver = self._get_resolver()
            res = resolver.resolve_evidence(ref)

            if res.status == EvidenceResolutionStatus.EVIDENCE_NOT_FOUND:
                failures.append((
                    "SOURCE_EVIDENCE_NOT_FOUND",
                    res.error_message or f"Evidence '{ref}' was not found in repository.",
                    ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                ))
                continue

            if res.status == EvidenceResolutionStatus.DOCUMENT_NOT_FOUND:
                failures.append((
                    "SOURCE_DOCUMENT_NOT_FOUND",
                    res.error_message
                    or f"Source document for evidence '{ref}' was not found in repository.",
                    ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                ))
                continue

            ev_item = res.evidence_item
            source_doc = res.source_document

            media_type = str(getattr(source_doc, "media_type", "") or "").lower()
            source_locator = str(getattr(source_doc, "source_locator", "") or "").lower()

            # B1: PDF Source Document
            if "pdf" in media_type or source_locator.endswith(".pdf"):
                page_num = extract_page_number(getattr(ev_item, "locator", None))
                if page_num is None or page_num < 1:
                    failures.append((
                        "SOURCE_PDF_PAGE_INVALID",
                        f"Evidence '{ref}' locator has missing or invalid 1-based page number: {getattr(ev_item, 'locator', None)}",
                        ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                    ))
                    continue

                pdf_path = resolve_source_file_path(getattr(source_doc, "source_locator", None))
                if pdf_path is None or not pdf_path.is_file():
                    failures.append((
                        "SOURCE_PDF_RENDER_FAILED",
                        f"Source PDF file not found on disk at '{getattr(source_doc, 'source_locator', None)}' for evidence '{ref}'",
                        ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                    ))
                    continue

                try:
                    dest_filename = f"{candidate.provider}_{safe_shot_id or 'shot'}_{ref}_p{page_num}.png"
                    dest_path = target_dir / dest_filename
                    aspect_ratio = getattr(request, "aspect_ratio", None) or "16:9"
                    render_pdf_page_to_png(
                        pdf_path=pdf_path,
                        page_number=page_num,
                        target_path=dest_path,
                        aspect_ratio=aspect_ratio,
                    )
                    return AdapterExecutionResult(
                        outcome_type=ProviderOutcomeType.SUCCESS,
                        file_path=str(dest_path),
                        raw_response={
                            "provider": candidate.provider,
                            "resolver_type": "evidence_pdf_page",
                            "evidence_id": ref,
                            "source_document_id": getattr(source_doc, "source_document_id", None),
                            "pdf_page_number": page_num,
                            "fallback_used": False,
                        },
                    )
                except ValueError as ve:
                    failures.append((
                        "SOURCE_PDF_PAGE_INVALID",
                        f"PDF page {page_num} invalid for evidence '{ref}': {ve}",
                        ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                    ))
                    continue
                except Exception as exc:  # noqa: BLE001
                    failures.append((
                        "SOURCE_PDF_RENDER_FAILED",
                        f"Failed to render PDF page {page_num} for evidence '{ref}': {exc}",
                        ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                    ))
                    continue

            # B2: Image Source Document
            elif media_type.startswith("image/") or source_locator.endswith(
                (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
            ):
                img_path = resolve_source_file_path(getattr(source_doc, "source_locator", None))
                if img_path is not None and img_path.is_file() and img_path.stat().st_size > 0:
                    dest_filename = f"{candidate.provider}_{safe_shot_id or 'shot'}_{ref}_{img_path.name}"
                    dest_path = target_dir / dest_filename
                    try:
                        shutil.copy2(img_path, dest_path)
                        return AdapterExecutionResult(
                            outcome_type=ProviderOutcomeType.SUCCESS,
                            file_path=str(dest_path),
                            raw_response={
                                "provider": candidate.provider,
                                "resolver_type": "evidence_image",
                                "evidence_id": ref,
                                "source_document_id": getattr(source_doc, "source_document_id", None),
                                "original_ref": str(img_path),
                                "fallback_used": False,
                            },
                        )
                    except Exception as exc:  # noqa: BLE001
                        failures.append((
                            "LOCAL_ASSET_COPY_FAILED",
                            f"Failed to copy image asset {img_path} to {dest_path}: {exc}",
                            ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                        ))
                        continue
                else:
                    failures.append((
                        "SOURCE_DOCUMENT_NOT_FOUND",
                        f"Image file not found on disk at '{getattr(source_doc, 'source_locator', None)}' for evidence '{ref}'",
                        ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                    ))
                    continue

            # B3: Valid non-visual evidence -> Knowledge Card Fallback
            else:
                try:
                    diagram_path = render_diagram_card(request, target_dir)
                    return AdapterExecutionResult(
                        outcome_type=ProviderOutcomeType.SUCCESS,
                        file_path=str(diagram_path),
                        raw_response={
                            "provider": candidate.provider,
                            "resolver_type": "evidence_knowledge_card_fallback",
                            "evidence_id": ref,
                            "source_document_id": getattr(source_doc, "source_document_id", None),
                            "fallback_used": True,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    failures.append((
                        "KNOWLEDGE_CARD_RENDER_FAILED",
                        f"Failed to render knowledge card fallback for evidence '{ref}': {exc}",
                        ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                    ))
                    continue

        if failures:
            err_code, err_msg, outcome_type = failures[0]
            return AdapterExecutionResult(
                outcome_type=outcome_type,
                error_code=err_code,
                error_message=f"Referenced assets could not be resolved: {err_msg}",
            )

        return AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
            error_code="LOCAL_ASSET_FILE_NOT_FOUND",
            error_message=f"Referenced files for {candidate.provider} do not exist or are empty: {refs}",
        )
