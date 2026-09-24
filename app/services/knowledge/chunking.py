from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from app.domain.evidence import (
    EvidenceLocator,
    KnowledgeChunk,
    SourceDocument,
)
from app.services.knowledge.document_parser import (
    DEFAULT_PROCESSING_VERSION,
    ParsedKnowledgeDocument,
    StructuralBlock,
    normalize_text,
)


@dataclass(frozen=True)
class ChunkingPolicy:
    """Configurable boundaries and target sizes for deterministic chunking."""

    target_size: int = 400
    max_size: int = 800
    overlap: int = 50
    processing_version: str = DEFAULT_PROCESSING_VERSION


class KnowledgeChunker:
    """Deterministic, structure-aware document chunking engine."""

    def __init__(self, policy: ChunkingPolicy | None = None) -> None:
        self.policy = policy or ChunkingPolicy()

    def chunk_document(
        self,
        source_document: SourceDocument,
        parsed_document: ParsedKnowledgeDocument,
    ) -> tuple[KnowledgeChunk, ...]:
        """Splits a parsed document into deterministic KnowledgeChunks respecting structure."""
        if not parsed_document.blocks:
            return ()

        chunks: list[KnowledgeChunk] = []
        current_blocks: list[StructuralBlock] = []
        current_len = 0
        chunk_idx = 0

        def flush_chunk(blocks_to_flush: list[StructuralBlock]) -> None:
            nonlocal chunk_idx
            if not blocks_to_flush:
                return

            # Combine text
            combined_text = "\n\n".join(b.text.strip() for b in blocks_to_flush if b.text.strip())
            norm_text = normalize_text(combined_text)
            if not norm_text:
                return

            # Merge locators
            first_loc = blocks_to_flush[0].locator
            last_loc = blocks_to_flush[-1].locator
            merged_locator = EvidenceLocator(
                page=first_loc.page,
                section=first_loc.section or last_loc.section,
                paragraph=first_loc.paragraph,
                line_start=first_loc.line_start,
                line_end=last_loc.line_end,
                extra={
                    "block_indices": [b.block_index for b in blocks_to_flush],
                    "block_types": [b.block_type for b in blocks_to_flush],
                }
            )

            chk = KnowledgeChunk.create(
                source_document=source_document,
                normalized_text=norm_text,
                chunk_index=chunk_idx,
                locator=merged_locator,
                processing_version=self.policy.processing_version,
            )
            chunks.append(chk)
            chunk_idx += 1

        for block in parsed_document.blocks:
            text = block.text.strip()
            if not text:
                continue

            text_len = len(text)

            # Case A: An individual block exceeds max_size -> split into sub-blocks
            if text_len > self.policy.max_size:
                # Flush pending blocks first
                if current_blocks:
                    flush_chunk(current_blocks)
                    current_blocks = []
                    current_len = 0

                # Split large block by sentences or sliding windows
                sub_chunks = self._split_large_block(block, source_document, chunk_idx)
                for sc in sub_chunks:
                    chunks.append(sc)
                    chunk_idx += 1
                continue

            # Case B: Adding this block would exceed target_size and we already have content
            if current_blocks and (current_len + text_len + 2 > self.policy.target_size):
                flush_chunk(current_blocks)
                current_blocks = [block]
                current_len = text_len
            else:
                current_blocks.append(block)
                current_len += text_len + 2

        if current_blocks:
            flush_chunk(current_blocks)

        return tuple(chunks)

    def _split_large_block(
        self,
        block: StructuralBlock,
        source_document: SourceDocument,
        starting_chunk_idx: int,
    ) -> list[KnowledgeChunk]:
        """Splits an oversized block along sentence boundaries with overlap."""
        text = block.text.strip()
        # Split on sentence terminals
        sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？\n])\s+", text) if s.strip()]
        if not sentences:
            sentences = [text]

        sub_chunks: list[KnowledgeChunk] = []
        cur_sentences: list[str] = []
        cur_len = 0
        local_idx = starting_chunk_idx

        for sent in sentences:
            s_len = len(sent)
            # If single sentence is itself larger than max_size, hard-slice with overlap
            if s_len > self.policy.max_size:
                if cur_sentences:
                    combined = " ".join(cur_sentences)
                    chk = KnowledgeChunk.create(
                        source_document=source_document,
                        normalized_text=normalize_text(combined),
                        chunk_index=local_idx,
                        locator=block.locator,
                        processing_version=self.policy.processing_version,
                    )
                    sub_chunks.append(chk)
                    local_idx += 1
                    cur_sentences = []
                    cur_len = 0

                # Slice large sentence
                step = self.policy.target_size - self.policy.overlap
                if step <= 0:
                    step = self.policy.target_size // 2 or 1
                for start in range(0, s_len, step):
                    window = sent[start : start + self.policy.target_size].strip()
                    if window:
                        chk = KnowledgeChunk.create(
                            source_document=source_document,
                            normalized_text=normalize_text(window),
                            chunk_index=local_idx,
                            locator=block.locator,
                            processing_version=self.policy.processing_version,
                        )
                        sub_chunks.append(chk)
                        local_idx += 1
                continue

            if cur_sentences and (cur_len + s_len + 1 > self.policy.target_size):
                combined = " ".join(cur_sentences)
                chk = KnowledgeChunk.create(
                    source_document=source_document,
                    normalized_text=normalize_text(combined),
                    chunk_index=local_idx,
                    locator=block.locator,
                    processing_version=self.policy.processing_version,
                )
                sub_chunks.append(chk)
                local_idx += 1

                # Keep overlap sentence if possible
                overlap_text = cur_sentences[-1] if cur_sentences else ""
                if len(overlap_text) < self.policy.overlap:
                    cur_sentences = [overlap_text, sent]
                    cur_len = len(overlap_text) + 1 + s_len
                else:
                    cur_sentences = [sent]
                    cur_len = s_len
            else:
                cur_sentences.append(sent)
                cur_len += s_len + 1

        if cur_sentences:
            combined = " ".join(cur_sentences)
            chk = KnowledgeChunk.create(
                source_document=source_document,
                normalized_text=normalize_text(combined),
                chunk_index=local_idx,
                locator=block.locator,
                processing_version=self.policy.processing_version,
            )
            sub_chunks.append(chk)

        return sub_chunks

    def chunk_raw_text(
        self,
        source_document: SourceDocument,
        raw_text: str,
    ) -> tuple[KnowledgeChunk, ...]:
        """Direct chunking of raw plain text."""
        cleaned = normalize_text(raw_text)
        if not cleaned:
            return ()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
        blocks = [
            StructuralBlock(
                block_index=i + 1,
                block_type="PARAGRAPH",
                text=p,
                locator=EvidenceLocator(paragraph=i + 1),
            )
            for i, p in enumerate(paragraphs)
        ]
        parsed = ParsedKnowledgeDocument(
            source_document_id=source_document.source_document_id,
            processing_version=self.policy.processing_version,
            title=source_document.title,
            blocks=tuple(blocks),
        )
        return self.chunk_document(source_document, parsed)
