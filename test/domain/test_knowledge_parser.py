from __future__ import annotations

import io
from unittest.mock import MagicMock, patch
import docx
import pytest
import requests

from app.domain.evidence import (
    DocumentTextNotExtractableError,
    EvidenceLocator,
    SourceDocument,
    SourceType,
    UrlFetchError,
)
from app.services.knowledge.document_parser import (
    DocumentParser,
    StructuralBlock,
    normalize_text,
)
from app.services.knowledge.source_fetcher import (
    SourceFetcher,
    validate_url_security,
)


def _make_source(source_type: SourceType = SourceType.FILE, title: str = "Test Doc", uri: str = "test.txt") -> SourceDocument:
    if source_type == SourceType.TEXT:
        return SourceDocument.create_text(text="Sample text content", title=title)
    elif source_type == SourceType.URL:
        return SourceDocument.register_url(url=uri, title=title)
    else:
        return SourceDocument.register_file(
            filename=uri,
            file_bytes=b"sample dummy content",
            media_type="application/octet-stream",
            storage_path=uri,
            title=title,
        )


# =============================================================================
# Normalization Tests
# =============================================================================


def test_normalize_text_unicode_nfkc():
    raw = "ＡＢＣ\u00a0\u3000ﬁle"
    norm = normalize_text(raw)
    assert norm == "ABC file"


def test_normalize_text_crlf_and_whitespace():
    raw = "Line 1\r\n\r\n  Line   2   \t\nLine 3\n\n\n\nLine 4"
    norm = normalize_text(raw)
    assert norm == "Line 1\n\nLine 2\nLine 3\n\nLine 4"


# =============================================================================
# Plain Text Parser Tests
# =============================================================================


def test_parse_plain_text():
    source = _make_source(SourceType.TEXT, "Notes", "inline")
    content = "Paragraph 1: Introduction to neural networks.\n\nParagraph 2: Transformer attention mechanisms."
    parser = DocumentParser()
    doc = parser.parse_text(source, content)

    assert doc.source_document_id == source.source_document_id
    assert len(doc.blocks) == 2
    assert "Introduction to neural networks" in doc.blocks[0].text
    assert doc.blocks[0].block_type == "PARAGRAPH"
    assert doc.blocks[0].locator.paragraph == 1
    assert "Transformer attention mechanisms" in doc.blocks[1].text
    assert doc.blocks[1].locator.paragraph == 2


# =============================================================================
# Markdown Parser Tests
# =============================================================================


def test_parse_markdown_with_hierarchy():
    source = _make_source(SourceType.FILE, "Guide", "guide.md")
    md = """# Title of Guide

Introductory context.

## Chapter 1: Foundations

First foundational paragraph.

### Section 1.1: Math

Mathematical explanations here.
"""
    parser = DocumentParser()
    doc = parser.parse_markdown(source, md)

    assert len(doc.blocks) >= 3
    math_block = next(b for b in doc.blocks if "Mathematical explanations" in b.text)
    assert math_block.locator.section is not None
    assert "Chapter 1: Foundations" in math_block.locator.section
    assert "Section 1.1: Math" in math_block.locator.section


# =============================================================================
# DOCX Parser Tests
# =============================================================================


def test_parse_docx():
    source = _make_source(SourceType.FILE, "Word Doc", "sample.docx")
    docx_doc = docx.Document()
    docx_doc.add_heading("Document Title", level=1)
    docx_doc.add_paragraph("First paragraph of docx content.")
    docx_doc.add_heading("Subheading", level=2)
    docx_doc.add_paragraph("Second paragraph under subheading.")

    buf = io.BytesIO()
    docx_doc.save(buf)
    docx_bytes = buf.getvalue()

    parser = DocumentParser()
    parsed = parser.parse_docx(source, docx_bytes)

    assert len(parsed.blocks) >= 2
    p1 = next(b for b in parsed.blocks if "First paragraph" in b.text)
    assert p1.locator.paragraph is not None
    p2 = next(b for b in parsed.blocks if "Second paragraph" in b.text)
    assert p2.locator.section is not None
    assert "Subheading" in p2.locator.section


# =============================================================================
# PDF Parser Tests
# =============================================================================


def test_parse_pdf_with_text():
    source = _make_source(SourceType.FILE, "Sample PDF", "sample.pdf")
    pdf_bytes = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 24 Tf 100 700 Td (Hello Knowledge PDF) Tj ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000234 00000 n 
0000000328 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
407
%%EOF"""

    parser = DocumentParser()
    parsed = parser.parse_pdf(source, pdf_bytes)

    assert len(parsed.blocks) == 1
    assert "Hello Knowledge PDF" in parsed.blocks[0].text
    assert parsed.blocks[0].locator.page == 1
    assert parsed.parser_metadata.get("page_count") == 1


def test_parse_pdf_scanned_raises_document_text_not_extractable():
    source = _make_source(SourceType.FILE, "Scanned PDF", "scanned.pdf")
    empty_pdf = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj
xref
0 4
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer << /Size 4 /Root 1 0 R >>
startxref
192
%%EOF"""

    parser = DocumentParser()
    with pytest.raises(DocumentTextNotExtractableError) as exc_info:
        parser.parse_pdf(source, empty_pdf)
    assert "extractable digital text" in str(exc_info.value) or "zero extractable text" in str(exc_info.value)


# =============================================================================
# HTML Parser Tests
# =============================================================================


def test_parse_html():
    source = _make_source(SourceType.URL, "Web Article", "https://example.com/article")
    html = """<!DOCTYPE html>
<html>
<head><title>Article Title</title></head>
<body>
<nav>Skip this nav</nav>
<h1>Main Heading</h1>
<p>First web paragraph with important information.</p>
<script>alert("bad");</script>
<h2>Sub Heading</h2>
<p>Second web paragraph under sub heading.</p>
<footer>Footer copyright</footer>
</body>
</html>"""

    parser = DocumentParser()
    parsed = parser.parse_html(source, html)

    assert parsed.title == "Article Title"
    assert len(parsed.blocks) >= 2
    full_text = " ".join(b.text for b in parsed.blocks)
    assert "alert" not in full_text
    assert "First web paragraph" in full_text
    assert "Second web paragraph" in full_text


# =============================================================================
# SourceFetcher & SSRF Security Tests
# =============================================================================


@pytest.mark.parametrize("bad_url", [
    "http://127.0.0.1/admin",
    "http://localhost/secret",
    "http://10.0.0.1/internal",
    "http://192.168.1.1/router",
    "http://172.16.0.1/docker",
    "http://169.254.169.254/latest/meta-data/",
    "ftp://ftp.example.com/file",
    "file:///etc/passwd",
    "gopher://example.com",
])
def test_source_fetcher_ssrf_protection(bad_url: str):
    fetcher = SourceFetcher()
    with pytest.raises(UrlFetchError) as exc_info:
        fetcher.validate_url_security(bad_url)
    assert any(w in str(exc_info.value).lower() for w in ("prohibited", "scheme", "blocked", "forbidden", "ssrf"))


def test_source_fetcher_bounded_size():
    fetcher = SourceFetcher(max_bytes=100)
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": "200"}
    mock_resp.status_code = 200

    with patch.object(fetcher.session, "get", return_value=mock_resp):
        with patch.object(fetcher, "validate_url_security"):
            with pytest.raises(UrlFetchError) as exc_info:
                fetcher.fetch_url("https://example.com/large")
            assert "exceeds maximum allowed size" in str(exc_info.value).lower() or "maximum allowed size" in str(exc_info.value).lower()


def test_source_fetcher_success():
    fetcher = SourceFetcher()
    html_content = "<html><head><title>Test Page</title></head><body><p>Clean text content</p></body></html>"
    
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Type": "text/html"}
    mock_resp.status_code = 200
    mock_resp.encoding = "utf-8"
    mock_resp.url = "https://example.com/page"
    mock_resp.iter_content.return_value = [html_content.encode("utf-8")]

    with patch.object(fetcher.session, "get", return_value=mock_resp):
        with patch.object(fetcher, "validate_url_security"):
            title, text = fetcher.fetch_and_extract_html("https://example.com/page")
            assert title == "Test Page"
            assert "Clean text content" in text
