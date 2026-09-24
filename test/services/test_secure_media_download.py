from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.asset_probe_service import (
    InvalidAssetFileError,
    _validate_safe_url,
    download_media_file,
    sanitize_url,
)


def test_sanitize_url_masks_sensitive_query_parameters():
    """Area 6: Verify sensitive credentials/tokens in URL are redacted with ***."""
    url = "https://cdn.example.com/asset.mp4?token=secret123&api_key=ak_999&signature=sig_abc&id=123"
    sanitized = sanitize_url(url)
    assert "secret123" not in sanitized
    assert "ak_999" not in sanitized
    assert "sig_abc" not in sanitized
    assert "id=123" in sanitized
    assert "token=***" in sanitized
    assert "api_key=***" in sanitized


def test_validate_safe_url_blocks_unsafe_protocols():
    """Area 6: Block non-http/https protocols (e.g. file://, ftp://, gopher://)."""
    with pytest.raises(InvalidAssetFileError, match="Only HTTP and HTTPS are permitted"):
        _validate_safe_url("file:///etc/passwd")

    with pytest.raises(InvalidAssetFileError, match="Only HTTP and HTTPS are permitted"):
        _validate_safe_url("ftp://example.com/movie.mp4")


def test_validate_safe_url_blocks_localhost_and_private_ips():
    """Area 6: Block localhost, 127.0.0.1, link-local, and RFC1918 private IP ranges."""
    prohibited_urls = [
        "http://localhost/video.mp4",
        "http://127.0.0.1:8080/video.mp4",
        "http://192.168.1.100/video.mp4",
        "http://10.0.0.1/video.mp4",
        "http://172.16.0.1/video.mp4",
        "http://169.254.169.254/latest/meta-data/",
    ]
    for u in prohibited_urls:
        with pytest.raises(InvalidAssetFileError) as exc_info:
            _validate_safe_url(u)
        assert "forbidden" in str(exc_info.value).lower() or "prohibited" in str(exc_info.value).lower()


def test_validate_safe_url_blocks_domain_resolving_to_private_ip():
    """Area 6: Block domain names resolving via DNS to internal/private IPs (DNS rebinding / SSRF)."""
    with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("127.0.0.1", 80))]):
        with pytest.raises(InvalidAssetFileError) as exc_info:
            _validate_safe_url("https://malicious-domain.com/video.mp4")
        assert "resolves to private/prohibited address" in str(exc_info.value)


def test_download_media_file_blocks_oversized_content_length(tmp_path):
    """Area 6: Abort immediately if remote Content-Length exceeds max_asset_download_bytes."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-length": str(10 * 1024 * 1024)}  # 10MB
    mock_resp.history = []
    mock_resp.url = "https://cdn.example.com/large.mp4"

    with (
        patch("app.services.asset_probe_service._validate_safe_url"),
        patch("requests.get", return_value=mock_resp),
        patch("app.config.config.app.get", return_value=1024 * 1024),  # 1MB limit
    ):
        with pytest.raises(InvalidAssetFileError) as exc_info:
            download_media_file("https://cdn.example.com/large.mp4", target_dir=tmp_path)
        assert "exceeds limit" in str(exc_info.value)


def test_download_media_file_aborts_and_cleans_up_on_streaming_size_limit(tmp_path):
    """
    Area 6: If remote sends chunks exceeding max bytes without content-length header,
    abort streaming and guarantee no .tmp file remains on disk.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "video/mp4"}
    mock_resp.history = []
    mock_resp.url = "https://cdn.example.com/infinite.mp4"
    # Yield 10 chunks of 100KB each = 1000KB
    mock_resp.iter_content.return_value = [b"x" * 102400] * 10

    with (
        patch("app.services.asset_probe_service._validate_safe_url"),
        patch("requests.get", return_value=mock_resp),
        patch("app.config.config.app.get", return_value=200 * 1024),  # 200KB limit
    ):
        with pytest.raises(InvalidAssetFileError) as exc_info:
            download_media_file("https://cdn.example.com/infinite.mp4", target_dir=tmp_path)
        assert "exceeded limit" in str(exc_info.value)

    # Verify no .tmp files remain on disk in target_dir
    tmp_files = list(Path(tmp_path).glob("*.tmp"))
    assert tmp_files == []


def test_download_media_file_cleans_up_temp_file_on_exception(tmp_path):
    """Area 6: Verify temp file is cleaned up even on unexpected network or OS exception."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {}
    mock_resp.history = []
    mock_resp.url = "https://cdn.example.com/stream.mp4"

    def fail_chunk(chunk_size):
        yield b"header_data"
        raise ConnectionResetError("Connection dropped by peer")

    mock_resp.iter_content.side_effect = fail_chunk

    with (
        patch("app.services.asset_probe_service._validate_safe_url"),
        patch("requests.get", return_value=mock_resp),
    ):
        with pytest.raises(InvalidAssetFileError) as exc_info:
            download_media_file("https://cdn.example.com/stream.mp4", target_dir=tmp_path)
        assert "Failed to save downloaded media" in str(exc_info.value)

    # Temp file must be cleaned up in finally block
    tmp_files = list(Path(tmp_path).glob("*.tmp"))
    assert tmp_files == []
