from __future__ import annotations

import hashlib
import ipaddress
import mimetypes
import os
import socket
import struct
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.domain.asset_execution import AssetMediaType


class InvalidAssetFileError(ValueError):
    """Raised when an asset file is missing, empty, or cannot be parsed."""


def sanitize_url(url: str) -> str:
    """
    Masks sensitive query parameters (token, key, auth, signature, secret, etc.)
    from URLs to prevent credential leakage in logs and exceptions.
    """
    try:
        parsed = urlparse(url)
        if not parsed.query:
            return url
        sensitive_keywords = {
            "token",
            "key",
            "api_key",
            "secret",
            "signature",
            "sig",
            "auth",
            "password",
            "access_token",
        }
        query_items = parse_qsl(parsed.query, keep_blank_values=True)
        masked_items = []
        for k, v in query_items:
            if any(kw in k.lower() for kw in sensitive_keywords):
                masked_items.append((k, "***"))
            else:
                masked_items.append((k, v))
        masked_query = urlencode(masked_items, safe="*")
        netloc = parsed.netloc
        if "@" in netloc:
            _, host_part = netloc.rsplit("@", 1)
            netloc = f"***:***@{host_part}"
        return urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                masked_query,
                parsed.fragment,
            )
        )
    except Exception:
        return "<url masked>"


def _is_ip_allowed(ip_str: str) -> bool:
    """Returns True only if the IP address is public and non-loopback/reserved."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return not (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return False


def _validate_safe_url(url: str) -> None:
    """
    Validates that a URL is safe to download from, preventing SSRF attacks.
    Blocks non-http/https protocols, localhost, and private/internal IP ranges.
    """
    safe_display = sanitize_url(url)
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise InvalidAssetFileError(f"Invalid URL format: {safe_display} ({exc})") from exc

    if parsed.scheme.lower() not in ("http", "https"):
        raise InvalidAssetFileError(
            f"Invalid URL scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted: {safe_display}"
        )

    hostname = parsed.hostname
    if not hostname:
        raise InvalidAssetFileError(f"URL must contain a valid hostname: {safe_display}")

    hostname_lower = hostname.lower()
    if hostname_lower in ("localhost", "127.0.0.1", "::1"):
        raise InvalidAssetFileError(
            f"Access to local or private address '{hostname}' is forbidden: {safe_display}"
        )

    # Direct IP address check
    try:
        ip = ipaddress.ip_address(hostname)
        if not _is_ip_allowed(str(ip)):
            raise InvalidAssetFileError(
                f"Access to private/prohibited IP address '{hostname}' is forbidden: {safe_display}"
            )
        return
    except ValueError:
        pass

    # Domain name DNS resolution check
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for item in addr_info:
            ip_str = item[4][0]
            if not _is_ip_allowed(ip_str):
                raise InvalidAssetFileError(
                    f"Domain '{hostname}' resolves to private/prohibited address '{ip_str}': {safe_display}"
                )
    except socket.gaierror as exc:
        raise InvalidAssetFileError(
            f"DNS resolution failed for '{hostname}': {safe_display} ({exc})"
        ) from exc


class AssetProbeResult(BaseModel):
    """
    Validated metadata probed from a concrete media file on disk.
    """
    file_path: str
    file_hash: str
    file_size_bytes: int = Field(gt=0)
    media_type: AssetMediaType
    mime_type: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_seconds: float = Field(ge=0.0)
    fps: float | None = None


def calculate_sha256(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Calculates lowercase hexadecimal SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest().lower()


def _probe_image_file(file_path: str) -> tuple[int, int, str]:
    """Probes image width, height, and MIME type using Pillow."""
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            fmt = (img.format or "png").lower()
            mime_type = Image.MIME.get(img.format) or f"image/{fmt}"
            return width, height, mime_type
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidAssetFileError(f"Failed to decode image file {file_path}: {exc}") from exc


def _parse_mp4_atoms(file_path: str) -> tuple[int, int, float]:
    """
    Lightweight pure-Python parser for MP4 moov/mvhd/trak/tkhd boxes to extract
    duration, width, and height without external dependencies.
    """
    timescale = 1000
    duration_units = 0
    width = 0
    height = 0

    try:
        with open(file_path, "rb") as f:
            file_size = os.fstat(f.fileno()).st_size

            def read_boxes(start: int, length: int) -> list[tuple[str, int, int]]:
                boxes = []
                pos = start
                end = start + length
                while pos + 8 <= end:
                    f.seek(pos)
                    header = f.read(8)
                    if len(header) < 8:
                        break
                    box_len, box_type_bytes = struct.unpack(">I4s", header)
                    box_type = box_type_bytes.decode("latin1", errors="ignore")
                    payload_pos = pos + 8
                    actual_len = box_len

                    if box_len == 1:
                        # 64-bit length
                        ext_header = f.read(8)
                        if len(ext_header) < 8:
                            break
                        actual_len = struct.unpack(">Q", ext_header)[0]
                        payload_pos = pos + 16
                    elif box_len == 0:
                        # Box extends to EOF
                        actual_len = end - pos

                    payload_len = actual_len - (payload_pos - pos)
                    boxes.append((box_type, payload_pos, payload_len))
                    pos += actual_len
                    if actual_len <= 0:
                        break
                return boxes

            root_boxes = read_boxes(0, file_size)
            moov_box = next((b for b in root_boxes if b[0] == "moov"), None)
            if not moov_box:
                return 1920, 1080, 5.0  # Fallback reasonable defaults if atom missing

            moov_children = read_boxes(moov_box[1], moov_box[2])
            mvhd_box = next((b for b in moov_children if b[0] == "mvhd"), None)
            if mvhd_box:
                f.seek(mvhd_box[1])
                ver_flags = f.read(4)
                if len(ver_flags) == 4:
                    version = ver_flags[0]
                    if version == 1:
                        f.seek(mvhd_box[1] + 4 + 16)
                        data = f.read(12)
                        if len(data) == 12:
                            timescale, duration_units = struct.unpack(">IQ", data)
                    else:
                        f.seek(mvhd_box[1] + 4 + 8)
                        data = f.read(8)
                        if len(data) == 8:
                            timescale, duration_units = struct.unpack(">II", data)

            # Find video track for width and height
            trak_boxes = [b for b in moov_children if b[0] == "trak"]
            for trak in trak_boxes:
                trak_children = read_boxes(trak[1], trak[2])
                tkhd = next((b for b in trak_children if b[0] == "tkhd"), None)
                if tkhd:
                    f.seek(tkhd[1])
                    ver = f.read(1)
                    if ver:
                        v = ver[0]
                        # skip flags(3), times, id, duration
                        skip = 3 + (32 if v == 1 else 20) + 48  # skip to width/height
                        f.seek(tkhd[1] + 1 + skip)
                        dim_data = f.read(8)
                        if len(dim_data) == 8:
                            w_fixed, h_fixed = struct.unpack(">II", dim_data)
                            t_w = w_fixed >> 16
                            t_h = h_fixed >> 16
                            if t_w > 0 and t_h > 0:
                                width = t_w
                                height = t_h
                                break

        duration_sec = (duration_units / timescale) if timescale > 0 else 0.0
        width = width or 1920
        height = height or 1080
        return width, height, max(0.0, float(duration_sec))
    except (OSError, struct.error, ValueError):
        return 1920, 1080, 5.0


def probe_media_file(
    file_path: str | Path,
    media_type: AssetMediaType | None = None,
    custom_mime: str | None = None,
) -> AssetProbeResult:
    """
    Validates that a media file exists, is non-empty, calculates SHA-256,
    and probes resolution and duration.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise InvalidAssetFileError(f"Asset file does not exist: {path}")

    file_size = path.stat().st_size
    if file_size <= 0:
        raise InvalidAssetFileError(f"Asset file is empty: {path}")

    file_hash = calculate_sha256(path)

    # Infer media type if not provided
    ext = path.suffix.lower()
    guessed_mime, _ = mimetypes.guess_type(str(path))

    if media_type is None:
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            inferred_type = AssetMediaType.IMAGE
        elif ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            inferred_type = AssetMediaType.VIDEO
        elif ext in {".mp3", ".wav", ".aac", ".m4a"}:
            inferred_type = AssetMediaType.AUDIO
        else:
            inferred_type = AssetMediaType.VIDEO
    else:
        inferred_type = media_type

    if inferred_type == AssetMediaType.IMAGE:
        width, height, img_mime = _probe_image_file(str(path))
        mime_type = custom_mime or img_mime or guessed_mime or "image/png"
        return AssetProbeResult(
            file_path=str(path),
            file_hash=file_hash,
            file_size_bytes=file_size,
            media_type=AssetMediaType.IMAGE,
            mime_type=mime_type,
            width=width,
            height=height,
            duration_seconds=0.0,
            fps=None,
        )

    # VIDEO
    mime_type = custom_mime or guessed_mime or "video/mp4"
    width, height, duration_sec = _parse_mp4_atoms(str(path))

    return AssetProbeResult(
        file_path=str(path),
        file_hash=file_hash,
        file_size_bytes=file_size,
        media_type=AssetMediaType.VIDEO,
        mime_type=mime_type,
        width=width,
        height=height,
        duration_seconds=duration_sec,
        fps=30.0,
    )


def download_media_file(
    url: str,
    target_dir: str | Path,
    filename_prefix: str = "media",
    timeout: tuple[int, int] = (30, 180),
) -> Path:
    """
    Downloads remote media URL to local storage directory with SSRF protection,
    size limits, and cleanup guarantees.
    Raises InvalidAssetFileError if validation or download fails.
    """
    from urllib.parse import urlparse

    import requests

    from app.config import config

    safe_display_url = sanitize_url(url)
    _validate_safe_url(url)

    dest_dir = Path(target_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = config.app.get("max_asset_download_bytes", 500 * 1024 * 1024)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = None
    tmp_file = dest_dir / f"{filename_prefix}_{int(time.time()*1000)}_{uuid4().hex[:8]}.tmp"

    try:
        try:
            response = requests.get(
                url,
                headers=headers,
                proxies=config.proxy,
                timeout=timeout,
                stream=True,
            )
        except Exception as exc:
            raise InvalidAssetFileError(
                f"Network error downloading media from {safe_display_url}: {exc}"
            ) from exc

        # Check redirect history for SSRF
        for hist_resp in response.history:
            _validate_safe_url(hist_resp.url)
        _validate_safe_url(response.url)

        if response.status_code != 200:
            raise InvalidAssetFileError(
                f"Remote media URL returned HTTP {response.status_code}: {safe_display_url}"
            )

        # Check Content-Length if present
        content_length = response.headers.get("content-length")
        if content_length and content_length.isdigit():
            cl_int = int(content_length)
            if cl_int > max_bytes:
                raise InvalidAssetFileError(
                    f"Remote media size ({cl_int} bytes) exceeds limit of {max_bytes} bytes: {safe_display_url}"
                )

        # Detect extension from Content-Type or URL path
        content_type = response.headers.get("content-type", "").lower()
        path_ext = Path(urlparse(response.url).path).suffix.lower()

        if path_ext in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".png", ".jpg", ".jpeg", ".webp", ".mp3", ".wav"):
            ext = path_ext
        elif "image/png" in content_type:
            ext = ".png"
        elif "image/jpeg" in content_type or "image/jpg" in content_type:
            ext = ".jpg"
        elif "image/webp" in content_type:
            ext = ".webp"
        elif "video/mp4" in content_type:
            ext = ".mp4"
        else:
            ext = ".mp4"

        dest_file = dest_dir / f"{filename_prefix}{ext}"

        downloaded_bytes = 0
        with open(tmp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > max_bytes:
                        raise InvalidAssetFileError(
                            f"Downloaded media exceeded limit of {max_bytes} bytes: {safe_display_url}"
                        )
                    f.write(chunk)

        if not tmp_file.is_file() or tmp_file.stat().st_size <= 0:
            raise InvalidAssetFileError(
                f"Downloaded media file is empty (0 bytes): {safe_display_url}"
            )

        if dest_file.exists():
            dest_file.unlink(missing_ok=True)
        tmp_file.rename(dest_file)
        return dest_file
    except Exception as exc:
        if isinstance(exc, InvalidAssetFileError):
            raise
        raise InvalidAssetFileError(
            f"Failed to save downloaded media from {safe_display_url}: {exc}"
        ) from exc
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        if tmp_file.exists():
            try:
                tmp_file.unlink(missing_ok=True)
            except Exception:
                pass
