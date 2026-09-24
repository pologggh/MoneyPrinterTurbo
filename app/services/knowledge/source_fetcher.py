from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import requests

from app.domain.evidence import UrlFetchError

# Private / reserved subnets for SSRF protection
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / cloud metadata (AWS, GCP, etc.)
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

MAX_CONTENT_BYTES = 5 * 1024 * 1024  # 5 MB
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_REDIRECTS = 3


class FetchedUrlContent:
    """Captured content and metadata from an explicit user URL fetch."""

    def __init__(
        self,
        url: str,
        final_url: str,
        status_code: int,
        content_type: str,
        title: str | None,
        extracted_text: str,
        raw_html: str | None = None,
    ) -> None:
        self.url = url
        self.final_url = final_url
        self.status_code = status_code
        self.content_type = content_type
        self.title = title
        self.extracted_text = extracted_text
        self.raw_html = raw_html


class SourceFetcher:
    """Acquisition client for explicit user-supplied URLs with SSRF protection and bounded sizes."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_bytes: int = MAX_CONTENT_BYTES,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.session = session or requests.Session()

    def validate_url_security(self, url: str, allow_private_for_tests: bool = False) -> None:
        return validate_url_security(url, allow_private_for_tests=allow_private_for_tests)

    def fetch_url(self, url: str, allow_private_for_tests: bool = False) -> FetchedUrlContent:
        return fetch_url_content(
            url=url,
            timeout=self.timeout,
            max_bytes=self.max_bytes,
            allow_private_for_tests=allow_private_for_tests,
            session=self.session,
        )

    def fetch_and_extract_html(self, url: str, allow_private_for_tests: bool = False) -> tuple[str | None, str]:
        res = self.fetch_url(url, allow_private_for_tests=allow_private_for_tests)
        return res.title, res.extracted_text


def validate_url_security(url: str, allow_private_for_tests: bool = False) -> None:
    """Validates that a URL uses http/https and does not target internal/private subnets."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise UrlFetchError(f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are allowed.")

    hostname = parsed.hostname
    if not hostname:
        raise UrlFetchError(f"Invalid URL '{url}': Missing hostname.")

    # Block localhost explicitly
    if hostname.lower() in ("localhost", "127.0.0.1", "::1"):
        if not allow_private_for_tests:
            raise UrlFetchError(f"Access to private or local address '{hostname}' is forbidden (SSRF protection).")

    if not allow_private_for_tests:
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for item in addr_info:
                ip_str = item[4][0]
                ip_obj = ipaddress.ip_address(ip_str)
                for blocked in _BLOCKED_NETWORKS:
                    if ip_obj in blocked:
                        raise UrlFetchError(
                            f"Access to private/internal network IP '{ip_str}' for host '{hostname}' is blocked."
                        )
        except socket.gaierror as exc:
            raise UrlFetchError(f"Failed to resolve host '{hostname}': {exc}") from exc


def fetch_url_content(
    url: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = MAX_CONTENT_BYTES,
    allow_private_for_tests: bool = False,
    session: requests.Session | None = None,
) -> FetchedUrlContent:
    """Bounded, secure fetch of an explicit user-supplied URL and main textual content extraction."""
    clean_url = url.strip()
    validate_url_security(clean_url, allow_private_for_tests=allow_private_for_tests)

    req_session = session or requests.Session()
    req_session.max_redirects = MAX_REDIRECTS

    try:
        response = req_session.get(
            clean_url,
            timeout=timeout,
            stream=True,
            headers={
                "User-Agent": "MoneyPrinterTurbo-KnowledgeAgent/1.0",
                "Accept": "text/html,text/plain;q=0.9,*/*;q=0.8",
            },
        )
    except requests.exceptions.TooManyRedirects as exc:
        raise UrlFetchError(f"URL exceeded maximum redirect limit of {MAX_REDIRECTS}: {clean_url}") from exc
    except requests.exceptions.Timeout as exc:
        raise UrlFetchError(f"URL fetch timed out after {timeout}s: {clean_url}") from exc
    except requests.exceptions.RequestException as exc:
        raise UrlFetchError(f"URL fetch failed for {clean_url}: {exc}") from exc

    if response.status_code >= 400:
        raise UrlFetchError(f"URL returned error HTTP {response.status_code}: {clean_url}")

    content_length_hdr = response.headers.get("Content-Length")
    if content_length_hdr and content_length_hdr.strip().isdigit():
        if int(content_length_hdr.strip()) > max_bytes:
            raise UrlFetchError(f"URL content length ({content_length_hdr} bytes) exceeds maximum allowed size of {max_bytes} bytes.")

    # Read bounded content
    content_chunks = []
    total_bytes = 0
    for chunk in response.iter_content(chunk_size=65536):
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise UrlFetchError(f"URL content exceeded maximum allowed size of {max_bytes} bytes.")
        content_chunks.append(chunk)

    raw_bytes = b"".join(content_chunks)
    content_type = response.headers.get("Content-Type", "").lower()
    charset = response.encoding if isinstance(response.encoding, str) else "utf-8"

    try:
        text_body = raw_bytes.decode(charset, errors="replace")
    except Exception as exc:
        raise UrlFetchError(f"Failed to decode response body as {charset}: {exc}") from exc

    title: str | None = None
    extracted_text: str = ""

    if "text/html" in content_type or "<html" in text_body[:500].lower():
        soup = BeautifulSoup(text_body, "html.parser")

        # Extract title
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Remove irrelevant elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]):
            tag.decompose()

        # Extract textual content preserving logical paragraph breaks
        extracted_text = soup.get_text(separator="\n", strip=True)
    else:
        # Plain text
        extracted_text = text_body.strip()
        first_line = extracted_text.splitlines()[0] if extracted_text else ""
        title = first_line[:100] if first_line else clean_url

    if not extracted_text:
        raise UrlFetchError(f"No textual content could be extracted from {clean_url}.")

    return FetchedUrlContent(
        url=clean_url,
        final_url=response.url,
        status_code=response.status_code,
        content_type=content_type,
        title=title,
        extracted_text=extracted_text,
        raw_html=text_body if "text/html" in content_type else None,
    )
