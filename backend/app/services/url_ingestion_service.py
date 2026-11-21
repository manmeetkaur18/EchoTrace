import ipaddress
import logging
from pathlib import Path
import re
import socket
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse, urljoin
import uuid
import xml.etree.ElementTree as ET

import httpx

from app.services.pipeline_service import (
    AUDIO_EXTENSIONS,
    MAX_AUDIO_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ingest_recording,
)

# Configure module logger for server-side diagnostics
logger = logging.getLogger(__name__)

# Base directory paths matching EchoTrace convention
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BACKEND_DIR / "data" / "uploads"

# ---------------------------------------------------------
# Security & Limit Configuration
# ---------------------------------------------------------
MAX_DOWNLOAD_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB max downloaded media
MAX_FEED_SIZE_BYTES: int = 5 * 1024 * 1024        # 5 MB max RSS feed size
MAX_REDIRECTS: int = 5
CONNECT_TIMEOUT_SECONDS: float = 10.0
READ_TIMEOUT_SECONDS: float = 30.0
STREAM_CHUNK_SIZE_BYTES: int = 64 * 1024          # 64 KB per stream chunk

ALLOWED_SCHEMES: Set[str] = {"http", "https"}

# Content-Type mapping to permitted file extensions
CONTENT_TYPE_MAP: Dict[str, str] = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-pn-wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".m4a",
    "video/mp4": ".mp4",
}

# Regex patterns for platform source detection
YOUTUBE_REGEX = re.compile(
    r"^https?:\/\/(?:(?:www|m)\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([\w-]{11})",
    re.IGNORECASE,
)
TED_REGEX = re.compile(
    r"^https?:\/\/(?:www\.)?ted\.com\/talks\/([\w-]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------
# 1. URL Security & SSRF Protection
# ---------------------------------------------------------
def is_ip_restricted(ip_obj: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> bool:
    """
    Rigorously verifies whether an IP address belongs to a non-public, internal,
    loopback, link-local, multicast, reserved, unspecified, or embedded private range.

    Covers:
    - IPv4: loopback (127.0.0.0/8), private (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16),
      link-local / cloud metadata (169.254.0.0/16), multicast (224.0.0.0/4),
      reserved (240.0.0.0/4), unspecified (0.0.0.0/8), broadcast (255.255.255.255).
    - IPv6: loopback (::1), unspecified (::), unique-local ULA (fc00::/7, fd00::/8),
      link-local (fe80::/10), multicast (ff00::/8), documentation (2001:db8::/32).
    - IPv6 transitions & embeddings:
      * IPv4-mapped IPv6 (::ffff:x.x.x.x)
      * 6to4 prefix (2002::/16 embedding IPv4 in bytes 2..6)
      * Teredo prefix (2001:0000::/32 embedding inverted IPv4 in bytes 12..16)

    Args:
        ip_obj: An instantiated IPv4Address or IPv6Address.

    Returns:
        bool: True if the address is restricted / private; False if publicly routable.
    """
    # 1. Base property checks
    if (
        not ip_obj.is_global
        or ip_obj.is_loopback
        or ip_obj.is_private
        or ip_obj.is_link_local
        or ip_obj.is_reserved
        or ip_obj.is_multicast
        or ip_obj.is_unspecified
    ):
        return True

    # 2. IPv6-specific transitions and embeddings
    if isinstance(ip_obj, ipaddress.IPv6Address):
        # IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1)
        if ip_obj.ipv4_mapped:
            return is_ip_restricted(ip_obj.ipv4_mapped)

        # 6to4 encapsulation (2002::/16) - bytes 2..6 contain an IPv4 address
        if ip_obj in ipaddress.IPv6Network("2002::/16"):
            embedded_ipv4 = ipaddress.IPv4Address(ip_obj.packed[2:6])
            if is_ip_restricted(embedded_ipv4):
                return True

        # Teredo tunneling (2001:0000::/32) - last 4 bytes contain XOR'd client IPv4
        if ip_obj in ipaddress.IPv6Network("2001::/32"):
            raw_bytes = bytes(b ^ 0xFF for b in ip_obj.packed[12:16])
            embedded_ipv4 = ipaddress.IPv4Address(raw_bytes)
            if is_ip_restricted(embedded_ipv4):
                return True

    return False


def validate_url_security(url: str) -> str:
    """
    Validates URL scheme, resolves hostname to IP addresses, and ensures the target
    is a publicly routable address to defend against SSRF (Server-Side Request Forgery)
    and unauthorized access to internal resources.

    Security Controls:
    1. Rejects schemes other than http and https.
    2. Resolves DNS via socket.getaddrinfo and inspects every resolved IP address.
    3. Blocks loopback, private RFC-1918, link-local, multicast, reserved, and IPv6 ULA.
    4. Rejects IPv4-mapped and IPv4-embedded IPv6 addresses targeting private networks.

    Args:
        url (str): The raw user-supplied URL.

    Returns:
        str: Validated, stripped URL string.

    Raises:
        ValueError: If scheme is disallowed, hostname is missing, or IP is private/restricted.
    """
    if isinstance(url, bool) or not isinstance(url, str):
        raise TypeError(f"URL must be a string, got {type(url).__name__}.")

    clean_url = url.strip()
    if not clean_url:
        raise ValueError("URL must not be empty or whitespace-only.")

    parsed = urlparse(clean_url)
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsupported URL scheme '{scheme}'. Only HTTP and HTTPS are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"URL '{clean_url}' does not contain a valid hostname.")

    port = parsed.port or (443 if scheme == "https" else 80)

    # Resolve hostname to all associated IP addresses
    try:
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(
            f"Failed to resolve hostname '{hostname}': {exc}"
        ) from exc

    resolved_ips: Set[str] = set()
    for entry in addr_info:
        sockaddr = entry[4]
        ip_str = sockaddr[0]
        resolved_ips.add(ip_str)

    if not resolved_ips:
        raise ValueError(f"No IP addresses resolved for host '{hostname}'.")

    # Inspect every resolved IP address
    for ip_str in resolved_ips:
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise ValueError(f"Invalid resolved IP '{ip_str}': {exc}") from exc

        if is_ip_restricted(ip_obj):
            logger.warning(
                "SSRF attempt blocked: Hostname '%s' resolved to restricted IP %s.",
                hostname,
                ip_obj,
            )
            raise ValueError(
                f"URL host '{hostname}' resolves to restricted internal/private IP address: {ip_obj}. Access denied."
            )

    return clean_url


# ---------------------------------------------------------
# 2. Source Category Classifier
# ---------------------------------------------------------
def classify_source_url(url: str) -> str:
    """
    Classifies a submitted URL into its corresponding EchoTrace source category.

    Categories:
    - "youtube": YouTube watch or share link.
    - "ted": TED Talk page link.
    - "podcast_rss": RSS/XML podcast feed.
    - "direct_media": Direct audio or video link (.mp3, .wav, .m4a, .mp4).
    - "unsupported": Unrecognized or unsupported URL.

    Args:
        url (str): The URL to inspect.

    Returns:
        str: Category identifier string.
    """
    parsed = urlparse(url)
    path_lower = parsed.path.lower()

    if YOUTUBE_REGEX.match(url):
        return "youtube"

    if TED_REGEX.match(url):
        return "ted"

    # Direct audio/video file extension
    for ext in SUPPORTED_EXTENSIONS:
        if path_lower.endswith(ext):
            return "direct_media"

    # Common RSS feed extensions or query patterns
    if (
        path_lower.endswith(".rss")
        or path_lower.endswith(".xml")
        or path_lower.endswith("/feed")
        or "feed=" in parsed.query.lower()
        or path_lower.endswith("/podcast")
    ):
        return "podcast_rss"

    return "unsupported"


# ---------------------------------------------------------
# 3. Magic Bytes & File Signature Detection
# ---------------------------------------------------------
def detect_extension_from_magic_bytes(header_bytes: bytes) -> Optional[str]:
    """
    Inspects initial bytes of a downloaded payload to determine verified media file format.
    Ensures that HTML, executables, archives, and arbitrary binaries are strictly rejected.

    Signatures:
    - MP3 (with ID3): starts with 'ID3' (bytes 0..3)
    - MP3 (frame sync): starts with 0xFF and second byte has sync bits set (0xE0) and Layer III bits (0x02)
    - WAV: starts with 'RIFF' and contains 'WAVE' at offset 8..12
    - MP4 / M4A: contains 'ftyp' at offset 4..8 with corresponding brand identifier
    """
    if len(header_bytes) < 4:
        return None

    # Immediate rejection of known non-media signatures (HTML, XML, executables, archives)
    if (
        header_bytes.startswith(b"<!DO")
        or header_bytes.startswith(b"<!do")
        or header_bytes.startswith(b"<htm")
        or header_bytes.startswith(b"<HTM")
        or header_bytes.startswith(b"<?xm")
        or header_bytes.startswith(b"MZ")        # Windows PE executable
        or header_bytes.startswith(b"\x7fELF")   # Linux ELF executable
        or header_bytes.startswith(b"PK\x03\x04")# ZIP archive / JAR
        or header_bytes.startswith(b"7z\xbc\xaf")# 7-Zip archive
        or header_bytes.startswith(b"Rar!\x1a")  # RAR archive
    ):
        return None

    # MP3 (ID3 tag header)
    if header_bytes[:3] == b"ID3":
        return ".mp3"

    # MP3 (MPEG audio frame sync: 11 bits sync word + Layer III)
    # Byte 0: 0xFF
    # Byte 1: 0b111xxxxx (sync bits) and bits 1..2 describe layer (0b01 = Layer III)
    if (
        header_bytes[0] == 0xFF
        and (header_bytes[1] & 0xE0) == 0xE0
        and ((header_bytes[1] >> 1) & 0x03) == 0x01
    ):
        return ".mp3"

    # WAV (RIFF....WAVE)
    if len(header_bytes) >= 12 and header_bytes[:4] == b"RIFF" and header_bytes[8:12] == b"WAVE":
        return ".wav"

    # MP4 / M4A (ISO Base Media File Format: ....ftyp)
    if len(header_bytes) >= 12 and header_bytes[4:8] == b"ftyp":
        brand = header_bytes[8:12]
        if brand in (b"M4A ", b"m4a "):
            return ".m4a"
        return ".mp4"

    return None


# ---------------------------------------------------------
# 4. Secure Streaming Downloader
# ---------------------------------------------------------
def download_media_from_url(
    url: str,
    recording_id: Optional[str] = None,
    max_bytes: int = MAX_DOWNLOAD_SIZE_BYTES,
) -> Tuple[Path, str]:
    """
    Downloads media from an external URL with comprehensive security controls:
    - Validates target IP against SSRF before connecting.
    - Manually tracks redirects (up to 5) and re-validates each redirect destination.
    - Disables environment proxy inheritance (`trust_env=False`) to prevent proxy bypass.
    - Streams directly to disk without loading full file into memory.
    - Immediately aborts and cleans up if byte count exceeds max_bytes (default: 100 MB).
    - Requires mandatory media signature (magic bytes) validation. Content-Type and URL
      extensions are never trusted without signature proof.
    - Generates a safe server-side UUID filename (never trusts user-provided filename).

    Args:
        url (str): Target media URL.
        recording_id (Optional[str]): Optional custom recording ID (generates UUID if None).
        max_bytes (int): Maximum permitted download size in bytes (default: 100 MB).

    Returns:
        Tuple[Path, str]: (Path to downloaded file on disk, detected file extension).

    Raises:
        ValueError: If security checks fail, redirect limit is exceeded, download exceeds size,
                    or file signature is invalid/unsupported.
        Exception: If download connection or streaming fails.
    """
    current_url = validate_url_security(url)
    rec_id = recording_id.strip() if recording_id else str(uuid.uuid4())
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    client_timeout = httpx.Timeout(
        timeout=READ_TIMEOUT_SECONDS,
        connect=CONNECT_TIMEOUT_SECONDS,
    )

    redirect_count = 0
    temp_target_path: Optional[Path] = None

    try:
        # trust_env=False prevents proxy environment variables (HTTP_PROXY, HTTPS_PROXY)
        # from bypassing intended IP validation.
        with httpx.Client(
            follow_redirects=False,
            timeout=client_timeout,
            trust_env=False,
        ) as client:

            # Handle redirects manually to re-verify SSRF on each hop
            while redirect_count <= MAX_REDIRECTS:
                response = client.get(current_url, headers={"User-Agent": "EchoTrace/1.0"})

                if response.is_redirect:
                    redirect_count += 1
                    if redirect_count > MAX_REDIRECTS:
                        raise ValueError(
                            f"Exceeded maximum redirect limit of {MAX_REDIRECTS}."
                        )

                    location = response.headers.get("Location")
                    if not location:
                        raise ValueError("Received redirect without Location header.")

                    next_url = urljoin(current_url, location)
                    current_url = validate_url_security(next_url)
                    logger.info("Redirect %d: -> %s", redirect_count, current_url)
                    continue

                # Final destination reached
                if response.status_code != 200:
                    raise ValueError(
                        f"Failed to fetch media from URL. HTTP status: {response.status_code}."
                    )

                # Check Content-Length if present
                content_length_header = response.headers.get("Content-Length")
                if content_length_header:
                    try:
                        content_length = int(content_length_header)
                        if content_length > max_bytes:
                            size_mb = content_length / (1024 * 1024)
                            limit_mb = max_bytes / (1024 * 1024)
                            raise ValueError(
                                f"Source media size ({size_mb:.2f} MB) exceeds maximum allowed limit of {limit_mb:.0f} MB."
                            )
                    except ValueError as parse_err:
                        if "exceeds maximum allowed limit" in str(parse_err):
                            raise

                # Determine candidate extension from URL path or Content-Type
                content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                detected_ext = CONTENT_TYPE_MAP.get(content_type)

                if not detected_ext:
                    parsed_path = urlparse(current_url).path.lower()
                    for ext in SUPPORTED_EXTENSIONS:
                        if parsed_path.endswith(ext):
                            detected_ext = ext
                            break

                # Stream response body to disk
                temp_filename = f"tmp_{rec_id}.bin"
                temp_target_path = UPLOAD_DIR / temp_filename

                total_downloaded = 0
                header_sample = bytearray()

                with open(temp_target_path, "wb") as f_out:
                    try:
                        # Stream chunks from response
                        for chunk in response.iter_bytes(chunk_size=STREAM_CHUNK_SIZE_BYTES):
                            if not chunk:
                                continue

                            if len(header_sample) < 64:
                                header_sample.extend(chunk[: 64 - len(header_sample)])

                            total_downloaded += len(chunk)
                            if total_downloaded > max_bytes:
                                size_mb = total_downloaded / (1024 * 1024)
                                limit_mb = max_bytes / (1024 * 1024)
                                raise ValueError(
                                    f"Download exceeded maximum allowed limit of {limit_mb:.0f} MB (read {size_mb:.2f} MB)."
                                )

                            f_out.write(chunk)
                    finally:
                        response.close()

                if total_downloaded == 0:
                    raise ValueError("Downloaded file is empty (0 bytes).")

                # Mandatory file signature (magic bytes) verification
                magic_ext = detect_extension_from_magic_bytes(bytes(header_sample))
                if not magic_ext or magic_ext not in SUPPORTED_EXTENSIONS:
                    allowed_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                    raise ValueError(
                        f"Downloaded content signature is invalid, unrecognized, or untrusted. "
                        f"Allowed media formats are: {allowed_list} (received Content-Type: '{content_type}')."
                    )

                # Ensure claimed extension is consistent with actual signature
                if detected_ext and detected_ext != magic_ext:
                    raise ValueError(
                        f"Mismatched media format: URL or Content-Type indicated '{detected_ext}', "
                        f"but verified file signature is '{magic_ext}'."
                    )

                final_ext = magic_ext

                # Rename to permanent server-side safe UUID filename
                final_filename = f"{rec_id}{final_ext}"
                final_path = UPLOAD_DIR / final_filename

                if temp_target_path.exists():
                    temp_target_path.replace(final_path)

                logger.info(
                    "Stream download complete: '%s' (%.2f MB, ext=%s).",
                    final_filename,
                    total_downloaded / (1024 * 1024),
                    final_ext,
                )

                return final_path, final_ext

    except Exception:
        # Clean up any partial or temporary files left behind on failure
        if temp_target_path and temp_target_path.exists():
            try:
                temp_target_path.unlink()
            except Exception:
                pass
        raise


# ---------------------------------------------------------
# 5. Incremental Stream Fetcher for Podcast RSS Feeds
# ---------------------------------------------------------
def fetch_podcast_rss_text(
    url: str,
    max_bytes: int = MAX_FEED_SIZE_BYTES,
) -> str:
    """
    Safely retrieves a podcast RSS feed via HTTP streaming with strict SSRF controls:
    - Disables automatic redirects; manually re-validates each redirect target against SSRF.
    - Enforces MAX_REDIRECTS (5).
    - Disables environment proxy inheritance (`trust_env=False`).
    - Streams response body incrementally to enforce MAX_FEED_SIZE_BYTES (5 MB) without
      loading unbounded payloads into memory.
    - Closes response streams and cleans up properly on any error or limit breach.

    Args:
        url (str): Target RSS feed URL.
        max_bytes (int): Maximum allowed feed size in bytes (default: 5 MB).

    Returns:
        str: Decoded XML text of the RSS feed.

    Raises:
        ValueError: If security checks fail, redirect limit is exceeded, or feed size > max_bytes.
    """
    current_url = validate_url_security(url)
    redirect_count = 0

    client_timeout = httpx.Timeout(
        timeout=READ_TIMEOUT_SECONDS,
        connect=CONNECT_TIMEOUT_SECONDS,
    )

    with httpx.Client(
        follow_redirects=False,
        timeout=client_timeout,
        trust_env=False,
    ) as client:
        while redirect_count <= MAX_REDIRECTS:
            response = client.get(current_url, headers={"User-Agent": "EchoTrace/1.0"})

            if response.is_redirect:
                redirect_count += 1
                if redirect_count > MAX_REDIRECTS:
                    raise ValueError(
                        f"Exceeded maximum redirect limit of {MAX_REDIRECTS}."
                    )

                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Received redirect without Location header.")

                next_url = urljoin(current_url, location)
                current_url = validate_url_security(next_url)
                logger.info("RSS Redirect %d: -> %s", redirect_count, current_url)
                continue

            # Final destination reached
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to fetch podcast RSS feed. HTTP status: {response.status_code}."
                )

            # Check Content-Length header if present
            content_length_header = response.headers.get("Content-Length")
            if content_length_header:
                try:
                    cl = int(content_length_header)
                    if cl > max_bytes:
                        limit_mb = max_bytes / (1024 * 1024)
                        raise ValueError(
                            f"Podcast RSS feed exceeds maximum allowed size of {limit_mb:.0f} MB."
                        )
                except ValueError as parse_err:
                    if "exceeds maximum allowed size" in str(parse_err):
                        raise

            # Stream body incrementally to enforce byte limit safely
            total_bytes = 0
            chunks: List[bytes] = []

            try:
                for chunk in response.iter_bytes(chunk_size=STREAM_CHUNK_SIZE_BYTES):
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        limit_mb = max_bytes / (1024 * 1024)
                        raise ValueError(
                            f"Podcast RSS feed exceeds maximum allowed size of {limit_mb:.0f} MB."
                        )
                    chunks.append(chunk)
            finally:
                response.close()

            if total_bytes == 0:
                raise ValueError("Podcast RSS feed is empty (0 bytes).")

            raw_bytes = b"".join(chunks)
            encoding = response.encoding or "utf-8"
            try:
                return raw_bytes.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                return raw_bytes.decode("utf-8", errors="replace")

    raise ValueError(f"Exceeded maximum redirect limit of {MAX_REDIRECTS}.")


# ---------------------------------------------------------
# 6. Podcast RSS Feed Parser
# ---------------------------------------------------------
def parse_podcast_rss(
    rss_text: str,
    requested_episode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parses a podcast RSS 2.0 feed using standard library xml.etree.ElementTree.
    Extracts the audio enclosure URL for the requested episode (or the latest episode
    if requested_episode is None).

    Args:
        rss_text (str): Raw XML text of the RSS feed.
        requested_episode (Optional[str]): Optional episode title or GUID substring to match.

    Returns:
        Dict[str, Any]: Episode metadata containing:
            - title: str (Episode title)
            - enclosure_url: str (Direct audio download link)
            - enclosure_type: str (MIME type)
            - duration: Optional[str]

    Raises:
        ValueError: If XML is malformed, has no items, or lacks a valid audio enclosure.
    """
    try:
        root = ET.fromstring(rss_text)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse podcast RSS feed XML: {exc}") from exc

    channel = root.find("channel")
    if channel is None:
        raise ValueError("Invalid podcast RSS: missing <channel> root element.")

    items = channel.findall("item")
    if not items:
        raise ValueError("Podcast feed contains no <item> episodes.")

    target_item: Optional[ET.Element] = None

    if requested_episode:
        req_clean = requested_episode.strip().lower()
        for item in items:
            title_elem = item.find("title")
            guid_elem = item.find("guid")
            title_text = title_elem.text.lower() if title_elem is not None and title_elem.text else ""
            guid_text = guid_elem.text.lower() if guid_elem is not None and guid_elem.text else ""

            if req_clean in title_text or req_clean in guid_text:
                target_item = item
                break

        if target_item is None:
            raise ValueError(
                f"Requested episode matching '{requested_episode}' was not found in podcast feed."
            )
    else:
        # Default to latest episode (first item in RSS channel)
        target_item = items[0]

    title_elem = target_item.find("title")
    title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Untitled Episode"

    enclosure = target_item.find("enclosure")
    if enclosure is None:
        raise ValueError(
            f"Podcast episode '{title}' does not contain an audio <enclosure> link."
        )

    enclosure_url = enclosure.get("url")
    if not enclosure_url or not enclosure_url.strip():
        raise ValueError(
            f"Podcast episode '{title}' has an invalid or empty enclosure URL."
        )

    enclosure_type = enclosure.get("type", "audio/mpeg").strip().lower()

    return {
        "title": title,
        "enclosure_url": enclosure_url.strip(),
        "enclosure_type": enclosure_type,
    }


# ---------------------------------------------------------
# 7. Unified URL Ingestion Orchestrator
# ---------------------------------------------------------
def ingest_from_url(
    url: str,
    recording_id: Optional[str] = None,
    authorized_transcript: Optional[Dict[str, Any]] = None,
    requested_episode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry point for URL-based recording ingestion:
    1. Validates URL security (blocks private IPs, loops, SSRF).
    2. Classifies URL into its source category (YouTube, TED, Podcast RSS, Direct Media).
    3. Handles platform permissions and policies:
       - YouTube: Only processed via authorized transcript; rejects unauthorized scraping/download.
       - TED: Protected by licensing terms; only processed via authorized transcript.
       - Podcast RSS: Extracts requested episode audio enclosure safely.
       - Direct Media: Downloads audio/video file securely via streaming with magic byte validation.
    4. Passes the local media file or authorized transcript to the common ingest_recording() pipeline.
    5. Cleans up any downloaded files in data/uploads/ if downstream ingestion fails.
    6. Returns a structured result containing source metadata, recording_id, and chunk count.

    Args:
        url (str): Source URL to ingest.
        recording_id (Optional[str]): Optional custom recording ID.
        authorized_transcript (Optional[Dict[str, Any]]): Pre-existing authorized transcript.
        requested_episode (Optional[str]): Optional podcast episode title filter.

    Returns:
        Dict[str, Any]: Ingestion summary outcome.

    Raises:
        ValueError: If URL is invalid, source is unsupported/unauthorized, or pipeline fails.
    """
    # 1. Validate URL security
    safe_url = validate_url_security(url)
    rec_id = recording_id.strip() if recording_id else str(uuid.uuid4())

    # 2. Classify source category
    category = classify_source_url(safe_url)
    logger.info("Classified URL '%s' as category: %s", safe_url, category)

    # ---------------------------------------------------------
    # Case A: YouTube URLs
    # ---------------------------------------------------------
    if category == "youtube":
        match = YOUTUBE_REGEX.match(safe_url)
        video_id = match.group(1) if match else "unknown"

        if authorized_transcript is not None:
            logger.info(
                "Ingesting YouTube recording '%s' (video_id=%s) using provided authorized transcript...",
                rec_id,
                video_id,
            )
            result = ingest_recording(
                recording_id=rec_id,
                existing_transcript=authorized_transcript,
            )
            result["source_category"] = "youtube"
            result["source_url"] = safe_url
            result["video_id"] = video_id
            return result
        else:
            raise ValueError(
                f"Direct downloading or unauthorized scraping of YouTube video '{video_id}' is restricted. "
                f"EchoTrace supports YouTube ingestion only through authorized timestamped transcripts. "
                f"Please provide an authorized transcript to ingest this recording."
            )

    # ---------------------------------------------------------
    # Case B: TED Talk URLs
    # ---------------------------------------------------------
    if category == "ted":
        match = TED_REGEX.match(safe_url)
        talk_slug = match.group(1) if match else "unknown"

        if authorized_transcript is not None:
            logger.info(
                "Ingesting TED Talk recording '%s' (talk=%s) using authorized transcript...",
                rec_id,
                talk_slug,
            )
            result = ingest_recording(
                recording_id=rec_id,
                existing_transcript=authorized_transcript,
            )
            result["source_category"] = "ted"
            result["source_url"] = safe_url
            result["talk_slug"] = talk_slug
            return result
        else:
            raise ValueError(
                f"TED Talk content ('{talk_slug}') is protected by terms that restrict automated downloading "
                f"and AI ingestion without written authorization. To index this talk, please provide an "
                f"authorized transcript under appropriate licensing."
            )

    # ---------------------------------------------------------
    # Case C: Podcast RSS Feed URLs
    # ---------------------------------------------------------
    if category == "podcast_rss":
        logger.info("Fetching podcast RSS feed from: %s", safe_url)
        # Fetch RSS XML via secure streaming with manual redirect re-validation
        feed_xml = fetch_podcast_rss_text(safe_url)

        episode_meta = parse_podcast_rss(feed_xml, requested_episode=requested_episode)
        logger.info(
            "Selected episode '%s' with enclosure URL: %s",
            episode_meta["title"],
            episode_meta["enclosure_url"],
        )

        # Download the episode audio enclosure via streaming
        downloaded_audio_path, _ = download_media_from_url(
            url=episode_meta["enclosure_url"],
            recording_id=rec_id,
        )

        # Pass downloaded audio to existing ingestion pipeline
        try:
            result = ingest_recording(
                recording_id=rec_id,
                media_path=downloaded_audio_path,
                existing_transcript=authorized_transcript,
            )
            result["source_category"] = "podcast"
            result["source_url"] = safe_url
            result["episode_title"] = episode_meta["title"]
            return result
        except Exception:
            # Clean up downloaded media file if pipeline processing fails
            if downloaded_audio_path.exists():
                try:
                    downloaded_audio_path.unlink()
                except Exception:
                    pass
            raise

    # ---------------------------------------------------------
    # Case D: Direct Media URLs (.mp3, .wav, .m4a, .mp4)
    # ---------------------------------------------------------
    if category == "direct_media":
        logger.info("Downloading direct media from: %s", safe_url)
        downloaded_media_path, _ = download_media_from_url(
            url=safe_url,
            recording_id=rec_id,
        )

        # Pass downloaded file to existing ingestion pipeline
        try:
            result = ingest_recording(
                recording_id=rec_id,
                media_path=downloaded_media_path,
                existing_transcript=authorized_transcript,
            )
            result["source_category"] = "direct_media"
            result["source_url"] = safe_url
            return result
        except Exception:
            # Clean up downloaded media file if pipeline processing fails
            if downloaded_media_path.exists():
                try:
                    downloaded_media_path.unlink()
                except Exception:
                    pass
            raise

    # ---------------------------------------------------------
    # Case E: Unsupported Source
    # ---------------------------------------------------------
    raise ValueError(
        f"URL '{safe_url}' is not a recognized direct media file (.mp4, .mp3, .wav, .m4a), "
        f"podcast RSS feed, YouTube video, or TED Talk URL."
    )
