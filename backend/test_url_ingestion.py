import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure 'backend' directory is in Python search path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import httpx

from app.services import url_ingestion_service
from app.services.url_ingestion_service import (
    MAX_DOWNLOAD_SIZE_BYTES,
    MAX_FEED_SIZE_BYTES,
    MAX_REDIRECTS,
    classify_source_url,
    detect_extension_from_magic_bytes,
    download_media_from_url,
    fetch_podcast_rss_text,
    ingest_from_url,
    is_ip_restricted,
    parse_podcast_rss,
    validate_url_security,
)


class TestURLSecurityAndSSRF(unittest.TestCase):
    """Verifies SSRF defense, scheme validation, and IP blocking."""

    def test_unsupported_scheme_rejected(self):
        """Non-HTTP/HTTPS schemes (file, ftp, gopher) are rejected."""
        unsupported = [
            "file:///etc/passwd",
            "ftp://example.com/audio.mp3",
            "gopher://example.com",
            "data:text/plain;base64,SGVsbG8=",
        ]
        for url in unsupported:
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as ctx:
                    validate_url_security(url)
                self.assertIn("Unsupported URL scheme", str(ctx.exception))

    def test_empty_or_non_string_url_rejected(self):
        """Invalid types and empty URLs raise TypeError or ValueError."""
        with self.assertRaises(TypeError):
            validate_url_security(123)
        with self.assertRaises(TypeError):
            validate_url_security(True)
        with self.assertRaises(ValueError):
            validate_url_security("   ")

    @patch("socket.getaddrinfo")
    def test_private_and_loopback_ips_blocked(self, mock_getaddrinfo):
        """Resolving to loopback, private, or link-local IPs raises ValueError."""
        blocked_ips = [
            "127.0.0.1",
            "127.0.0.2",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",  # AWS/Cloud metadata link-local
            "::1",
            "::ffff:127.0.0.1",
        ]
        for ip in blocked_ips:
            with self.subTest(ip=ip):
                mock_getaddrinfo.return_value = [
                    (2, 1, 6, "", (ip, 443))
                ]
                with self.assertRaises(ValueError) as ctx:
                    validate_url_security("https://internal-service.local/media.mp3")
                self.assertIn("Access denied", str(ctx.exception))

    @patch("socket.getaddrinfo")
    def test_ipv6_non_public_addresses_blocked(self, mock_getaddrinfo):
        """IPv6 ULA, link-local, loopback, multicast, documentation, and 6to4 private embeddings are blocked."""
        restricted_ipv6 = [
            "::1",                      # Loopback
            "::",                       # Unspecified
            "fc00::1",                  # Unique Local Address (ULA)
            "fd00::1",                  # Unique Local Address (ULA)
            "fe80::1",                  # Link-local
            "ff02::1",                  # Multicast
            "2001:db8::1",              # Documentation / reserved
            "::ffff:127.0.0.1",         # IPv4-mapped loopback
            "::ffff:10.0.0.1",          # IPv4-mapped private
            "::ffff:169.254.169.254",   # IPv4-mapped metadata
            "2002:7f00:0001::",         # 6to4 embedding 127.0.0.1
            "2002:0a00:0001::",         # 6to4 embedding 10.0.0.1
            "2002:a9fe:a9fe::",         # 6to4 embedding 169.254.169.254
        ]
        for ip in restricted_ipv6:
            with self.subTest(ip=ip):
                mock_getaddrinfo.return_value = [
                    (23, 1, 6, "", (ip, 443, 0, 0))
                ]
                with self.assertRaises(ValueError) as ctx:
                    validate_url_security("https://ipv6-target.local/feed.xml")
                self.assertIn("Access denied", str(ctx.exception))

    @patch("socket.getaddrinfo")
    def test_public_ip_allowed(self, mock_getaddrinfo):
        """Publicly routable IP addresses pass security validation."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))  # example.com public IP
        ]
        url = "https://example.com/audio.mp3"
        result = validate_url_security(url)
        self.assertEqual(result, url)


class TestURLClassification(unittest.TestCase):
    """Verifies URL source category classification."""

    def test_classify_youtube_urls(self):
        """YouTube URLs are correctly identified."""
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(classify_source_url(url), "youtube")

    def test_classify_ted_urls(self):
        """TED Talk URLs are correctly identified."""
        urls = [
            "https://www.ted.com/talks/jane_goodall_what_makes_us_human",
            "https://ted.com/talks/ken_robinson_says_schools_kill_creativity",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(classify_source_url(url), "ted")

    def test_classify_direct_media_urls(self):
        """Direct media file URLs are correctly identified."""
        urls = [
            ("https://example.com/recording.mp3", "direct_media"),
            ("https://example.com/audio.wav", "direct_media"),
            ("https://example.com/speech.m4a", "direct_media"),
            ("https://example.com/video.mp4", "direct_media"),
        ]
        for url, expected in urls:
            with self.subTest(url=url):
                self.assertEqual(classify_source_url(url), expected)

    def test_classify_podcast_rss_urls(self):
        """Podcast RSS/XML feeds are correctly identified."""
        urls = [
            "https://feeds.simplecast.com/54nAGcIl.rss",
            "https://example.com/podcast.xml",
            "https://example.com/channel/feed",
            "https://example.com/shows/podcast",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(classify_source_url(url), "podcast_rss")

    def test_classify_unsupported_urls(self):
        """Unrecognized web pages are classified as unsupported."""
        self.assertEqual(classify_source_url("https://example.com/article"), "unsupported")


class TestMagicBytesDetection(unittest.TestCase):
    """Verifies file signature/magic bytes recognition and strict non-media rejection."""

    def test_wav_magic_bytes(self):
        header = b"RIFF\x00\x00\x00\x00WAVEfmt "
        self.assertEqual(detect_extension_from_magic_bytes(header), ".wav")

    def test_mp3_id3_magic_bytes(self):
        header = b"ID3\x04\x00\x00\x00\x00\x00\x00"
        self.assertEqual(detect_extension_from_magic_bytes(header), ".mp3")

    def test_mp3_sync_magic_bytes(self):
        """MP3 with frame sync (without ID3 tag) is recognized for MPEG Layer III."""
        header = b"\xff\xfb\x90\x64\x00\x00\x00\x00\x00\x00\x00\x00"
        self.assertEqual(detect_extension_from_magic_bytes(header), ".mp3")

        header_layer2 = b"\xff\xf3\x90\x64\x00\x00\x00\x00\x00\x00\x00\x00"
        self.assertEqual(detect_extension_from_magic_bytes(header_layer2), ".mp3")

    def test_mp4_magic_bytes(self):
        header = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00"
        self.assertEqual(detect_extension_from_magic_bytes(header), ".mp4")

    def test_m4a_magic_bytes(self):
        header = b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00"
        self.assertEqual(detect_extension_from_magic_bytes(header), ".m4a")

    def test_non_media_signatures_rejected(self):
        """HTML, XML, executables, and archives return None from magic bytes detection."""
        invalid_headers = [
            b"<!DOCTYPE html><html><body>Error</body></html>",
            b"<!doctype html>",
            b"<html><head><title>Test</title></head></html>",
            b"<?xml version='1.0' encoding='UTF-8'?>",
            b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff",  # Windows PE
            b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00",     # Linux ELF
            b"PK\x03\x04\x14\x00\x00\x00\x08\x00",                   # ZIP
            b"7z\xbc\xaf\x27\x1c\x00\x04",                           # 7-Zip
            b"Rar!\x1a\x07\x00",                                     # RAR
            b"Arbitrary text content that is not media",
        ]
        for h in invalid_headers:
            with self.subTest(header=h[:10]):
                self.assertIsNone(detect_extension_from_magic_bytes(h))


class TestStreamingDownloaderAndLimits(unittest.TestCase):
    """Verifies streaming download, size limits, redirect boundaries, and file cleanup."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.patch_upload_dir = patch.object(
            url_ingestion_service,
            "UPLOAD_DIR",
            Path(self.temp_dir.name),
        )
        self.patch_upload_dir.start()

    def tearDown(self):
        self.patch_upload_dir.stop()
        self.temp_dir.cleanup()

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("httpx.Client.get")
    def test_successful_stream_download_with_id3(self, mock_get, mock_val_url):
        """Direct media stream download succeeds and saves MP3 with ID3 header."""
        mock_val_url.side_effect = lambda u: u

        mock_resp = MagicMock()
        mock_resp.is_redirect = False
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "audio/mpeg", "Content-Length": "100"}
        mock_resp.iter_bytes.return_value = [b"ID3\x04" + b"A" * 96]
        mock_get.return_value = mock_resp

        path, ext = download_media_from_url(
            url="https://example.com/audio.mp3",
            recording_id="rec-download-1",
        )

        self.assertTrue(path.is_file())
        self.assertEqual(ext, ".mp3")
        self.assertEqual(path.name, "rec-download-1.mp3")

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("httpx.Client.get")
    def test_successful_stream_download_without_id3(self, mock_get, mock_val_url):
        """Direct media stream download succeeds and saves MP3 with raw sync frame."""
        mock_val_url.side_effect = lambda u: u

        mock_resp = MagicMock()
        mock_resp.is_redirect = False
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "audio/mpeg", "Content-Length": "100"}
        mock_resp.iter_bytes.return_value = [b"\xff\xfb\x90\x64" + b"A" * 96]
        mock_get.return_value = mock_resp

        path, ext = download_media_from_url(
            url="https://example.com/audio.mp3",
            recording_id="rec-download-raw-mp3",
        )

        self.assertTrue(path.is_file())
        self.assertEqual(ext, ".mp3")
        self.assertEqual(path.name, "rec-download-raw-mp3.mp3")

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("httpx.Client.get")
    def test_html_returned_with_audio_content_type_rejected(self, mock_get, mock_val_url):
        """HTML body returned with Content-Type: audio/mpeg is strictly rejected by magic byte check."""
        mock_val_url.side_effect = lambda u: u

        mock_resp = MagicMock()
        mock_resp.is_redirect = False
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "audio/mpeg", "Content-Length": "150"}
        mock_resp.iter_bytes.return_value = [b"<!DOCTYPE html><html><body>Error 404 Not Found</body></html>"]
        mock_get.return_value = mock_resp

        with self.assertRaises(ValueError) as ctx:
            download_media_from_url(
                url="https://example.com/spoofed_audio.mp3",
                recording_id="rec-spoof-html",
            )

        self.assertIn("signature is invalid, unrecognized, or untrusted", str(ctx.exception))
        # Ensure temporary file was removed
        files = list(Path(self.temp_dir.name).glob("*"))
        self.assertEqual(len(files), 0)

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("httpx.Client.get")
    def test_executable_returned_with_audio_content_type_rejected(self, mock_get, mock_val_url):
        """Executable returned with Content-Type: audio/mpeg is strictly rejected by magic byte check."""
        mock_val_url.side_effect = lambda u: u

        mock_resp = MagicMock()
        mock_resp.is_redirect = False
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "audio/mpeg", "Content-Length": "200"}
        mock_resp.iter_bytes.return_value = [b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff" + b"X" * 100]
        mock_get.return_value = mock_resp

        with self.assertRaises(ValueError) as ctx:
            download_media_from_url(
                url="https://example.com/trojan.mp3",
                recording_id="rec-spoof-exe",
            )

        self.assertIn("signature is invalid, unrecognized, or untrusted", str(ctx.exception))
        # Ensure temporary file was removed
        files = list(Path(self.temp_dir.name).glob("*"))
        self.assertEqual(len(files), 0)

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("httpx.Client.get")
    def test_mismatched_extension_and_signature_rejected(self, mock_get, mock_val_url):
        """MP4 video served under .wav URL/MIME is rejected as a mismatch."""
        mock_val_url.side_effect = lambda u: u

        mock_resp = MagicMock()
        mock_resp.is_redirect = False
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "audio/wav"}
        mock_resp.iter_bytes.return_value = [b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00" + b"Z" * 50]
        mock_get.return_value = mock_resp

        with self.assertRaises(ValueError) as ctx:
            download_media_from_url(
                url="https://example.com/wrong.wav",
                recording_id="rec-mismatch",
            )

        self.assertIn("Mismatched media format", str(ctx.exception))

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("httpx.Client.get")
    def test_redirect_limit_exceeded_raises_error(self, mock_get, mock_val_url):
        """Redirects exceeding limit of 5 raise ValueError."""
        mock_val_url.side_effect = lambda u: u

        mock_resp = MagicMock()
        mock_resp.is_redirect = True
        mock_resp.headers = {"Location": "https://example.com/next"}
        mock_get.return_value = mock_resp

        with self.assertRaises(ValueError) as ctx:
            download_media_from_url("https://example.com/start")

        self.assertIn("Exceeded maximum redirect limit of 5", str(ctx.exception))

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("httpx.Client.get")
    def test_content_length_exceeding_100mb_aborts_immediately(self, mock_get, mock_val_url):
        """Content-Length exceeding 100 MB aborts before streaming."""
        mock_val_url.side_effect = lambda u: u

        mock_resp = MagicMock()
        mock_resp.is_redirect = False
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Length": str(105 * 1024 * 1024)}  # 105 MB
        mock_get.return_value = mock_resp

        with self.assertRaises(ValueError) as ctx:
            download_media_from_url("https://example.com/large.mp4")

        self.assertIn("exceeds maximum allowed limit of 100 MB", str(ctx.exception))

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("httpx.Client.get")
    def test_stream_bytes_exceeding_limit_aborts_and_cleans_up(self, mock_get, mock_val_url):
        """Streaming download exceeding 100 MB aborts and cleans up partial file."""
        mock_val_url.side_effect = lambda u: u

        mock_resp = MagicMock()
        mock_resp.is_redirect = False
        mock_resp.status_code = 200
        mock_resp.headers = {}  # No Content-Length
        chunk = b"X" * (10 * 1024 * 1024)  # 10 MB per chunk
        mock_resp.iter_bytes.return_value = [chunk] * 11  # 110 MB total
        mock_get.return_value = mock_resp

        with self.assertRaises(ValueError) as ctx:
            download_media_from_url(
                url="https://example.com/huge.mp4",
                recording_id="rec-huge",
            )

        self.assertIn("Download exceeded maximum allowed limit of 100 MB", str(ctx.exception))

        # Ensure no lingering files exist in upload directory
        files = list(Path(self.temp_dir.name).glob("*"))
        self.assertEqual(len(files), 0)


class TestPodcastRSSFetcherAndSecurity(unittest.TestCase):
    """Verifies RSS streaming, redirect validation, and SSRF prevention."""

    SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Tech Insights Podcast</title>
        <item>
          <title>Episode 42: The Future of AI</title>
          <guid>ep-42</guid>
          <enclosure url="https://cdn.example.com/ep42.mp3" length="123456" type="audio/mpeg" />
        </item>
      </channel>
    </rss>
    """

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("httpx.Client.get")
    def test_public_rss_redirect_to_loopback_blocked(self, mock_get, mock_val_url):
        """Public RSS URL redirecting to 127.0.0.1 is blocked on redirect hop."""
        # Initial URL passes validation
        mock_val_url.side_effect = [
            "https://public-podcast.com/feed.xml",
            ValueError("URL host '127.0.0.1' resolves to restricted internal/private IP address: 127.0.0.1. Access denied."),
        ]

        mock_redirect = MagicMock()
        mock_redirect.is_redirect = True
        mock_redirect.headers = {"Location": "http://127.0.0.1/admin/secret-feed.xml"}
        mock_get.return_value = mock_redirect

        with self.assertRaises(ValueError) as ctx:
            fetch_podcast_rss_text("https://public-podcast.com/feed.xml")

        self.assertIn("restricted internal/private IP address: 127.0.0.1", str(ctx.exception))

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("httpx.Client.get")
    def test_public_rss_redirect_to_cloud_metadata_blocked(self, mock_get, mock_val_url):
        """Public RSS URL redirecting to 169.254.169.254 is blocked on redirect hop."""
        mock_val_url.side_effect = [
            "https://public-podcast.com/feed.xml",
            ValueError("URL host '169.254.169.254' resolves to restricted internal/private IP address: 169.254.169.254. Access denied."),
        ]

        mock_redirect = MagicMock()
        mock_redirect.is_redirect = True
        mock_redirect.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}
        mock_get.return_value = mock_redirect

        with self.assertRaises(ValueError) as ctx:
            fetch_podcast_rss_text("https://public-podcast.com/feed.xml")

        self.assertIn("restricted internal/private IP address: 169.254.169.254", str(ctx.exception))

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("httpx.Client.get")
    def test_rss_redirect_chain_exceeding_limit_rejected(self, mock_get, mock_val_url):
        """RSS feed redirect chain exceeding 5 hops is rejected."""
        mock_val_url.side_effect = lambda u: u

        mock_redirect = MagicMock()
        mock_redirect.is_redirect = True
        mock_redirect.headers = {"Location": "https://public-podcast.com/feed-next.xml"}
        mock_get.return_value = mock_redirect

        with self.assertRaises(ValueError) as ctx:
            fetch_podcast_rss_text("https://public-podcast.com/feed.xml")

        self.assertIn("Exceeded maximum redirect limit of 5", str(ctx.exception))

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("httpx.Client.get")
    def test_rss_response_exceeding_5mb_aborts(self, mock_get, mock_val_url):
        """Streaming RSS feed that exceeds 5 MB aborts immediately."""
        mock_val_url.side_effect = lambda u: u

        mock_resp = MagicMock()
        mock_resp.is_redirect = False
        mock_resp.status_code = 200
        mock_resp.headers = {}
        # Stream 6 MB in chunks
        chunk = b"A" * (1024 * 1024)
        mock_resp.iter_bytes.return_value = [chunk] * 6
        mock_get.return_value = mock_resp

        with self.assertRaises(ValueError) as ctx:
            fetch_podcast_rss_text("https://example.com/huge-feed.xml")

        self.assertIn("Podcast RSS feed exceeds maximum allowed size of 5 MB", str(ctx.exception))
        mock_resp.close.assert_called_once()

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("httpx.Client.get")
    def test_interrupted_rss_download_cleanup(self, mock_get, mock_val_url):
        """Interrupted connection during RSS streaming triggers cleanup and closes response."""
        mock_val_url.side_effect = lambda u: u

        def failing_iter(**kwargs):
            yield b"<rss><channel>"
            raise httpx.ReadTimeout("Connection dropped")

        mock_resp = MagicMock()
        mock_resp.is_redirect = False
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.iter_bytes.side_effect = failing_iter
        mock_get.return_value = mock_resp

        with self.assertRaises(httpx.ReadTimeout):
            fetch_podcast_rss_text("https://example.com/interrupted.xml")

        mock_resp.close.assert_called_once()

    def test_parse_latest_episode_by_default(self):
        """Default selection extracts the first/latest episode enclosure."""
        meta = parse_podcast_rss(self.SAMPLE_RSS)
        self.assertEqual(meta["title"], "Episode 42: The Future of AI")
        self.assertEqual(meta["enclosure_url"], "https://cdn.example.com/ep42.mp3")
        self.assertEqual(meta["enclosure_type"], "audio/mpeg")

    def test_parse_requested_episode_by_title_substring(self):
        """Requested episode title filters and selects the matching episode."""
        sample = """<rss version="2.0"><channel>
          <item><title>Ep 1</title><enclosure url="https://e.com/1.mp3" /></item>
          <item><title>Special Episode</title><enclosure url="https://e.com/special.mp3" /></item>
        </channel></rss>"""
        meta = parse_podcast_rss(sample, requested_episode="special")
        self.assertEqual(meta["title"], "Special Episode")
        self.assertEqual(meta["enclosure_url"], "https://e.com/special.mp3")

    def test_missing_enclosure_raises_value_error(self):
        """Episode lacking an audio enclosure raises ValueError."""
        bad_rss = """<rss version="2.0"><channel><item><title>Text Only</title></item></channel></rss>"""
        with self.assertRaises(ValueError) as ctx:
            parse_podcast_rss(bad_rss)
        self.assertIn("does not contain an audio <enclosure>", str(ctx.exception))


class TestSecurityArchitectureAndTrustEnv(unittest.TestCase):
    """Verifies client security settings (trust_env) and DNS rebinding architecture."""

    @patch("httpx.Client")
    def test_proxy_environment_variables_not_trusted(self, mock_client_cls):
        """httpx.Client is instantiated with trust_env=False to prevent proxy bypass."""
        mock_client_cls.return_value.__enter__.return_value.get.return_value = MagicMock(
            is_redirect=False, status_code=200, headers={}, iter_bytes=lambda **kw: [b"ID3\x04" + b"X" * 60]
        )

        with patch("app.services.url_ingestion_service.validate_url_security", return_value="https://example.com/audio.mp3"):
            try:
                download_media_from_url("https://example.com/audio.mp3", recording_id="rec-env")
            except Exception:
                pass

        mock_client_cls.assert_called()
        _, kwargs = mock_client_cls.call_args
        self.assertFalse(kwargs.get("trust_env"), "trust_env must be explicitly False")

    @patch("socket.getaddrinfo")
    def test_dns_rebinding_simulation_detection(self, mock_getaddrinfo):
        """
        Simulates a DNS rebinding condition: DNS resolution returns a public IP during
        the initial check, but subsequent resolutions return a private loopback IP.
        Proves that re-validation at any subsequent step catches the re-bound IP.
        """
        # Call 1: returns public IP (93.184.216.34)
        # Call 2: returns private loopback (127.0.0.1) due to TTL expiry / DNS rebinding
        mock_getaddrinfo.side_effect = [
            [(2, 1, 6, "", ("93.184.216.34", 443))],
            [(2, 1, 6, "", ("127.0.0.1", 443))],
        ]

        url = "https://rebound-host.com/media.mp3"

        # Step 1: Initial validation passes
        validated_1 = validate_url_security(url)
        self.assertEqual(validated_1, url)

        # Step 2: Second check (e.g. before hop or secondary connection) catches rebinding
        with self.assertRaises(ValueError) as ctx:
            validate_url_security(url)

        self.assertIn("Access denied", str(ctx.exception))


class TestPlatformURLPolicies(unittest.TestCase):
    """Verifies YouTube and TED policy enforcement and transcript reuse routing."""

    @patch("app.services.url_ingestion_service.validate_url_security", side_effect=lambda u: u)
    def test_youtube_without_transcript_rejected(self, mock_val_url):
        """YouTube URLs without pre-existing authorized transcript are rejected."""
        with self.assertRaises(ValueError) as ctx:
            ingest_from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertIn("unauthorized scraping of YouTube video", str(ctx.exception))

    @patch("app.services.url_ingestion_service.validate_url_security", side_effect=lambda u: u)
    @patch("app.services.url_ingestion_service.ingest_recording")
    def test_youtube_with_authorized_transcript_succeeds(self, mock_ingest, mock_val_url):
        """YouTube URL with authorized transcript routes to transcript reuse pipeline."""
        mock_ingest.return_value = {
            "status": "success",
            "recording_id": "rec-yt-1",
            "chunk_count": 3,
        }
        transcript = {
            "segments": [{"start": 0.0, "end": 10.0, "text": "Authorized caption"}]
        }

        result = ingest_from_url(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            recording_id="rec-yt-1",
            authorized_transcript=transcript,
        )

        mock_ingest.assert_called_once_with(
            recording_id="rec-yt-1",
            existing_transcript=transcript,
        )
        self.assertEqual(result["source_category"], "youtube")
        self.assertEqual(result["video_id"], "dQw4w9WgXcQ")

    @patch("app.services.url_ingestion_service.validate_url_security", side_effect=lambda u: u)
    def test_ted_without_transcript_rejected(self, mock_val_url):
        """TED Talk URLs without authorized transcript are rejected with licensing notice."""
        with self.assertRaises(ValueError) as ctx:
            ingest_from_url("https://www.ted.com/talks/jane_goodall_what_makes_us_human")
        self.assertIn("TED Talk content", str(ctx.exception))
        self.assertIn("restrict automated downloading", str(ctx.exception))

    @patch("app.services.url_ingestion_service.validate_url_security", side_effect=lambda u: u)
    @patch("app.services.url_ingestion_service.ingest_recording")
    def test_ted_with_authorized_transcript_succeeds(self, mock_ingest, mock_val_url):
        """TED Talk URL with authorized transcript routes to transcript reuse pipeline."""
        mock_ingest.return_value = {
            "status": "success",
            "recording_id": "rec-ted-1",
            "chunk_count": 2,
        }
        transcript = {
            "segments": [{"start": 0.0, "end": 15.0, "text": "Licensed TED transcript text"}]
        }

        result = ingest_from_url(
            url="https://www.ted.com/talks/jane_goodall_what_makes_us_human",
            recording_id="rec-ted-1",
            authorized_transcript=transcript,
        )

        mock_ingest.assert_called_once_with(
            recording_id="rec-ted-1",
            existing_transcript=transcript,
        )
        self.assertEqual(result["source_category"], "ted")
        self.assertEqual(result["talk_slug"], "jane_goodall_what_makes_us_human")


class TestDownloadedMediaRetentionAndCleanup(unittest.TestCase):
    """
    Verifies that downloaded media files are preserved when ingestion succeeds
    (for application audio/video playback), but deleted immediately if downstream
    ingestion fails.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.patch_upload_dir = patch.object(
            url_ingestion_service,
            "UPLOAD_DIR",
            Path(self.temp_dir.name),
        )
        self.patch_upload_dir.start()

    def tearDown(self):
        self.patch_upload_dir.stop()
        self.temp_dir.cleanup()

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("app.services.url_ingestion_service.download_media_from_url")
    @patch("app.services.url_ingestion_service.ingest_recording")
    def test_successfully_ingested_downloaded_media_remains_for_playback(
        self, mock_ingest, mock_download, mock_val_url
    ):
        mock_val_url.side_effect = lambda u: u

        # Create actual downloaded media file in upload temp dir
        media_file = Path(self.temp_dir.name) / "downloaded_sample.mp3"
        media_file.write_bytes(b"ID3\x04" + b"A" * 100)
        mock_download.return_value = (media_file, ".mp3")

        mock_ingest.return_value = {
            "status": "success",
            "recording_id": "rec-download-success",
            "chunk_count": 2,
        }

        result = ingest_from_url("https://example.com/audio.mp3", recording_id="rec-download-success")

        self.assertEqual(result["status"], "success")
        # Downloaded media must still exist for playback
        self.assertTrue(media_file.exists())

    @patch("app.services.url_ingestion_service.validate_url_security")
    @patch("app.services.url_ingestion_service.download_media_from_url")
    @patch("app.services.url_ingestion_service.ingest_recording")
    def test_downloaded_media_deleted_after_downstream_failure(
        self, mock_ingest, mock_download, mock_val_url
    ):
        mock_val_url.side_effect = lambda u: u

        # Create actual downloaded media file
        media_file = Path(self.temp_dir.name) / "downloaded_to_fail.mp3"
        media_file.write_bytes(b"ID3\x04" + b"A" * 100)
        mock_download.return_value = (media_file, ".mp3")

        # Downstream pipeline failure (e.g. Whisper API or ChromaDB storage failure)
        mock_ingest.side_effect = RuntimeError("Downstream ingestion failure")

        with self.assertRaises(RuntimeError):
            ingest_from_url("https://example.com/audio.mp3", recording_id="rec-download-fail")

        # Downloaded media must have been deleted to avoid disk clutter
        self.assertFalse(media_file.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
