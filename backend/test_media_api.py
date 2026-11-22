import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure 'backend' directory is in Python search path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import status
from fastapi.testclient import TestClient

from app.main import app
from app.services import media_service


class TestMediaAPI(unittest.TestCase):
    """
    Tests for media upload, streaming with HTTP Range requests,
    metadata tracking, status reporting, and transcript retrieval.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.db_path = self.tmp_path / "test_registry.sqlite3"
        media_service.init_recordings_table(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    # -----------------------------------------------------------------------
    # 1. Media Streaming & HTTP Range Requests
    # -----------------------------------------------------------------------
    def test_stream_media_returns_200_and_accept_ranges(self):
        """Streaming endpoint returns 200 OK with Accept-Ranges: bytes when no Range header is sent."""
        dummy_file = self.tmp_path / "test-rec-1.mp3"
        dummy_content = b"EchoTraceAudioByteStreamContent12345"
        dummy_file.write_bytes(dummy_content)

        with patch("app.api.media.get_media_file_path", return_value=dummy_file):
            response = self.client.get("/media/stream/test-rec-1")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.headers.get("accept-ranges"), "bytes")
        self.assertIn("audio/mpeg", response.headers.get("content-type", ""))
        self.assertEqual(response.content, dummy_content)

    def test_stream_media_supports_http_range_requests(self):
        """Streaming endpoint returns 206 Partial Content when Range header is provided."""
        dummy_file = self.tmp_path / "test-rec-seek.mp4"
        dummy_content = b"0123456789abcdefghijklmnopqrstuvwxyz"
        dummy_file.write_bytes(dummy_content)

        with patch("app.api.media.get_media_file_path", return_value=dummy_file):
            response = self.client.get(
                "/media/stream/test-rec-seek",
                headers={"Range": "bytes=0-9"},
            )

        self.assertEqual(response.status_code, status.HTTP_206_PARTIAL_CONTENT)
        self.assertEqual(response.headers.get("content-range"), f"bytes 0-9/{len(dummy_content)}")
        self.assertEqual(response.content, b"0123456789")

    def test_stream_media_not_found(self):
        """Streaming non-existent recording returns 404."""
        with patch("app.api.media.get_media_file_path", return_value=None):
            response = self.client.get("/media/stream/nonexistent-id-123")

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_stream_media_invalid_id_rejected(self):
        """Path traversal or invalid characters in recording_id return 400 Bad Request."""
        response = self.client.get("/media/stream/../../etc/passwd")
        # FastAPI router will either match 404 or our handler 400
        self.assertIn(response.status_code, (status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND))

    # -----------------------------------------------------------------------
    # 2. Media Upload & Ingestion Integration
    # -----------------------------------------------------------------------
    @patch("app.api.media.ingest_recording")
    def test_upload_media_triggers_ingestion(self, mock_ingest):
        """POST /media/upload saves file and triggers ingestion."""
        mock_ingest.return_value = {
            "status": "success",
            "recording_id": "test-rec-up",
            "duration_seconds": 45.5,
            "chunk_count": 3,
        }

        audio_bytes = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 200
        file_payload = ("test_speech.mp3", io.BytesIO(audio_bytes), "audio/mpeg")

        response = self.client.post(
            "/media/upload?async_process=false",
            files={"file": file_payload},
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("recording_id", data)
        self.assertEqual(data["original_filename"], "test_speech.mp3")
        self.assertEqual(data["media_type"], "audio")
        mock_ingest.assert_called_once()

    # -----------------------------------------------------------------------
    # 3. Status and Transcript Retrieval
    # -----------------------------------------------------------------------
    def test_get_recording_status(self):
        """GET /media/{recording_id}/status returns recording metadata."""
        rec_id = "test-status-rec-1"
        media_service.upsert_recording_meta(
            recording_id=rec_id,
            original_filename="lecture.mp4",
            saved_filename=f"{rec_id}.mp4",
            media_type="video",
            file_size=10240,
            status="ready",
            duration_seconds=120.0,
            chunk_count=5,
            db_path=self.db_path,
        )

        with patch("app.api.media.get_recording_meta", side_effect=lambda rid: media_service.get_recording_meta(rid, db_path=self.db_path)):
            response = self.client.get(f"/media/{rec_id}/status")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["recording_id"], rec_id)
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["duration_seconds"], 120.0)
        self.assertEqual(data["chunk_count"], 5)

    def test_get_recording_status_not_found(self):
        """GET /media/{recording_id}/status for non-existent recording returns 404."""
        with patch("app.api.media.get_recording_meta", return_value=None):
            response = self.client.get("/media/definitely-not-found-999/status")

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_get_recording_transcript(self):
        """GET /media/{recording_id}/transcript returns timestamped chunks."""
        rec_id = "test-transcript-rec"
        mock_chunks = [
            {
                "chunk_id": f"{rec_id}_000000",
                "recording_id": rec_id,
                "chunk_index": 0,
                "start": 0.0,
                "end": 10.5,
                "timestamp_range": "00:00–00:10",
                "text": "First chunk of transcribed speech.",
                "segment_count": 1,
            },
            {
                "chunk_id": f"{rec_id}_000001",
                "recording_id": rec_id,
                "chunk_index": 1,
                "start": 10.5,
                "end": 22.0,
                "timestamp_range": "00:10–00:22",
                "text": "Second chunk of transcribed speech.",
                "segment_count": 1,
            },
        ]

        with patch("app.api.media.get_recording_transcript_chunks", return_value=mock_chunks):
            response = self.client.get(f"/media/{rec_id}/transcript")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["recording_id"], rec_id)
        self.assertEqual(data["chunk_count"], 2)
        self.assertEqual(len(data["chunks"]), 2)
        self.assertEqual(data["chunks"][0]["start"], 0.0)
        self.assertEqual(data["chunks"][0]["timestamp_range"], "00:00–00:10")

    def test_list_recordings(self):
        """GET /media/recordings returns array of all recordings."""
        media_service.upsert_recording_meta(
            recording_id="rec-list-1",
            original_filename="a.mp3",
            saved_filename="rec-list-1.mp3",
            media_type="audio",
            file_size=1000,
            status="ready",
            db_path=self.db_path,
        )
        media_service.upsert_recording_meta(
            recording_id="rec-list-2",
            original_filename="b.mp4",
            saved_filename="rec-list-2.mp4",
            media_type="video",
            file_size=2000,
            status="processing",
            db_path=self.db_path,
        )

        with patch("app.api.media.list_recordings_meta", side_effect=lambda: media_service.list_recordings_meta(db_path=self.db_path)):
            response = self.client.get("/media/recordings")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 2)
        rec_ids = [r["recording_id"] for r in data]
        self.assertIn("rec-list-1", rec_ids)
        self.assertIn("rec-list-2", rec_ids)

    # -----------------------------------------------------------------------
    # 4. CORS Support
    # -----------------------------------------------------------------------
    def test_cors_headers_present(self):
        """Cross-origin requests receive CORS allow headers."""
        response = self.client.options(
            "/media/upload",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertIn(response.status_code, (status.HTTP_200_OK, status.HTTP_204_NO_CONTENT))
        # With allow_credentials=True, Starlette mirrors the specific Origin
        self.assertIn(response.headers.get("access-control-allow-origin"), ("*", "http://localhost:5173"))


if __name__ == "__main__":
    unittest.main()
