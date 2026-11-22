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
from app.services.registry_service import init_registry


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
        init_registry(db_path=self.db_path)
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
        """Cross-origin requests from localhost:5173 receive CORS allow headers."""
        response = self.client.options(
            "/media/upload",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertIn(response.status_code, (status.HTTP_200_OK, status.HTTP_204_NO_CONTENT))
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")

    def test_cors_127_0_0_1_allowed(self):
        """Cross-origin requests from 127.0.0.1:5173 receive CORS allow headers."""
        response = self.client.options(
            "/media/upload",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertIn(response.status_code, (status.HTTP_200_OK, status.HTTP_204_NO_CONTENT))
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://127.0.0.1:5173")

    def test_cors_unauthorized_origin_rejected(self):
        """Unauthorized origins do not receive CORS allow header."""
        response = self.client.options(
            "/media/upload",
            headers={
                "Origin": "http://malicious-site.org",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertNotEqual(response.headers.get("access-control-allow-origin"), "http://malicious-site.org")

    # -----------------------------------------------------------------------
    # 5. Error Sanitization Tests
    # -----------------------------------------------------------------------
    def test_sanitize_error_message_api_keys(self):
        """API keys are redacted from error messages."""
        raw = "Groq service error with key gsk_abc123secretToken456 and sk-secretKey789"
        sanitized = media_service.sanitize_error_message(raw)
        self.assertNotIn("gsk_abc123secretToken456", sanitized)
        self.assertNotIn("sk-secretKey789", sanitized)

    def test_sanitize_error_message_passwords_and_internals(self):
        """Database connection strings and passwords are redacted."""
        raw = "Exception: CRITICAL: DATABASE_PASSWORD=secret_db_pass_123 failed on internal-db:5432"
        sanitized = media_service.sanitize_error_message(raw)
        self.assertNotIn("secret_db_pass_123", sanitized)
        self.assertNotIn("internal-db", sanitized)
        self.assertIn("unexpected server error", sanitized.lower())

    def test_sanitize_error_message_filesystem_paths(self):
        """Local filesystem paths are not leaked to clients."""
        raw = "FileNotFoundError: D:\\EchoTrace\\backend\\data\\uploads\\private_user_file.mp4 does not exist"
        sanitized = media_service.sanitize_error_message(raw)
        self.assertNotIn("D:\\EchoTrace", sanitized)
        self.assertNotIn("private_user_file", sanitized)

    def test_sanitize_error_message_actionable_categories(self):
        """Actionable diagnostic categories are preserved in friendly format."""
        yt_msg = "Direct downloading or unauthorized scraping of YouTube video 'dQw4w9WgXcQ' is restricted."
        self.assertIn("YouTube downloading is restricted", media_service.sanitize_error_message(yt_msg))

        size_msg = "Download exceeded maximum allowed limit of 100 MB (read 105.20 MB)."
        self.assertIn("exceeds the maximum allowed", media_service.sanitize_error_message(size_msg))

        empty_msg = "Downloaded file is empty (0 bytes)."
        self.assertIn("empty (0 bytes)", media_service.sanitize_error_message(empty_msg))

        fmt_msg = "Downloaded content signature is invalid, unrecognized, or untrusted."
        self.assertIn("unsupported or corrupted", media_service.sanitize_error_message(fmt_msg))

        seg_msg = "Audio extraction failed: FFmpeg segment 2 extraction failed with exit code 1"
        self.assertIn("Audio extraction or segmentation failed", media_service.sanitize_error_message(seg_msg))

    def test_ingestion_failure_stores_sanitized_error(self):
        """Ingestion failure stores a sanitized error in recordings_meta."""
        with patch("app.api.media.ingest_recording", side_effect=RuntimeError("Groq key gsk_superSecret999 failed on C:\\secret\\file.wav")):
            with patch("app.api.media.upsert_recording_meta", side_effect=lambda **kw: media_service.upsert_recording_meta(**kw, db_path=self.db_path)):
                from app.api.media import run_ingestion_background
                run_ingestion_background("rec-err-test", self.tmp_path / "dummy.mp3")

        meta = media_service.get_recording_meta("rec-err-test", db_path=self.db_path)
        self.assertEqual(meta["status"], "failed")
        self.assertNotIn("gsk_superSecret999", meta["error_message"])
        self.assertNotIn("C:\\secret", meta["error_message"])
        self.assertIn("Speech transcription failed", meta["error_message"])

    # -----------------------------------------------------------------------
    # 6. URL Ingestion Media Type Distinction
    # -----------------------------------------------------------------------
    def test_url_ingestion_mp4_sets_video_media_type(self):
        """MP4 URL ingestion records media_type as 'video'."""
        dummy_mp4 = self.tmp_path / "rec-url-vid.mp4"
        dummy_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100)

        with patch("app.services.url_ingestion_service.ingest_from_url", return_value={"status": "success", "media_type": "video", "duration_seconds": 12.0, "chunk_count": 2}):
            with patch("app.api.media.get_media_file_path", return_value=dummy_mp4):
                with patch("app.api.media.upsert_recording_meta", side_effect=lambda **kw: media_service.upsert_recording_meta(**kw, db_path=self.db_path)):
                    from app.api.media import run_url_ingestion_background
                    run_url_ingestion_background("https://example.com/recording.mp4", "rec-url-vid")

        meta = media_service.get_recording_meta("rec-url-vid", db_path=self.db_path)
        self.assertEqual(meta["status"], "ready")
        self.assertEqual(meta["media_type"], "video")

    def test_url_ingestion_audio_sets_audio_media_type(self):
        """MP3 URL ingestion records media_type as 'audio'."""
        dummy_mp3 = self.tmp_path / "rec-url-aud.mp3"
        dummy_mp3.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 100)

        with patch("app.services.url_ingestion_service.ingest_from_url", return_value={"status": "success", "media_type": "audio", "duration_seconds": 25.0, "chunk_count": 3}):
            with patch("app.api.media.get_media_file_path", return_value=dummy_mp3):
                with patch("app.api.media.upsert_recording_meta", side_effect=lambda **kw: media_service.upsert_recording_meta(**kw, db_path=self.db_path)):
                    from app.api.media import run_url_ingestion_background
                    run_url_ingestion_background("https://example.com/podcast.mp3", "rec-url-aud")

        meta = media_service.get_recording_meta("rec-url-aud", db_path=self.db_path)
        self.assertEqual(meta["status"], "ready")
        self.assertEqual(meta["media_type"], "audio")

    # -----------------------------------------------------------------------
    # 7. BackgroundTasks Injection & Synchronous Ingestion Execution
    # -----------------------------------------------------------------------
    def test_background_tasks_dependency_injection(self):
        """FastAPI route signatures inject BackgroundTasks without default instances."""
        import inspect
        from app.api.media import upload_media, ingest_url_endpoint

        for fn in (upload_media, ingest_url_endpoint):
            sig = inspect.signature(fn)
            bg_param = sig.parameters.get("background_tasks")
            self.assertIsNotNone(bg_param)
            # Default should be empty (dependency injection), not a pre-instantiated BackgroundTasks()
            self.assertEqual(bg_param.default, inspect.Parameter.empty)

    @patch("app.api.media.run_in_threadpool")
    @patch("app.api.media.save_uploaded_media")
    def test_upload_media_sync_uses_threadpool(self, mock_save, mock_threadpool):
        """When async_process=False, ingestion runs in threadpool to prevent blocking the event loop."""
        mock_save.return_value = {
            "saved_filename": "sync-test.mp3",
            "original_filename": "sync-test.mp3",
            "file_size": 500,
            "media_type": "audio",
            "file_path": str(self.tmp_path / "sync-test.mp3"),
        }
        mock_threadpool.return_value = None

        audio_bytes = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 100
        file_payload = ("sync-test.mp3", io.BytesIO(audio_bytes), "audio/mpeg")

        with patch("app.api.media.upsert_recording_meta", side_effect=lambda **kw: media_service.upsert_recording_meta(**kw, db_path=self.db_path)):
            with patch("app.api.media.get_recording_meta", side_effect=lambda rid: media_service.get_recording_meta(rid, db_path=self.db_path)):
                response = self.client.post(
                    "/media/upload?async_process=false",
                    files={"file": file_payload},
                )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        mock_threadpool.assert_called_once()

    # -----------------------------------------------------------------------
    # 8. Recording Deletion (DELETE /media/{recording_id})
    # -----------------------------------------------------------------------
    def test_delete_failed_recording_success(self):
        """DELETE /media/{recording_id} removes failed recording metadata and files."""
        # 1. Seed failed recording metadata
        media_service.upsert_recording_meta(
            recording_id="rec-del-fail-1",
            original_filename="https://youtu.be/bad123",
            saved_filename="rec-del-fail-1.media",
            media_type="audio",
            file_size=0,
            status="failed",
            error_message="YouTube downloading is restricted.",
            db_path=self.db_path,
        )

        # 2. Seed a mock media file in temporary uploads dir
        media_file = self.tmp_path / "rec-del-fail-1.mp3"
        media_file.write_bytes(b"dummy")

        with patch("app.services.media_service.UPLOAD_DIR", self.tmp_path):
            with patch("app.api.media.get_recording_meta", side_effect=lambda rid: media_service.get_recording_meta(rid, db_path=self.db_path)):
                with patch("app.api.media.delete_recording_data", side_effect=lambda rid: media_service.delete_recording_data(rid, db_path=self.db_path)):
                    response = self.client.delete("/media/rec-del-fail-1")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data.get("success"))

        # Verify metadata is removed from SQLite
        self.assertIsNone(media_service.get_recording_meta("rec-del-fail-1", db_path=self.db_path))

    def test_delete_processing_recording_returns_409_conflict(self):
        """Attempting to delete an actively processing recording returns 409 Conflict."""
        media_service.upsert_recording_meta(
            recording_id="rec-del-proc",
            original_filename="processing_lecture.mp4",
            status="processing",
            db_path=self.db_path,
        )

        with patch("app.api.media.get_recording_meta", side_effect=lambda rid: media_service.get_recording_meta(rid, db_path=self.db_path)):
            response = self.client.delete("/media/rec-del-proc")

        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)
        self.assertIn("actively processing", response.json().get("detail", ""))

    def test_delete_nonexistent_recording_returns_404(self):
        """Deleting a non-existent recording returns 404 Not Found."""
        with patch("app.api.media.get_recording_meta", return_value=None):
            response = self.client.delete("/media/rec-does-not-exist")

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_delete_invalid_recording_id_returns_400(self):
        """Path traversal or invalid characters in recording_id return 400 Bad Request."""
        response = self.client.delete("/media/../../etc/passwd")
        self.assertIn(response.status_code, (status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND))

    def test_delete_cleans_chromadb_vectors_safely(self):
        """ChromaDB vectors matching recording_id are deleted while others remain untouched."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["rec-target_v_1_000000", "rec-target_v_1_000001"]}

        media_service.upsert_recording_meta(
            recording_id="rec-target",
            original_filename="target.mp4",
            status="ready",
            db_path=self.db_path,
        )

        result = media_service.delete_recording_data(
            "rec-target",
            db_path=self.db_path,
            collection=mock_collection,
        )

        self.assertTrue(result["success"])
        mock_collection.delete.assert_called_once_with(ids=["rec-target_v_1_000000", "rec-target_v_1_000001"])

    def test_delete_leaves_other_recordings_intact(self):
        """Deleting recording A does not touch recording B's metadata."""
        media_service.upsert_recording_meta(
            recording_id="rec-keep",
            original_filename="keep_me.mp4",
            status="ready",
            db_path=self.db_path,
        )
        media_service.upsert_recording_meta(
            recording_id="rec-delete",
            original_filename="delete_me.mp4",
            status="failed",
            db_path=self.db_path,
        )

        media_service.delete_recording_data("rec-delete", db_path=self.db_path)

        # Confirm rec-delete is gone
        self.assertIsNone(media_service.get_recording_meta("rec-delete", db_path=self.db_path))
        # Confirm rec-keep is still present and intact
        keep_meta = media_service.get_recording_meta("rec-keep", db_path=self.db_path)
        self.assertIsNotNone(keep_meta)
        self.assertEqual(keep_meta["original_filename"], "keep_me.mp4")


if __name__ == "__main__":
    unittest.main()
