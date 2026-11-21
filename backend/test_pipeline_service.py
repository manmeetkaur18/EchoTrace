from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure 'backend' directory is in Python search path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services import pipeline_service
from app.services.pipeline_service import (
    AUDIO_EXTENSIONS,
    MAX_AUDIO_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ingest_recording,
    validate_normalized_transcript,
)


class TestPipelineServiceExtensionsAndValidation(unittest.TestCase):
    """Verifies media format validation and extension allowlists."""

    def test_extension_allowlist_matches_upload_service(self):
        """Supported extensions in pipeline must strictly match upload allowlist."""
        self.assertEqual(VIDEO_EXTENSIONS, {".mp4"})
        self.assertEqual(AUDIO_EXTENSIONS, {".mp3", ".wav", ".m4a"})
        self.assertEqual(SUPPORTED_EXTENSIONS, {".mp4", ".mp3", ".wav", ".m4a"})

    def test_invalid_recording_id_rejected(self):
        """Invalid or empty recording_id raises TypeError or ValueError."""
        with self.assertRaises(TypeError):
            ingest_recording(recording_id=123, media_path="test.mp4")

        with self.assertRaises(TypeError):
            ingest_recording(recording_id=True, media_path="test.mp4")

        with self.assertRaises(ValueError):
            ingest_recording(recording_id="   ", media_path="test.mp4")

    def test_missing_media_and_transcript_rejected(self):
        """Calling without media_path and without transcript raises ValueError."""
        with self.assertRaises(ValueError):
            ingest_recording(recording_id="rec-1")

    def test_nonexistent_media_path_raises_file_not_found(self):
        """Passing non-existent media file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            ingest_recording(
                recording_id="rec-1",
                media_path="nonexistent_media_file.mp4",
            )

    def test_unsupported_extension_rejected(self):
        """Extensions not in allowlist (.avi, .mov, .flac, .ogg, .mkv) are rejected."""
        unsupported = [".avi", ".mov", ".flac", ".ogg", ".mkv", ".webm"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            for ext in unsupported:
                test_file = Path(tmp_dir) / f"sample{ext}"
                test_file.write_bytes(b"dummy")
                with self.subTest(ext=ext):
                    with self.assertRaises(ValueError) as ctx:
                        ingest_recording(recording_id="rec-test", media_path=test_file)
                    self.assertIn("Unsupported media format", str(ctx.exception))


class TestPipelineServiceAudioSizeLimit(unittest.TestCase):
    """Verifies strict 25 MB audio file limit enforcement."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("app.services.pipeline_service.transcribe_audio")
    def test_direct_audio_exceeding_25mb_rejected(self, mock_transcribe):
        """Direct audio files exceeding 25 MB are rejected before transcription."""
        audio_file = self.tmp_path / "large_audio.mp3"
        audio_file.write_bytes(b"A" * 1024)

        real_stat = audio_file.stat()
        mock_stat_res = MagicMock()
        mock_stat_res.st_mode = real_stat.st_mode
        mock_stat_res.st_size = 26 * 1024 * 1024  # 26 MB

        with patch.object(Path, "stat", return_value=mock_stat_res):
            with self.assertRaises(ValueError) as ctx:
                ingest_recording(recording_id="rec-large", media_path=audio_file)

            self.assertIn("exceeds maximum allowed transcription limit of 25 MB", str(ctx.exception))
            mock_transcribe.assert_not_called()

    @patch("app.services.pipeline_service.extract_audio_from_video")
    @patch("app.services.pipeline_service.transcribe_audio")
    def test_extracted_video_wav_exceeding_25mb_rejected(self, mock_transcribe, mock_extract):
        """WAV audio extracted from video exceeding 25 MB is rejected before transcription."""
        video_file = self.tmp_path / "long_video.mp4"
        video_file.write_bytes(b"video_bytes")

        extracted_wav = self.tmp_path / "long_video.wav"
        extracted_wav.write_bytes(b"RIFFWAVE")
        mock_extract.return_value = str(extracted_wav)

        real_stat = extracted_wav.stat()
        mock_stat_res = MagicMock()
        mock_stat_res.st_mode = real_stat.st_mode
        mock_stat_res.st_size = 30 * 1024 * 1024

        with patch.object(Path, "stat", return_value=mock_stat_res):
            with self.assertRaises(ValueError) as ctx:
                ingest_recording(recording_id="rec-video-large", media_path=video_file)

            self.assertIn("exceeds maximum allowed transcription limit of 25 MB", str(ctx.exception))
            mock_transcribe.assert_not_called()


class TestPipelineServiceDurationAndMetadata(unittest.TestCase):
    """Verifies duration derivation and metadata handling."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.transcribe_audio")
    def test_duration_derived_from_segment_end_timestamps(
        self, mock_transcribe, mock_chunk, mock_embed, mock_store
    ):
        """Duration is derived accurately from segment end timestamps, ignoring invalid ones."""
        audio_file = self.tmp_path / "test.mp3"
        audio_file.write_bytes(b"dummy_audio")

        # Mock transcription with various segments
        mock_transcribe.return_value = {
            "text": "Hello world",
            "segments": [
                {"id": 0, "start": 0.0, "end": 5.25, "text": "Hello"},
                {"id": 1, "start": 5.25, "end": 14.88, "text": "world"},
                {"id": 2, "start": 14.88, "end": "invalid", "text": "corrupted"},
            ],
        }

        mock_chunk.return_value = [
            {"chunk_index": 0, "start": 0.0, "end": 14.88, "text": "Hello world", "segment_count": 2}
        ]
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = {"recording_id": "rec-1", "stored_count": 1, "ids": ["rec-1_000000"]}

        result = ingest_recording(recording_id="rec-1", media_path=audio_file)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["duration_seconds"], 14.88)
        self.assertEqual(result["transcript_language"], "unknown")
        self.assertEqual(result["segment_count"], 3)
        self.assertEqual(result["chunk_count"], 1)
        self.assertEqual(result["stored_chunk_ids"], ["rec-1_000000"])


class TestPipelineServiceTranscriptReuse(unittest.TestCase):
    """Verifies transcript validation and reuse without calling Whisper."""

    @patch("app.services.pipeline_service.transcribe_audio")
    @patch("app.services.pipeline_service.extract_audio_from_video")
    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    def test_valid_transcript_reuse_skips_whisper_and_ffmpeg(
        self, mock_chunk, mock_embed, mock_store, mock_extract, mock_transcribe
    ):
        """When an authorized transcript is provided, extraction and Whisper are bypassed."""
        valid_transcript = {
            "text": "Pre-existing authorized transcript text.",
            "language": "en",
            "segments": [
                {"id": 0, "start": 0.0, "end": 10.0, "text": "Pre-existing authorized"},
                {"id": 1, "start": 10.0, "end": 20.5, "text": "transcript text."},
            ],
        }

        mock_chunk.return_value = [
            {"chunk_index": 0, "start": 0.0, "end": 20.5, "text": "Pre-existing authorized transcript text.", "segment_count": 2}
        ]
        mock_embed.return_value = [0.05] * 768
        mock_store.return_value = {"recording_id": "rec-reuse", "stored_count": 1, "ids": ["rec-reuse_000000"]}

        result = ingest_recording(
            recording_id="rec-reuse",
            existing_transcript=valid_transcript,
        )

        # Confirm Whisper and FFmpeg were NEVER called
        mock_transcribe.assert_not_called()
        mock_extract.assert_not_called()

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["transcript_reused"])
        self.assertEqual(result["duration_seconds"], 20.5)
        self.assertEqual(result["transcript_language"], "en")
        self.assertEqual(result["stored_chunk_ids"], ["rec-reuse_000000"])

    def test_transcript_validation_rejects_empty_segments(self):
        """Transcripts with missing or empty segments list are rejected."""
        with self.assertRaises(ValueError):
            validate_normalized_transcript({"text": "Hello", "segments": []})

        with self.assertRaises(ValueError):
            validate_normalized_transcript({"text": "Hello"})

    def test_transcript_validation_rejects_invalid_timestamps(self):
        """Transcripts with non-numeric, negative, or end < start timestamps are rejected."""
        # Non-numeric timestamp
        with self.assertRaises(TypeError):
            validate_normalized_transcript({
                "segments": [{"text": "Hello", "start": "bad", "end": 5.0}]
            })

        # Boolean timestamp
        with self.assertRaises(TypeError):
            validate_normalized_transcript({
                "segments": [{"text": "Hello", "start": True, "end": 5.0}]
            })

        # Negative timestamp
        with self.assertRaises(ValueError):
            validate_normalized_transcript({
                "segments": [{"text": "Hello", "start": -1.0, "end": 5.0}]
            })

        # end < start
        with self.assertRaises(ValueError):
            validate_normalized_transcript({
                "segments": [{"text": "Hello", "start": 10.0, "end": 5.0}]
            })

        # Empty segment text
        with self.assertRaises(ValueError):
            validate_normalized_transcript({
                "segments": [{"text": "   ", "start": 0.0, "end": 5.0}]
            })


class TestPipelineServiceResourceCleanup(unittest.TestCase):
    """
    Verifies that intermediate extracted WAV files are deleted after both successful
    and failed pipeline runs, while original source videos and user-owned audio files
    are strictly preserved.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _setup_video_and_extracted_wav(self):
        video_file = self.tmp_path / "test_video.mp4"
        video_file.write_bytes(b"dummy_video_stream")

        extracted_wav = self.tmp_path / "test_video.wav"
        extracted_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        return video_file, extracted_wav

    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.transcribe_audio")
    @patch("app.services.pipeline_service.extract_audio_from_video")
    def test_extracted_wav_deleted_after_successful_processing(
        self, mock_extract, mock_transcribe, mock_chunk, mock_embed, mock_store
    ):
        video_file, extracted_wav = self._setup_video_and_extracted_wav()
        mock_extract.return_value = str(extracted_wav)
        mock_transcribe.return_value = {
            "text": "Hello world",
            "segments": [{"start": 0.0, "end": 5.0, "text": "Hello world"}],
        }
        mock_chunk.return_value = [{"chunk_index": 0, "start": 0.0, "end": 5.0, "text": "Hello world", "segment_count": 1}]
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = {"recording_id": "rec-clean-1", "stored_count": 1, "ids": ["rec-clean-1_000000"]}

        result = ingest_recording("rec-clean-1", media_path=video_file)

        self.assertEqual(result["status"], "success")
        # Intermediate extracted WAV must be deleted
        self.assertFalse(extracted_wav.exists())
        # Original source video must remain
        self.assertTrue(video_file.exists())

    @patch("app.services.pipeline_service.transcribe_audio")
    @patch("app.services.pipeline_service.extract_audio_from_video")
    def test_extracted_wav_deleted_after_transcription_failure(
        self, mock_extract, mock_transcribe
    ):
        video_file, extracted_wav = self._setup_video_and_extracted_wav()
        mock_extract.return_value = str(extracted_wav)
        mock_transcribe.side_effect = RuntimeError("Whisper service failure")

        with self.assertRaises(RuntimeError):
            ingest_recording("rec-clean-fail-transcribe", media_path=video_file)

        self.assertFalse(extracted_wav.exists())
        self.assertTrue(video_file.exists())

    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.transcribe_audio")
    @patch("app.services.pipeline_service.extract_audio_from_video")
    def test_extracted_wav_deleted_after_embedding_failure(
        self, mock_extract, mock_transcribe, mock_chunk, mock_embed
    ):
        video_file, extracted_wav = self._setup_video_and_extracted_wav()
        mock_extract.return_value = str(extracted_wav)
        mock_transcribe.return_value = {
            "text": "Speech",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Speech"}],
        }
        mock_chunk.return_value = [{"chunk_index": 0, "start": 0.0, "end": 2.0, "text": "Speech", "segment_count": 1}]
        mock_embed.side_effect = RuntimeError("Gemini quota exhausted")

        with self.assertRaises(RuntimeError):
            ingest_recording("rec-clean-fail-embed", media_path=video_file)

        self.assertFalse(extracted_wav.exists())
        self.assertTrue(video_file.exists())

    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.transcribe_audio")
    @patch("app.services.pipeline_service.extract_audio_from_video")
    def test_extracted_wav_deleted_after_vector_storage_failure(
        self, mock_extract, mock_transcribe, mock_chunk, mock_embed, mock_store
    ):
        video_file, extracted_wav = self._setup_video_and_extracted_wav()
        mock_extract.return_value = str(extracted_wav)
        mock_transcribe.return_value = {
            "text": "Speech",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Speech"}],
        }
        mock_chunk.return_value = [{"chunk_index": 0, "start": 0.0, "end": 2.0, "text": "Speech", "segment_count": 1}]
        mock_embed.return_value = [0.1] * 768
        mock_store.side_effect = RuntimeError("ChromaDB storage error")

        with self.assertRaises(RuntimeError):
            ingest_recording("rec-clean-fail-store", media_path=video_file)

        self.assertFalse(extracted_wav.exists())
        self.assertTrue(video_file.exists())

    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.transcribe_audio")
    def test_user_uploaded_audio_never_deleted_on_success_or_failure(
        self, mock_transcribe, mock_chunk, mock_embed, mock_store
    ):
        audio_file = self.tmp_path / "user_audio.mp3"
        audio_file.write_bytes(b"user_audio_bytes")

        mock_transcribe.return_value = {
            "text": "User audio speech",
            "segments": [{"start": 0.0, "end": 5.0, "text": "User audio speech"}],
        }
        mock_chunk.return_value = [{"chunk_index": 0, "start": 0.0, "end": 5.0, "text": "User audio speech", "segment_count": 1}]
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = {"recording_id": "rec-audio-1", "stored_count": 1, "ids": ["rec-audio-1_000000"]}

        # 1. Success case: user audio must remain on disk
        res = ingest_recording("rec-audio-1", media_path=audio_file)
        self.assertEqual(res["status"], "success")
        self.assertTrue(audio_file.exists())

        # 2. Failure case: user audio must STILL remain on disk
        mock_store.side_effect = RuntimeError("ChromaDB storage error")
        with self.assertRaises(RuntimeError):
            ingest_recording("rec-audio-fail", media_path=audio_file)

        self.assertTrue(audio_file.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
