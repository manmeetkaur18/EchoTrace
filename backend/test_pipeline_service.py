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
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mock_transcription(text="Hello", segments=None, words=None):
    if segments is None:
        segments = [{"id": 0, "start": 0.0, "end": 5.0, "text": text}]
    if words is None:
        words = []
    return {"text": text, "segments": segments, "words": words}


def _make_mock_chunk(text="Hello", start=0.0, end=5.0):
    return [{"chunk_index": 0, "start": start, "end": end, "text": text, "segment_count": 1}]


def _make_mock_store(recording_id="rec-1"):
    return {"recording_id": recording_id, "stored_count": 1, "ids": [f"{recording_id}_000000"]}


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
    """Verifies that large audio files trigger the segmented transcription path."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.get_media_duration")
    @patch("app.services.pipeline_service.transcribe_segmented_audio")
    @patch("app.services.pipeline_service.split_audio_into_segments")
    @patch("app.services.pipeline_service.transcribe_audio")
    def test_long_audio_uses_segmented_path(
        self, mock_transcribe, mock_split, mock_transcribe_seg, mock_duration,
        mock_chunk, mock_embed, mock_store
    ):
        """Direct audio files exceeding 25 MB use the segmented transcription path."""
        audio_file = self.tmp_path / "large_audio.mp3"
        audio_file.write_bytes(b"A" * 1024)

        real_stat = audio_file.stat()
        mock_stat_res = MagicMock()
        mock_stat_res.st_mode = real_stat.st_mode
        mock_stat_res.st_size = 26 * 1024 * 1024  # 26 MB

        mock_split.return_value = [(audio_file, 0.0, 600.0)]
        mock_transcribe_seg.return_value = _make_mock_transcription()
        mock_duration.return_value = 1800.0
        mock_chunk.return_value = _make_mock_chunk()
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = _make_mock_store("rec-large")

        with patch.object(Path, "stat", return_value=mock_stat_res):
            result = ingest_recording(recording_id="rec-large", media_path=audio_file)

        mock_split.assert_called_once()
        mock_transcribe_seg.assert_called_once()
        mock_transcribe.assert_not_called()
        self.assertEqual(result["status"], "success")

    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.get_media_duration")
    @patch("app.services.pipeline_service.transcribe_segmented_audio")
    @patch("app.services.pipeline_service.split_audio_into_segments")
    @patch("app.services.pipeline_service.transcribe_audio")
    @patch("app.services.pipeline_service.extract_audio_from_video")
    def test_long_video_uses_extraction_then_segmentation(
        self, mock_extract, mock_transcribe, mock_split, mock_transcribe_seg,
        mock_duration, mock_chunk, mock_embed, mock_store
    ):
        """WAV audio extracted from large video triggers the segmented transcription path."""
        video_file = self.tmp_path / "long_video.mp4"
        video_file.write_bytes(b"video_bytes")

        extracted_wav = self.tmp_path / "long_video.wav"
        extracted_wav.write_bytes(b"RIFFWAVE")
        mock_extract.return_value = str(extracted_wav)

        real_stat = extracted_wav.stat()
        mock_stat_res = MagicMock()
        mock_stat_res.st_mode = real_stat.st_mode
        mock_stat_res.st_size = 30 * 1024 * 1024

        mock_split.return_value = [(extracted_wav, 0.0, 600.0)]
        mock_transcribe_seg.return_value = _make_mock_transcription()
        mock_duration.return_value = 1800.0
        mock_chunk.return_value = _make_mock_chunk()
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = _make_mock_store("rec-video-large")

        with patch.object(Path, "stat", return_value=mock_stat_res):
            result = ingest_recording(recording_id="rec-video-large", media_path=video_file)

        mock_split.assert_called_once()
        mock_transcribe_seg.assert_called_once()
        mock_transcribe.assert_not_called()
        self.assertEqual(result["status"], "success")


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


class TestPipelineServiceLongRecording(unittest.TestCase):
    """
    Integration tests for the long recording pipeline path (Phase 3.3).
    Verifies correct path selection, duration calculation, and resource cleanup
    for both single-file (<=25 MB) and segmented (>25 MB) recordings.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_audio_file(self, name="audio.mp3", content=b"audio_bytes"):
        path = self.tmp_path / name
        path.write_bytes(content)
        return path

    def _make_video_file(self, name="video.mp4", content=b"video_bytes"):
        path = self.tmp_path / name
        path.write_bytes(content)
        return path

    def _large_stat_mock(self, base_path: Path, size_mb: int = 26):
        real_stat = base_path.stat()
        mock_stat_res = MagicMock()
        mock_stat_res.st_mode = real_stat.st_mode
        mock_stat_res.st_size = size_mb * 1024 * 1024
        return mock_stat_res

    def _small_stat_mock(self, base_path: Path, size_bytes: int = 1024):
        real_stat = base_path.stat()
        mock_stat_res = MagicMock()
        mock_stat_res.st_mode = real_stat.st_mode
        mock_stat_res.st_size = size_bytes
        return mock_stat_res

    # -----------------------------------------------------------------------
    # 1. Short audio → single-file transcription
    # -----------------------------------------------------------------------
    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.transcribe_segmented_audio")
    @patch("app.services.pipeline_service.transcribe_audio")
    def test_short_audio_uses_single_file_transcription(
        self, mock_ta, mock_tsa, mock_chunk, mock_embed, mock_store
    ):
        """Small audio (<= 25 MB) calls transcribe_audio(); transcribe_segmented_audio() NOT called."""
        audio = self._make_audio_file()
        mock_ta.return_value = _make_mock_transcription()
        mock_chunk.return_value = _make_mock_chunk()
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = _make_mock_store()

        result = ingest_recording("rec-short", media_path=audio)

        mock_ta.assert_called_once()
        mock_tsa.assert_not_called()
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["segmented_audio"])
        self.assertEqual(result["transcription_mode"], "single_file")

    # -----------------------------------------------------------------------
    # 2. Long audio → segmented transcription
    # -----------------------------------------------------------------------
    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.get_media_duration")
    @patch("app.services.pipeline_service.transcribe_segmented_audio")
    @patch("app.services.pipeline_service.split_audio_into_segments")
    @patch("app.services.pipeline_service.transcribe_audio")
    def test_long_audio_uses_segmented_transcription(
        self, mock_ta, mock_split, mock_tsa, mock_duration,
        mock_chunk, mock_embed, mock_store
    ):
        """Large audio (> 25 MB) calls split_audio_into_segments() + transcribe_segmented_audio(); transcribe_audio() NOT called."""
        audio = self._make_audio_file()
        large_stat = self._large_stat_mock(audio)

        mock_split.return_value = [(audio, 0.0, 600.0)]
        mock_tsa.return_value = _make_mock_transcription()
        mock_duration.return_value = 1800.0
        mock_chunk.return_value = _make_mock_chunk()
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = _make_mock_store("rec-long")

        with patch.object(Path, "stat", return_value=large_stat):
            result = ingest_recording("rec-long", media_path=audio)

        mock_ta.assert_not_called()
        mock_split.assert_called_once()
        mock_tsa.assert_called_once()
        self.assertTrue(result["segmented_audio"])
        self.assertEqual(result["transcription_mode"], "segmented")
        self.assertEqual(result["status"], "success")

    # -----------------------------------------------------------------------
    # 3. Large video → extraction → segmentation → transcription
    # -----------------------------------------------------------------------
    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.get_media_duration")
    @patch("app.services.pipeline_service.transcribe_segmented_audio")
    @patch("app.services.pipeline_service.split_audio_into_segments")
    @patch("app.services.pipeline_service.transcribe_audio")
    @patch("app.services.pipeline_service.extract_audio_from_video")
    def test_long_video_uses_extraction_segmentation_and_transcription(
        self, mock_extract, mock_ta, mock_split, mock_tsa, mock_duration,
        mock_chunk, mock_embed, mock_store
    ):
        """Large video: extraction → split → transcribe_segmented_audio (not transcribe_audio)."""
        video = self._make_video_file()
        extracted_wav = self.tmp_path / "video.wav"
        extracted_wav.write_bytes(b"WAV")
        mock_extract.return_value = str(extracted_wav)

        large_stat = self._large_stat_mock(extracted_wav)
        mock_split.return_value = [(extracted_wav, 0.0, 600.0)]
        mock_tsa.return_value = _make_mock_transcription()
        mock_duration.return_value = 1800.0
        mock_chunk.return_value = _make_mock_chunk()
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = _make_mock_store("rec-long-vid")

        with patch.object(Path, "stat", return_value=large_stat):
            result = ingest_recording("rec-long-vid", media_path=video)

        mock_extract.assert_called_once()
        mock_ta.assert_not_called()
        mock_split.assert_called_once()
        mock_tsa.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["segmented_audio"])

    # -----------------------------------------------------------------------
    # 4. FFprobe duration used for long audio
    # -----------------------------------------------------------------------
    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.get_media_duration")
    @patch("app.services.pipeline_service.transcribe_segmented_audio")
    @patch("app.services.pipeline_service.split_audio_into_segments")
    @patch("app.services.pipeline_service.transcribe_audio")
    def test_ffprobe_duration_used_for_long_audio(
        self, mock_ta, mock_split, mock_tsa, mock_duration,
        mock_chunk, mock_embed, mock_store
    ):
        """For large audio, duration comes from get_media_duration() (FFprobe), not segment timestamps."""
        audio = self._make_audio_file()
        large_stat = self._large_stat_mock(audio)

        mock_split.return_value = [(audio, 0.0, 600.0)]
        # Transcription returns a segment ending at 300s
        mock_tsa.return_value = _make_mock_transcription(
            segments=[{"id": 0, "start": 0.0, "end": 300.0, "text": "speech"}]
        )
        # FFprobe says 1800s — this should win for long recordings
        mock_duration.return_value = 1800.0
        mock_chunk.return_value = _make_mock_chunk(end=300.0)
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = _make_mock_store("rec-dur")

        with patch.object(Path, "stat", return_value=large_stat):
            result = ingest_recording("rec-dur", media_path=audio)

        self.assertEqual(result["duration_seconds"], 1800.0)
        mock_duration.assert_called_once()

    # -----------------------------------------------------------------------
    # 5. Short audio duration from segment timestamps
    # -----------------------------------------------------------------------
    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.get_media_duration")
    @patch("app.services.pipeline_service.transcribe_audio")
    def test_short_audio_duration_from_segment_timestamps(
        self, mock_ta, mock_duration, mock_chunk, mock_embed, mock_store
    ):
        """For short audio, duration comes from segment end timestamps; get_media_duration NOT called."""
        audio = self._make_audio_file()
        mock_ta.return_value = _make_mock_transcription(
            segments=[{"id": 0, "start": 0.0, "end": 123.45, "text": "text"}]
        )
        mock_chunk.return_value = _make_mock_chunk(end=123.45)
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = _make_mock_store("rec-short-dur")

        result = ingest_recording("rec-short-dur", media_path=audio)

        self.assertEqual(result["duration_seconds"], 123.45)
        mock_duration.assert_not_called()

    # -----------------------------------------------------------------------
    # 6. Empty final segment does not affect FFprobe duration
    # -----------------------------------------------------------------------
    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.get_media_duration")
    @patch("app.services.pipeline_service.transcribe_segmented_audio")
    @patch("app.services.pipeline_service.split_audio_into_segments")
    @patch("app.services.pipeline_service.transcribe_audio")
    def test_empty_final_segment_does_not_affect_duration(
        self, mock_ta, mock_split, mock_tsa, mock_duration,
        mock_chunk, mock_embed, mock_store
    ):
        """Long audio with empty final segment; duration still correct from FFprobe."""
        audio = self._make_audio_file()
        large_stat = self._large_stat_mock(audio)

        mock_split.return_value = [(audio, 0.0, 600.0), (audio, 600.0, 1200.0)]
        # Segmented transcription returns non-empty segments (empty last segment is handled inside tsa)
        mock_tsa.return_value = _make_mock_transcription(
            segments=[{"id": 0, "start": 0.0, "end": 580.0, "text": "audio content"}]
        )
        mock_duration.return_value = 1200.0
        mock_chunk.return_value = _make_mock_chunk(end=580.0)
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = _make_mock_store("rec-empty-seg")

        with patch.object(Path, "stat", return_value=large_stat):
            result = ingest_recording("rec-empty-seg", media_path=audio)

        self.assertEqual(result["duration_seconds"], 1200.0)
        self.assertEqual(result["status"], "success")

    # -----------------------------------------------------------------------
    # 7. Segmentation failure propagates and no temp files leaked
    # -----------------------------------------------------------------------
    @patch("app.services.pipeline_service.split_audio_into_segments")
    @patch("app.services.pipeline_service.transcribe_audio")
    def test_segmentation_failure_cleans_temp_files_and_raises(
        self, mock_ta, mock_split
    ):
        """split_audio_into_segments raises RuntimeError; exception propagates; no leaked temp dirs."""
        audio = self._make_audio_file()
        large_stat = self._large_stat_mock(audio)
        mock_split.side_effect = RuntimeError("FFmpeg segmentation failed")

        with patch.object(Path, "stat", return_value=large_stat):
            with self.assertRaises(RuntimeError) as ctx:
                ingest_recording("rec-seg-fail", media_path=audio)

        self.assertIn("FFmpeg segmentation failed", str(ctx.exception))
        mock_ta.assert_not_called()

    # -----------------------------------------------------------------------
    # 8. Transcription failure on long audio propagates; extracted WAV cleaned
    # -----------------------------------------------------------------------
    @patch("app.services.pipeline_service.transcribe_segmented_audio")
    @patch("app.services.pipeline_service.split_audio_into_segments")
    @patch("app.services.pipeline_service.transcribe_audio")
    @patch("app.services.pipeline_service.extract_audio_from_video")
    def test_transcription_failure_on_long_audio_raises(
        self, mock_extract, mock_ta, mock_split, mock_tsa
    ):
        """transcribe_segmented_audio raises HTTPException; propagates; extracted WAV cleaned up for video."""
        video = self._make_video_file()
        extracted_wav = self.tmp_path / "video.wav"
        extracted_wav.write_bytes(b"WAV")
        mock_extract.return_value = str(extracted_wav)

        large_stat = self._large_stat_mock(extracted_wav)
        mock_split.return_value = [(extracted_wav, 0.0, 600.0)]
        mock_tsa.side_effect = HTTPException(status_code=500, detail="Groq error")

        with patch.object(Path, "stat", return_value=large_stat):
            with self.assertRaises(HTTPException) as ctx:
                ingest_recording("rec-tsa-fail", media_path=video)

        self.assertEqual(ctx.exception.status_code, 500)
        # Extracted WAV must be cleaned up even on failure
        self.assertFalse(extracted_wav.exists())

    # -----------------------------------------------------------------------
    # 9. Chunking failure on long audio — extracted WAV cleaned up
    # -----------------------------------------------------------------------
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.get_media_duration")
    @patch("app.services.pipeline_service.transcribe_segmented_audio")
    @patch("app.services.pipeline_service.split_audio_into_segments")
    @patch("app.services.pipeline_service.transcribe_audio")
    @patch("app.services.pipeline_service.extract_audio_from_video")
    def test_chunking_failure_on_long_audio_cleans_up(
        self, mock_extract, mock_ta, mock_split, mock_tsa, mock_duration, mock_chunk
    ):
        """chunk_transcript raises; extracted WAV cleaned up."""
        video = self._make_video_file()
        extracted_wav = self.tmp_path / "video.wav"
        extracted_wav.write_bytes(b"WAV")
        mock_extract.return_value = str(extracted_wav)

        large_stat = self._large_stat_mock(extracted_wav)
        mock_split.return_value = [(extracted_wav, 0.0, 600.0)]
        mock_tsa.return_value = _make_mock_transcription()
        mock_duration.return_value = 1800.0
        mock_chunk.side_effect = RuntimeError("Chunking error")

        with patch.object(Path, "stat", return_value=large_stat):
            with self.assertRaises(RuntimeError):
                ingest_recording("rec-chunk-fail", media_path=video)

        self.assertFalse(extracted_wav.exists())

    # -----------------------------------------------------------------------
    # 10. Embedding failure on long audio — extracted WAV cleaned up
    # -----------------------------------------------------------------------
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.get_media_duration")
    @patch("app.services.pipeline_service.transcribe_segmented_audio")
    @patch("app.services.pipeline_service.split_audio_into_segments")
    @patch("app.services.pipeline_service.transcribe_audio")
    @patch("app.services.pipeline_service.extract_audio_from_video")
    def test_embedding_failure_on_long_audio_cleans_up(
        self, mock_extract, mock_ta, mock_split, mock_tsa, mock_duration, mock_chunk, mock_embed
    ):
        """embed_document raises; extracted WAV cleaned up."""
        video = self._make_video_file()
        extracted_wav = self.tmp_path / "video.wav"
        extracted_wav.write_bytes(b"WAV")
        mock_extract.return_value = str(extracted_wav)

        large_stat = self._large_stat_mock(extracted_wav)
        mock_split.return_value = [(extracted_wav, 0.0, 600.0)]
        mock_tsa.return_value = _make_mock_transcription()
        mock_duration.return_value = 1800.0
        mock_chunk.return_value = _make_mock_chunk()
        mock_embed.side_effect = RuntimeError("Gemini quota")

        with patch.object(Path, "stat", return_value=large_stat):
            with self.assertRaises(RuntimeError):
                ingest_recording("rec-embed-fail", media_path=video)

        self.assertFalse(extracted_wav.exists())

    # -----------------------------------------------------------------------
    # 11. Original audio untouched after segmented success
    # -----------------------------------------------------------------------
    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.get_media_duration")
    @patch("app.services.pipeline_service.transcribe_segmented_audio")
    @patch("app.services.pipeline_service.split_audio_into_segments")
    @patch("app.services.pipeline_service.transcribe_audio")
    def test_original_audio_untouched_after_segmented_success(
        self, mock_ta, mock_split, mock_tsa, mock_duration,
        mock_chunk, mock_embed, mock_store
    ):
        """Original audio file remains on disk after successful segmented processing."""
        audio = self._make_audio_file()
        large_stat = self._large_stat_mock(audio)

        mock_split.return_value = [(audio, 0.0, 600.0)]
        mock_tsa.return_value = _make_mock_transcription()
        mock_duration.return_value = 1800.0
        mock_chunk.return_value = _make_mock_chunk()
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = _make_mock_store("rec-orig")

        with patch.object(Path, "stat", return_value=large_stat):
            result = ingest_recording("rec-orig", media_path=audio)

        self.assertEqual(result["status"], "success")
        self.assertTrue(audio.exists(), "Original audio file must not be deleted.")

    # -----------------------------------------------------------------------
    # 12. Original video untouched; extracted WAV cleaned after segmented processing
    # -----------------------------------------------------------------------
    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.get_media_duration")
    @patch("app.services.pipeline_service.transcribe_segmented_audio")
    @patch("app.services.pipeline_service.split_audio_into_segments")
    @patch("app.services.pipeline_service.transcribe_audio")
    @patch("app.services.pipeline_service.extract_audio_from_video")
    def test_original_video_untouched_after_segmented_processing(
        self, mock_extract, mock_ta, mock_split, mock_tsa, mock_duration,
        mock_chunk, mock_embed, mock_store
    ):
        """Original video remains; extracted WAV cleaned up after segmented processing."""
        video = self._make_video_file()
        extracted_wav = self.tmp_path / "video.wav"
        extracted_wav.write_bytes(b"WAV")
        mock_extract.return_value = str(extracted_wav)

        large_stat = self._large_stat_mock(extracted_wav)
        mock_split.return_value = [(extracted_wav, 0.0, 600.0)]
        mock_tsa.return_value = _make_mock_transcription()
        mock_duration.return_value = 1800.0
        mock_chunk.return_value = _make_mock_chunk()
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = _make_mock_store("rec-vid-seg")

        with patch.object(Path, "stat", return_value=large_stat):
            result = ingest_recording("rec-vid-seg", media_path=video)

        self.assertEqual(result["status"], "success")
        self.assertTrue(video.exists(), "Original video file must not be deleted.")
        self.assertFalse(extracted_wav.exists(), "Extracted WAV must be deleted.")

    # -----------------------------------------------------------------------
    # 13. Existing transcript bypasses segmentation
    # -----------------------------------------------------------------------
    @patch("app.services.pipeline_service.store_transcript_chunks")
    @patch("app.services.pipeline_service.embed_document")
    @patch("app.services.pipeline_service.chunk_transcript")
    @patch("app.services.pipeline_service.transcribe_segmented_audio")
    @patch("app.services.pipeline_service.split_audio_into_segments")
    @patch("app.services.pipeline_service.transcribe_audio")
    def test_existing_transcript_bypasses_segmentation(
        self, mock_ta, mock_split, mock_tsa, mock_chunk, mock_embed, mock_store
    ):
        """existing_transcript provided → split_audio_into_segments and transcribe_segmented_audio never called."""
        valid_transcript = {
            "text": "Pre-existing transcript",
            "language": "en",
            "segments": [
                {"id": 0, "start": 0.0, "end": 30.0, "text": "Pre-existing transcript"},
            ],
        }
        mock_chunk.return_value = _make_mock_chunk(end=30.0)
        mock_embed.return_value = [0.1] * 768
        mock_store.return_value = _make_mock_store("rec-bypass")

        result = ingest_recording("rec-bypass", existing_transcript=valid_transcript)

        mock_ta.assert_not_called()
        mock_split.assert_not_called()
        mock_tsa.assert_not_called()
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["transcript_reused"])
        self.assertFalse(result["segmented_audio"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
