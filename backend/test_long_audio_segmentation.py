import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest.mock import MagicMock, patch

# Ensure 'backend' directory is in Python search path so 'app' imports cleanly
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services import audio_service
from app.services.audio_service import (
    FFMPEG_SEGMENT_TIMEOUT_SECONDS,
    FFPROBE_TIMEOUT_SECONDS,
    MAX_SEGMENT_SIZE_BYTES,
    get_media_duration,
    split_audio_into_segments,
)


class TestLongAudioSegmentation(unittest.TestCase):
    """
    Exhaustive unit and integration test suite for EchoTrace Phase 3.1:
    Audio Probing & Segmentation Utility (FFmpeg / FFprobe).
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="echotrace_phase31_test_")
        self.temp_path = Path(self.temp_dir)
        self.output_dir = self.temp_path / "segments_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create a dummy media file
        self.dummy_media = self.temp_path / "dummy_lecture.mp3"
        self.dummy_media.write_bytes(b"DUMMY_AUDIO_BYTES" * 100)

    def tearDown(self):
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    # ---------------------------------------------------------
    # 1. Valid duration extraction
    # ---------------------------------------------------------
    @patch("subprocess.run")
    def test_1_valid_duration_extraction(self, mock_run):
        """FFprobe extracts accurate media duration from format output."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout="125.456\n",
            stderr="",
        )

        duration = get_media_duration(self.dummy_media)
        self.assertEqual(duration, 125.456)

        # Verify command arguments passed to subprocess
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "ffprobe")
        self.assertIn("format=duration", cmd)
        self.assertEqual(cmd[-1], str(self.dummy_media.resolve()))

    # ---------------------------------------------------------
    # 2. Missing media file
    # ---------------------------------------------------------
    def test_2_missing_media_file(self):
        """Non-existent media file raises FileNotFoundError."""
        missing = self.temp_path / "does_not_exist.mp4"

        with self.assertRaises(FileNotFoundError) as ctx:
            get_media_duration(missing)
        self.assertIn("not found", str(ctx.exception).lower())

        with self.assertRaises(FileNotFoundError) as ctx2:
            split_audio_into_segments(missing, self.output_dir)
        self.assertIn("not found", str(ctx2.exception).lower())

    # ---------------------------------------------------------
    # 3. Corrupt FFprobe output
    # ---------------------------------------------------------
    @patch("subprocess.run")
    def test_3_corrupt_ffprobe_output(self, mock_run):
        """Non-numeric, zero, negative, or N/A FFprobe output raises ValueError."""
        # Case A: Non-numeric output
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffprobe"], returncode=0, stdout="NOT_A_NUMBER\n", stderr=""
        )
        with self.assertRaises(ValueError) as ctx:
            get_media_duration(self.dummy_media)
        self.assertIn("non-numeric duration", str(ctx.exception).lower())

        # Case B: Zero duration
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffprobe"], returncode=0, stdout="0.000\n", stderr=""
        )
        with self.assertRaises(ValueError) as ctx:
            get_media_duration(self.dummy_media)
        self.assertIn("must be a positive finite number", str(ctx.exception).lower())

        # Case C: Negative duration
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffprobe"], returncode=0, stdout="-12.5\n", stderr=""
        )
        with self.assertRaises(ValueError) as ctx:
            get_media_duration(self.dummy_media)
        self.assertIn("must be a positive finite number", str(ctx.exception).lower())

        # Case D: Empty output
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffprobe"], returncode=0, stdout="N/A\n", stderr=""
        )
        with self.assertRaises(ValueError) as ctx:
            get_media_duration(self.dummy_media)
        self.assertIn("could not extract duration", str(ctx.exception).lower())

    # ---------------------------------------------------------
    # 4. Missing FFprobe executable
    # ---------------------------------------------------------
    @patch("subprocess.run", side_effect=FileNotFoundError("ffprobe executable not found"))
    def test_4_missing_ffprobe_executable(self, mock_run):
        """Missing FFprobe executable raises RuntimeError."""
        with self.assertRaises(RuntimeError) as ctx:
            get_media_duration(self.dummy_media)
        self.assertIn("ffprobe is not installed or not available", str(ctx.exception).lower())

    # ---------------------------------------------------------
    # 5. FFprobe timeout
    # ---------------------------------------------------------
    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=30))
    def test_5_ffprobe_timeout(self, mock_run):
        """FFprobe timeout raises TimeoutError."""
        with self.assertRaises(TimeoutError) as ctx:
            get_media_duration(self.dummy_media)
        self.assertIn("timed out", str(ctx.exception).lower())

    # ---------------------------------------------------------
    # 6. Correct number of segments
    # ---------------------------------------------------------
    @patch("app.services.audio_service.get_media_duration", return_value=1500.0)
    @patch("subprocess.run")
    def test_6_correct_number_of_segments(self, mock_run, mock_duration):
        """1500-second media with 600s slices and 4s overlap produces exactly 3 segments."""
        def fake_ffmpeg(cmd, **kwargs):
            out_file = Path(cmd[-1])
            out_file.write_bytes(b"RIFFWAVE" + b"\x00" * 1024)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_ffmpeg

        segments = split_audio_into_segments(
            audio_path=self.dummy_media,
            output_dir=self.output_dir,
            segment_duration=600.0,
            overlap=4.0,
        )

        self.assertEqual(len(segments), 3)

    # ---------------------------------------------------------
    # 7. Correct global start and end offsets
    # ---------------------------------------------------------
    @patch("app.services.audio_service.get_media_duration", return_value=1500.0)
    @patch("subprocess.run")
    def test_7_correct_global_start_and_end_offsets(self, mock_run, mock_duration):
        """Segments have exact global start and end offsets across timeline."""
        def fake_ffmpeg(cmd, **kwargs):
            out_file = Path(cmd[-1])
            out_file.write_bytes(b"RIFFWAVE" + b"\x00" * 1024)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_ffmpeg

        segments = split_audio_into_segments(
            audio_path=self.dummy_media,
            output_dir=self.output_dir,
            segment_duration=600.0,
            overlap=4.0,
        )

        # Segment 0: 0.0 -> 604.0
        self.assertEqual(segments[0][1], 0.0)
        self.assertEqual(segments[0][2], 604.0)

        # Segment 1: 600.0 -> 1204.0
        self.assertEqual(segments[1][1], 600.0)
        self.assertEqual(segments[1][2], 1204.0)

        # Segment 2: 1200.0 -> 1500.0
        self.assertEqual(segments[2][1], 1200.0)
        self.assertEqual(segments[2][2], 1500.0)

    # ---------------------------------------------------------
    # 8. Four-second overlap between consecutive segments
    # ---------------------------------------------------------
    @patch("app.services.audio_service.get_media_duration", return_value=1500.0)
    @patch("subprocess.run")
    def test_8_four_second_overlap_between_consecutive_segments(self, mock_run, mock_duration):
        """Consecutive segments share exactly the configured 4-second overlap."""
        def fake_ffmpeg(cmd, **kwargs):
            out_file = Path(cmd[-1])
            out_file.write_bytes(b"RIFFWAVE" + b"\x00" * 1024)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_ffmpeg

        segments = split_audio_into_segments(
            audio_path=self.dummy_media,
            output_dir=self.output_dir,
            segment_duration=600.0,
            overlap=4.0,
        )

        # Overlap between seg 0 and seg 1: [600.0, 604.0] -> 4.0 seconds
        overlap_0_1 = segments[0][2] - segments[1][1]
        self.assertAlmostEqual(overlap_0_1, 4.0, places=3)

        # Overlap between seg 1 and seg 2: [1200.0, 1204.0] -> 4.0 seconds
        overlap_1_2 = segments[1][2] - segments[2][1]
        self.assertAlmostEqual(overlap_1_2, 4.0, places=3)

    # ---------------------------------------------------------
    # 9. Final segment ending at actual duration
    # ---------------------------------------------------------
    @patch("app.services.audio_service.get_media_duration", return_value=753.25)
    @patch("subprocess.run")
    def test_9_final_segment_ending_at_actual_duration(self, mock_run, mock_duration):
        """Final segment ends at exact total media duration, not an arbitrary interval."""
        def fake_ffmpeg(cmd, **kwargs):
            out_file = Path(cmd[-1])
            out_file.write_bytes(b"RIFFWAVE" + b"\x00" * 1024)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_ffmpeg

        segments = split_audio_into_segments(
            audio_path=self.dummy_media,
            output_dir=self.output_dir,
            segment_duration=600.0,
            overlap=4.0,
        )

        self.assertEqual(len(segments), 2)
        # Seg 0: 0.0 -> 604.0
        self.assertEqual(segments[0][1], 0.0)
        self.assertEqual(segments[0][2], 604.0)
        # Seg 1: 600.0 -> 753.25
        self.assertEqual(segments[1][1], 600.0)
        self.assertEqual(segments[1][2], 753.25)

    # ---------------------------------------------------------
    # 10. Output WAV normalization arguments
    # ---------------------------------------------------------
    @patch("app.services.audio_service.get_media_duration", return_value=300.0)
    @patch("subprocess.run")
    def test_10_output_wav_normalization_arguments(self, mock_run, mock_duration):
        """FFmpeg command includes strict 16 kHz mono PCM 16-bit parameters."""
        def fake_ffmpeg(cmd, **kwargs):
            out_file = Path(cmd[-1])
            out_file.write_bytes(b"RIFFWAVE" + b"\x00" * 1024)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_ffmpeg

        split_audio_into_segments(
            audio_path=self.dummy_media,
            output_dir=self.output_dir,
            segment_duration=600.0,
            overlap=4.0,
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("-vn", cmd)
        self.assertIn("-acodec", cmd)
        idx_codec = cmd.index("-acodec")
        self.assertEqual(cmd[idx_codec + 1], "pcm_s16le")

        self.assertIn("-ar", cmd)
        idx_ar = cmd.index("-ar")
        self.assertEqual(cmd[idx_ar + 1], "16000")

        self.assertIn("-ac", cmd)
        idx_ac = cmd.index("-ac")
        self.assertEqual(cmd[idx_ac + 1], "1")

        self.assertIn("-ss", cmd)
        self.assertIn("-t", cmd)

    # ---------------------------------------------------------
    # 11. Oversized segment rejection
    # ---------------------------------------------------------
    @patch("app.services.audio_service.get_media_duration", return_value=1200.0)
    @patch("subprocess.run")
    def test_11_oversized_segment_rejection(self, mock_run, mock_duration):
        """Generated segment exceeding MAX_SEGMENT_SIZE_BYTES (20 MB) raises ValueError and is deleted."""
        def fake_oversized_ffmpeg(cmd, **kwargs):
            out_file = Path(cmd[-1])
            # Write 21,000,000 bytes (> 20 MB limit)
            out_file.write_bytes(b"X" * (MAX_SEGMENT_SIZE_BYTES + 500))
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_oversized_ffmpeg

        with self.assertRaises(ValueError) as ctx:
            split_audio_into_segments(
                audio_path=self.dummy_media,
                output_dir=self.output_dir,
                segment_duration=600.0,
            )

        self.assertIn("exceeds maximum allowed safety limit", str(ctx.exception).lower())

        # Confirm the oversized file was cleaned up
        remaining_files = list(self.output_dir.glob("*.wav"))
        self.assertEqual(len(remaining_files), 0)

    # ---------------------------------------------------------
    # 12. FFmpeg failure cleanup
    # ---------------------------------------------------------
    @patch("app.services.audio_service.get_media_duration", return_value=1800.0)
    @patch("subprocess.run")
    def test_12_ffmpeg_failure_cleanup(self, mock_run, mock_duration):
        """Failure on segment 2 cleans up segment 0, segment 1, and any partial files."""
        call_count = 0

        def failing_ffmpeg(cmd, **kwargs):
            nonlocal call_count
            out_file = Path(cmd[-1])
            call_count += 1
            if call_count <= 2:
                out_file.write_bytes(b"RIFFWAVE" + b"\x00" * 1024)
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            else:
                out_file.write_bytes(b"PARTIAL_FAIL")
                return subprocess.CompletedProcess(
                    args=cmd, returncode=1, stdout="", stderr="Simulated FFmpeg disk full"
                )

        mock_run.side_effect = failing_ffmpeg

        with self.assertRaises(RuntimeError) as ctx:
            split_audio_into_segments(
                audio_path=self.dummy_media,
                output_dir=self.output_dir,
                segment_duration=600.0,
            )

        self.assertIn("ffmpeg failed to extract segment", str(ctx.exception).lower())

        # Confirm all 3 segments were purged on failure
        remaining_files = list(self.output_dir.glob("*.wav"))
        self.assertEqual(len(remaining_files), 0)

    # ---------------------------------------------------------
    # 13. Preservation of original input media
    # ---------------------------------------------------------
    @patch("app.services.audio_service.get_media_duration", return_value=1200.0)
    @patch("subprocess.run")
    def test_13_preservation_of_original_input_media(self, mock_run, mock_duration):
        """Segmentation failure never deletes or modifies the source media file."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=1, stdout="", stderr="Corrupted stream"
        )

        original_bytes = self.dummy_media.read_bytes()

        with self.assertRaises(RuntimeError):
            split_audio_into_segments(
                audio_path=self.dummy_media,
                output_dir=self.output_dir,
            )

        # Source media must still exist and be completely unaltered
        self.assertTrue(self.dummy_media.is_file())
        self.assertEqual(self.dummy_media.read_bytes(), original_bytes)

    # ---------------------------------------------------------
    # 14. Preservation of unrelated files in output directory
    # ---------------------------------------------------------
    @patch("app.services.audio_service.get_media_duration", return_value=1200.0)
    @patch("subprocess.run")
    def test_14_preservation_of_unrelated_files_in_output_dir(self, mock_run, mock_duration):
        """Cleanup on error only removes files generated during this operation, preserving other files."""
        unrelated_file = self.output_dir / "unrelated_document.txt"
        unrelated_file.write_text("IMPORTANT_USER_NOTES")

        unrelated_wav = self.output_dir / "pre_existing_unrelated.wav"
        unrelated_wav.write_bytes(b"OTHER_RECORDING")

        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=1, stdout="", stderr="Segmentation error"
        )

        with self.assertRaises(RuntimeError):
            split_audio_into_segments(
                audio_path=self.dummy_media,
                output_dir=self.output_dir,
            )

        # Unrelated files must still exist untouched
        self.assertTrue(unrelated_file.is_file())
        self.assertEqual(unrelated_file.read_text(), "IMPORTANT_USER_NOTES")

        self.assertTrue(unrelated_wav.is_file())
        self.assertEqual(unrelated_wav.read_bytes(), b"OTHER_RECORDING")

    # ---------------------------------------------------------
    # 15. Invalid duration and overlap configuration
    # ---------------------------------------------------------
    def test_15_invalid_duration_and_overlap_configuration(self):
        """Invalid segment_duration and overlap values raise TypeError or ValueError."""
        # Type errors
        with self.assertRaises(TypeError):
            split_audio_into_segments(self.dummy_media, self.output_dir, segment_duration="600")

        with self.assertRaises(TypeError):
            split_audio_into_segments(self.dummy_media, self.output_dir, segment_duration=True)

        with self.assertRaises(TypeError):
            split_audio_into_segments(self.dummy_media, self.output_dir, overlap="4")

        with self.assertRaises(TypeError):
            split_audio_into_segments(self.dummy_media, self.output_dir, overlap=False)

        # Value errors: non-positive segment_duration
        with self.assertRaises(ValueError):
            split_audio_into_segments(self.dummy_media, self.output_dir, segment_duration=0)

        with self.assertRaises(ValueError):
            split_audio_into_segments(self.dummy_media, self.output_dir, segment_duration=-100)

        # Value errors: negative overlap
        with self.assertRaises(ValueError):
            split_audio_into_segments(self.dummy_media, self.output_dir, overlap=-1.0)

        # Value errors: overlap >= segment_duration
        with self.assertRaises(ValueError) as ctx:
            split_audio_into_segments(self.dummy_media, self.output_dir, segment_duration=10.0, overlap=10.0)
        self.assertIn("must be strictly less than", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx2:
            split_audio_into_segments(self.dummy_media, self.output_dir, segment_duration=10.0, overlap=15.0)
        self.assertIn("must be strictly less than", str(ctx2.exception).lower())

    # ---------------------------------------------------------
    # 16. Integration test: Real FFmpeg and FFprobe with synthetic audio
    # ---------------------------------------------------------
    def test_16_synthetic_audio_ffmpeg_integration(self):
        """Integration test verifying real FFprobe and FFmpeg on a synthetic 4-second WAV file."""
        synthetic_wav = self.temp_path / "synthetic_audio.wav"

        # Generate a real 4-second 16 kHz mono 16-bit PCM WAV file
        sample_rate = 16000
        duration_s = 4.0
        total_samples = int(sample_rate * duration_s)

        with wave.open(str(synthetic_wav), "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(sample_rate)
            # Generate sine wave pulses
            raw_frames = bytearray()
            for i in range(total_samples):
                val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))
                raw_frames.extend(struct.pack("<h", val))
            wav_out.writeframes(raw_frames)

        self.assertTrue(synthetic_wav.is_file())

        # 1. Test real FFprobe duration extraction
        probed_duration = get_media_duration(synthetic_wav)
        self.assertAlmostEqual(probed_duration, 4.0, places=1)

        # 2. Test real FFmpeg segmentation with 2.0s duration and 0.5s overlap
        segments = split_audio_into_segments(
            audio_path=synthetic_wav,
            output_dir=self.output_dir,
            segment_duration=2.0,
            overlap=0.5,
        )

        # Expecting:
        # Seg 0: [0.0, 2.5]
        # Seg 1: [2.0, 4.0]
        self.assertEqual(len(segments), 2)

        seg0_path, seg0_start, seg0_end = segments[0]
        seg1_path, seg1_start, seg1_end = segments[1]

        self.assertEqual(seg0_start, 0.0)
        self.assertEqual(seg0_end, 2.5)

        self.assertEqual(seg1_start, 2.0)
        self.assertAlmostEqual(seg1_end, 4.0, places=1)

        # Confirm files exist on disk and are valid WAV files
        self.assertTrue(seg0_path.is_file())
        self.assertTrue(seg1_path.is_file())

        for p in [seg0_path, seg1_path]:
            with wave.open(str(p), "rb") as w:
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getsampwidth(), 2)
                self.assertEqual(w.getframerate(), 16000)

        # Confirm file size < 20 MB
        self.assertLess(seg0_path.stat().st_size, MAX_SEGMENT_SIZE_BYTES)
        self.assertLess(seg1_path.stat().st_size, MAX_SEGMENT_SIZE_BYTES)


if __name__ == "__main__":
    unittest.main()
