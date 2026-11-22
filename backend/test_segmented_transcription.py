"""
Phase 3.2 — Tests for transcribe_segmented_audio()

All 20 tests use mocks exclusively. No real Groq API calls are made.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Path bootstrap: allow running from repository root or backend/ directory
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.chunking_service import chunk_transcript
from app.services.transcription_service import (
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    transcribe_segmented_audio,
)

from fastapi import HTTPException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment_file(tmp_dir: str, name: str = "seg.wav", size: int = 1024) -> Path:
    """Create a small dummy WAV file in tmp_dir and return its Path."""
    path = Path(tmp_dir) / name
    path.write_bytes(b"\x00" * size)
    return path


def _make_raw_result(
    text: str = "hello",
    seg_start: float = 0.0,
    seg_end: float = 5.0,
    word_start: float = 0.0,
    word_end: float = 2.0,
    word: str = "hello",
    include_words: bool = True,
) -> Dict[str, Any]:
    """Build a minimal transcribe_audio()-style return value."""
    result: Dict[str, Any] = {
        "text": text,
        "segments": [{"id": 0, "start": seg_start, "end": seg_end, "text": text}],
        "words": [],
    }
    if include_words:
        result["words"] = [{"word": word, "start": word_start, "end": word_end}]
    return result


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestTranscribeSegmentedAudio(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    # ------------------------------------------------------------------
    # test_01 — single segment, verify timestamp passthrough
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_01_single_segment_transcription(self, mock_ta: MagicMock) -> None:
        """Single segment: translated timestamps equal global_start + local."""
        path = _make_segment_file(self.tmp, "seg0.wav")
        mock_ta.return_value = _make_raw_result(
            text="hello world",
            seg_start=1.0,
            seg_end=4.0,
            word_start=1.0,
            word_end=3.0,
        )

        result = transcribe_segmented_audio([(path, 10.0, 20.0)])

        mock_ta.assert_called_once_with(path)
        self.assertEqual(result["text"], "hello world")
        self.assertEqual(len(result["segments"]), 1)
        self.assertAlmostEqual(result["segments"][0]["start"], 11.0)
        self.assertAlmostEqual(result["segments"][0]["end"], 14.0)
        self.assertEqual(len(result["words"]), 1)
        self.assertAlmostEqual(result["words"][0]["start"], 11.0)
        self.assertAlmostEqual(result["words"][0]["end"], 13.0)

    # ------------------------------------------------------------------
    # test_02 — multiple segments merge
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_02_multiple_segment_transcription(self, mock_ta: MagicMock) -> None:
        """3 segments: all sub-segments and words appear in merged output."""
        paths = [_make_segment_file(self.tmp, f"seg{i}.wav") for i in range(3)]
        mock_ta.side_effect = [
            {
                "text": "alpha",
                "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "alpha"}],
                "words": [{"word": "alpha", "start": 0.0, "end": 5.0}],
            },
            {
                "text": "beta",
                "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "beta"}],
                "words": [{"word": "beta", "start": 0.0, "end": 5.0}],
            },
            {
                "text": "gamma",
                "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "gamma"}],
                "words": [{"word": "gamma", "start": 0.0, "end": 5.0}],
            },
        ]
        # Segment tuples: 0-600, 600-1200, 1200-1500
        segs: List[Tuple[Path, float, float]] = [
            (paths[0], 0.0, 604.0),
            (paths[1], 600.0, 1204.0),
            (paths[2], 1200.0, 1500.0),
        ]
        result = transcribe_segmented_audio(segs)

        self.assertEqual(mock_ta.call_count, 3)
        # Each segment contributes 1 sub-segment
        self.assertEqual(len(result["segments"]), 3)
        # Words: seg0 word at 0.0 < boundary 600 → kept;
        #        seg1 word at 600.0 >= boundary 1200 → kept (600 < 1200);
        #        seg2 word at 1200.0 (last) → kept
        self.assertEqual(len(result["words"]), 3)

    # ------------------------------------------------------------------
    # test_03 — global offsets applied to segment timestamps
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_03_correct_global_offsets_for_segments(self, mock_ta: MagicMock) -> None:
        """Translated segment timestamps = global_start + local timestamps."""
        path = _make_segment_file(self.tmp, "seg0.wav")
        mock_ta.return_value = {
            "text": "test",
            "segments": [{"id": 0, "start": 2.5, "end": 8.0, "text": "test"}],
            "words": [{"word": "test", "start": 2.5, "end": 8.0}],
        }
        result = transcribe_segmented_audio([(path, 300.0, 904.0)])

        seg = result["segments"][0]
        self.assertAlmostEqual(seg["start"], 302.5)
        self.assertAlmostEqual(seg["end"], 308.0)

    # ------------------------------------------------------------------
    # test_04 — global offsets applied to word timestamps
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_04_correct_global_offsets_for_words(self, mock_ta: MagicMock) -> None:
        """Translated word timestamps = global_start + local timestamps."""
        path = _make_segment_file(self.tmp, "seg0.wav")
        mock_ta.return_value = {
            "text": "quick fox",
            "segments": [{"id": 0, "start": 0.5, "end": 3.0, "text": "quick fox"}],
            "words": [
                {"word": "quick", "start": 0.5, "end": 1.5},
                {"word": "fox", "start": 1.8, "end": 3.0},
            ],
        }
        result = transcribe_segmented_audio([(path, 1200.0, 1800.0)])

        self.assertAlmostEqual(result["words"][0]["start"], 1200.5)
        self.assertAlmostEqual(result["words"][0]["end"], 1201.5)
        self.assertAlmostEqual(result["words"][1]["start"], 1201.8)
        self.assertAlmostEqual(result["words"][1]["end"], 1203.0)

    # ------------------------------------------------------------------
    # test_05 — merged segment IDs are monotonically increasing from 0
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_05_monotonically_increasing_segment_ids(self, mock_ta: MagicMock) -> None:
        """After merge, all segment IDs are 0, 1, 2, ... regardless of original IDs."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(3)]
        mock_ta.side_effect = [
            {
                "text": f"text{i}",
                "segments": [{"id": 99, "start": 0.0, "end": 5.0, "text": f"text{i}"}],
                "words": [{"word": f"word{i}", "start": 0.0, "end": 5.0}],
            }
            for i in range(3)
        ]
        segs: List[Tuple[Path, float, float]] = [
            (paths[0], 0.0, 604.0),
            (paths[1], 600.0, 1204.0),
            (paths[2], 1200.0, 1500.0),
        ]
        result = transcribe_segmented_audio(segs)

        ids = [s["id"] for s in result["segments"]]
        self.assertEqual(ids, list(range(len(ids))))

    # ------------------------------------------------------------------
    # test_06 — merged output is sorted chronologically by start time
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_06_chronological_ordering(self, mock_ta: MagicMock) -> None:
        """Merged segments are sorted by (start, end)."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(2)]
        mock_ta.side_effect = [
            {
                "text": "b a",
                "segments": [
                    {"id": 1, "start": 3.0, "end": 5.0, "text": "b"},
                    {"id": 0, "start": 0.0, "end": 2.0, "text": "a"},
                ],
                "words": [
                    {"word": "b", "start": 3.0, "end": 5.0},
                    {"word": "a", "start": 0.0, "end": 2.0},
                ],
            },
            {
                "text": "c",
                "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "c"}],
                "words": [{"word": "c", "start": 0.0, "end": 3.0}],
            },
        ]
        segs: List[Tuple[Path, float, float]] = [
            (paths[0], 0.0, 604.0),
            (paths[1], 600.0, 900.0),
        ]
        result = transcribe_segmented_audio(segs)

        starts = [s["start"] for s in result["segments"]]
        self.assertEqual(starts, sorted(starts))

    # ------------------------------------------------------------------
    # test_07 — word before boundary is kept; duplicate in next seg dropped
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_07_duplicate_word_removal_at_boundary(self, mock_ta: MagicMock) -> None:
        """Words from seg 0 that fall at or after the boundary are dropped from seg 0.

        The boundary between seg 0 and seg 1 is segments[1][1] (global_start of seg 1 = 600.0).
        Words in seg 0 with translated start >= 600.0 are dropped.
        Words in the overlap region are only carried by seg 1 (which owns from 600.0 onward).
        This prevents seg 0 from contributing duplicate content beyond its nominal boundary.
        """
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(2)]
        # Boundary between seg 0 and seg 1 = 600.0
        mock_ta.side_effect = [
            {
                # Seg 0: local timestamps relative to seg 0's start (global 0.0)
                # "early" at local 598.0 → global 598.0 < 600.0 → KEPT by seg 0
                # "overlap_word" at local 600.5 → global 600.5 >= 600.0 → DROPPED from seg 0
                "text": "early overlap_word",
                "segments": [
                    {"id": 0, "start": 598.0, "end": 600.0, "text": "early"},
                    {"id": 1, "start": 600.5, "end": 602.0, "text": "overlap_word"},
                ],
                "words": [
                    {"word": "early", "start": 598.0, "end": 600.0},
                    {"word": "overlap_word", "start": 600.5, "end": 602.0},
                ],
            },
            {
                # Seg 1 (last): all words kept. "overlap_word" at local 0.5 → global 600.5
                # "continuing" at local 2.0 → global 602.0
                "text": "overlap_word continuing",
                "segments": [
                    {"id": 0, "start": 0.5, "end": 2.0, "text": "overlap_word"},
                    {"id": 1, "start": 2.0, "end": 5.0, "text": "continuing"},
                ],
                "words": [
                    {"word": "overlap_word", "start": 0.5, "end": 2.0},
                    {"word": "continuing", "start": 2.0, "end": 5.0},
                ],
            },
        ]
        segs: List[Tuple[Path, float, float]] = [
            (paths[0], 0.0, 604.0),
            (paths[1], 600.0, 900.0),
        ]
        result = transcribe_segmented_audio(segs)

        word_texts = [w["word"] for w in result["words"]]
        # "early" from seg 0 (global 598.0 < 600.0) → KEPT
        self.assertIn("early", word_texts)
        # "overlap_word" from seg 0 (global 600.5 >= 600.0) → DROPPED from seg 0
        # "overlap_word" from seg 1 (last segment) → KEPT
        # So exactly 1 "overlap_word" total
        self.assertEqual(word_texts.count("overlap_word"), 1)
        # Verify the "overlap_word" that survived came from seg 1 (global_start=600.0, local=0.5)
        ow = next(w for w in result["words"] if w["word"] == "overlap_word")
        self.assertAlmostEqual(ow["start"], 600.5)  # 600.0 + 0.5
        self.assertIn("continuing", word_texts)

    # ------------------------------------------------------------------
    # test_08 — word starting before boundary but ending after it → kept (belongs to seg 0)
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_08_word_before_boundary_ending_after_it(self, mock_ta: MagicMock) -> None:
        """Word starting at 598s, ending at 601s → start < 600 → belongs to seg 0."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(2)]
        mock_ta.side_effect = [
            {
                "text": "spanning",
                "segments": [{"id": 0, "start": 598.0, "end": 601.0, "text": "spanning"}],
                "words": [{"word": "spanning", "start": 598.0, "end": 601.0}],
            },
            {
                "text": "next",
                "segments": [{"id": 0, "start": 1.0, "end": 3.0, "text": "next"}],
                "words": [{"word": "next", "start": 1.0, "end": 3.0}],
            },
        ]
        segs: List[Tuple[Path, float, float]] = [
            (paths[0], 0.0, 604.0),
            (paths[1], 600.0, 900.0),
        ]
        result = transcribe_segmented_audio(segs)

        word_texts = [w["word"] for w in result["words"]]
        self.assertIn("spanning", word_texts)
        # Find "spanning" word and verify it was translated from seg 0
        spanning_word = next(w for w in result["words"] if w["word"] == "spanning")
        # Translated start = 0.0 + 598.0 = 598.0 (seg 0 global_start = 0.0)
        self.assertAlmostEqual(spanning_word["start"], 598.0)

    # ------------------------------------------------------------------
    # test_09 — word exactly at boundary belongs to next segment
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_09_word_exactly_at_boundary(self, mock_ta: MagicMock) -> None:
        """Word starting at exactly 600.0 (boundary) → belongs to seg 1, dropped from seg 0."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(2)]
        mock_ta.side_effect = [
            {
                "text": "boundary",
                "segments": [{"id": 0, "start": 600.0, "end": 602.0, "text": "boundary"}],
                # In seg 0's local coords: start=600.0, end=602.0
                # Translated: 0.0 + 600.0 = 600.0 >= boundary 600.0 → dropped from seg 0
                "words": [{"word": "boundary", "start": 600.0, "end": 602.0}],
            },
            {
                "text": "boundary",
                "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "boundary"}],
                # In seg 1 local: 0.0, translated: 600.0 + 0.0 = 600.0
                "words": [{"word": "boundary", "start": 0.0, "end": 2.0}],
            },
        ]
        segs: List[Tuple[Path, float, float]] = [
            (paths[0], 0.0, 604.0),
            (paths[1], 600.0, 900.0),
        ]
        result = transcribe_segmented_audio(segs)

        word_texts = [w["word"] for w in result["words"]]
        # Only 1 "boundary" word should appear (from seg 1); seg 0's copy dropped
        self.assertEqual(word_texts.count("boundary"), 1)
        boundary_word = next(w for w in result["words"] if w["word"] == "boundary")
        # It comes from seg 1: global_start=600.0, local start=0.0 → translated 600.0
        self.assertAlmostEqual(boundary_word["start"], 600.0)

    # ------------------------------------------------------------------
    # test_10 — final partial segment handled correctly (no upper boundary)
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_10_final_partial_segment_handling(self, mock_ta: MagicMock) -> None:
        """Final (shorter) segment owns all its words unconditionally."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(2)]
        mock_ta.side_effect = [
            {
                "text": "first",
                "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "first"}],
                "words": [{"word": "first", "start": 0.0, "end": 5.0}],
            },
            {
                "text": "last words here",
                "segments": [
                    {"id": 0, "start": 0.0, "end": 2.0, "text": "last words"},
                    {"id": 1, "start": 2.0, "end": 90.0, "text": "here"},
                ],
                "words": [
                    {"word": "last", "start": 0.0, "end": 1.0},
                    {"word": "words", "start": 1.0, "end": 2.0},
                    {"word": "here", "start": 2.0, "end": 90.0},
                ],
            },
        ]
        segs: List[Tuple[Path, float, float]] = [
            (paths[0], 0.0, 604.0),
            (paths[1], 600.0, 694.0),  # Final segment, clamped duration
        ]
        result = transcribe_segmented_audio(segs)

        # All 3 words from final segment must be present
        word_texts = [w["word"] for w in result["words"]]
        self.assertIn("last", word_texts)
        self.assertIn("words", word_texts)
        self.assertIn("here", word_texts)

    # ------------------------------------------------------------------
    # test_11 — empty successful segment does not abort
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_11_empty_successful_segment(self, mock_ta: MagicMock) -> None:
        """A segment returning empty text/segments/words is a valid success."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(2)]
        mock_ta.side_effect = [
            {"text": "", "segments": [], "words": []},  # silence segment
            {
                "text": "hello",
                "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "hello"}],
                "words": [{"word": "hello", "start": 0.0, "end": 3.0}],
            },
        ]
        segs: List[Tuple[Path, float, float]] = [
            (paths[0], 0.0, 604.0),
            (paths[1], 600.0, 900.0),
        ]
        result = transcribe_segmented_audio(segs)

        self.assertEqual(mock_ta.call_count, 2)
        self.assertIn("hello", result["text"])
        self.assertEqual(len(result["segments"]), 1)

    # ------------------------------------------------------------------
    # test_12 — missing word timestamps trigger segment-level fallback dedup
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_12_missing_word_timestamps_fallback(self, mock_ta: MagicMock) -> None:
        """When no words are returned, segment-level dedup is used at boundary."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(2)]
        # Boundary = 600.0
        mock_ta.side_effect = [
            {
                "text": "early late",
                "segments": [
                    {"id": 0, "start": 595.0, "end": 599.0, "text": "early"},   # < 600 → kept
                    {"id": 1, "start": 600.0, "end": 604.0, "text": "late"},    # >= 600 → dropped
                ],
                "words": [],  # No words → segment-level fallback
            },
            {
                "text": "late more",
                "segments": [
                    {"id": 0, "start": 0.0, "end": 4.0, "text": "late"},        # → 600.0 (last seg, all kept)
                    {"id": 1, "start": 4.0, "end": 8.0, "text": "more"},
                ],
                "words": [],
            },
        ]
        segs: List[Tuple[Path, float, float]] = [
            (paths[0], 0.0, 604.0),
            (paths[1], 600.0, 900.0),
        ]
        result = transcribe_segmented_audio(segs)

        texts = [s["text"] for s in result["segments"]]
        # "early" (start=595 < 600) kept from seg 0
        self.assertIn("early", texts)
        # "late" from seg 0 (start=600 >= 600) dropped by fallback
        # "late" from seg 1 (start=600, last seg) kept
        late_count = texts.count("late")
        self.assertEqual(late_count, 1)
        # "more" from seg 1 always kept
        self.assertIn("more", texts)

    # ------------------------------------------------------------------
    # test_13 — oversized segment raises ValueError before any API call
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_13_oversized_segment_rejected_before_api_upload(self, mock_ta: MagicMock) -> None:
        """File > MAX_SEGMENT_UPLOAD_BYTES raises ValueError; transcribe_audio not called."""
        from app.services.transcription_service import MAX_SEGMENT_UPLOAD_BYTES

        path = _make_segment_file(self.tmp, "big.wav", size=1)
        # Patch stat().st_size to simulate oversized file
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=MAX_SEGMENT_UPLOAD_BYTES + 1)
            with patch.object(Path, "is_file", return_value=True):
                with self.assertRaises(ValueError) as ctx:
                    transcribe_segmented_audio([(path, 0.0, 600.0)])

        self.assertIn("MAX_SEGMENT_UPLOAD_BYTES", str(ctx.exception) + "size")
        mock_ta.assert_not_called()

    # ------------------------------------------------------------------
    # test_14 — invalid segment tuple raises TypeError / ValueError
    # ------------------------------------------------------------------
    def test_14_invalid_segment_tuple_rejected(self) -> None:
        """Non-tuple, wrong-length tuple, and wrong types raise TypeError/ValueError."""
        path = _make_segment_file(self.tmp, "seg.wav")

        # Non-list
        with self.assertRaises(TypeError):
            transcribe_segmented_audio("not a list")  # type: ignore[arg-type]

        # Empty list
        with self.assertRaises(ValueError):
            transcribe_segmented_audio([])

        # Non-tuple element
        with self.assertRaises(TypeError):
            transcribe_segmented_audio(["not_a_tuple"])  # type: ignore[list-item]

        # Wrong-length tuple
        with self.assertRaises(TypeError):
            transcribe_segmented_audio([(path, 0.0)])  # type: ignore[list-item]

        # Non-numeric global_start
        with self.assertRaises(TypeError):
            transcribe_segmented_audio([(path, "bad", 600.0)])  # type: ignore[list-item]

        # global_end <= global_start
        with self.assertRaises(ValueError):
            transcribe_segmented_audio([(path, 600.0, 100.0)])

    # ------------------------------------------------------------------
    # test_15 — failure on one segment aborts the entire merge
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_15_segment_failure_aborts_merged_transcription(self, mock_ta: MagicMock) -> None:
        """If segment 2 of 3 fails permanently, the exception propagates immediately."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(3)]
        mock_ta.side_effect = [
            {
                "text": "first",
                "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "first"}],
                "words": [{"word": "first", "start": 0.0, "end": 5.0}],
            },
            HTTPException(status_code=404, detail="not found"),  # permanent error
            {
                "text": "third",
                "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "third"}],
                "words": [{"word": "third", "start": 0.0, "end": 5.0}],
            },
        ]
        segs: List[Tuple[Path, float, float]] = [
            (paths[0], 0.0, 604.0),
            (paths[1], 600.0, 1204.0),
            (paths[2], 1200.0, 1500.0),
        ]
        with self.assertRaises(HTTPException) as ctx:
            transcribe_segmented_audio(segs)

        self.assertEqual(ctx.exception.status_code, 404)
        # Third segment was never attempted
        self.assertEqual(mock_ta.call_count, 2)

    # ------------------------------------------------------------------
    # test_16 — HTTP 429 is retried with exponential backoff, then raised
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.time")
    @patch("app.services.transcription_service.transcribe_audio")
    def test_16_bounded_retry_on_http_429(self, mock_ta: MagicMock, mock_time: MagicMock) -> None:
        """HTTP 429 triggers MAX_RETRIES retries then propagates the exception."""
        path = _make_segment_file(self.tmp, "seg.wav")
        mock_ta.side_effect = HTTPException(status_code=429, detail="rate limited")

        with self.assertRaises(HTTPException) as ctx:
            transcribe_segmented_audio([(path, 0.0, 600.0)])

        self.assertEqual(ctx.exception.status_code, 429)
        # Total attempts = MAX_RETRIES + 1 (initial + retries)
        self.assertEqual(mock_ta.call_count, MAX_RETRIES + 1)
        # sleep called MAX_RETRIES times (between attempts)
        self.assertEqual(mock_time.sleep.call_count, MAX_RETRIES)

        # Verify exponential backoff values (capped at RETRY_MAX_DELAY)
        expected_delays = [
            min(RETRY_BASE_DELAY * (2 ** i), RETRY_MAX_DELAY)
            for i in range(MAX_RETRIES)
        ]
        actual_delays = [c.args[0] for c in mock_time.sleep.call_args_list]
        for expected, actual in zip(expected_delays, actual_delays):
            self.assertAlmostEqual(expected, actual)

    # ------------------------------------------------------------------
    # test_17 — HTTP 503 retried, then fails
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.time")
    @patch("app.services.transcription_service.transcribe_audio")
    def test_17_bounded_retry_on_5xx_network_failure(self, mock_ta: MagicMock, mock_time: MagicMock) -> None:
        """HTTP 503 triggers retry policy identical to 429."""
        path = _make_segment_file(self.tmp, "seg.wav")
        mock_ta.side_effect = HTTPException(status_code=503, detail="unavailable")

        with self.assertRaises(HTTPException) as ctx:
            transcribe_segmented_audio([(path, 0.0, 600.0)])

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(mock_ta.call_count, MAX_RETRIES + 1)
        self.assertEqual(mock_time.sleep.call_count, MAX_RETRIES)

    # ------------------------------------------------------------------
    # test_18 — HTTP 400/404/413/422 are NOT retried
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.time")
    @patch("app.services.transcription_service.transcribe_audio")
    def test_18_permanent_error_not_retried(self, mock_ta: MagicMock, mock_time: MagicMock) -> None:
        """HTTP 400, 404, 413, 422 fail immediately without any retry."""
        path = _make_segment_file(self.tmp, "seg.wav")

        for status_code in (400, 404, 413, 422):
            mock_ta.reset_mock()
            mock_time.reset_mock()
            mock_ta.side_effect = HTTPException(status_code=status_code, detail="permanent")

            with self.assertRaises(HTTPException) as ctx:
                transcribe_segmented_audio([(path, 0.0, 600.0)])

            self.assertEqual(ctx.exception.status_code, status_code)
            self.assertEqual(mock_ta.call_count, 1, f"Expected 1 attempt for HTTP {status_code}")
            mock_time.sleep.assert_not_called()

    # ------------------------------------------------------------------
    # test_19 — original response dicts from transcribe_audio() are not mutated
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_19_original_response_not_mutated(self, mock_ta: MagicMock) -> None:
        """transcribe_segmented_audio must not modify the dicts returned by transcribe_audio."""
        path = _make_segment_file(self.tmp, "seg.wav")
        original_seg = {"id": 0, "start": 1.0, "end": 3.0, "text": "hi"}
        original_word = {"word": "hi", "start": 1.0, "end": 3.0}
        mock_ta.return_value = {
            "text": "hi",
            "segments": [original_seg],
            "words": [original_word],
        }

        result = transcribe_segmented_audio([(path, 10.0, 20.0)])

        # Translated timestamps in result
        self.assertAlmostEqual(result["segments"][0]["start"], 11.0)
        self.assertAlmostEqual(result["words"][0]["start"], 11.0)

        # Original dicts must still have their original local values
        self.assertAlmostEqual(original_seg["start"], 1.0)
        self.assertAlmostEqual(original_seg["end"], 3.0)
        self.assertAlmostEqual(original_word["start"], 1.0)
        self.assertAlmostEqual(original_word["end"], 3.0)

    # ------------------------------------------------------------------
    # test_20 — output is compatible with chunk_transcript()
    # ------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_20_compatibility_with_chunk_transcript(self, mock_ta: MagicMock) -> None:
        """Output of transcribe_segmented_audio() can be passed to chunk_transcript() without error."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(2)]
        mock_ta.side_effect = [
            {
                "text": "Hello world this is EchoTrace",
                "segments": [
                    {"id": 0, "start": 0.0, "end": 5.0, "text": "Hello world"},
                    {"id": 1, "start": 5.0, "end": 10.0, "text": "this is EchoTrace"},
                ],
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 1.0},
                    {"word": "world", "start": 1.0, "end": 2.0},
                    {"word": "this", "start": 5.0, "end": 6.0},
                    {"word": "is", "start": 6.0, "end": 7.0},
                    {"word": "EchoTrace", "start": 7.0, "end": 10.0},
                ],
            },
            {
                "text": "phase three is live",
                "segments": [
                    {"id": 0, "start": 0.0, "end": 4.0, "text": "phase three is live"},
                ],
                "words": [
                    {"word": "phase", "start": 0.0, "end": 1.0},
                    {"word": "three", "start": 1.0, "end": 2.0},
                    {"word": "is", "start": 2.0, "end": 3.0},
                    {"word": "live", "start": 3.0, "end": 4.0},
                ],
            },
        ]
        segs: List[Tuple[Path, float, float]] = [
            (paths[0], 0.0, 604.0),
            (paths[1], 600.0, 900.0),
        ]
        merged = transcribe_segmented_audio(segs)

        # Must not raise
        chunks = chunk_transcript(merged, max_duration_seconds=30, max_characters=500)

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("text", chunk)
            self.assertIn("start", chunk)
            self.assertIn("end", chunk)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


class TestSubSegmentConsistencyFix(unittest.TestCase):
    """
    Phase 3.3 regression tests for the sub-segment boundary dedup fix.

    Previously, when word-level timestamps were present, sub-segments were
    included unconditionally. This caused overlap-region sub-segments to appear
    in BOTH adjacent audio-segment lists, resulting in duplicate text reaching
    chunk_transcript(). The fix applies the boundary filter to sub-segments
    unconditionally, regardless of whether word timestamps are present.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    # -----------------------------------------------------------------------
    # test_21 — sub-segment starting before boundary is included from seg 0
    # -----------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_sub_segment_before_boundary_included(self, mock_ta: MagicMock) -> None:
        """Sub-segment with translated start=598s < boundary=600s is included from seg 0."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(2)]
        mock_ta.side_effect = [
            {
                "text": "early content",
                "segments": [
                    # local 598.0 → global 598.0 < boundary 600.0 → MUST be included
                    {"id": 0, "start": 598.0, "end": 599.5, "text": "early content"},
                ],
                "words": [{"word": "early", "start": 598.0, "end": 598.8},
                           {"word": "content", "start": 598.8, "end": 599.5}],
            },
            {
                "text": "next",
                "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "next"}],
                "words": [{"word": "next", "start": 0.0, "end": 3.0}],
            },
        ]
        segs = [(paths[0], 0.0, 604.0), (paths[1], 600.0, 900.0)]
        result = transcribe_segmented_audio(segs)

        texts = [s["text"] for s in result["segments"]]
        self.assertIn("early content", texts)

    # -----------------------------------------------------------------------
    # test_22 — sub-segment at boundary=600s excluded from seg 0, present in seg 1
    # -----------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_sub_segment_at_boundary_excluded_from_seg0(self, mock_ta: MagicMock) -> None:
        """Sub-segment starting at exactly boundary (600s) excluded from seg 0 and owned by seg 1."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(2)]
        mock_ta.side_effect = [
            {
                "text": "overlap text",
                "segments": [
                    # local 600.0 → global 600.0 == boundary 600.0 → MUST be excluded from seg 0
                    {"id": 0, "start": 600.0, "end": 602.0, "text": "overlap text"},
                ],
                "words": [{"word": "overlap", "start": 600.0, "end": 601.0},
                           {"word": "text", "start": 601.0, "end": 602.0}],
            },
            {
                "text": "overlap text continued",
                "segments": [
                    # seg 1 owns from 600.0 onward (last seg, no boundary)
                    {"id": 0, "start": 0.0, "end": 2.0, "text": "overlap text"},
                    {"id": 1, "start": 2.0, "end": 5.0, "text": "continued"},
                ],
                "words": [{"word": "overlap", "start": 0.0, "end": 1.0},
                           {"word": "text", "start": 1.0, "end": 2.0},
                           {"word": "continued", "start": 2.0, "end": 5.0}],
            },
        ]
        segs = [(paths[0], 0.0, 604.0), (paths[1], 600.0, 900.0)]
        result = transcribe_segmented_audio(segs)

        texts = [s["text"] for s in result["segments"]]
        # "overlap text" should appear exactly once: from seg 1 (last segment, no boundary)
        # It should NOT appear from seg 0 (its start 600.0 >= boundary 600.0)
        overlap_count = texts.count("overlap text")
        self.assertEqual(overlap_count, 1,
                         f"Expected exactly 1 'overlap text' sub-segment, got {overlap_count}. "
                         f"All texts: {texts}")
        # "continued" from seg 1 (last) is always included
        self.assertIn("continued", texts)

    # -----------------------------------------------------------------------
    # test_23 — sub-segment spanning boundary: start < boundary → included
    # -----------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_sub_segment_spanning_boundary_uses_start_time(self, mock_ta: MagicMock) -> None:
        """Sub-segment starting at 599s (< 600s boundary), ending at 603s → included (start < boundary)."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(2)]
        mock_ta.side_effect = [
            {
                "text": "spanning",
                "segments": [
                    # local 599.0 → global 599.0 < boundary 600.0 → included
                    {"id": 0, "start": 599.0, "end": 603.0, "text": "spanning"},
                ],
                "words": [{"word": "spanning", "start": 599.0, "end": 603.0}],
            },
            {
                "text": "after",
                "segments": [{"id": 0, "start": 3.0, "end": 6.0, "text": "after"}],
                "words": [{"word": "after", "start": 3.0, "end": 6.0}],
            },
        ]
        segs = [(paths[0], 0.0, 604.0), (paths[1], 600.0, 900.0)]
        result = transcribe_segmented_audio(segs)

        texts = [s["text"] for s in result["segments"]]
        self.assertIn("spanning", texts,
                      "Sub-segment starting before boundary must be included from seg 0.")
        # Verify the spanning sub-segment's translated start
        spanning_seg = next(s for s in result["segments"] if s["text"] == "spanning")
        self.assertAlmostEqual(spanning_seg["start"], 599.0)

    # -----------------------------------------------------------------------
    # test_24 — overlap text appears exactly once (not duplicated)
    # -----------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_overlap_text_appears_exactly_once(self, mock_ta: MagicMock) -> None:
        """Same overlap text in both audio segments after merge appears only once (from seg 1)."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(2)]
        # Boundary = 600.0
        # Both audio segments have a sub-segment with text "repeated content" at different local times
        # Seg 0: "repeated content" starts at local 600.5 → global 600.5 >= boundary → EXCLUDED from seg 0
        # Seg 1: "repeated content" starts at local 0.5 → global 600.5 (last segment) → INCLUDED
        mock_ta.side_effect = [
            {
                "text": "early repeated content",
                "segments": [
                    {"id": 0, "start": 595.0, "end": 599.0, "text": "early"},
                    {"id": 1, "start": 600.5, "end": 604.0, "text": "repeated content"},  # >= boundary → excluded
                ],
                "words": [
                    {"word": "early", "start": 595.0, "end": 599.0},
                    {"word": "repeated", "start": 600.5, "end": 602.0},
                    {"word": "content", "start": 602.0, "end": 604.0},
                ],
            },
            {
                "text": "repeated content and more",
                "segments": [
                    {"id": 0, "start": 0.5, "end": 4.0, "text": "repeated content"},  # last seg → included
                    {"id": 1, "start": 4.0, "end": 8.0, "text": "and more"},
                ],
                "words": [
                    {"word": "repeated", "start": 0.5, "end": 2.0},
                    {"word": "content", "start": 2.0, "end": 4.0},
                    {"word": "and", "start": 4.0, "end": 5.0},
                    {"word": "more", "start": 5.0, "end": 8.0},
                ],
            },
        ]
        segs = [(paths[0], 0.0, 604.0), (paths[1], 600.0, 900.0)]
        result = transcribe_segmented_audio(segs)

        texts = [s["text"] for s in result["segments"]]
        repeated_count = texts.count("repeated content")
        self.assertEqual(repeated_count, 1,
                         f"Expected exactly 1 'repeated content' sub-segment, got {repeated_count}. "
                         f"All texts: {texts}")
        self.assertIn("early", texts)
        self.assertIn("and more", texts)

    # -----------------------------------------------------------------------
    # test_25 — merged text, segments, words all describe same content without duplicates
    # -----------------------------------------------------------------------
    @patch("app.services.transcription_service.transcribe_audio")
    def test_merged_text_words_segments_consistent(self, mock_ta: MagicMock) -> None:
        """After merge, text, segments, and words all describe the same content without duplicates."""
        paths = [_make_segment_file(self.tmp, f"s{i}.wav") for i in range(2)]
        # Boundary = 600.0
        # Seg 0: "alpha" (local 0–5) + "beta" (local 600.5–603) — beta is in overlap region, excluded from seg 0
        # Seg 1: "beta" (local 0.5–3) + "gamma" (local 3–6) — last segment, all included
        mock_ta.side_effect = [
            {
                "text": "alpha beta",
                "segments": [
                    {"id": 0, "start": 0.0, "end": 5.0, "text": "alpha"},
                    {"id": 1, "start": 600.5, "end": 603.0, "text": "beta"},  # >= boundary → excluded
                ],
                "words": [
                    {"word": "alpha", "start": 0.0, "end": 5.0},
                    {"word": "beta", "start": 600.5, "end": 603.0},  # >= boundary → excluded
                ],
            },
            {
                "text": "beta gamma",
                "segments": [
                    {"id": 0, "start": 0.5, "end": 3.0, "text": "beta"},   # last → included
                    {"id": 1, "start": 3.0, "end": 6.0, "text": "gamma"},  # last → included
                ],
                "words": [
                    {"word": "beta", "start": 0.5, "end": 3.0},    # last → included
                    {"word": "gamma", "start": 3.0, "end": 6.0},   # last → included
                ],
            },
        ]
        segs = [(paths[0], 0.0, 604.0), (paths[1], 600.0, 900.0)]
        result = transcribe_segmented_audio(segs)

        # Segments should contain alpha, beta, gamma — each exactly once
        seg_texts = [s["text"] for s in result["segments"]]
        self.assertEqual(seg_texts.count("alpha"), 1)
        self.assertEqual(seg_texts.count("beta"), 1)
        self.assertEqual(seg_texts.count("gamma"), 1)

        # Words should contain alpha, beta, gamma — each exactly once
        word_texts = [w["word"] for w in result["words"]]
        self.assertEqual(word_texts.count("alpha"), 1)
        self.assertEqual(word_texts.count("beta"), 1)
        self.assertEqual(word_texts.count("gamma"), 1)

        # Merged text should contain all three words
        merged_text = result["text"]
        self.assertIn("alpha", merged_text)
        self.assertIn("beta", merged_text)
        self.assertIn("gamma", merged_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
