import math
from pathlib import Path
import sys
import unittest

# Ensure 'backend' directory is on sys.path so 'app' imports cleanly
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from pydantic import ValidationError
from app.schemas.vector_store import TranscriptChunkRecord


# =====================================================================
# Reusable Test Fixture Helper
# =====================================================================
def get_valid_record_payload(**overrides) -> dict:
    """
    Returns a valid baseline payload for TranscriptChunkRecord,
    allowing specific fields to be overridden for targeted testing.
    """
    base = {
        "recording_id": "rec-5aa51179-2c02-4def-ab08-015caf455998",
        "chunk_index": 0,
        "text": "This is a clean, timestamped transcript chunk.",
        "start": 0.0,
        "end": 14.5,
        "segment_count": 3,
        "embedding": [0.01] * 768,
    }
    base.update(overrides)
    return base


# =====================================================================
# Valid Records Test Cases (Tests 1 - 4)
# =====================================================================
class TestTranscriptChunkRecordValid(unittest.TestCase):
    def test_valid_record_with_768_dim_embedding(self):
        """1. A valid record with a 768-dimensional embedding."""
        payload = get_valid_record_payload()
        record = TranscriptChunkRecord(**payload)

        self.assertEqual(record.recording_id, payload["recording_id"])
        self.assertEqual(record.chunk_index, 0)
        self.assertEqual(record.text, payload["text"])
        self.assertEqual(record.start, 0.0)
        self.assertEqual(record.end, 14.5)
        self.assertEqual(record.segment_count, 3)
        self.assertEqual(len(record.embedding), 768)

    def test_integer_timestamps_converted_to_floats(self):
        """2. Integer timestamps are accepted and converted to floats."""
        payload = get_valid_record_payload(start=0, end=15)
        record = TranscriptChunkRecord(**payload)

        self.assertIsInstance(record.start, float)
        self.assertIsInstance(record.end, float)
        self.assertEqual(record.start, 0.0)
        self.assertEqual(record.end, 15.0)

    def test_whitespace_stripped(self):
        """3. Whitespace around recording_id and text is stripped."""
        payload = get_valid_record_payload(
            recording_id="   padded-recording-id   ",
            text="   \n  Padded text with surrounding whitespace.  \t  ",
        )
        record = TranscriptChunkRecord(**payload)

        self.assertEqual(record.recording_id, "padded-recording-id")
        self.assertEqual(record.text, "Padded text with surrounding whitespace.")

    def test_start_equals_end_accepted(self):
        """4. A timestamp where start equals end is accepted."""
        payload = get_valid_record_payload(start=5.5, end=5.5)
        record = TranscriptChunkRecord(**payload)

        self.assertEqual(record.start, 5.5)
        self.assertEqual(record.end, 5.5)


# =====================================================================
# Invalid Records Test Cases (Tests 5 - 18)
# =====================================================================
class TestTranscriptChunkRecordInvalid(unittest.TestCase):
    def test_empty_recording_id(self):
        """5. Empty recording_id raises ValidationError."""
        for empty_val in ["", "   ", "\t\n"]:
            with self.subTest(recording_id=empty_val):
                with self.assertRaises(ValidationError):
                    TranscriptChunkRecord(**get_valid_record_payload(recording_id=empty_val))

    def test_empty_transcript_text(self):
        """6. Empty transcript text raises ValidationError."""
        for empty_val in ["", "   ", "\n\t"]:
            with self.subTest(text=empty_val):
                with self.assertRaises(ValidationError):
                    TranscriptChunkRecord(**get_valid_record_payload(text=empty_val))

    def test_negative_chunk_index(self):
        """7. Negative chunk_index raises ValidationError."""
        with self.assertRaises(ValidationError):
            TranscriptChunkRecord(**get_valid_record_payload(chunk_index=-1))

    def test_boolean_chunk_index(self):
        """8. Boolean chunk_index raises ValidationError."""
        for bool_val in [True, False]:
            with self.subTest(chunk_index=bool_val):
                with self.assertRaises(ValidationError):
                    TranscriptChunkRecord(**get_valid_record_payload(chunk_index=bool_val))

    def test_negative_timestamps(self):
        """9. Negative timestamps raise ValidationError."""
        with self.assertRaises(ValidationError):
            TranscriptChunkRecord(**get_valid_record_payload(start=-0.1))

        with self.assertRaises(ValidationError):
            TranscriptChunkRecord(**get_valid_record_payload(start=0.0, end=-5.0))

    def test_boolean_timestamps(self):
        """10. Boolean timestamps raise ValidationError."""
        for bool_val in [True, False]:
            with self.subTest(start=bool_val):
                with self.assertRaises(ValidationError):
                    TranscriptChunkRecord(**get_valid_record_payload(start=bool_val))

            with self.subTest(end=bool_val):
                with self.assertRaises(ValidationError):
                    TranscriptChunkRecord(**get_valid_record_payload(end=bool_val))

    def test_nan_and_infinity_timestamps(self):
        """11. NaN and infinity timestamps raise ValidationError."""
        for non_finite in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(start=non_finite):
                with self.assertRaises(ValidationError):
                    TranscriptChunkRecord(**get_valid_record_payload(start=non_finite))

            with self.subTest(end=non_finite):
                with self.assertRaises(ValidationError):
                    TranscriptChunkRecord(**get_valid_record_payload(end=non_finite))

    def test_end_earlier_than_start(self):
        """12. end earlier than start raises ValidationError."""
        with self.assertRaises(ValidationError):
            TranscriptChunkRecord(**get_valid_record_payload(start=10.0, end=5.0))

    def test_zero_or_negative_segment_count(self):
        """13. Zero or negative segment_count raises ValidationError."""
        for non_positive in [0, -1, -10]:
            with self.subTest(segment_count=non_positive):
                with self.assertRaises(ValidationError):
                    TranscriptChunkRecord(**get_valid_record_payload(segment_count=non_positive))

    def test_boolean_segment_count(self):
        """14. Boolean segment_count raises ValidationError."""
        for bool_val in [True, False]:
            with self.subTest(segment_count=bool_val):
                with self.assertRaises(ValidationError):
                    TranscriptChunkRecord(**get_valid_record_payload(segment_count=bool_val))

    def test_embedding_fewer_than_768_dimensions(self):
        """15. Embedding with fewer than 768 dimensions raises ValidationError."""
        for dim in [0, 100, 767, 769]:
            with self.subTest(dimensions=dim):
                with self.assertRaises(ValidationError):
                    TranscriptChunkRecord(**get_valid_record_payload(embedding=[0.05] * dim))

    def test_embedding_containing_boolean(self):
        """16. Embedding containing a boolean raises ValidationError."""
        bad_emb_true = [0.01] * 767 + [True]
        bad_emb_false = [0.01] * 767 + [False]

        with self.assertRaises(ValidationError):
            TranscriptChunkRecord(**get_valid_record_payload(embedding=bad_emb_true))

        with self.assertRaises(ValidationError):
            TranscriptChunkRecord(**get_valid_record_payload(embedding=bad_emb_false))

    def test_embedding_containing_nan_or_infinity(self):
        """17. Embedding containing NaN or infinity raises ValidationError."""
        for non_finite in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(value=non_finite):
                bad_emb = [0.01] * 767 + [non_finite]
                with self.assertRaises(ValidationError):
                    TranscriptChunkRecord(**get_valid_record_payload(embedding=bad_emb))

    def test_unexpected_extra_fields(self):
        """18. Unexpected extra fields raise ValidationError due to extra='forbid'."""
        with self.assertRaises(ValidationError):
            TranscriptChunkRecord(
                **get_valid_record_payload(unauthorized_field="malicious_payload")
            )


# =====================================================================
# Main Test Runner
# =====================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
