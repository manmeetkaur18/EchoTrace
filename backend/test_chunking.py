import sys
from pathlib import Path

# Ensure 'backend' directory is in Python search path so 'app' imports cleanly
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.chunking_service import chunk_transcript


# =====================================================================
# Test 1: Multiple consecutive segments merging into one chunk
# =====================================================================
def test_multiple_consecutive_segments_merge():
    transcription = {
        "text": "Hello world. Welcome to EchoTrace. Video search made easy.",
        "segments": [
            {"id": 0, "start": 0.0, "end": 4.5, "text": "Hello world."},
            {"id": 1, "start": 5.0, "end": 9.2, "text": "Welcome to EchoTrace."},
            {"id": 2, "start": 9.5, "end": 14.0, "text": "Video search made easy."},
        ],
    }

    chunks = chunk_transcript(transcription, max_duration_seconds=30, max_characters=1000)

    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    chunk = chunks[0]
    assert chunk["chunk_index"] == 0, f"Expected chunk_index 0, got {chunk['chunk_index']}"
    assert chunk["start"] == 0.0, f"Expected start 0.0, got {chunk['start']}"
    assert chunk["end"] == 14.0, f"Expected end 14.0, got {chunk['end']}"
    assert chunk["segment_count"] == 3, f"Expected 3 segments, got {chunk['segment_count']}"
    expected_text = "Hello world. Welcome to EchoTrace. Video search made easy."
    assert chunk["text"] == expected_text, f"Text mismatch: '{chunk['text']}' != '{expected_text}'"


# =====================================================================
# Test 2: A new chunk being created when the 30-second duration limit is exceeded
# =====================================================================
def test_duration_limit_split():
    transcription = {
        "segments": [
            {"start": 0.0, "end": 12.0, "text": "First segment (0-12s)."},
            {"start": 12.5, "end": 25.0, "text": "Second segment (12.5-25s)."},
            # Adding third segment reaches 35.0s -> 35.0 - 0.0 = 35s > 30s duration limit!
            {"start": 26.0, "end": 35.0, "text": "Third segment (26-35s)."},
        ]
    }

    chunks = chunk_transcript(transcription, max_duration_seconds=30, max_characters=1000)

    assert len(chunks) == 2, f"Expected 2 chunks due to 30s limit, got {len(chunks)}"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 25.0
    assert chunks[0]["segment_count"] == 2
    assert chunks[0]["text"] == "First segment (0-12s). Second segment (12.5-25s)."

    assert chunks[1]["chunk_index"] == 1
    assert chunks[1]["start"] == 26.0
    assert chunks[1]["end"] == 35.0
    assert chunks[1]["segment_count"] == 1
    assert chunks[1]["text"] == "Third segment (26-35s)."


# =====================================================================
# Test 3: A new chunk being created when the 1000-character limit is exceeded
# =====================================================================
def test_character_limit_split():
    # Two segments whose individual lengths are <= 1000, but combined exceed 1000 chars
    seg1_text = "A" * 600
    seg2_text = "B" * 500  # 600 + 1 space + 500 = 1101 chars > 1000 max

    transcription = {
        "segments": [
            {"start": 0.0, "end": 5.0, "text": seg1_text},
            {"start": 5.5, "end": 10.0, "text": seg2_text},
        ]
    }

    chunks = chunk_transcript(transcription, max_duration_seconds=60, max_characters=1000)

    assert len(chunks) == 2, f"Expected 2 chunks due to character limit, got {len(chunks)}"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["text"] == seg1_text
    assert chunks[0]["segment_count"] == 1

    assert chunks[1]["chunk_index"] == 1
    assert chunks[1]["text"] == seg2_text
    assert chunks[1]["segment_count"] == 1


# =====================================================================
# Test 4: A single segment exceeding the character limit remaining intact
# =====================================================================
def test_single_oversized_segment_remains_intact():
    oversized_text = "X" * 1500  # Exceeds the 1000-character threshold on its own

    transcription = {
        "segments": [
            {"start": 0.0, "end": 10.0, "text": oversized_text},
            {"start": 10.5, "end": 15.0, "text": "Following normal segment."},
        ]
    }

    chunks = chunk_transcript(transcription, max_duration_seconds=30, max_characters=1000)

    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
    # First chunk must contain the complete 1500-char segment without truncation or splitting
    assert chunks[0]["chunk_index"] == 0
    assert len(chunks[0]["text"]) == 1500, f"Oversized segment text was altered"
    assert chunks[0]["text"] == oversized_text
    assert chunks[0]["segment_count"] == 1

    assert chunks[1]["chunk_index"] == 1
    assert chunks[1]["text"] == "Following normal segment."
    assert chunks[1]["segment_count"] == 1


# =====================================================================
# Test 5: Segments being sorted chronologically
# =====================================================================
def test_chronological_sorting():
    # Pass segments deliberately shuffled / out of order
    transcription = {
        "segments": [
            {"start": 20.0, "end": 28.0, "text": "Part 3 (20-28s)"},
            {"start": 0.0, "end": 8.0, "text": "Part 1 (0-8s)"},
            {"start": 9.0, "end": 15.0, "text": "Part 2 (9-15s)"},
        ]
    }

    chunks = chunk_transcript(transcription, max_duration_seconds=30, max_characters=1000)

    assert len(chunks) == 1, f"Expected 1 chunk within 30s, got {len(chunks)}"
    expected_ordered_text = "Part 1 (0-8s) Part 2 (9-15s) Part 3 (20-28s)"
    assert chunks[0]["text"] == expected_ordered_text, (
        f"Segments were not sorted chronologically. Got: '{chunks[0]['text']}'"
    )
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 28.0


# =====================================================================
# Test 6: Invalid timestamps being skipped
# =====================================================================
def test_invalid_timestamps_skipped():
    transcription = {
        "segments": [
            {"start": -5.0, "end": 10.0, "text": "Negative start time"},
            {"start": 5.0, "end": 2.0, "text": "Start time greater than end time"},
            {"start": float("nan"), "end": 10.0, "text": "NaN timestamp"},
            {"start": 0.0, "end": float("inf"), "text": "Infinity timestamp"},
            {"start": "not_a_number", "end": 10.0, "text": "String timestamp"},
            {"start": True, "end": 10.0, "text": "Boolean timestamp"},
            # Only this segment is valid:
            {"start": 0.0, "end": 8.0, "text": "The only valid segment."},
        ]
    }

    chunks = chunk_transcript(transcription, max_duration_seconds=30, max_characters=1000)

    assert len(chunks) == 1, f"Expected 1 chunk from the valid segment, got {len(chunks)}"
    assert chunks[0]["text"] == "The only valid segment."
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 8.0
    assert chunks[0]["segment_count"] == 1


# =====================================================================
# Test 7: Empty segment text being skipped
# =====================================================================
def test_empty_segment_text_skipped():
    transcription = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": ""},          # Empty string
            {"start": 2.0, "end": 4.0, "text": "     "},      # Whitespace only
            {"start": 4.0, "end": 6.0, "text": None},        # None value
            {"start": 6.0, "end": 10.0, "text": "Valid speech segment."},
        ]
    }

    chunks = chunk_transcript(transcription, max_duration_seconds=30, max_characters=1000)

    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert chunks[0]["text"] == "Valid speech segment."
    assert chunks[0]["start"] == 6.0
    assert chunks[0]["end"] == 10.0
    assert chunks[0]["segment_count"] == 1


# =====================================================================
# Test 8: Empty or missing segments raising ValueError
# =====================================================================
def test_empty_or_missing_segments_raise_value_error():
    # Case A: Missing 'segments' key
    try:
        chunk_transcript({"text": "No segments here"})
        assert False, "Should have raised ValueError for missing segments"
    except ValueError:
        pass

    # Case B: Empty 'segments' list
    try:
        chunk_transcript({"segments": []})
        assert False, "Should have raised ValueError for empty segments list"
    except ValueError:
        pass

    # Case C: Segments is not a list
    try:
        chunk_transcript({"segments": "not_a_list"})
        assert False, "Should have raised ValueError when segments is not a list"
    except ValueError:
        pass

    # Case D: List contains only invalid/corrupt segments so no valid segments remain
    try:
        chunk_transcript({
            "segments": [
                {"start": -10, "end": -5, "text": "bad timestamps"},
                {"start": 10, "end": 5, "text": "start > end"},
                {"start": 0, "end": 5, "text": "   "},
            ]
        })
        assert False, "Should have raised ValueError when no valid segments remain"
    except ValueError:
        pass


# =====================================================================
# Test 9: Invalid configuration parameters raising TypeError or ValueError
# =====================================================================
def test_invalid_configuration_parameters():
    valid_transcription = {
        "segments": [{"start": 0.0, "end": 5.0, "text": "Valid test segment."}]
    }

    # --- transcription input checks ---
    try:
        chunk_transcript("not_a_dict")
        assert False, "Should have raised TypeError for non-dict transcription"
    except TypeError:
        pass

    # --- max_duration_seconds type checks (TypeError) ---
    for bad_val in [True, False, "30", None, [30]]:
        try:
            chunk_transcript(valid_transcription, max_duration_seconds=bad_val)
            assert False, f"Should have raised TypeError for max_duration_seconds={bad_val}"
        except TypeError:
            pass

    # --- max_duration_seconds value checks (ValueError) ---
    for bad_val in [0, -5, float("nan"), float("inf"), float("-inf")]:
        try:
            chunk_transcript(valid_transcription, max_duration_seconds=bad_val)
            assert False, f"Should have raised ValueError for max_duration_seconds={bad_val}"
        except ValueError:
            pass

    # --- max_characters type checks (TypeError) ---
    for bad_val in [True, False, 100.5, "1000", None, [1000]]:
        try:
            chunk_transcript(valid_transcription, max_characters=bad_val)
            assert False, f"Should have raised TypeError for max_characters={bad_val}"
        except TypeError:
            pass

    # --- max_characters value checks (ValueError) ---
    for bad_val in [0, -1, -500]:
        try:
            chunk_transcript(valid_transcription, max_characters=bad_val)
            assert False, f"Should have raised ValueError for max_characters={bad_val}"
        except ValueError:
            pass


# =====================================================================
# Main Test Runner
# =====================================================================
def main():
    test_cases = [
        ("Test 1: Multiple segments merge into one chunk", test_multiple_consecutive_segments_merge),
        ("Test 2: Duration limit (30s) splits into new chunk", test_duration_limit_split),
        ("Test 3: Character limit (1000 chars) splits into new chunk", test_character_limit_split),
        ("Test 4: Standalone oversized segment remains intact", test_single_oversized_segment_remains_intact),
        ("Test 5: Segments sorted chronologically", test_chronological_sorting),
        ("Test 6: Invalid timestamps are skipped", test_invalid_timestamps_skipped),
        ("Test 7: Empty or whitespace segment text is skipped", test_empty_segment_text_skipped),
        ("Test 8: Empty or missing segments raise ValueError", test_empty_or_missing_segments_raise_value_error),
        ("Test 9: Invalid config parameters raise TypeError or ValueError", test_invalid_configuration_parameters),
    ]

    print("=" * 70)
    print("EchoTrace: Running Chunking Service Unit Tests")
    print("=" * 70)

    passed_count = 0
    failed_count = 0

    for name, test_func in test_cases:
        try:
            test_func()
            print(f"[PASS] {name}")
            passed_count += 1
        except AssertionError as exc:
            print(f"[FAIL] {name}: Assertion failed -> {exc}")
            failed_count += 1
        except Exception as exc:
            print(f"[FAIL] {name}: Unexpected {type(exc).__name__} -> {exc}")
            failed_count += 1

    print("=" * 70)
    print(f"Results: {passed_count} passed, {failed_count} failed out of {len(test_cases)} tests.")
    print("=" * 70)

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
