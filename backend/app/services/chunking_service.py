import logging
import math
from typing import Any, Dict, List, Union

# Configure module logger for server-side diagnostics
logger = logging.getLogger(__name__)


def _is_valid_timestamp(value: Any) -> bool:
    """
    Validates that a timestamp is numeric, finite, non-negative, and not a boolean.

    Args:
        value (Any): The timestamp candidate to validate.

    Returns:
        bool: True if valid float/int, finite, and >= 0.0; False otherwise.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0.0


def chunk_transcript(
    transcription: Dict[str, Any],
    max_duration_seconds: Union[int, float] = 30,
    max_characters: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Groups consecutive transcript segments into timestamp-aligned chunks for vector embeddings.

    Why segment-based chunking instead of raw string chunking?
    - Preserves exact audio/video timestamp alignment for search citations.
    - Avoids splitting words or sentences unnaturally across arbitrary character offsets.
    - Each chunk represents a coherent thought or topic boundary with defined start and end times.

    Chunking Rules:
    1. Segments are processed in strict chronological order.
    2. A new chunk is started when adding the next segment would exceed either:
       - `max_duration_seconds` (default: 30 seconds from the chunk's start time)
       - `max_characters` (default: 1,000 characters of combined text)
    3. Running character counts include single joining spaces between consecutive segments.
    4. Original segments are never split. If an individual segment exceeds limits on its own,
       it is preserved intact as its own standalone chunk.

    Args:
        transcription (Dict[str, Any]): The dictionary returned by transcribe_audio(),
                                        containing at least a 'segments' list.
        max_duration_seconds (Union[int, float], optional): Maximum allowed duration per chunk in seconds.
                                                            Must be a positive finite int or float (not bool).
                                                            Defaults to 30.
        max_characters (int, optional): Maximum allowed character length per chunk.
                                        Must be a positive integer (not bool).
                                        Defaults to 1000.

    Returns:
        List[Dict[str, Any]]: List of structured chunks, each containing:
            - chunk_index (int): Zero-based sequential index (0, 1, 2, ...).
            - start (float): Timestamp (in seconds) where this chunk begins.
            - end (float): Timestamp (in seconds) where this chunk ends.
            - text (str): Combined transcript text for the chunk.
            - segment_count (int): Number of original segments merged into this chunk.

    Raises:
        TypeError: If transcription is not a dict, max_duration_seconds is not an int/float (or is bool),
                   or max_characters is not an int (or is bool).
        ValueError: If max_duration_seconds is non-finite or <= 0, max_characters <= 0,
                    'segments' is missing/empty, or no valid segments remain.
    """
    # ---------------------------------------------------------
    # 1. Parameter and Type Validation
    # ---------------------------------------------------------
    if not isinstance(transcription, dict):
        raise TypeError(
            f"Expected 'transcription' to be a dict, got {type(transcription).__name__}."
        )

    # Validate max_duration_seconds: int or float, not bool, finite, > 0
    if isinstance(max_duration_seconds, bool) or not isinstance(max_duration_seconds, (int, float)):
        raise TypeError(
            f"max_duration_seconds must be an int or float, got {type(max_duration_seconds).__name__}."
        )
    if not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0:
        raise ValueError(
            f"max_duration_seconds must be a finite number greater than 0, got {max_duration_seconds}."
        )

    # Validate max_characters: int, not bool, > 0
    if isinstance(max_characters, bool) or not isinstance(max_characters, int):
        raise TypeError(
            f"max_characters must be an int, got {type(max_characters).__name__}."
        )
    if max_characters <= 0:
        raise ValueError(
            f"max_characters must be an integer greater than 0, got {max_characters}."
        )

    raw_segments = transcription.get("segments")
    if not isinstance(raw_segments, list) or len(raw_segments) == 0:
        raise ValueError(
            "Transcription input must contain a non-empty 'segments' list."
        )

    # ---------------------------------------------------------
    # 2. Sanitize and Validate Segments
    # ---------------------------------------------------------
    valid_segments: List[Dict[str, Any]] = []

    for idx, seg in enumerate(raw_segments):
        # Ignore malformed non-dictionary segments
        if not isinstance(seg, dict):
            logger.warning(
                "Skipping malformed segment at index %d (expected dict, got %s)",
                idx,
                type(seg).__name__,
            )
            continue

        # Extract and validate text
        text_raw = seg.get("text")
        if text_raw is None:
            continue
        clean_text = str(text_raw).strip()
        if not clean_text:
            continue

        # Extract and validate timestamps
        start_val = seg.get("start")
        end_val = seg.get("end")

        if not _is_valid_timestamp(start_val) or not _is_valid_timestamp(end_val):
            logger.warning(
                "Skipping segment %d with invalid timestamps (start=%s, end=%s)",
                idx,
                start_val,
                end_val,
            )
            continue

        start_f = float(start_val)
        end_f = float(end_val)

        # Reject negative timestamps and segments where start > end
        if start_f > end_f:
            logger.warning(
                "Skipping segment %d: start timestamp (%.3f) exceeds end timestamp (%.3f)",
                idx,
                start_f,
                end_f,
            )
            continue

        valid_segments.append({
            "start": start_f,
            "end": end_f,
            "text": clean_text,
        })

    if not valid_segments:
        raise ValueError("No valid segments found to chunk in the transcription input.")

    # ---------------------------------------------------------
    # 3. Preserve Chronological Order
    # ---------------------------------------------------------
    # Sort by start timestamp (and end timestamp as tie-breaker)
    valid_segments.sort(key=lambda s: (s["start"], s["end"]))

    # ---------------------------------------------------------
    # 4. Group Segments into Chunks with Running Character Count
    # ---------------------------------------------------------
    chunks: List[Dict[str, Any]] = []
    current_chunk_segments: List[Dict[str, Any]] = []
    current_chunk_chars: int = 0  # Running character count including single joining spaces

    for seg in valid_segments:
        seg_len = len(seg["text"])

        if not current_chunk_segments:
            # First segment in a new chunk
            current_chunk_segments.append(seg)
            current_chunk_chars = seg_len
            continue

        # Calculate potential chunk duration if this segment were added
        chunk_start = current_chunk_segments[0]["start"]
        candidate_end = seg["end"]
        candidate_duration = candidate_end - chunk_start

        # Calculate potential character length with 1 space between segments
        candidate_length = current_chunk_chars + 1 + seg_len

        # Check if limits would be exceeded
        duration_exceeded = candidate_duration > max_duration_seconds
        length_exceeded = candidate_length > max_characters

        if duration_exceeded or length_exceeded:
            # Flush the current chunk
            combined_text = " ".join(s["text"] for s in current_chunk_segments)
            chunks.append({
                "chunk_index": len(chunks),
                "start": current_chunk_segments[0]["start"],
                "end": current_chunk_segments[-1]["end"],
                "text": combined_text,
                "segment_count": len(current_chunk_segments),
            })
            # Start the new chunk with the current segment
            current_chunk_segments = [seg]
            current_chunk_chars = seg_len
        else:
            # Add segment to the current chunk and update running character count
            current_chunk_segments.append(seg)
            current_chunk_chars = candidate_length

    # ---------------------------------------------------------
    # 5. Flush Final Remaining Chunk
    # ---------------------------------------------------------
    if current_chunk_segments:
        combined_text = " ".join(s["text"] for s in current_chunk_segments)
        chunks.append({
            "chunk_index": len(chunks),
            "start": current_chunk_segments[0]["start"],
            "end": current_chunk_segments[-1]["end"],
            "text": combined_text,
            "segment_count": len(current_chunk_segments),
        })

    logger.info(
        "Chunking complete: %d segments grouped into %d timestamp-aligned chunks.",
        len(valid_segments),
        len(chunks),
    )

    return chunks
