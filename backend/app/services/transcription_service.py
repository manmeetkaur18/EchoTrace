import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from dotenv import load_dotenv
from fastapi import HTTPException, status
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    Groq,
    GroqError,
    RateLimitError,
)

# Configure module logger for server-side diagnostics
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Environment & Path Configuration
# ---------------------------------------------------------
# __file__ points to: backend/app/services/transcription_service.py
# .parent -> services/
# .parent.parent -> app/
# .parent.parent.parent -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BACKEND_DIR / ".env"

# Explicitly load .env from backend/.env so keys are available
if ENV_PATH.is_file():
    load_dotenv(dotenv_path=ENV_PATH)

# Model configuration: Groq hosted Whisper large v3
WHISPER_MODEL = "whisper-large-v3"

# Maximum audio upload size supported by Groq API: 25 Megabytes (in bytes)
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

# Conservative maximum segment size for segmented transcription (20 MB)
MAX_SEGMENT_UPLOAD_BYTES = 20_000_000  # 20 MB

# Retry configuration for segmented transcription
MAX_RETRIES = 3  # Maximum total attempts (1 initial + 2 retries)
RETRY_BASE_DELAY = 2.0  # Exponential backoff base delay in seconds
RETRY_MAX_DELAY = 30.0  # Maximum delay between retries in seconds

# HTTP status codes that are transient and warrant a retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def get_groq_client() -> Groq:
    """
    Initializes and returns an authenticated Groq client using GROQ_API_KEY.

    Raises:
        HTTPException (500): If GROQ_API_KEY is not configured or empty.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        logger.error("GROQ_API_KEY is missing from backend/.env or environment.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transcription service is not configured. Missing GROQ_API_KEY.",
        )

    # Note: We pass api_key without ever printing or logging its value
    return Groq(api_key=api_key.strip())


def _is_valid_timestamp(value: Any) -> bool:
    """
    Validates that a timestamp is numeric, finite, non-negative, and not boolean.

    Args:
        value (Any): The timestamp candidate to validate.

    Returns:
        bool: True if valid float/int, finite, and >= 0.0; False otherwise.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0.0


def transcribe_audio(audio_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Transcribes an audio file (e.g. WAV, MP3, M4A) using Groq's hosted Whisper Large v3 model.

    Features:
    1. Validates that the audio file exists, is not empty, and does not exceed 25 MB.
    2. Requests verbose_json response format from Groq to retrieve both segment-level
       and word-level timestamps.
    3. Rigorously validates API response integrity and timestamp sanity (numeric, finite,
       non-negative, start <= end) to prevent corrupted data from entering the database.
    4. Returns a standardized, structured dictionary ready for SQLite indexing and ChromaDB embeddings.
    5. Provides clean error handling for authentication, rate limits, network timeouts,
       and service issues without leaking credentials.

    Args:
        audio_path (Union[str, Path]): Path to the audio file on disk.

    Returns:
        Dict[str, Any]: Structured transcription data containing:
            - text: str (Complete transcript text)
            - segments: List[Dict] (Timestamped transcript chunks with start, end, and text)
            - words: List[Dict] (Individual words with start, end, and word text)

    Raises:
        HTTPException (404): If the input audio file does not exist.
        HTTPException (400): If the input audio file is empty (0 bytes).
        HTTPException (413): If the audio file exceeds the 25 MB limit.
        HTTPException (422): If no speech could be transcribed (empty transcript text).
        HTTPException (429): If Groq rate limits are encountered.
        HTTPException (500/502/503): If Groq authentication, network, or server errors occur.
    """
    input_file = Path(audio_path).resolve()

    # 1. Validate that the audio file exists on disk
    if not input_file.is_file():
        logger.error("Audio file not found for transcription: %s", input_file)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file not found: {input_file.name}",
        )

    # 2. Validate file size constraints (must not be empty, must not exceed 25 MB)
    file_size_bytes = input_file.stat().st_size
    if file_size_bytes == 0:
        logger.error("Audio file is 0 bytes: %s", input_file)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio file is empty: {input_file.name}",
        )

    if file_size_bytes > MAX_AUDIO_SIZE_BYTES:
        size_mb = file_size_bytes / (1024 * 1024)
        logger.error(
            "Audio file size (%.2f MB) exceeds Groq limit of 25 MB: %s",
            size_mb,
            input_file,
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file size ({size_mb:.2f} MB) exceeds maximum allowed limit of 25 MB.",
        )

    # 3. Retrieve authenticated Groq client
    client = get_groq_client()

    logger.info(
        "Sending audio file '%s' (%.2f KB) to Groq Whisper (%s)...",
        input_file.name,
        file_size_bytes / 1024,
        WHISPER_MODEL,
    )

    # 4. Invoke Groq Speech-to-Text API with verbose_json and granular timestamps
    try:
        with open(input_file, "rb") as file_handle:
            response = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=file_handle,
                response_format="verbose_json",
                timestamp_granularities=["segment", "word"],
            )
    except AuthenticationError as exc:
        logger.error("Groq API authentication failed. Invalid API key provided.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transcription failed: Invalid Groq API credentials.",
        ) from exc
    except RateLimitError as exc:
        logger.warning("Groq API rate limit reached.")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Transcription rate limit reached. Please wait a moment and try again.",
        ) from exc
    except APIConnectionError as exc:
        logger.error("Unable to connect to Groq API: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to reach the transcription server. Please check your network connection.",
        ) from exc
    except APIStatusError as exc:
        logger.error("Groq API returned an HTTP error (Status %d): %s", exc.status_code, exc.message)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Transcription provider returned an error while processing audio.",
        ) from exc
    except GroqError as exc:
        logger.exception("Groq SDK error during transcription: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="A transcription service error occurred.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while transcribing '%s': %s", input_file.name, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during audio transcription.",
        ) from exc

    # 5. Extract and validate raw response structure
    raw_data = response.model_dump() if hasattr(response, "model_dump") else dict(response)

    if not isinstance(raw_data, dict):
        logger.error("Groq API response is not a valid dictionary: %s", type(raw_data))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Transcription provider returned an invalid response structure.",
        )

    # Validate that transcript text is present and is a string
    raw_text = raw_data.get("text")
    if raw_text is None or not isinstance(raw_text, str):
        logger.error(
            "Groq API response is missing a valid 'text' string field (received type: %s)",
            type(raw_text).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Transcription provider response is missing the transcript text field.",
        )

    full_text: str = raw_text.strip()
    if not full_text:
        logger.warning("No speech could be transcribed from audio file: '%s'", input_file.name)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No speech could be transcribed from the audio file.",
        )

    # 6. Validate and normalize segment-level timestamps
    # Segments must be dictionaries with valid numeric, finite, non-negative timestamps and start <= end
    segments: List[Dict[str, Any]] = []
    raw_segments = raw_data.get("segments")
    if isinstance(raw_segments, list):
        for idx, seg in enumerate(raw_segments):
            if not isinstance(seg, dict):
                logger.warning(
                    "Skipping malformed non-dict segment at index %d (received type: %s)",
                    idx,
                    type(seg).__name__,
                )
                continue

            start_val = seg.get("start")
            end_val = seg.get("end")

            if not _is_valid_timestamp(start_val) or not _is_valid_timestamp(end_val):
                logger.warning(
                    "Skipping segment %d: timestamps must be numeric, finite, and non-negative (start=%s, end=%s)",
                    idx,
                    start_val,
                    end_val,
                )
                continue

            start_f = float(start_val)
            end_f = float(end_val)

            if start_f > end_f:
                logger.warning(
                    "Skipping segment %d: start time (%.3f) exceeds end time (%.3f)",
                    idx,
                    start_f,
                    end_f,
                )
                continue

            seg_text = seg.get("text")
            clean_text = str(seg_text).strip() if seg_text is not None else ""

            segments.append({
                "id": seg.get("id", idx),
                "start": start_f,
                "end": end_f,
                "text": clean_text,
            })
    else:
        logger.warning("Response 'segments' field is missing or not a list; proceeding with empty segments.")

    # 7. Validate and normalize word-level timestamps
    # Words must be dictionaries with valid numeric, finite, non-negative timestamps and start <= end
    words: List[Dict[str, Any]] = []
    raw_words = raw_data.get("words")
    if isinstance(raw_words, list):
        for idx, item in enumerate(raw_words):
            if not isinstance(item, dict):
                logger.warning(
                    "Skipping malformed non-dict word at index %d (received type: %s)",
                    idx,
                    type(item).__name__,
                )
                continue

            start_val = item.get("start")
            end_val = item.get("end")

            if not _is_valid_timestamp(start_val) or not _is_valid_timestamp(end_val):
                logger.warning(
                    "Skipping word %d: timestamps must be numeric, finite, and non-negative (start=%s, end=%s)",
                    idx,
                    start_val,
                    end_val,
                )
                continue

            start_f = float(start_val)
            end_f = float(end_val)

            if start_f > end_f:
                logger.warning(
                    "Skipping word %d: start time (%.3f) exceeds end time (%.3f)",
                    idx,
                    start_f,
                    end_f,
                )
                continue

            word_val = item.get("word")
            clean_word = str(word_val).strip() if word_val is not None else ""

            words.append({
                "word": clean_word,
                "start": start_f,
                "end": end_f,
            })
    else:
        logger.warning("Response 'words' field is missing or not a list; proceeding with empty words.")

    logger.info(
        "Transcription complete for '%s': %d characters, %d valid segments, %d valid words.",
        input_file.name,
        len(full_text),
        len(segments),
        len(words),
    )

    return {
        "text": full_text,
        "segments": segments,
        "words": words,
    }


def transcribe_segmented_audio(
    segments: List[Tuple[Path, float, float]],
) -> Dict[str, Any]:
    """
    Transcribes a list of audio segment files and merges their results into a single,
    timestamp-correct transcription dictionary compatible with chunk_transcript().

    Each segment tuple is produced by split_audio_into_segments() and contains:
        (segment_file_path, global_start_seconds, global_end_seconds)

    Key behaviors:
    - Transcribes segments sequentially using transcribe_audio().
    - Applies a global time offset to every segment and word timestamp so that all
      timestamps are expressed in terms of the original recording timeline.
    - Deduplicates overlapping content at segment boundaries using word-level timestamps
      when available. A word belongs to segment k if its translated start time is strictly
      less than the global_start of segment k+1 (the nominal boundary). The final segment
      owns all of its words unconditionally.
    - Falls back to segment-level deduplication when a segment returns no word-level
      timestamps. In this case, a translated segment is included if its translated start
      time is strictly less than the nominal boundary. This is less precise than
      word-level deduplication and may include or exclude a small number of boundary
      words incorrectly.
    - Retries transient failures (HTTP 429, 500, 502, 503, 504) with exponential backoff
      up to MAX_RETRIES total attempts. Permanent failures (HTTP 400, 404, 413, 422) are
      not retried.
    - A segment returning an empty transcript is treated as a valid success (not an error).
    - Does not mutate the dictionaries returned by transcribe_audio().
    - Re-assigns monotonically increasing segment IDs (0, 1, 2, ...) across the merged
      result.

    Args:
        segments (List[Tuple[Path, float, float]]): Non-empty list of tuples, each
            containing (audio_file_path, global_start_seconds, global_end_seconds).
            global_start >= 0.0, global_end > global_start.
            Each file must exist and must not exceed MAX_SEGMENT_UPLOAD_BYTES.

    Returns:
        Dict[str, Any]: Merged transcription result:
            {
                "text": str,              # Concatenated transcript text across all segments
                "segments": List[Dict],   # Chronological, re-indexed segments
                "words": List[Dict],      # Deduplicated, chronological words
            }

    Raises:
        TypeError: If segments is not a list, or any element is not a tuple of 3 items
            with the correct types.
        ValueError: If segments is empty, any global_start/global_end is invalid,
            or any segment file is missing or oversized.
        HTTPException: Propagated from transcribe_audio() after exhausting retries.

    Limitations:
        - Segment-level fallback deduplication (when words are absent) may be slightly
          imprecise at boundaries if a segment's text spans the boundary without word
          timestamps.
        - Concurrent transcription is not supported; segments are processed sequentially.
    """
    # ----------------------------------------------------------------
    # 1. Input validation
    # ----------------------------------------------------------------
    if not isinstance(segments, list):
        raise TypeError(
            f"segments must be a list of (Path, float, float) tuples, got {type(segments).__name__}"
        )
    if len(segments) == 0:
        raise ValueError("segments list must not be empty")

    for idx, item in enumerate(segments):
        if not isinstance(item, tuple) or len(item) != 3:
            raise TypeError(
                f"segments[{idx}] must be a tuple of exactly 3 elements "
                f"(Path, float, float), got {type(item).__name__} of length "
                f"{len(item) if isinstance(item, (tuple, list)) else 'N/A'}"
            )

        seg_path_raw, global_start, global_end = item

        # Validate path
        try:
            seg_path = Path(seg_path_raw)
        except TypeError as exc:
            raise TypeError(
                f"segments[{idx}][0] must be a path-like object, "
                f"got {type(seg_path_raw).__name__}"
            ) from exc

        # Validate timestamps
        if isinstance(global_start, bool) or not isinstance(global_start, (int, float)):
            raise TypeError(
                f"segments[{idx}][1] (global_start) must be a float, "
                f"got {type(global_start).__name__}"
            )
        if isinstance(global_end, bool) or not isinstance(global_end, (int, float)):
            raise TypeError(
                f"segments[{idx}][2] (global_end) must be a float, "
                f"got {type(global_end).__name__}"
            )
        global_start = float(global_start)
        global_end = float(global_end)

        if not (math.isfinite(global_start) and global_start >= 0.0):
            raise ValueError(
                f"segments[{idx}][1] (global_start={global_start}) must be finite and >= 0.0"
            )
        if not (math.isfinite(global_end) and global_end > global_start):
            raise ValueError(
                f"segments[{idx}][2] (global_end={global_end}) must be finite "
                f"and strictly greater than global_start={global_start}"
            )

        # Validate file existence and size
        if not seg_path.is_file():
            raise ValueError(
                f"segments[{idx}]: audio file does not exist: {seg_path}"
            )
        file_size = seg_path.stat().st_size
        if file_size > MAX_SEGMENT_UPLOAD_BYTES:
            size_mb = file_size / (1_000_000)
            raise ValueError(
                f"segments[{idx}]: file size ({size_mb:.2f} MB) exceeds "
                f"MAX_SEGMENT_UPLOAD_BYTES ({MAX_SEGMENT_UPLOAD_BYTES / 1_000_000:.0f} MB): "
                f"{seg_path.name}"
            )

    # Nominal boundaries: for segment k, the boundary is segments[k+1][1] (the next
    # segment's global_start). The final segment has no upper boundary.
    nominal_boundaries: List[float] = [float(segments[i + 1][1]) for i in range(len(segments) - 1)]

    # ----------------------------------------------------------------
    # 2. Sequential transcription with retry
    # ----------------------------------------------------------------
    per_segment_results: List[Dict[str, Any]] = []

    for seg_idx, (seg_path_raw, global_start, global_end) in enumerate(segments):
        seg_path = Path(seg_path_raw)
        global_start = float(global_start)
        global_end = float(global_end)

        last_exc: Exception = RuntimeError("unreachable")
        raw_result: Dict[str, Any] = {}

        for attempt in range(MAX_RETRIES + 1):
            try:
                raw_result = transcribe_audio(seg_path)
                last_exc = None  # type: ignore[assignment]
                break
            except HTTPException as exc:
                if exc.status_code in _RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    logger.warning(
                        "Segment %d/%d: transient error (HTTP %d), retry %d/%d after %.1fs",
                        seg_idx + 1,
                        len(segments),
                        exc.status_code,
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                    )
                    time.sleep(delay)
                    last_exc = exc
                else:
                    # Permanent error or retries exhausted — propagate immediately
                    logger.error(
                        "Segment %d/%d: non-retryable error (HTTP %d) or retries exhausted",
                        seg_idx + 1,
                        len(segments),
                        exc.status_code,
                    )
                    raise

        if last_exc is not None:
            raise last_exc  # Should not be reached if the loop raised above

        per_segment_results.append({
            "seg_idx": seg_idx,
            "global_start": global_start,
            "global_end": global_end,
            "raw": raw_result,
        })

    # ----------------------------------------------------------------
    # 3. Timestamp translation, boundary deduplication, and merge
    # ----------------------------------------------------------------
    merged_segments: List[Dict[str, Any]] = []
    merged_words: List[Dict[str, Any]] = []

    BOUNDARY_TOLERANCE = 2.0  # seconds of lenience for model imprecision at segment end

    for seg_idx, seg_result in enumerate(per_segment_results):
        global_start: float = seg_result["global_start"]
        global_end: float = seg_result["global_end"]
        raw: Dict[str, Any] = seg_result["raw"]

        is_last = seg_idx == len(per_segment_results) - 1
        boundary = None if is_last else nominal_boundaries[seg_idx]

        raw_words = raw.get("words", [])
        has_words = isinstance(raw_words, list) and len(raw_words) > 0

        # --- Translate and collect words (if available) ---
        translated_words: List[Dict[str, Any]] = []
        for w in (raw_words if isinstance(raw_words, list) else []):
            w_start = global_start + w["start"]
            w_end = global_start + w["end"]

            # Validate translated timestamps
            if not (math.isfinite(w_start) and w_start >= 0.0):
                logger.warning(
                    "Segment %d word '%s': translated start=%.3f invalid; skipping",
                    seg_idx, w.get("word", ""), w_start,
                )
                continue
            if not (math.isfinite(w_end) and w_end >= w_start):
                logger.warning(
                    "Segment %d word '%s': translated end=%.3f invalid; skipping",
                    seg_idx, w.get("word", ""), w_end,
                )
                continue
            if w_end > global_end + BOUNDARY_TOLERANCE:
                logger.warning(
                    "Segment %d word '%s': translated end=%.3f exceeds "
                    "global_end=%.3f + tolerance=%.1f; skipping",
                    seg_idx, w.get("word", ""), w_end, global_end, BOUNDARY_TOLERANCE,
                )
                continue

            # Deduplication: word belongs to this segment if its start < boundary
            if boundary is not None and w_start >= boundary:
                continue  # Owned by the next segment

            translated_words.append({
                **w,
                "start": w_start,
                "end": w_end,
            })

        # --- Translate segments and deduplicate ---
        raw_segs = raw.get("segments", [])
        for s in (raw_segs if isinstance(raw_segs, list) else []):
            s_start = global_start + s["start"]
            s_end = global_start + s["end"]

            # Validate translated timestamps
            if not (math.isfinite(s_start) and s_start >= 0.0):
                logger.warning(
                    "Segment %d sub-segment: translated start=%.3f invalid; skipping",
                    seg_idx, s_start,
                )
                continue
            if not (math.isfinite(s_end) and s_end >= s_start):
                logger.warning(
                    "Segment %d sub-segment: translated end=%.3f invalid; skipping",
                    seg_idx, s_end,
                )
                continue
            if s_end > global_end + BOUNDARY_TOLERANCE:
                logger.warning(
                    "Segment %d sub-segment: translated end=%.3f exceeds "
                    "global_end=%.3f + tolerance=%.1f; skipping",
                    seg_idx, s_end, global_end, BOUNDARY_TOLERANCE,
                )
                continue

            # Deduplication strategy:
            #   - If word-level timestamps are available, include this sub-segment
            #     unconditionally (word-level dedup handles precision).
            #   - If no word-level timestamps, apply segment-level dedup:
            #     include only if s_start < boundary.
            if not has_words and boundary is not None and s_start >= boundary:
                continue  # Segment-level fallback dedup

            merged_segments.append({
                **s,
                "start": s_start,
                "end": s_end,
            })

        merged_words.extend(translated_words)

    # ----------------------------------------------------------------
    # 4. Sort, re-index, and build final result
    # ----------------------------------------------------------------
    merged_segments.sort(key=lambda s: (s["start"], s["end"]))

    for new_id, seg in enumerate(merged_segments):
        seg["id"] = new_id

    merged_text = " ".join(s["text"] for s in merged_segments if s.get("text", "").strip())

    logger.info(
        "Segmented transcription complete: %d input segments → %d merged sub-segments, "
        "%d merged words, %d characters",
        len(segments),
        len(merged_segments),
        len(merged_words),
        len(merged_text),
    )

    return {
        "text": merged_text.strip(),
        "segments": merged_segments,
        "words": merged_words,
    }
