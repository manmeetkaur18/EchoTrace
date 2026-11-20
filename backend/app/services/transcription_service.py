import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Union

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
