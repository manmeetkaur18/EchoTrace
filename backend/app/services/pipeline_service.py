import logging
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from app.services.audio_service import (
    extract_audio_from_video,
    get_media_duration,
    split_audio_into_segments,
)
from app.services.chunking_service import chunk_transcript
from app.services.embedding_service import embed_document
from app.services.transcription_service import MAX_AUDIO_SIZE_BYTES  # 25 MB, Groq limit
from app.services.transcription_service import transcribe_audio, transcribe_segmented_audio
from app.services.vector_store_service import store_transcript_chunks

# Configure module logger for server-side diagnostics
logger = logging.getLogger(__name__)

# Supported media extensions matching media_service.py upload allowlist exactly
VIDEO_EXTENSIONS: Set[str] = {".mp4"}
AUDIO_EXTENSIONS: Set[str] = {".mp3", ".wav", ".m4a"}
SUPPORTED_EXTENSIONS: Set[str] = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def validate_normalized_transcript(transcript: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates that a pre-existing or external transcript conforms to the required
    timestamped segment structure for EchoTrace chunking, embedding, and retrieval.

    Validation Rules:
    1. Must be a dictionary containing a non-empty 'segments' list.
    2. Each segment must be a dictionary with non-empty 'text', valid numeric 'start',
       and valid numeric 'end' timestamps.
    3. Timestamps must be finite, non-negative, and end must be >= start.
    4. Timestamps cannot be booleans.
    5. Transcripts lacking valid timestamps are rejected (EchoTrace schema requires timestamps).

    Args:
        transcript (Dict[str, Any]): Candidate transcript dictionary.

    Returns:
        Dict[str, Any]: Validated transcript dictionary with normalized segments.

    Raises:
        TypeError: If transcript or fields have invalid types.
        ValueError: If segments are missing, empty, or have invalid timestamp order.
    """
    if isinstance(transcript, bool) or not isinstance(transcript, dict):
        raise TypeError(
            f"Transcript must be a dictionary, got {type(transcript).__name__}."
        )

    raw_segments = transcript.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(
            "Transcript must contain a non-empty 'segments' list."
        )

    validated_segments: List[Dict[str, Any]] = []
    for idx, seg in enumerate(raw_segments):
        if isinstance(seg, bool) or not isinstance(seg, dict):
            raise TypeError(
                f"Transcript segment at index {idx} must be a dictionary, got {type(seg).__name__}."
            )

        text = seg.get("text")
        if isinstance(text, bool) or not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Transcript segment at index {idx} has invalid or empty 'text'."
            )

        start = seg.get("start")
        end = seg.get("end")

        # Validate start timestamp
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise TypeError(
                f"Transcript segment at index {idx} 'start' must be numeric, got {type(start).__name__}."
            )
        f_start = float(start)
        if not math.isfinite(f_start) or f_start < 0.0:
            raise ValueError(
                f"Transcript segment at index {idx} 'start' must be finite and non-negative, got {f_start}."
            )

        # Validate end timestamp
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            raise TypeError(
                f"Transcript segment at index {idx} 'end' must be numeric, got {type(end).__name__}."
            )
        f_end = float(end)
        if not math.isfinite(f_end) or f_end < 0.0:
            raise ValueError(
                f"Transcript segment at index {idx} 'end' must be finite and non-negative, got {f_end}."
            )

        if f_end < f_start:
            raise ValueError(
                f"Transcript segment at index {idx} 'end' ({f_end}) cannot be earlier than 'start' ({f_start})."
            )

        validated_segments.append({
            "id": int(seg.get("id", idx)),
            "start": f_start,
            "end": f_end,
            "text": text.strip(),
        })

    # Combine text if not provided or empty
    combined_text = transcript.get("text")
    if not isinstance(combined_text, str) or not combined_text.strip():
        combined_text = " ".join(s["text"] for s in validated_segments)

    return {
        "text": combined_text.strip(),
        "segments": validated_segments,
        "language": str(transcript.get("language") or "unknown"),
    }


def ingest_recording(
    recording_id: str,
    media_path: Optional[Union[str, Path]] = None,
    existing_transcript: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Orchestrates the complete end-to-end ingestion pipeline for a recording:
    1. Validates inputs (recording_id, media_path, or existing_transcript).
    2. If an authorized pre-existing transcript is provided, validates it and reuses it directly,
       bypassing audio extraction and Groq transcription.
    3. If no pre-existing transcript is provided:
       a. Validates media file format against allowlist (.mp4, .mp3, .wav, .m4a).
       b. Extracts 16 kHz mono WAV audio from video via FFmpeg (or reuses audio directly).
       c. Enforces the strict 25 MB audio file limit on the audio to be transcribed.
       d. Transcribes speech into timestamped segments via Groq Whisper Large v3.
       e. Derives recording duration safely from segment end timestamps.
    4. Partitions transcript segments into timestamp-aligned chunks.
    5. Generates 768-dimensional Gemini document embeddings for each chunk.
    6. Persists chunks, text, metadata, and vectors into ChromaDB.

    Args:
        recording_id (str): Unique identifier of the recording.
        media_path (Optional[Union[str, Path]]): Path to media file on disk (optional if transcript provided).
        existing_transcript (Optional[Dict[str, Any]]): Pre-existing validated timestamped transcript.

    Returns:
        Dict[str, Any]: Structured summary containing:
            - status (str): "success"
            - recording_id (str): Validated recording identifier.
            - media_type (str): "video", "audio", or "transcript_reuse"
            - source_path (Optional[str]): Path to source media file if present.
            - audio_path (Optional[str]): Path to audio file if transcribed.
            - transcript_language (str): Verified spoken language, or "unknown".
            - duration_seconds (float): Derived total duration in seconds.
            - segment_count (int): Count of validated transcript segments.
            - chunk_count (int): Count of timestamp-aligned chunks created.
            - stored_chunk_ids (List[str]): List of deterministic ChromaDB chunk IDs.
            - transcript_reused (bool): True if external transcript was reused without calling Whisper.

    Raises:
        TypeError: If recording_id or media_path have invalid types.
        ValueError: If recording_id is empty, media format is unsupported, audio exceeds 25 MB,
                    or no segments/chunks could be generated.
        FileNotFoundError: If the specified media file does not exist on disk.
        Exception: Propagates any underlying extraction, transcription, embedding, or storage error.
    """
    # ---------------------------------------------------------
    # 1. Input Validation
    # ---------------------------------------------------------
    if isinstance(recording_id, bool) or not isinstance(recording_id, str):
        raise TypeError(
            f"recording_id must be a string, got {type(recording_id).__name__}."
        )

    clean_recording_id = recording_id.strip()
    if not clean_recording_id:
        raise ValueError("recording_id must not be empty or whitespace-only.")

    if media_path is None and existing_transcript is None:
        raise ValueError(
            "Either 'media_path' or 'existing_transcript' must be provided for ingestion."
        )

    resolved_media_path: Optional[Path] = None
    media_type: str = "transcript_reuse"

    if media_path is not None:
        if isinstance(media_path, bool) or not isinstance(media_path, (str, Path)):
            raise TypeError(
                f"media_path must be a str or Path, got {type(media_path).__name__}."
            )
        resolved_media_path = Path(media_path).resolve()
        if not resolved_media_path.is_file():
            logger.error(
                "Media file not found for ingestion pipeline: %s",
                resolved_media_path,
            )
            raise FileNotFoundError(
                f"Media file does not exist at: {resolved_media_path}"
            )

        file_ext = resolved_media_path.suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            allowed_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            logger.error(
                "Unsupported media format '%s' for recording '%s'. Allowed: %s",
                file_ext,
                clean_recording_id,
                allowed_list,
            )
            raise ValueError(
                f"Unsupported media format '{file_ext}'. Allowed formats are: {allowed_list}"
            )

        media_type = "video" if file_ext in VIDEO_EXTENSIONS else "audio"

    logger.info(
        "Starting ingestion pipeline for recording '%s' (type=%s, media=%s, transcript_reuse=%s)...",
        clean_recording_id,
        media_type,
        resolved_media_path.name if resolved_media_path else "None",
        existing_transcript is not None,
    )

    audio_path: Optional[Path] = None
    is_extracted_audio: bool = False

    try:
        transcription: Dict[str, Any]
        transcript_reused: bool = False
        segmented_audio: bool = False

        # ---------------------------------------------------------
        # Stage A: Transcript Reuse OR Audio Extraction & Transcription
        # ---------------------------------------------------------
        if existing_transcript is not None:
            logger.info(
                "Stage 1-2: Reusing authorized pre-existing transcript for recording '%s' (bypassing Groq)...",
                clean_recording_id,
            )
            transcription = validate_normalized_transcript(existing_transcript)
            transcript_reused = True
            segments = transcription.get("segments") or []
            if not segments:
                raise ValueError(
                    f"Transcription produced no segments for recording '{clean_recording_id}'."
                )
            # Duration from segment timestamps for reused transcripts
            valid_ends = [
                float(s["end"])
                for s in segments
                if isinstance(s, dict)
                and not isinstance(s.get("end"), bool)
                and isinstance(s.get("end"), (int, float))
                and math.isfinite(s["end"])
                and s["end"] >= 0.0
            ]
            duration: float = round(max(valid_ends), 2) if valid_ends else 0.0
        else:
            assert resolved_media_path is not None
            is_video = resolved_media_path.suffix.lower() in VIDEO_EXTENSIONS

            # Stage 1: Audio Extraction (Video) or Direct Audio Path
            if is_video:
                logger.info(
                    "Stage 1/5: Extracting 16 kHz mono WAV audio from video '%s'...",
                    resolved_media_path.name,
                )
                extracted_wav_str = extract_audio_from_video(resolved_media_path)
                audio_path = Path(extracted_wav_str).resolve()
                is_extracted_audio = True
                logger.info(
                    "Stage 1/5: Audio extraction complete -> '%s' (%.2f KB).",
                    audio_path.name,
                    audio_path.stat().st_size / 1024,
                )
            else:
                logger.info(
                    "Stage 1/5: Media file '%s' is already audio; using directly for transcription.",
                    resolved_media_path.name,
                )
                audio_path = resolved_media_path
                is_extracted_audio = False

            # Stage 2: Transcription — single-file or segmented based on audio size
            audio_size_bytes = audio_path.stat().st_size

            if audio_size_bytes <= MAX_AUDIO_SIZE_BYTES:
                # Stage 2a: Single-file transcription (Groq Whisper Large v3, ≤25 MB)
                logger.info(
                    "Stage 2/5: Transcribing audio '%s' (%.2f MB) via Groq Whisper Large v3 (single-file path)...",
                    audio_path.name,
                    audio_size_bytes / (1024 * 1024),
                )
                transcription = transcribe_audio(audio_path)
                segmented_audio = False

                # Derive duration from segment end timestamps (existing behavior for short audio)
                segments = transcription.get("segments") or []
                if not segments:
                    raise ValueError(
                        f"Transcription produced no segments for recording '{clean_recording_id}'."
                    )
                valid_ends = [
                    float(s["end"])
                    for s in segments
                    if isinstance(s, dict)
                    and not isinstance(s.get("end"), bool)
                    and isinstance(s.get("end"), (int, float))
                    and math.isfinite(s["end"])
                    and s["end"] >= 0.0
                ]
                duration: float = round(max(valid_ends), 2) if valid_ends else 0.0

            else:
                # Stage 2b: Segmented transcription for long audio (>25 MB)
                size_mb = audio_size_bytes / (1024 * 1024)
                logger.info(
                    "Stage 2/5: Audio '%s' (%.2f MB) exceeds 25 MB limit — using segmented transcription path...",
                    audio_path.name,
                    size_mb,
                )
                with tempfile.TemporaryDirectory(prefix="echotrace_segments_") as seg_tmp_dir:
                    segments_list = split_audio_into_segments(audio_path, seg_tmp_dir)
                    logger.info(
                        "Stage 2/5: Split into %d segment(s); transcribing via Groq Whisper Large v3 (segmented path)...",
                        len(segments_list),
                    )
                    transcription = transcribe_segmented_audio(segments_list)
                # TemporaryDirectory context manager cleans up segment files here

                segmented_audio = True
                segments = transcription.get("segments") or []
                if not segments:
                    raise ValueError(
                        f"Segmented transcription produced no segments for recording '{clean_recording_id}'."
                    )

                # Duration from FFprobe for long recordings (segment timestamps may not span full duration)
                duration = round(get_media_duration(audio_path), 2)
                logger.info(
                    "Stage 2/5: Segmented transcription complete; FFprobe duration=%.2fs.",
                    duration,
                )

        segments = transcription.get("segments") or []

        # Language: keep "unknown" unless verified information is provided
        language: str = str(transcription.get("language") or "unknown")

        logger.info(
            "Transcript ready (segments=%d, derived_duration=%.2fs, language=%s).",
            len(segments),
            duration,
            language,
        )

        # ---------------------------------------------------------
        # Stage 3: Transcript Chunking
        # ---------------------------------------------------------
        logger.info(
            "Stage 3/5: Chunking transcript (%d segment(s)) into timestamp-aligned chunks...",
            len(segments),
        )
        chunks = chunk_transcript(
            transcription=transcription,
            max_duration_seconds=30,
            max_characters=1000,
        )

        if not chunks:
            raise ValueError(
                f"Chunking produced 0 chunks for recording '{clean_recording_id}'."
            )

        logger.info(
            "Stage 3/5: Chunking complete -> %d timestamp-aligned chunk(s).",
            len(chunks),
        )

        # ---------------------------------------------------------
        # Stage 4: Document Embedding Generation (Gemini)
        # ---------------------------------------------------------
        logger.info(
            "Stage 4/5: Generating 768-dim Gemini embeddings for %d chunk(s)...",
            len(chunks),
        )
        embeddings: List[List[float]] = []
        for idx, chunk in enumerate(chunks):
            emb = embed_document(chunk["text"])
            embeddings.append(emb)

        logger.info(
            "Stage 4/5: Embedding generation complete (%d vectors of dimension %d).",
            len(embeddings),
            len(embeddings[0]) if embeddings else 0,
        )

        # ---------------------------------------------------------
        # Stage 5: Persistent ChromaDB Storage
        # ---------------------------------------------------------
        logger.info(
            "Stage 5/5: Persisting %d chunk(s) and vectors to ChromaDB for recording '%s'...",
            len(chunks),
            clean_recording_id,
        )
        store_result = store_transcript_chunks(
            recording_id=clean_recording_id,
            chunks=chunks,
            embeddings=embeddings,
        )
        stored_ids = store_result.get("ids", [])

        logger.info(
            "Stage 5/5: ChromaDB storage complete -> %d chunk(s) stored for recording '%s'.",
            len(stored_ids),
            clean_recording_id,
        )

        # ---------------------------------------------------------
        # Return Structured Outcome
        # ---------------------------------------------------------
        pipeline_outcome = {
            "status": "success",
            "recording_id": clean_recording_id,
            "media_type": media_type,
            "source_path": str(resolved_media_path) if resolved_media_path else None,
            "audio_path": str(audio_path) if audio_path else None,
            "transcript_language": language,
            "duration_seconds": duration,
            "segment_count": len(segments),
            "chunk_count": len(chunks),
            "stored_chunk_ids": stored_ids,
            "transcript_reused": transcript_reused,
            "segmented_audio": segmented_audio,
            "transcription_mode": "segmented" if segmented_audio else "single_file",
        }

        logger.info(
            "Ingestion pipeline completed successfully for recording '%s' (%d chunks indexed, reused=%s).",
            clean_recording_id,
            len(chunks),
            transcript_reused,
        )

        return pipeline_outcome

    except Exception as exc:
        logger.error(
            "Ingestion pipeline failed for recording '%s': %s: %s",
            clean_recording_id,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        # Re-raise so caller or background task can record and handle failure
        raise
    finally:
        # Reliable cleanup: clean up intermediate extracted WAV file regardless of success or failure.
        # User-owned audio files (is_extracted_audio=False) and original source videos are never deleted.
        if is_extracted_audio and audio_path and audio_path.exists():
            try:
                audio_path.unlink()
                logger.info(
                    "Cleaned up intermediate extracted WAV file: %s",
                    audio_path,
                )
            except Exception as clean_err:
                logger.warning(
                    "Failed to delete intermediate extracted WAV file %s: %s",
                    audio_path,
                    clean_err,
                )
