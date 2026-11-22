import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi import HTTPException, UploadFile, status

# Configure module logger for server-side error diagnostics
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
# Only allow these specific audio and video extensions
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".mp4", ".m4a"}

# Maximum allowed file size: 100 Megabytes (in bytes)
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

# Chunk size for streaming file writes (1 Megabyte at a time)
CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MB

# Base directory resolution:
# __file__ gives us the path to this file: backend/app/services/media_service.py
# .parent -> services/
# .parent.parent -> app/
# .parent.parent.parent -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

# Destination directory: backend/data/uploads/
UPLOAD_DIR = BACKEND_DIR / "data" / "uploads"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"

# Regex for safe recording ID: alphanumeric, dash, underscore, 1..64 chars
SAFE_RECORDING_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def validate_recording_id(recording_id: str) -> str:
    """
    Validates that a recording ID consists strictly of alphanumeric characters,
    hyphens, or underscores, defending against path traversal attacks.
    """
    if isinstance(recording_id, bool) or not isinstance(recording_id, str):
        raise ValueError("recording_id must be a string.")
    clean = recording_id.strip()
    if not clean or not SAFE_RECORDING_ID_REGEX.match(clean):
        raise ValueError(f"Invalid recording_id '{recording_id}'. Must be 1-64 alphanumeric/dash/underscore characters.")
    return clean


def get_media_file_path(recording_id: str) -> Optional[Path]:
    """
    Safely locates the media file on disk for a given recording_id.
    Inspects UPLOAD_DIR (and PROCESSED_DIR as fallback).
    Prevents path traversal by strictly validating the recording_id.
    """
    clean_id = validate_recording_id(recording_id)

    # 1. Search UPLOAD_DIR for matching allowed media extensions
    if UPLOAD_DIR.is_dir():
        for ext in sorted(ALLOWED_EXTENSIONS):
            candidate = (UPLOAD_DIR / f"{clean_id}{ext}").resolve()
            # Guard against path traversal escaping UPLOAD_DIR
            if candidate.is_file() and UPLOAD_DIR.resolve() in candidate.parents:
                return candidate

    # 2. Search PROCESSED_DIR fallback (e.g. for extracted WAVs)
    if PROCESSED_DIR.is_dir():
        candidate_wav = (PROCESSED_DIR / f"{clean_id}.wav").resolve()
        if candidate_wav.is_file() and PROCESSED_DIR.resolve() in candidate_wav.parents:
            return candidate_wav

    return None


def init_recordings_table(db_path: Optional[Union[str, Path]] = None) -> None:
    """
    Initializes the recordings_meta table in the registry database if not already present.
    """
    from app.services.registry_service import get_db_connection

    with get_db_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recordings_meta (
                    recording_id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    saved_filename TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    duration_seconds REAL,
                    chunk_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise


def upsert_recording_meta(
    recording_id: str,
    original_filename: Optional[str] = None,
    saved_filename: Optional[str] = None,
    media_type: Optional[str] = None,
    file_size: Optional[int] = None,
    status: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    chunk_count: Optional[int] = None,
    error_message: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Inserts or updates metadata for a recording in the SQLite recordings_meta table.
    """
    from app.services.registry_service import get_db_connection

    clean_id = validate_recording_id(recording_id)
    init_recordings_table(db_path)
    now = time.time()

    with get_db_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            cursor = conn.execute(
                "SELECT * FROM recordings_meta WHERE recording_id = ?;",
                (clean_id,),
            )
            existing = cursor.fetchone()

            if existing is None:
                orig_fn = original_filename or f"{clean_id}.mp4"
                saved_fn = saved_filename or orig_fn
                m_type = media_type or ("video" if saved_fn.endswith(".mp4") else "audio")
                f_size = file_size or 0
                st = status or "processing"

                conn.execute(
                    """
                    INSERT INTO recordings_meta (
                        recording_id, original_filename, saved_filename, media_type,
                        file_size, status, duration_seconds, chunk_count, error_message,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        clean_id,
                        orig_fn,
                        saved_fn,
                        m_type,
                        f_size,
                        st,
                        duration_seconds,
                        chunk_count or 0,
                        error_message,
                        now,
                        now,
                    ),
                )
            else:
                updates = ["updated_at = ?"]
                params: List[Any] = [now]

                if original_filename is not None:
                    updates.append("original_filename = ?")
                    params.append(original_filename)
                if saved_filename is not None:
                    updates.append("saved_filename = ?")
                    params.append(saved_filename)
                if media_type is not None:
                    updates.append("media_type = ?")
                    params.append(media_type)
                if file_size is not None:
                    updates.append("file_size = ?")
                    params.append(file_size)
                if status is not None:
                    updates.append("status = ?")
                    params.append(status)
                if duration_seconds is not None:
                    updates.append("duration_seconds = ?")
                    params.append(duration_seconds)
                if chunk_count is not None:
                    updates.append("chunk_count = ?")
                    params.append(chunk_count)
                if error_message is not None:
                    updates.append("error_message = ?")
                    params.append(error_message)

                params.append(clean_id)
                conn.execute(
                    f"UPDATE recordings_meta SET {', '.join(updates)} WHERE recording_id = ?;",
                    tuple(params),
                )

            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    meta = get_recording_meta(clean_id, db_path=db_path)
    return meta or {}


def get_recording_meta(
    recording_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieves metadata for a specific recording ID from recordings_meta.
    Falls back to detecting files on disk and active versions if no metadata row exists.
    """
    from app.services.registry_service import get_active_version, get_db_connection

    clean_id = validate_recording_id(recording_id)
    init_recordings_table(db_path)

    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM recordings_meta WHERE recording_id = ?;",
            (clean_id,),
        )
        row = cursor.fetchone()
        if row is not None:
            return dict(row)

    # Fallback: check if media file exists on disk
    media_path = get_media_file_path(clean_id)
    if media_path and media_path.is_file():
        active_ver = None
        try:
            active_ver = get_active_version(clean_id, db_path=db_path)
        except Exception:
            pass

        return {
            "recording_id": clean_id,
            "original_filename": media_path.name,
            "saved_filename": media_path.name,
            "media_type": "video" if media_path.suffix.lower() == ".mp4" else "audio",
            "file_size": media_path.stat().st_size,
            "status": "ready" if active_ver is not None else "uploaded",
            "duration_seconds": None,
            "chunk_count": 0,
            "error_message": None,
            "created_at": media_path.stat().st_ctime,
            "updated_at": media_path.stat().st_mtime,
        }

    return None


def list_recordings_meta(db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """
    Lists all recorded media metadata, sorted by creation timestamp descending.
    """
    from app.services.registry_service import get_db_connection

    init_recordings_table(db_path)

    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM recordings_meta ORDER BY created_at DESC;"
        )
        return [dict(row) for row in cursor.fetchall()]


def get_recording_transcript_chunks(
    recording_id: str,
    db_path: Optional[Union[str, Path]] = None,
    collection: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves all transcript chunks for a recording from ChromaDB, filtered by active version.
    Returns chronologically sorted chunks with formatted time ranges.
    """
    from app.services.rag_service import format_timestamp_range
    from app.services.registry_service import get_active_version
    from app.services.vector_store_service import get_vector_collection

    clean_id = validate_recording_id(recording_id)
    target_col = collection if collection is not None else get_vector_collection()

    active_ver = None
    try:
        active_ver = get_active_version(clean_id, db_path=db_path)
    except Exception as exc:
        logger.warning("Could not resolve active version for transcript retrieval: %s", exc)

    where_filter: Dict[str, Any]
    if active_ver is not None:
        where_filter = {
            "$and": [
                {"recording_id": clean_id},
                {"version": active_ver},
            ]
        }
    else:
        where_filter = {"recording_id": clean_id}

    try:
        raw = target_col.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        logger.error("ChromaDB get failed for recording '%s': %s", clean_id, exc)
        return []

    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    chunks: List[Dict[str, Any]] = []
    for idx, chunk_id in enumerate(ids):
        doc = docs[idx] if idx < len(docs) else ""
        meta = metas[idx] if idx < len(metas) else {}
        if not isinstance(meta, dict):
            continue

        start = float(meta.get("start", 0.0))
        end = float(meta.get("end", 0.0))

        try:
            time_range = format_timestamp_range(start, end)
        except Exception:
            time_range = f"{start:.1f}s–{end:.1f}s"

        chunks.append({
            "chunk_id": str(chunk_id),
            "recording_id": clean_id,
            "chunk_index": int(meta.get("chunk_index", idx)),
            "start": start,
            "end": end,
            "timestamp_range": time_range,
            "text": str(doc).strip(),
            "segment_count": int(meta.get("segment_count", 1)),
        })

    # Sort chunks chronologically
    chunks.sort(key=lambda c: (c["start"], c["end"], c["chunk_index"]))
    return chunks


async def save_uploaded_media(file: UploadFile) -> Dict[str, Any]:
    """
    Saves an uploaded audio or video file securely in chunks to disk.

    Key Features:
    1. Validates that the file extension is allowed (.mp3, .wav, .mp4, .m4a).
    2. Enforces a maximum file size limit (100 MB) during stream writing.
    3. Replaces user-provided filenames with a secure, unique UUID to prevent
       path-traversal attacks and filename collisions.
    4. Writes in 1 MB chunks to prevent loading large media into RAM all at once.
    5. Leverages Python context manager ('with open') to ensure file handles
       are automatically closed before cleanup.
    6. Automatically removes any partially written file if an exception occurs.
    7. Keeps server-side error logging detailed while returning safe, generic
       messages to API clients to prevent leaking filesystem details.
    8. Guarantees that the uploaded file handle is closed in all scenarios.

    Args:
        file (UploadFile): The FastAPI uploaded file object from the request.

    Returns:
        Dict[str, Any]: Internal service metadata containing:
            - saved_filename: The generated UUID filename stored on disk.
            - original_filename: The client's original uploaded filename.
            - file_size: Total bytes written to disk.
            - media_type: "audio" or "video" (useful for downstream FFmpeg processing).
            - file_path: Absolute path to the saved file on the filesystem (internal use).

    Raises:
        HTTPException (400): If the file has an unsupported extension or missing name.
        HTTPException (413): If the file size exceeds the 100 MB limit.
        HTTPException (500): If an unexpected filesystem or I/O error occurs.
    """
    original_filename = file.filename or ""

    # 1. Validate file extension (case-insensitive)
    file_extension = Path(original_filename).suffix.lower()
    if not file_extension or file_extension not in ALLOWED_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_extension}'. Allowed formats are: {allowed_list}",
        )

    # 2. Determine general media type ("video" for .mp4, "audio" for .mp3, .wav, .m4a)
    media_type = "video" if file_extension == ".mp4" else "audio"

    # 3. Generate a secure, unique filename using uuid4
    unique_filename = f"{uuid.uuid4()}{file_extension}"

    # 4. Ensure the destination upload directory exists (backend/data/uploads/)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination_path = UPLOAD_DIR / unique_filename

    total_bytes_written = 0

    try:
        # Open destination file in binary write mode
        # Python's 'with' statement guarantees buffer closure upon block exit or exception
        with open(destination_path, "wb") as buffer:
            while True:
                # Read from upload stream in 1 MB chunks
                chunk = await file.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    # End of file reached
                    break

                total_bytes_written += len(chunk)

                # Check if upload exceeded the 100 MB limit
                if total_bytes_written > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File size exceeds the 100 MB limit.",
                    )

                buffer.write(chunk)

    except HTTPException:
        # If an HTTPException occurred (e.g. 413 limit exceeded), delete the partially written file
        if destination_path.exists():
            destination_path.unlink()
        raise

    except Exception as exc:
        # Clean up partial file on any unexpected disk / network write failure
        if destination_path.exists():
            destination_path.unlink()

        # Log the full traceback on the server for debugging purposes
        logger.exception("Unexpected error saving uploaded media to %s", destination_path)

        # Return a safe, generic message to the client without exposing internal filesystem paths
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while saving the uploaded file.",
        ) from exc

    finally:
        # Guarantee that FastAPI's underlying upload file descriptor is always closed
        await file.close()

    return {
        "saved_filename": unique_filename,
        "original_filename": original_filename,
        "file_size": total_bytes_written,
        "media_type": media_type,
        "file_path": str(destination_path),
    }
