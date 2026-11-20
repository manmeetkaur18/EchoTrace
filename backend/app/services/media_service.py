import logging
import uuid
from pathlib import Path
from typing import Any, Dict
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
