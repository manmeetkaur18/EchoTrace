import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.services.media_service import (
    ALLOWED_EXTENSIONS,
    get_media_file_path,
    get_recording_meta,
    get_recording_transcript_chunks,
    list_recordings_meta,
    save_uploaded_media,
    upsert_recording_meta,
    validate_recording_id,
)
from app.services.pipeline_service import ingest_recording

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# APIRouter Configuration
# ---------------------------------------------------------
router = APIRouter(prefix="/media", tags=["Media"])

MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
}


# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------
class MediaUploadResponse(BaseModel):
    success: bool
    message: str
    recording_id: str
    original_filename: str
    saved_filename: str
    file_size: int
    media_type: str
    status: Optional[str] = "processing"


class RecordingStatusResponse(BaseModel):
    recording_id: str
    original_filename: str
    saved_filename: str
    media_type: str
    file_size: int
    status: str
    duration_seconds: Optional[float] = None
    chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


class TranscriptChunkItem(BaseModel):
    chunk_id: str
    recording_id: str
    chunk_index: int
    start: float
    end: float
    timestamp_range: str
    text: str
    segment_count: int = 1


class RecordingTranscriptResponse(BaseModel):
    recording_id: str
    chunk_count: int
    chunks: List[TranscriptChunkItem]


class UrlIngestRequest(BaseModel):
    url: str
    recording_id: Optional[str] = None
    requested_episode: Optional[str] = None


# ---------------------------------------------------------
# Background Worker Function
# ---------------------------------------------------------
def run_ingestion_background(recording_id: str, file_path: Union[str, Path]) -> None:
    """
    Executes the ingestion pipeline in a background thread without blocking the FastAPI event loop.
    Updates the recording metadata with the outcome.
    """
    path_obj = Path(file_path).resolve()
    try:
        logger.info("Executing background ingestion for recording '%s'...", recording_id)
        outcome = ingest_recording(recording_id=recording_id, media_path=path_obj)
        upsert_recording_meta(
            recording_id=recording_id,
            status="ready",
            duration_seconds=outcome.get("duration_seconds"),
            chunk_count=outcome.get("chunk_count", 0),
        )
        logger.info("Background ingestion succeeded for recording '%s'.", recording_id)
    except Exception as exc:
        logger.error("Background ingestion failed for recording '%s': %s", recording_id, exc, exc_info=True)
        upsert_recording_meta(
            recording_id=recording_id,
            status="failed",
            error_message=str(exc),
        )


def run_url_ingestion_background(
    url: str,
    recording_id: str,
    requested_episode: Optional[str] = None,
) -> None:
    """
    Executes the URL ingestion pipeline in a background worker thread.
    """
    from app.services.url_ingestion_service import ingest_from_url

    try:
        logger.info("Executing URL ingestion for '%s' (recording_id='%s')...", url, recording_id)
        outcome = ingest_from_url(
            url=url,
            recording_id=recording_id,
            requested_episode=requested_episode,
        )
        upsert_recording_meta(
            recording_id=recording_id,
            status="ready",
            duration_seconds=outcome.get("duration_seconds"),
            chunk_count=outcome.get("chunk_count", 0),
        )
        logger.info("URL ingestion succeeded for recording '%s'.", recording_id)
    except Exception as exc:
        logger.error("URL ingestion failed for recording '%s': %s", recording_id, exc, exc_info=True)
        upsert_recording_meta(
            recording_id=recording_id,
            status="failed",
            error_message=str(exc),
        )


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@router.post(
    "/upload",
    response_model=MediaUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload audio or video media and trigger ingestion",
    description="Uploads media file, saves it securely, and starts ingestion pipeline.",
)
async def upload_media(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    async_process: bool = Query(True, description="Process ingestion asynchronously in background."),
) -> MediaUploadResponse:
    """
    Receives an uploaded audio or video file, streams it to disk, stores metadata,
    and triggers the transcription and ChromaDB ingestion pipeline.
    """
    # 1. Stream file securely to disk
    saved_media = await save_uploaded_media(file)
    recording_id = Path(saved_media["saved_filename"]).stem

    # 2. Record initial metadata
    upsert_recording_meta(
        recording_id=recording_id,
        original_filename=saved_media["original_filename"],
        saved_filename=saved_media["saved_filename"],
        media_type=saved_media["media_type"],
        file_size=saved_media["file_size"],
        status="processing",
    )

    # 3. Trigger ingestion
    if async_process:
        background_tasks.add_task(run_ingestion_background, recording_id, saved_media["file_path"])
    else:
        run_ingestion_background(recording_id, saved_media["file_path"])

    meta = get_recording_meta(recording_id)
    current_status = meta.get("status", "processing") if meta else "processing"

    return MediaUploadResponse(
        success=True,
        message="File uploaded successfully; processing started.",
        recording_id=recording_id,
        original_filename=saved_media["original_filename"],
        saved_filename=saved_media["saved_filename"],
        file_size=saved_media["file_size"],
        media_type=saved_media["media_type"],
        status=current_status,
    )


@router.post(
    "/url",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest media or podcast from URL",
)
async def ingest_url_endpoint(
    payload: UrlIngestRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    async_process: bool = Query(True),
) -> Dict[str, Any]:
    """
    Ingests media from a secure URL (podcast RSS or direct audio/video).
    """
    import uuid

    clean_url = payload.url.strip()
    if not clean_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="URL must not be empty.")

    rec_id = payload.recording_id.strip() if payload.recording_id else str(uuid.uuid4())

    upsert_recording_meta(
        recording_id=rec_id,
        original_filename=clean_url,
        saved_filename=f"{rec_id}.media",
        media_type="audio",
        file_size=0,
        status="processing",
    )

    if async_process:
        background_tasks.add_task(
            run_url_ingestion_background,
            clean_url,
            rec_id,
            payload.requested_episode,
        )
    else:
        run_url_ingestion_background(
            clean_url,
            rec_id,
            payload.requested_episode,
        )

    return {
        "success": True,
        "recording_id": rec_id,
        "message": f"URL ingestion initiated for recording '{rec_id}'.",
        "status": "processing",
    }


@router.get(
    "/recordings",
    response_model=List[RecordingStatusResponse],
    summary="List all recordings and their processing statuses",
)
async def list_recordings() -> List[RecordingStatusResponse]:
    """
    Returns a list of all uploaded recordings and their current status.
    """
    records = list_recordings_meta()
    return [RecordingStatusResponse(**r) for r in records]


@router.get(
    "/{recording_id}/status",
    response_model=RecordingStatusResponse,
    summary="Get recording processing status",
)
@router.get(
    "/status/{recording_id}",
    response_model=RecordingStatusResponse,
    include_in_schema=False,
)
async def get_status(recording_id: str) -> RecordingStatusResponse:
    """
    Returns the processing status, duration, chunk count, and any error message for a recording.
    """
    try:
        meta = get_recording_meta(recording_id)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))

    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording '{recording_id}' not found.",
        )
    return RecordingStatusResponse(**meta)


@router.get(
    "/{recording_id}/transcript",
    response_model=RecordingTranscriptResponse,
    summary="Get timestamped transcript chunks for a recording",
)
async def get_transcript(recording_id: str) -> RecordingTranscriptResponse:
    """
    Retrieves all indexed transcript chunks for a recording from ChromaDB.
    """
    try:
        chunks = get_recording_transcript_chunks(recording_id)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))

    return RecordingTranscriptResponse(
        recording_id=recording_id,
        chunk_count=len(chunks),
        chunks=chunks,
    )


@router.get(
    "/stream/{recording_id}",
    summary="Stream recording audio or video with HTTP Range support",
)
@router.get(
    "/{recording_id}",
    summary="Stream recording media file",
    include_in_schema=False,
)
async def stream_media(recording_id: str) -> FileResponse:
    """
    Serves the original audio or video file supporting standard HTTP Range requests (206 Partial Content).
    Enables seeking in HTML5 audio and video players when users click evidence timestamps.
    """
    try:
        file_path = get_media_file_path(recording_id)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )

    if not file_path or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Media file not found for recording '{recording_id}'.",
        )

    media_type = MIME_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
    )
