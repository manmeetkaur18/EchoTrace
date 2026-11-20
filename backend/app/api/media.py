from pathlib import Path
from fastapi import APIRouter, File, UploadFile, status
from pydantic import BaseModel

# Import the upload service function created in the previous step
from app.services.media_service import save_uploaded_media

# ---------------------------------------------------------
# APIRouter Configuration
# ---------------------------------------------------------
# - prefix="/media": All routes defined in this router will start with /media
# - tags=["Media"]: Groups endpoints under "Media" in FastAPI's Swagger UI docs (/docs)
router = APIRouter(prefix="/media", tags=["Media"])


# ---------------------------------------------------------
# Response Schema
# ---------------------------------------------------------
class MediaUploadResponse(BaseModel):
    """
    Defines the shape of the JSON response returned to the client.
    This provides automatic validation and Swagger UI documentation.
    Notice that the internal server 'file_path' is deliberately omitted.
    """
    success: bool
    message: str
    recording_id: str
    original_filename: str
    saved_filename: str
    file_size: int
    media_type: str


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@router.post(
    "/upload",
    response_model=MediaUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload audio or video media",
    description="Uploads an MP3, WAV, MP4, or M4A file up to 100 MB for processing.",
)
async def upload_media(file: UploadFile = File(...)) -> MediaUploadResponse:
    """
    Receives an uploaded audio or video file from the client, validates its format
    and size, streams it securely to disk, and returns the recording metadata.

    Args:
        file (UploadFile): The uploaded file sent via multipart/form-data.

    Returns:
        MediaUploadResponse: Response model containing recording_id and file metadata.
    """
    # 1. Delegate file validation and disk streaming to the service layer
    # Any validation error (invalid format, oversized file) will raise an HTTPException
    # which FastAPI automatically catches and converts into a proper JSON error response.
    saved_media = await save_uploaded_media(file)

    # 2. Extract the UUID portion without the file extension to act as the recording ID
    # Example: "a1b2c3d4-...-ef.mp4" -> Path(...).stem gives "a1b2c3d4-...-ef"
    recording_id = Path(saved_media["saved_filename"]).stem

    # 3. Construct and return the typed response model
    # Note: 'file_path' is NOT returned to the client to avoid exposing internal server paths.
    return MediaUploadResponse(
        success=True,
        message="File uploaded successfully",
        recording_id=recording_id,
        original_filename=saved_media["original_filename"],
        saved_filename=saved_media["saved_filename"],
        file_size=saved_media["file_size"],
        media_type=saved_media["media_type"],
    )
