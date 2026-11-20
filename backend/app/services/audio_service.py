import logging
from pathlib import Path
import subprocess
from typing import Union
from fastapi import HTTPException, status

# Configure module logger for server-side troubleshooting
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Configuration & Directory Resolution
# ---------------------------------------------------------
# Maximum time (in seconds) allowed for FFmpeg to extract audio before timing out
FFMPEG_TIMEOUT_SECONDS = 600  # 10 minutes

# __file__ points to: backend/app/services/audio_service.py
# .parent -> services/
# .parent.parent -> app/
# .parent.parent.parent -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

# Destination directory for extracted audio: backend/data/processed/
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"


def extract_audio_from_video(video_path: Union[str, Path]) -> str:
    """
    Extracts an audio track from a video file using FFmpeg and saves it as
    a standardized 16 kHz mono PCM 16-bit WAV file.

    Why these specific audio settings?
    - WAV (PCM 16-bit): Uncompressed, lossless format universally accepted by STT models.
    - 16 kHz (16,000 Hz) sample rate: Optimal standard for speech-to-text engines
      (e.g., Whisper, Google Speech-to-Text), minimizing API payload size without losing clarity.
    - 1 Channel (Mono): Human speech is mono; reduces file size by ~50% compared to stereo.

    Args:
        video_path (Union[str, Path]): Path to the source video file (e.g. backend/data/uploads/<uuid>.mp4).

    Returns:
        str: Absolute path to the extracted .wav audio file (backend/data/processed/<uuid>.wav).

    Raises:
        HTTPException (404): If the input video file does not exist.
        HTTPException (500): If FFmpeg is missing, execution fails, or the output file is not generated.
        HTTPException (504): If audio extraction exceeds the 600-second timeout limit.
    """
    input_path = Path(video_path).resolve()

    # 1. Verify that the source video file exists on disk
    if not input_path.is_file():
        logger.error("Input video file not found: %s", input_path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file not found: {input_path.name}",
        )

    # 2. Ensure output directory exists (backend/data/processed/)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Use the recording UUID (file stem) as the output audio filename
    # Example: "e7b99c85-23c1-48e0-bb15-38dbfa188a10.mp4" -> "e7b99c85-23c1-48e0-bb15-38dbfa188a10.wav"
    recording_id = input_path.stem
    output_path = PROCESSED_DIR / f"{recording_id}.wav"

    # 4. Construct FFmpeg command arguments as a list (avoiding shell=True for security)
    cmd = [
        "ffmpeg",
        "-y",                   # Overwrite existing output file without prompting
        "-i", str(input_path),  # Input video path
        "-vn",                  # Ignore/disable video stream (audio only)
        "-acodec", "pcm_s16le", # Codec: PCM signed 16-bit little-endian
        "-ar", "16000",         # Sample rate: 16,000 Hz
        "-ac", "1",             # Audio channels: 1 (mono)
        str(output_path),       # Output destination path
    ]

    logger.info("Extracting audio from '%s' to '%s' using FFmpeg...", input_path.name, output_path.name)

    # 5. Execute FFmpeg safely using subprocess.run
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,        # We manually inspect returncode for precise logging
            shell=False,        # Prevents shell injection vulnerabilities
            timeout=FFMPEG_TIMEOUT_SECONDS,  # Failsafe: abort if extraction takes > 10 minutes
        )
    except FileNotFoundError as exc:
        # Occurs if FFmpeg is not installed or not added to system PATH
        logger.error(
            "FFmpeg executable not found. Please ensure FFmpeg is installed and added to the system PATH."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Media processing failed: FFmpeg is not installed or not available in the server PATH.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        # Clean up any partial/incomplete WAV file left behind by timeout
        if output_path.exists():
            output_path.unlink()
        logger.error("FFmpeg audio extraction timed out after %d seconds.", FFMPEG_TIMEOUT_SECONDS)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Audio extraction took too long. Please try a shorter video.",
        ) from exc
    except Exception as exc:
        # Clean up any partial file on any unexpected failure
        if output_path.exists():
            output_path.unlink()
        logger.exception("Unexpected error while launching FFmpeg: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while starting audio extraction.",
        ) from exc

    # 6. Check if FFmpeg exited with an error code
    if result.returncode != 0:
        # Clean up incomplete output file if FFmpeg failed halfway through
        if output_path.exists():
            output_path.unlink()

        # Log stderr server-side for debugging, but hide internal details from the client
        logger.error(
            "FFmpeg failed with exit code %d. Stderr output:\n%s",
            result.returncode,
            result.stderr.strip(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract audio track from the uploaded video.",
        )

    # 7. Confirm that the extracted audio file exists and is not empty (0 bytes)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        if output_path.exists():
            output_path.unlink()
        logger.error("Audio extraction output file missing or 0 bytes at %s", output_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audio extraction completed, but the resulting audio file is missing or invalid.",
        )

    logger.info(
        "Audio extracted successfully: '%s' (%.2f KB)",
        output_path.name,
        output_path.stat().st_size / 1024,
    )

    return str(output_path)
