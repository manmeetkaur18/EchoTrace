import logging
import math
from pathlib import Path
import subprocess
from typing import List, Optional, Tuple, Union
from fastapi import HTTPException, status

# Configure module logger for server-side troubleshooting
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Configuration & Directory Resolution
# ---------------------------------------------------------
# Maximum time (in seconds) allowed for FFmpeg to extract audio before timing out
FFMPEG_TIMEOUT_SECONDS = 600  # 10 minutes

# Timeout for FFprobe duration queries
FFPROBE_TIMEOUT_SECONDS = 30  # 30 seconds

# Timeout per FFmpeg segment slice extraction
FFMPEG_SEGMENT_TIMEOUT_SECONDS = 300  # 5 minutes per segment

# Conservative maximum segment size: 20 MB (Groq upload limit is 25 MB)
MAX_SEGMENT_SIZE_BYTES = 20_000_000

# __file__ points to: backend/app/services/audio_service.py
# .parent -> services/
# .parent.parent -> app/
# .parent.parent.parent -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

# Destination directory for extracted audio: backend/data/processed/
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"


def get_media_duration(file_path: Union[str, Path]) -> float:
    """
    Extracts the duration of a media file (audio or video) in seconds using FFprobe.

    Args:
        file_path (Union[str, Path]): Path to the target media file on disk.

    Returns:
        float: Accurate media duration in seconds.

    Raises:
        TypeError: If file_path is invalid.
        FileNotFoundError: If the media file does not exist.
        ValueError: If file is empty, duration is invalid, zero, or non-finite.
        RuntimeError: If FFprobe fails, is missing, or produces corrupt output.
        TimeoutError: If FFprobe execution exceeds the timeout limit.
    """
    if isinstance(file_path, bool) or not isinstance(file_path, (str, Path)):
        raise TypeError(f"file_path must be a string or Path, got {type(file_path).__name__}.")

    input_path = Path(file_path).resolve()
    if not input_path.is_file():
        logger.error("Media file not found for duration probe: %s", input_path)
        raise FileNotFoundError(f"Media file not found: {input_path}")

    if input_path.stat().st_size == 0:
        logger.error("Media file is empty (0 bytes): %s", input_path)
        raise ValueError(f"Media file is empty (0 bytes): {input_path.name}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        logger.error("FFprobe executable not found in system PATH.")
        raise RuntimeError(
            "Media processing failed: FFprobe is not installed or not available in the server PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        logger.error("FFprobe duration probe timed out after %ds on '%s'.", FFPROBE_TIMEOUT_SECONDS, input_path.name)
        raise TimeoutError(
            f"FFprobe duration probe timed out after {FFPROBE_TIMEOUT_SECONDS}s on '{input_path.name}'."
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while launching FFprobe: %s", exc)
        raise RuntimeError(f"Unexpected error running FFprobe on '{input_path.name}': {exc}") from exc

    if result.returncode != 0:
        err_msg = result.stderr.strip()
        logger.error("FFprobe failed with returncode %d on '%s': %s", result.returncode, input_path.name, err_msg)
        raise RuntimeError(
            f"FFprobe failed to inspect '{input_path.name}' (exit code {result.returncode}): {err_msg}"
        )

    output = result.stdout.strip()
    # Fallback to audio stream duration if container format duration is missing or N/A
    if not output or output == "N/A":
        cmd_stream = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ]
        try:
            res_stream = subprocess.run(
                cmd_stream,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                timeout=FFPROBE_TIMEOUT_SECONDS,
            )
            if res_stream.returncode == 0 and res_stream.stdout.strip() and res_stream.stdout.strip() != "N/A":
                output = res_stream.stdout.strip()
        except Exception:
            pass

    if not output or output == "N/A":
        logger.error("FFprobe produced empty or N/A duration for '%s'.", input_path.name)
        raise ValueError(f"Could not extract duration from '{input_path.name}'. Corrupt or unsupported media.")

    try:
        duration = float(output)
    except (ValueError, TypeError) as parse_err:
        logger.error("FFprobe returned unparseable duration string '%s' for '%s'.", output, input_path.name)
        raise ValueError(
            f"FFprobe returned invalid non-numeric duration '{output}' for '{input_path.name}'."
        ) from parse_err

    if isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0.0:
        logger.error("Duration for '%s' is not finite and positive: %s", input_path.name, duration)
        raise ValueError(
            f"Invalid duration ({duration}) parsed for '{input_path.name}'. Duration must be a positive finite number."
        )

    return duration


def split_audio_into_segments(
    audio_path: Union[str, Path],
    output_dir: Union[str, Path],
    segment_duration: float = 600.0,
    overlap: float = 4.0,
) -> List[Tuple[Path, float, float]]:
    """
    Splits an audio or video file into standardized 16 kHz mono PCM 16-bit WAV segments
    safely below Groq's 25 MB upload limit, preserving global timeline offsets and overlap.

    Args:
        audio_path (Union[str, Path]): Path to the input media file.
        output_dir (Union[str, Path]): Directory where segment WAV files will be written.
        segment_duration (float, optional): Maximum duration of each segment in seconds (default 600.0s = 10m).
        overlap (float, optional): Overlap duration between consecutive segments in seconds (default 4.0s).

    Returns:
        List[Tuple[Path, float, float]]: List of tuples containing:
            - Path: Absolute path to the generated segment WAV file.
            - float: Global start timestamp in seconds.
            - float: Global end timestamp in seconds.

    Raises:
        TypeError: If parameter types are invalid.
        FileNotFoundError: If input media file does not exist.
        ValueError: If parameters or durations are invalid, or if an oversized segment (> 20 MB) is generated.
        RuntimeError: If FFmpeg fails, is missing, or produces corrupt output.
        TimeoutError: If FFmpeg execution times out.
    """
    # 1. Parameter and Type Validation
    if isinstance(audio_path, bool) or not isinstance(audio_path, (str, Path)):
        raise TypeError(f"audio_path must be a string or Path, got {type(audio_path).__name__}.")

    input_path = Path(audio_path).resolve()
    if not input_path.is_file():
        logger.error("Input media file not found for segmentation: %s", input_path)
        raise FileNotFoundError(f"Input media file not found: {input_path}")

    if input_path.stat().st_size == 0:
        logger.error("Input media file is empty (0 bytes): %s", input_path)
        raise ValueError(f"Input media file is empty (0 bytes): {input_path.name}")

    if isinstance(output_dir, bool) or not isinstance(output_dir, (str, Path)):
        raise TypeError(f"output_dir must be a string or Path, got {type(output_dir).__name__}.")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(segment_duration, bool) or not isinstance(segment_duration, (int, float)):
        raise TypeError(f"segment_duration must be an int or float, got {type(segment_duration).__name__}.")
    seg_dur = float(segment_duration)
    if not math.isfinite(seg_dur) or seg_dur <= 0.0:
        raise ValueError(f"segment_duration must be a finite number greater than 0, got {seg_dur}.")

    if isinstance(overlap, bool) or not isinstance(overlap, (int, float)):
        raise TypeError(f"overlap must be an int or float, got {type(overlap).__name__}.")
    ovl = float(overlap)
    if not math.isfinite(ovl) or ovl < 0.0:
        raise ValueError(f"overlap must be a finite non-negative number, got {ovl}.")

    if ovl >= seg_dur:
        raise ValueError(
            f"overlap ({ovl}s) must be strictly less than segment_duration ({seg_dur}s)."
        )

    # 2. Extract media duration via FFprobe
    total_duration = get_media_duration(input_path)
    if total_duration <= 0.0:
        raise ValueError(f"Invalid media duration for '{input_path.name}': {total_duration}")

    segments: List[Tuple[Path, float, float]] = []
    created_files: List[Path] = []
    stem = input_path.stem

    try:
        current_start = 0.0
        seg_idx = 0

        while current_start < total_duration:
            tentative_end = current_start + seg_dur + ovl
            current_end = min(tentative_end, total_duration)

            seg_length = current_end - current_start
            if seg_length <= 0:
                break

            seg_filename = f"{stem}_part_{seg_idx:04d}.wav"
            seg_path = out_dir / seg_filename

            # Safety guard: Never overwrite the original input media
            if seg_path.resolve() == input_path.resolve():
                raise ValueError(
                    f"Output segment path '{seg_path}' conflicts with input media file '{input_path}'."
                )

            # FFmpeg extraction arguments:
            # -ss before -i enables fast sample-accurate seeking
            # Output normalized to: WAV, PCM 16-bit, 16 kHz, Mono
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", f"{current_start:.3f}",
                "-t", f"{seg_length:.3f}",
                "-i", str(input_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(seg_path),
            ]

            logger.info(
                "Extracting segment %d [%.2fs -> %.2fs] (duration=%.2fs) from '%s' to '%s'...",
                seg_idx,
                current_start,
                current_end,
                seg_length,
                input_path.name,
                seg_path.name,
            )

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    shell=False,
                    timeout=FFMPEG_SEGMENT_TIMEOUT_SECONDS,
                )
            except FileNotFoundError as exc:
                if seg_path.exists():
                    try:
                        seg_path.unlink()
                    except Exception:
                        pass
                logger.error("FFmpeg executable not found in system PATH.")
                raise RuntimeError(
                    "Media processing failed: FFmpeg is not installed or not available in the server PATH."
                ) from exc
            except subprocess.TimeoutExpired as exc:
                if seg_path.exists():
                    try:
                        seg_path.unlink()
                    except Exception:
                        pass
                logger.error(
                    "FFmpeg segment extraction timed out after %ds for segment %d of '%s'.",
                    FFMPEG_SEGMENT_TIMEOUT_SECONDS,
                    seg_idx,
                    input_path.name,
                )
                raise TimeoutError(
                    f"FFmpeg segment extraction timed out after {FFMPEG_SEGMENT_TIMEOUT_SECONDS}s "
                    f"for segment {seg_idx} of '{input_path.name}'."
                ) from exc
            except Exception as exc:
                if seg_path.exists():
                    try:
                        seg_path.unlink()
                    except Exception:
                        pass
                logger.exception("Unexpected error launching FFmpeg for segment %d: %s", seg_idx, exc)
                raise RuntimeError(
                    f"Unexpected error running FFmpeg for segment {seg_idx}: {exc}"
                ) from exc

            if result.returncode != 0:
                if seg_path.exists():
                    try:
                        seg_path.unlink()
                    except Exception:
                        pass
                err_msg = result.stderr.strip()
                logger.error(
                    "FFmpeg segment %d extraction failed with exit code %d: %s",
                    seg_idx,
                    result.returncode,
                    err_msg,
                )
                raise RuntimeError(
                    f"FFmpeg failed to extract segment {seg_idx} from '{input_path.name}' "
                    f"(exit code {result.returncode}): {err_msg}"
                )

            # Confirm segment was created on disk
            if not seg_path.is_file() or seg_path.stat().st_size == 0:
                if seg_path.exists():
                    try:
                        seg_path.unlink()
                    except Exception:
                        pass
                raise RuntimeError(
                    f"FFmpeg completed but segment file '{seg_path.name}' is missing or 0 bytes."
                )

            # Track created file for lifecycle cleanup if subsequent steps fail
            created_files.append(seg_path)

            # Enforce strict file-size limit: max 20,000,000 bytes (Groq limit is 25 MB)
            actual_size = seg_path.stat().st_size
            if actual_size > MAX_SEGMENT_SIZE_BYTES:
                size_mb = actual_size / (1024 * 1024)
                limit_mb = MAX_SEGMENT_SIZE_BYTES / (1024 * 1024)
                raise ValueError(
                    f"Generated segment '{seg_path.name}' size ({size_mb:.2f} MB, {actual_size} bytes) "
                    f"exceeds maximum allowed safety limit of {limit_mb:.2f} MB ({MAX_SEGMENT_SIZE_BYTES} bytes)."
                )

            segments.append((seg_path, current_start, current_end))

            # If current_end reached or exceeded total_duration, all audio has been captured
            if current_end >= total_duration:
                break

            current_start += seg_dur
            seg_idx += 1

        logger.info(
            "Successfully split '%s' (duration=%.2fs) into %d segment(s).",
            input_path.name,
            total_duration,
            len(segments),
        )
        return segments

    except Exception:
        # Atomic cleanup: remove all segment files created during this failed operation
        logger.warning(
            "Segmentation failed for '%s'. Cleaning up %d partially created segment file(s)...",
            input_path.name,
            len(created_files),
        )
        for f in created_files:
            if f.exists():
                try:
                    f.unlink()
                except Exception as unl_err:
                    logger.warning("Failed to clean up temporary segment '%s': %s", f, unl_err)
        raise


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
