import sys
from pathlib import Path
from fastapi import HTTPException

# Ensure 'backend' directory is in Python search path so 'app' imports cleanly
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.audio_service import extract_audio_from_video


def test_audio_extraction():
    # 1. Define input video path
    video_path = BACKEND_DIR / "data" / "uploads" / "5aa51179-2c02-4def-ab08-015caf455998.mp4"
    print(f"--- Audio Extraction Test ---")
    print(f"Input video path: {video_path}")

    # 2. Call extraction function and handle exceptions
    try:
        wav_path_str = extract_audio_from_video(video_path)
        print(f"Returned WAV path: {wav_path_str}")

        # 3. Check output file existence and size
        wav_path = Path(wav_path_str)
        exists = wav_path.is_file()
        print(f"Output file exists: {exists}")

        if exists:
            size_kb = wav_path.stat().st_size / 1024
            print(f"Output file size: {size_kb:.2f} KB")

    except HTTPException as http_exc:
        print(f"\n[HTTPException Caught]")
        print(f"Status Code : {http_exc.status_code}")
        print(f"Detail      : {http_exc.detail}")

    except Exception as exc:
        print(f"\n[Unexpected Error Caught]: {exc}")


if __name__ == "__main__":
    test_audio_extraction()
