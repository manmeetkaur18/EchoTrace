import sys
from pathlib import Path
from fastapi import HTTPException

# Ensure 'backend' directory is in Python search path so 'app' imports cleanly
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.transcription_service import transcribe_audio


def main():
    # 1. Define target audio file path
    audio_path = BACKEND_DIR / "data" / "processed" / "5aa51179-2c02-4def-ab08-015caf455998.wav"

    print("=" * 60)
    print("EchoTrace: Testing Audio Transcription with Groq Whisper")
    print("=" * 60)
    print(f"Target Audio File: {audio_path.name}")
    print(f"Full Path        : {audio_path}")

    if not audio_path.is_file():
        print(f"\n[ERROR] Audio file does not exist at: {audio_path}")
        return

    # 2. Call transcribe_audio and display results
    try:
        print("\nSending audio to Groq Whisper API (whisper-large-v3)...")
        result = transcribe_audio(audio_path)

        # Print success indicator
        print("\n[SUCCESS] Transcription completed successfully!")
        print("-" * 60)

        # Print full transcript text
        print("Full Transcript Text:")
        print(result.get("text", ""))
        print("-" * 60)

        # Print segment-level timestamps
        segments = result.get("segments", [])
        print(f"\nTotal Segments: {len(segments)}")
        print("First 5 Segments:")
        for idx, seg in enumerate(segments[:5], 1):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "")
            print(f"  [{idx}] [{start:06.2f}s -> {end:06.2f}s]: {text}")

        # Print word-level timestamps
        words = result.get("words", [])
        print(f"\nTotal Words: {len(words)}")
        print("First 10 Words:")
        for idx, w in enumerate(words[:10], 1):
            start = w.get("start", 0.0)
            end = w.get("end", 0.0)
            word = w.get("word", "")
            print(f"  [{idx:2d}] [{start:06.2f}s -> {end:06.2f}s]: '{word}'")

        print("\n" + "=" * 60)
        print("Test Completed Successfully")
        print("=" * 60)

    except HTTPException as http_exc:
        print("\n[HTTPException Caught]")
        print(f"Status Code: {http_exc.status_code}")
        print(f"Detail     : {http_exc.detail}")

    except Exception as exc:
        print(f"\n[Unexpected Error Caught]: {exc}")


if __name__ == "__main__":
    main()
