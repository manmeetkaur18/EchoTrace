
import math
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from fastapi import HTTPException
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. Configure Project Paths
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.audio_service import extract_audio_from_video
from app.services.chunking_service import chunk_transcript
from app.services.transcription_service import transcribe_audio

ENV_PATH = BACKEND_DIR / ".env"

if ENV_PATH.is_file():
    load_dotenv(dotenv_path=ENV_PATH)


# ---------------------------------------------------------
# 2. Locate Uploaded Video
# ---------------------------------------------------------

def locate_uploaded_video(uploads_dir: Path) -> Path:
    """
    Locate upload1.mp4 or the most recently uploaded MP4.
    """

    direct_file = uploads_dir / "upload1.mp4"

    if direct_file.is_file():
        return direct_file

    mp4_files = sorted(
        uploads_dir.glob("*.mp4"),
        key=lambda file: file.stat().st_mtime,
        reverse=True,
    )

    if mp4_files:
        return mp4_files[0]

    raise FileNotFoundError(
        f"No uploaded MP4 video found in {uploads_dir}"
    )


# ---------------------------------------------------------
# 3. Main Embedding Pipeline Test
# ---------------------------------------------------------

def test_embedding_pipeline():

    print("=" * 70)
    print("EchoTrace: End-to-End Pipeline & Gemini Embedding Test")
    print("=" * 70)

    # -----------------------------------------------------
    # STEP 1: Locate Uploaded Video
    # -----------------------------------------------------

    uploads_dir = BACKEND_DIR / "data" / "uploads"

    try:
        video_path = locate_uploaded_video(uploads_dir)

        print("\n--- Step 1: Video Location ---")
        print(f"Target Video : {video_path.name}")
        print(f"Full Path    : {video_path}")
        print(
            f"Video Size   : "
            f"{video_path.stat().st_size / (1024 * 1024):.2f} MB"
        )

    except FileNotFoundError as exc:
        print(f"\n[FAIL] {exc}")
        return

    # -----------------------------------------------------
    # STEP 2: Extract Audio
    # -----------------------------------------------------

    print("\n--- Step 2: Audio Extraction ---")

    try:
        wav_path_str = extract_audio_from_video(video_path)
        wav_path = Path(wav_path_str)

        extraction_succeeded = (
            wav_path.is_file()
            and wav_path.stat().st_size > 0
        )

        if not extraction_succeeded:
            print("[FAIL] Extracted WAV is missing or empty.")
            return

        print(f"Audio extraction succeeded: {extraction_succeeded}")
        print(f"Extracted WAV Path       : {wav_path}")
        print(
            f"Extracted WAV Size       : "
            f"{wav_path.stat().st_size / 1024:.2f} KB"
        )

    except HTTPException as exc:
        print(
            f"[FAIL] Audio extraction failed: "
            f"Status {exc.status_code} - {exc.detail}"
        )
        return

    except Exception as exc:
        print(
            f"[FAIL] Unexpected audio extraction error: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    # -----------------------------------------------------
    # STEP 3: Transcribe Audio Using Groq Whisper
    # -----------------------------------------------------

    print("\n--- Step 3: Audio Transcription (Groq Whisper) ---")

    try:
        transcription = transcribe_audio(wav_path)

        full_text = transcription.get("text", "")

        transcription_succeeded = (
            isinstance(full_text, str)
            and bool(full_text.strip())
        )

        print(f"Transcription succeeded : {transcription_succeeded}")
        print(f"Transcript Length       : {len(full_text)} characters")
        print(
            f"Raw Transcript Segments : "
            f"{len(transcription.get('segments', []))}"
        )

        if not transcription_succeeded:
            print("[FAIL] Transcription returned empty text.")
            return

    except HTTPException as exc:
        print(
            f"[FAIL] Transcription failed: "
            f"Status {exc.status_code} - {exc.detail}"
        )
        return

    except Exception as exc:
        print(
            f"[FAIL] Unexpected transcription error: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    # -----------------------------------------------------
    # STEP 4: Chunk Transcript
    # -----------------------------------------------------

    print("\n--- Step 4: Transcript Chunking ---")

    try:
        chunks = chunk_transcript(
            transcription,
            max_duration_seconds=30,
            max_characters=1000,
        )

        print(f"Transcript chunks generated: {len(chunks)}")

    except Exception as exc:
        print(
            f"[FAIL] Chunking failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    # -----------------------------------------------------
    # STEP 5: Select First Non-Empty Chunk
    # -----------------------------------------------------

    selected_chunk = next(
        (
            chunk
            for chunk in chunks
            if isinstance(chunk.get("text"), str)
            and chunk["text"].strip()
        ),
        None,
    )

    if selected_chunk is None:
        print("[FAIL] No non-empty transcript chunk available.")
        return

    chunk_text = selected_chunk["text"]

    print("\n--- Selected Transcript Chunk ---")
    print(f"Chunk Index    : {selected_chunk['chunk_index']}")
    print(
        f"Timestamp Span : "
        f"{selected_chunk['start']:.2f}s -> "
        f"{selected_chunk['end']:.2f}s"
    )
    print(f"Segment Count  : {selected_chunk['segment_count']}")
    print(f"Chunk Text     :\n{chunk_text}")

    # -----------------------------------------------------
    # STEP 6: Load Gemini API Key
    # -----------------------------------------------------

    print("\n--- Step 5: Gemini API Configuration ---")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or not api_key.strip():
        print("[FAIL] GEMINI_API_KEY is missing from backend/.env.")
        return

    try:
        client = genai.Client(api_key=api_key.strip())
        print("Google Gen AI client initialized successfully.")

    except Exception as exc:
        print(
            f"[FAIL] Gemini client initialization failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    # -----------------------------------------------------
    # STEP 7: Generate Gemini Embedding
    # -----------------------------------------------------

    print("\n--- Step 6: Gemini Embedding Generation ---")

    model_name = "gemini-embedding-2"

    print(f"Model       : {model_name}")
    print("Dimensions  : 768")
    print("Task Type   : RETRIEVAL_DOCUMENT")

    try:
        response = client.models.embed_content(
            model=model_name,
            contents=chunk_text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )

        if not response.embeddings:
            print("[FAIL] Gemini returned an empty embeddings list.")
            return

        embedding_values = response.embeddings[0].values

        if embedding_values is None:
            print("[FAIL] Gemini returned no embedding values.")
            return

        vector_length = len(embedding_values)

        print("\n--- Embedding Results ---")
        print("Embedding generation succeeded: True")
        print(f"Embedding vector length       : {vector_length}")

        first_five = [
            round(float(value), 6)
            for value in embedding_values[:5]
        ]

        print(f"First five embedding values   : {first_five}")

        # -------------------------------------------------
        # STEP 8: Validate Embedding
        # -------------------------------------------------

        if vector_length != 768:
            print(
                f"[FAIL] Expected 768 dimensions, "
                f"received {vector_length}."
            )
            return

        all_numeric = all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            for value in embedding_values
        )

        if not all_numeric:
            print(
                "[FAIL] Embedding contains invalid or "
                "non-finite numerical values."
            )
            return

        print("\n" + "=" * 70)
        print("[SUCCESS] EMBEDDING VALIDATION PASSED")
        print("Exactly 768 finite numerical values received.")
        print("Upload -> Extraction -> Transcription -> Chunking -> Embedding")
        print("Entire test pipeline completed successfully!")
        print("=" * 70)

    except Exception as exc:
        print(
            f"[FAIL] Gemini embedding API request failed: "
            f"{type(exc).__name__}: {exc}"
        )


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    test_embedding_pipeline()
