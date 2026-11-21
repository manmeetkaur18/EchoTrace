from pathlib import Path
import sys

from dotenv import load_dotenv

# ---------------------------------------------------------
# 1. Project Root & Path Resolution
# ---------------------------------------------------------
# parents[0] -> backend/
# parents[1] -> EchoTrace/ (project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

# Ensure backend directory is in sys.path for app module imports
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load environment variables securely from backend/.env
ENV_PATH = BACKEND_DIR / ".env"
if ENV_PATH.is_file():
    load_dotenv(dotenv_path=ENV_PATH)

from app.services.audio_service import extract_audio_from_video
from app.services.chunking_service import chunk_transcript
from app.services.embedding_service import embed_document, EMBEDDING_DIMENSIONS
from app.services.transcription_service import transcribe_audio
from app.services.vector_store_service import (
    get_vector_collection,
    store_transcript_chunks,
)


def run_store_pipeline():
    """
    Executes the complete ingestion and persistent storage pipeline:
    1. Locates the existing uploaded video.
    2. Reuses existing extracted WAV or extracts audio via FFmpeg.
    3. Transcribes audio to timestamped segments via Groq Whisper.
    4. Groups segments into timestamp-aligned chunks.
    5. Generates 768-dimensional Gemini document embeddings.
    6. Persists chunks, text, metadata, and vectors into ChromaDB.
    """
    print("=" * 70)
    print("EchoTrace: Complete Ingestion & ChromaDB Storage Pipeline")
    print("=" * 70)

    # ---------------------------------------------------------
    # Step 1: Locate Existing Uploaded Video
    # ---------------------------------------------------------
    print("\n--- Step 1: Video File Verification ---")
    video_path = BACKEND_DIR / "data" / "uploads" / "5aa51179-2c02-4def-ab08-015caf455998.mp4"

    if not video_path.is_file():
        print(f"[ERROR] Target video file does not exist at: {video_path}")
        return

    recording_id = video_path.stem
    print(f"Video File   : {video_path.name}")
    print(f"Recording ID : {recording_id}")
    print(f"Video Path   : {video_path}")
    print(f"Video Size   : {video_path.stat().st_size / (1024 * 1024):.2f} MB")

    # ---------------------------------------------------------
    # Step 2: Audio Extraction (Reuse Existing WAV if Present)
    # ---------------------------------------------------------
    print("\n--- Step 2: Audio Extraction / WAV Reuse ---")
    wav_path = BACKEND_DIR / "data" / "processed" / f"{recording_id}.wav"

    if wav_path.is_file() and wav_path.stat().st_size > 0:
        print(f"[REUSED] Found existing WAV file: {wav_path.name}")
        print(f"WAV Path   : {wav_path}")
        print(f"WAV Size   : {wav_path.stat().st_size / 1024:.2f} KB")
    else:
        print("[EXTRACTING] WAV not found or empty. Extracting audio from video...")
        try:
            extracted_path_str = extract_audio_from_video(video_path)
            wav_path = Path(extracted_path_str)
            print(f"[SUCCESS] Audio extracted: {wav_path.name}")
            print(f"WAV Path   : {wav_path}")
            print(f"WAV Size   : {wav_path.stat().st_size / 1024:.2f} KB")
        except Exception as exc:
            print(f"[ERROR] Audio extraction failed: {type(exc).__name__}: {exc}")
            return

    # ---------------------------------------------------------
    # Step 3: Transcription via Groq Whisper
    # ---------------------------------------------------------
    print("\n--- Step 3: Speech Transcription (Groq Whisper) ---")
    try:
        transcription = transcribe_audio(wav_path)
        full_text = transcription.get("text", "")
        segments = transcription.get("segments", [])

        print(f"[SUCCESS] Audio transcription complete.")
        print(f"Detected Language : {transcription.get('language', 'unknown')}")
        print(f"Duration (sec)    : {transcription.get('duration', 0.0):.2f}s")
        print(f"Total Segments    : {len(segments)}")
        print(f"Transcript Length : {len(full_text)} characters")

        if not full_text.strip() or not segments:
            print("[ERROR] Transcription returned empty text or no segments.")
            return

    except Exception as exc:
        print(f"[ERROR] Audio transcription failed: {type(exc).__name__}: {exc}")
        return

    # ---------------------------------------------------------
    # Step 4: Transcript Chunking
    # ---------------------------------------------------------
    print("\n--- Step 4: Transcript Chunking ---")
    try:
        chunks = chunk_transcript(
            transcription,
            max_duration_seconds=30,
            max_characters=1000,
        )
        print(f"[SUCCESS] Transcript chunked into {len(chunks)} timestamp-aligned chunk(s).")
        for chunk in chunks:
            print(
                f"  - Chunk {chunk['chunk_index']:02d}: "
                f"[{chunk['start']:6.2f}s -> {chunk['end']:6.2f}s] "
                f"({chunk['segment_count']} segment(s), {len(chunk['text'])} chars)"
            )
    except Exception as exc:
        print(f"[ERROR] Transcript chunking failed: {type(exc).__name__}: {exc}")
        return

    # ---------------------------------------------------------
    # Step 5: Embedding Generation via Gemini
    # ---------------------------------------------------------
    print("\n--- Step 5: Document Embedding Generation (Gemini) ---")
    embeddings = []
    try:
        for idx, chunk in enumerate(chunks):
            print(
                f"  - Generating embedding for Chunk {chunk['chunk_index']:02d} "
                f"({len(chunk['text'])} chars)...",
                end="",
                flush=True,
            )
            emb = embed_document(chunk["text"])
            embeddings.append(emb)
            print(f" Done ({len(emb)} dims)")

        # Verify embedding dimensions
        all_correct_dim = all(len(e) == EMBEDDING_DIMENSIONS for e in embeddings)
        if not all_correct_dim:
            print(f"[ERROR] Not all embeddings matched expected dimension {EMBEDDING_DIMENSIONS}.")
            return

        print(f"[SUCCESS] Generated {len(embeddings)} embedding(s) of dimension {EMBEDDING_DIMENSIONS}.")

    except Exception as exc:
        print(f"\n[ERROR] Embedding generation failed: {type(exc).__name__}: {exc}")
        return

    # ---------------------------------------------------------
    # Step 6: Persistent ChromaDB Storage
    # ---------------------------------------------------------
    print("\n--- Step 6: Persistent ChromaDB Storage ---")
    try:
        # Pre-check current database count
        collection = get_vector_collection()
        initial_count = collection.count()
        print(f"ChromaDB Collection Name : {collection.name}")
        print(f"Pre-insertion Record Count: {initial_count}")

        # Store chunks and embeddings
        result = store_transcript_chunks(
            recording_id=recording_id,
            chunks=chunks,
            embeddings=embeddings,
        )

        final_count = collection.count()
        print(f"[SUCCESS] ChromaDB storage complete.")
        print(f"Post-insertion Count    : {final_count}")
        print(f"ChromaDB Stored Count   : {result['stored_count']}")
        print(f"Stored Chunk IDs        : {result['ids']}")

    except Exception as exc:
        print(f"[ERROR] ChromaDB storage failed: {type(exc).__name__}: {exc}")
        return

    # ---------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Recording ID         : {recording_id}")
    print(f"WAV Path             : {wav_path}")
    print(f"Transcript Chunks    : {len(chunks)}")
    print(f"Embedding Dimensions : {EMBEDDING_DIMENSIONS}")
    print(f"ChromaDB Stored IDs  : {len(result['ids'])} record(s)")
    print(f"ChromaDB Total Count : {final_count}")
    print("=" * 70)
    print("Video ingestion and ChromaDB persistence completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_store_pipeline()
