import logging
from pathlib import Path
from typing import Any, Dict, List, Set
import chromadb
from chromadb.api.models.Collection import Collection

from app.schemas.vector_store import TranscriptChunkRecord

# Configure module logger for server-side diagnostics
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Storage Path & Collection Configuration
# ---------------------------------------------------------
# Dynamic path resolution anchored to the project structure:
# __file__ -> backend/app/services/vector_store_service.py
# .parent -> services/
# .parent.parent -> app/
# .parent.parent.parent -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

# Persistent database directory: backend/data/chroma/
# Excluded by .gitignore to keep vector index binaries out of version control
CHROMA_DATA_DIR = BACKEND_DIR / "data" / "chroma"

# Name of the collection dedicated to storing timestamp-aligned transcript chunks
COLLECTION_NAME = "echotrace_transcript_chunks"


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Initializes and returns a persistent ChromaDB client anchored to backend/data/chroma/.

    Returns:
        chromadb.PersistentClient: Connected persistent client instance.

    Raises:
        Exception: If the storage directory cannot be created or ChromaDB fails to initialize.
    """
    try:
        # Ensure the destination data directory exists before client startup
        CHROMA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(CHROMA_DATA_DIR))
    except Exception as exc:
        logger.exception(
            "Failed to initialize ChromaDB PersistentClient at '%s': %s",
            CHROMA_DATA_DIR,
            exc,
        )
        raise


def get_vector_collection() -> Collection:
    """
    Initializes and returns the ChromaDB collection for EchoTrace transcript chunks.

    Configuration:
    - Collection Name: 'echotrace_transcript_chunks'
    - Distance Metric: Cosine similarity ('hnsw:space': 'cosine')
      Cosine distance is optimal for normalized text embeddings (such as Gemini 768-dim vectors).
    - Idempotency: Uses 'get_or_create_collection' so repeated invocations return the
      existing collection without creating duplicates or raising collision errors.

    Returns:
        Collection: The active ChromaDB collection ready for vector storage operations.

    Raises:
        Exception: If collection creation or connection fails.
    """
    try:
        client = get_chroma_client()

        logger.info(
            "Accessing ChromaDB collection '%s' (storage path: %s)...",
            COLLECTION_NAME,
            CHROMA_DATA_DIR,
        )

        # Retrieve the collection if it exists, or create it with cosine distance configuration
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "ChromaDB collection '%s' ready. Current record count: %d",
            COLLECTION_NAME,
            collection.count(),
        )

        return collection

    except Exception as exc:
        logger.exception(
            "Failed to retrieve or create ChromaDB collection '%s': %s",
            COLLECTION_NAME,
            exc,
        )
        raise


# Required keys present in each chunk dictionary from chunk_transcript()
EXPECTED_CHUNK_FIELDS = {"chunk_index", "start", "end", "text", "segment_count"}


def store_transcript_chunks(
    recording_id: str,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> Dict[str, Any]:
    """
    Validates, formats, and persists timestamp-aligned transcript chunks with their
    corresponding 768-dimensional vector embeddings into the ChromaDB vector store.

    Validation & Safety Rules:
    1. Validates that recording_id is a non-empty string (not boolean).
    2. Validates that chunks and embeddings lists are non-empty and of equal length.
    3. Validates each chunk dictionary has exact expected fields: chunk_index, start, end, text, segment_count.
    4. Validates each chunk and embedding pair using TranscriptChunkRecord schema (rejects booleans,
       requires 768 finite numeric embedding dimensions, finite non-negative timestamps, start <= end).
    5. Validates that every chunk within the batch has a unique chunk_index (rejects duplicate indices).
    6. Validates the entire batch before calling ChromaDB upsert; if any record is invalid,
       no partial insertion takes place.
    7. Generates deterministic chunk IDs: '{recording_id}_{chunk_index:06d}'.
    8. Stores chunks via collection.upsert() with:
       - ids: generated deterministic chunk IDs
       - documents: transcript chunk text
       - embeddings: 768-dimensional vector embeddings
       - metadatas: recording_id, chunk_index, start, end, segment_count (embeddings excluded).

    Args:
        recording_id (str): Unique identifier of the parent media recording.
        chunks (List[Dict[str, Any]]): List of chunk dictionaries produced by chunk_transcript().
        embeddings (List[List[float]]): List of 768-dimensional float embedding vectors.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - recording_id (str): Sanitized recording identifier.
            - stored_count (int): Number of chunks successfully upserted.
            - ids (List[str]): List of generated deterministic chunk IDs.

    Raises:
        TypeError: If input types are invalid (non-string recording_id, non-list chunks/embeddings).
        ValueError: If lists are empty/mismatched or chunk fields are missing/unexpected.
        pydantic.ValidationError: If any chunk record fails schema validation.
        Exception: If ChromaDB upsert fails.
    """
    # ---------------------------------------------------------
    # 1. Validate Input Types & High-Level Constraints
    # ---------------------------------------------------------
    if isinstance(recording_id, bool) or not isinstance(recording_id, str):
        raise TypeError(
            f"recording_id must be a string, got {type(recording_id).__name__}."
        )

    clean_recording_id = recording_id.strip()
    if not clean_recording_id:
        raise ValueError("recording_id must be a non-empty string.")

    if isinstance(chunks, bool) or not isinstance(chunks, list):
        raise TypeError(
            f"chunks must be a list of dictionaries, got {type(chunks).__name__}."
        )

    if isinstance(embeddings, bool) or not isinstance(embeddings, list):
        raise TypeError(
            f"embeddings must be a list of embedding vectors, got {type(embeddings).__name__}."
        )

    if len(chunks) == 0:
        raise ValueError("chunks list cannot be empty.")

    if len(embeddings) == 0:
        raise ValueError("embeddings list cannot be empty.")

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks and embeddings must have equal lengths. "
            f"Got {len(chunks)} chunks and {len(embeddings)} embeddings."
        )

    # ---------------------------------------------------------
    # 2. Validate All Records Before Persistence (Zero Partial Writes)
    # ---------------------------------------------------------
    logger.info(
        "Validating %d transcript chunk(s) for recording_id '%s'...",
        len(chunks),
        clean_recording_id,
    )

    validated_records: List[TranscriptChunkRecord] = []

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        if isinstance(chunk, bool) or not isinstance(chunk, dict):
            raise TypeError(
                f"Chunk at index {idx} must be a dictionary, got {type(chunk).__name__}."
            )

        missing_fields = EXPECTED_CHUNK_FIELDS - chunk.keys()
        if missing_fields:
            raise ValueError(
                f"Chunk at index {idx} is missing required fields: {sorted(missing_fields)}."
            )

        extra_fields = chunk.keys() - EXPECTED_CHUNK_FIELDS
        if extra_fields:
            raise ValueError(
                f"Chunk at index {idx} contains unexpected fields: {sorted(extra_fields)}."
            )

        # Validate through TranscriptChunkRecord schema
        # This rigorously checks types, booleans, finiteness, dimensionality (768), and start <= end
        record = TranscriptChunkRecord(
            recording_id=clean_recording_id,
            chunk_index=chunk["chunk_index"],
            start=chunk["start"],
            end=chunk["end"],
            text=chunk["text"],
            segment_count=chunk["segment_count"],
            embedding=embedding,
        )
        validated_records.append(record)

    # ---------------------------------------------------------
    # 3. Validate Unique chunk_index Across All Chunks
    # ---------------------------------------------------------
    seen_indices: Set[int] = set()
    duplicate_indices: Set[int] = set()

    for record in validated_records:
        if record.chunk_index in seen_indices:
            duplicate_indices.add(record.chunk_index)
        seen_indices.add(record.chunk_index)

    if duplicate_indices:
        sorted_duplicates = sorted(duplicate_indices)
        raise ValueError(
            f"Duplicate chunk_index values found in chunks for recording_id '{clean_recording_id}': {sorted_duplicates}."
        )

    # ---------------------------------------------------------
    # 4. Construct ChromaDB Batch Payloads
    # ---------------------------------------------------------
    batch_ids: List[str] = []
    batch_documents: List[str] = []
    batch_embeddings: List[List[float]] = []
    batch_metadatas: List[Dict[str, Any]] = []

    for record in validated_records:
        # Deterministic chunk ID: {recording_id}_{chunk_index:06d}
        chunk_id = f"{record.recording_id}_{record.chunk_index:06d}"

        batch_ids.append(chunk_id)
        batch_documents.append(record.text)
        batch_embeddings.append(record.embedding)
        batch_metadatas.append({
            "recording_id": record.recording_id,
            "chunk_index": record.chunk_index,
            "start": record.start,
            "end": record.end,
            "segment_count": record.segment_count,
        })

    # ---------------------------------------------------------
    # 5. Upsert into ChromaDB Collection
    # ---------------------------------------------------------
    collection = get_vector_collection()

    logger.info(
        "Upserting %d transcript chunk(s) into ChromaDB collection '%s' for recording_id '%s'...",
        len(batch_ids),
        collection.name,
        clean_recording_id,
    )

    try:
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )
    except Exception as exc:
        logger.exception(
            "Failed to upsert %d transcript chunk(s) into ChromaDB for recording_id '%s': %s",
            len(batch_ids),
            clean_recording_id,
            exc,
        )
        raise

    logger.info(
        "Successfully stored %d chunk(s) in ChromaDB collection '%s' for recording_id '%s'.",
        len(batch_ids),
        collection.name,
        clean_recording_id,
    )

    return {
        "recording_id": clean_recording_id,
        "stored_count": len(batch_ids),
        "ids": batch_ids,
    }
