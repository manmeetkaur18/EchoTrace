import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import chromadb
from chromadb.api.models.Collection import Collection

from app.schemas.vector_store import TranscriptChunkRecord
from app.services.registry_service import (
    RegistryError,
    RegistryUnavailableError,
    StaleVersionError,
    acquire_recording_lock,
    activate_version,
    get_active_version,
    get_recording_generation,
    get_version_statuses,
    init_registry,
    mark_version_failed,
    stage_version,
    update_version_heartbeat,
)

# Configure module logger for server-side diagnostics
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Storage Path & Collection Configuration
# ---------------------------------------------------------
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
    - Idempotency: Uses 'get_or_create_collection' so repeated invocations return the
      existing collection without creating duplicates.

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
    version: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
    collection: Optional[Collection] = None,
) -> Dict[str, Any]:
    """
    Persists timestamp-aligned transcript chunks with their vector embeddings into ChromaDB
    under an isolated UUID-based version, atomically activating the version in the SQLite registry
    only after full verification.

    Safe Version Activation Flow:
    1. Validate input parameters and all chunk/embedding pairs.
    2. Acquire a recording-scoped lock to serialize simultaneous ingestions of this recording.
    3. Generate a unique version UUID (or use provided version) and read current generation.
    4. Stage the version in SQLite registry.
    5. Construct versioned chunk IDs: '{recording_id}_v_{version}_{chunk_index:06d}'.
    6. Upsert the new isolated version into ChromaDB with periodic heartbeat checkpoints.
    7. Read back from ChromaDB to verify all chunk IDs and metadata exist.
    8. Atomically activate the new version in SQLite (with generation CAS concurrency check).
    9. Post-activation cleanup: prune previous version or legacy chunks for this recording.
       If cleanup fails, log warning; retrieval remains unaffected because inactive versions are excluded.

    Error Handling & Resilience:
    - Every failure between staging and activation marks the staged version as 'failed'.
    - Already activated versions are NEVER downgraded or marked failed on post-activation cleanup issues.
    - Original exceptions are preserved and propagated without being masked.

    Args:
        recording_id (str): Unique identifier of the parent media recording.
        chunks (List[Dict[str, Any]]): List of chunk dictionaries produced by chunk_transcript().
        embeddings (List[List[float]]): List of 768-dimensional float embedding vectors.
        version (Optional[str]): Optional custom UUID version string (generated automatically if omitted).
        db_path (Optional[Union[str, Path]]): Optional path to registry SQLite database (for test isolation).
        collection (Optional[Collection]): Optional ChromaDB collection instance (for test isolation).

    Returns:
        Dict[str, Any]:
            - recording_id (str): Sanitized recording identifier.
            - version (str): Activated version UUID string.
            - stored_count (int): Number of chunks successfully stored and verified.
            - ids (List[str]): List of versioned chunk IDs.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If lists are empty/mismatched or chunk data is malformed.
        RuntimeError: If verification or vector store storage fails.
        StaleVersionError: If a concurrent ingestion already activated a newer generation.
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

    # Validate version string if explicitly passed
    version_id: str
    if version is not None:
        if isinstance(version, bool) or not isinstance(version, str):
            raise TypeError(f"version must be a string, got {type(version).__name__}.")
        clean_version = version.strip()
        if not clean_version:
            raise ValueError("version must be a non-empty string if provided.")
        version_id = clean_version
    else:
        version_id = uuid.uuid4().hex

    # ---------------------------------------------------------
    # 2. Acquire Recording-Scoped Lock & Concurrency Setup
    # ---------------------------------------------------------
    with acquire_recording_lock(clean_recording_id):
        # Initialize registry schema if not already present
        init_registry(db_path=db_path)

        # Inspect current generation for optimistic concurrency check
        expected_generation = get_recording_generation(clean_recording_id, db_path=db_path)

        # Stage the new version in the registry
        stage_version(clean_recording_id, version_id, expected_generation, db_path=db_path)

        version_activated = False

        try:
            # ---------------------------------------------------------
            # 3. Validate All Records Before Persistence (Zero Partial Writes)
            # ---------------------------------------------------------
            logger.info(
                "Validating %d transcript chunk(s) for recording_id '%s' (version '%s')...",
                len(chunks),
                clean_recording_id,
                version_id,
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

                record = TranscriptChunkRecord(
                    recording_id=clean_recording_id,
                    chunk_index=chunk["chunk_index"],
                    start=chunk["start"],
                    end=chunk["end"],
                    text=chunk["text"],
                    segment_count=chunk["segment_count"],
                    embedding=embedding,
                    version=version_id,
                )
                validated_records.append(record)

            # ---------------------------------------------------------
            # 4. Validate Unique chunk_index Across All Chunks
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
            # 5. Construct ChromaDB Batch Payloads with Versioned Chunk IDs
            # Format: {recording_id}_v_{version}_{chunk_index:06d}
            # ---------------------------------------------------------
            batch_ids: List[str] = []
            batch_documents: List[str] = []
            batch_embeddings: List[List[float]] = []
            batch_metadatas: List[Dict[str, Any]] = []

            for record in validated_records:
                chunk_id = f"{record.recording_id}_v_{version_id}_{record.chunk_index:06d}"

                batch_ids.append(chunk_id)
                batch_documents.append(record.text)
                batch_embeddings.append(record.embedding)
                batch_metadatas.append({
                    "recording_id": record.recording_id,
                    "version": version_id,
                    "chunk_index": record.chunk_index,
                    "start": record.start,
                    "end": record.end,
                    "segment_count": record.segment_count,
                })

            target_col = collection if collection is not None else get_vector_collection()

            # Heartbeat checkpoint 1: prior to ChromaDB upsert
            if not update_version_heartbeat(clean_recording_id, version_id, db_path=db_path):
                raise StaleVersionError(
                    f"Ingestion aborted for recording '{clean_recording_id}' version '{version_id}': "
                    f"heartbeat lease expired or version is no longer staged."
                )

            # ---------------------------------------------------------
            # 6. Upsert Versioned Chunks into ChromaDB
            # ---------------------------------------------------------
            logger.info(
                "Upserting %d isolated chunk(s) into ChromaDB for recording '%s' under version '%s'...",
                len(batch_ids),
                clean_recording_id,
                version_id,
            )

            target_col.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )

            # Heartbeat checkpoint 2: prior to verification
            if not update_version_heartbeat(clean_recording_id, version_id, db_path=db_path):
                raise StaleVersionError(
                    f"Ingestion aborted for recording '{clean_recording_id}' version '{version_id}': "
                    f"heartbeat lease expired or version is no longer staged."
                )

            # ---------------------------------------------------------
            # 7. Verification: Confirm all expected new chunks and metadata exist
            # ---------------------------------------------------------
            logger.info(
                "Verifying %d stored chunk(s) for recording '%s' version '%s'...",
                len(batch_ids),
                clean_recording_id,
                version_id,
            )

            verification_records = target_col.get(
                ids=batch_ids,
                include=["metadatas"],
            )
            found_ids = set(verification_records.get("ids", []) if verification_records else [])
            expected_ids = set(batch_ids)

            if found_ids != expected_ids:
                missing_ids = expected_ids - found_ids
                raise RuntimeError(
                    f"Verification failed for recording '{clean_recording_id}' version '{version_id}': "
                    f"missing {len(missing_ids)} chunk(s) in ChromaDB."
                )

            # Confirm metadata integrity on verified records
            metas = verification_records.get("metadatas", []) or []
            for meta in metas:
                if not isinstance(meta, dict):
                    raise RuntimeError("Verification failed: stored metadata is not a dictionary.")
                if meta.get("recording_id") != clean_recording_id or meta.get("version") != version_id:
                    raise RuntimeError(
                        f"Verification failed: metadata mismatch for recording '{clean_recording_id}' version '{version_id}'."
                    )

            logger.info(
                "Verification successful: all %d chunks confirmed for version '%s'.",
                len(batch_ids),
                version_id,
            )

            # Heartbeat checkpoint 3: prior to activation
            if not update_version_heartbeat(clean_recording_id, version_id, db_path=db_path):
                raise StaleVersionError(
                    f"Ingestion aborted for recording '{clean_recording_id}' version '{version_id}': "
                    f"heartbeat lease expired or version is no longer staged."
                )

            # ---------------------------------------------------------
            # 8. Atomically Activate Version in SQLite Registry
            # ---------------------------------------------------------
            logger.info(
                "Activating version '%s' in registry for recording '%s' (expected_generation=%d)...",
                version_id,
                clean_recording_id,
                expected_generation,
            )

            activate_version(
                recording_id=clean_recording_id,
                version=version_id,
                expected_generation=expected_generation,
                db_path=db_path,
            )
            version_activated = True

        except Exception as exc:
            # Mark version failed ONLY if failure occurred before successful activation
            if not version_activated:
                logger.warning(
                    "Ingestion failed before activation for recording '%s' version '%s': %s",
                    clean_recording_id,
                    version_id,
                    exc,
                )
                try:
                    mark_version_failed(clean_recording_id, version_id, db_path=db_path)
                except Exception as mark_err:
                    logger.warning(
                        "Failed to mark version '%s' as failed for recording '%s': %s",
                        version_id,
                        clean_recording_id,
                        mark_err,
                    )
            raise

        # ---------------------------------------------------------
        # 9. Post-Activation Cleanup of Previous / Legacy Chunks
        # ---------------------------------------------------------
        # Only executed AFTER successful activation in the registry.
        # If cleanup fails, retrieval is unaffected because inactive versions are excluded by queries.
        try:
            existing_records = target_col.get(where={"recording_id": clean_recording_id})
            all_existing_ids = set(existing_records.get("ids", []) if existing_records else [])
            obsolete_ids = list(all_existing_ids - set(batch_ids))

            if obsolete_ids:
                logger.info(
                    "Post-activation cleanup: pruning %d obsolete chunk(s) for recording '%s'...",
                    len(obsolete_ids),
                    clean_recording_id,
                )
                target_col.delete(ids=obsolete_ids)
                logger.info(
                    "Successfully pruned %d obsolete chunk(s) for recording '%s'.",
                    len(obsolete_ids),
                    clean_recording_id,
                )
        except Exception as clean_exc:
            logger.warning(
                "Version '%s' activated successfully, but post-activation cleanup of obsolete chunks "
                "failed for recording '%s': %s. Active retrieval remains consistent.",
                version_id,
                clean_recording_id,
                clean_exc,
            )

        logger.info(
            "Versioned ingestion complete for recording '%s' (version '%s', %d chunks).",
            clean_recording_id,
            version_id,
            len(batch_ids),
        )

        return {
            "recording_id": clean_recording_id,
            "version": version_id,
            "stored_count": len(batch_ids),
            "ids": batch_ids,
        }


def cleanup_inactive_chunks(
    recording_id: str,
    collection: Optional[Collection] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Safely purges abandoned or superseded chunks for a given recording from ChromaDB.

    Safety Guarantees:
    1. Acquires the recording-scoped lock to synchronize with active ingestions.
    2. Reads the authoritative active version and version status mapping from SQLite.
    3. Always preserves the active version.
    4. Always preserves versions whose status is 'staged' (in-flight ingestions).
    5. Deletes ONLY versions confirmed safe to remove ('failed', 'superseded').
    6. Preserves legacy unversioned chunks when there is no active version and the recording
       is still an unversioned legacy recording.
    7. If the registry is unavailable or corrupted, aborts cleanup safely without deleting data.
    8. Does not delete chunks based only on their absence from the active version.
    9. Does not hold a SQLite write transaction open while calling ChromaDB.

    Args:
        recording_id (str): Recording identifier.
        collection (Optional[Collection]): ChromaDB collection instance.
        db_path (Optional[Union[str, Path]]): Path to SQLite registry database.

    Returns:
        List[str]: List of chunk IDs successfully deleted.

    Raises:
        RegistryUnavailableError: If SQLite registry is unreachable or corrupted.
    """
    clean_id = str(recording_id).strip()
    if not clean_id:
        return []

    # 1. Acquire the recording-scoped lock
    with acquire_recording_lock(clean_id):
        # 2. Read authoritative state from registry
        try:
            active_version = get_active_version(clean_id, db_path=db_path)
            version_statuses = get_version_statuses(clean_id, db_path=db_path)
        except RegistryError as reg_err:
            logger.error(
                "Registry error during cleanup for recording '%s': %s. Aborting cleanup safely.",
                clean_id,
                reg_err,
            )
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error accessing registry during cleanup for recording '%s': %s. Aborting cleanup safely.",
                clean_id,
                exc,
            )
            raise RegistryUnavailableError(
                f"Registry unavailable during cleanup for recording '{clean_id}': {exc}"
            ) from exc

        target_col = collection if collection is not None else get_vector_collection()
        existing = target_col.get(
            where={"recording_id": clean_id},
            include=["metadatas"],
        )
        if not existing or not existing.get("ids"):
            return []

        ids = existing.get("ids", [])
        metadatas = existing.get("metadatas", []) or []

        to_delete: List[str] = []
        for chunk_id, meta in zip(ids, metadatas):
            chunk_ver = meta.get("version") if isinstance(meta, dict) else None

            # Rule A: Unversioned legacy chunks
            if chunk_ver is None:
                if active_version is None:
                    # Preserved: recording is still in legacy state
                    continue
                else:
                    # Recording has migrated to versioned; legacy unversioned chunks are safe to purge
                    to_delete.append(chunk_id)
                    continue

            # Rule B: Always preserve the active version
            if chunk_ver == active_version:
                continue

            # Rule C: Check ingestion status from SQLite
            status = version_statuses.get(chunk_ver)

            # Rule D: Always preserve versions whose ingestion status is 'staged'
            if status == "staged":
                logger.info(
                    "Preserving chunk %s: version '%s' is currently staged in active ingestion.",
                    chunk_id,
                    chunk_ver,
                )
                continue

            # Rule E: Delete ONLY versions confirmed safe to remove ('failed', 'superseded')
            if status in ("failed", "superseded"):
                to_delete.append(chunk_id)
            else:
                logger.warning(
                    "Chunk %s has version '%s' with unconfirmed status '%s'; skipping deletion for safety.",
                    chunk_id,
                    chunk_ver,
                    status,
                )

        if to_delete:
            logger.info(
                "Purging %d confirmed safe obsolete chunk(s) for recording '%s'...",
                len(to_delete),
                clean_id,
            )
            target_col.delete(ids=to_delete)

        return to_delete
