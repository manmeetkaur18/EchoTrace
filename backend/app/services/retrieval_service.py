import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from chromadb.api.models.Collection import Collection

from app.services.embedding_service import embed_query
from app.services.registry_service import (
    RegistryError,
    get_active_version,
    list_active_versions,
)
from app.services.vector_store_service import get_vector_collection

# Configure module logger for server-side diagnostics
logger = logging.getLogger(__name__)


def _is_safe_finite_number(val: Any) -> bool:
    """
    Safely checks if a value is a numeric finite float or int without raising OverflowError
    on unusually large integers.
    """
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    try:
        return math.isfinite(val)
    except (TypeError, ValueError, OverflowError):
        return False


def search_transcript_chunks(
    query: str,
    recording_id: Optional[str] = None,
    top_k: int = 5,
    db_path: Optional[Union[str, Path]] = None,
    collection: Optional[Collection] = None,
) -> List[Dict[str, Any]]:
    """
    Searches indexed transcript chunks using vector similarity (cosine distance).

    Versioned Retrieval Flow:
    1. Validates input parameters (query, recording_id, top_k).
    2. If recording_id is provided, queries the active-version registry:
       - If active version exists: applies compound filter {"$and": [{"recording_id": ...}, {"version": active_version}]}.
       - If registry confirms no entry (legacy recording): retains legacy {"recording_id": ...} filter.
       - If registry is unavailable or corrupted: fails safely by raising RuntimeError (NEVER falls back to unfiltered/legacy!).
    3. If recording_id is None (global search): validates active versions map from registry to filter out
       superseded or unactivated version chunks at response time.
    4. Obtains the persistent ChromaDB collection.
    5. Handles empty collections gracefully by returning an empty list without unnecessary queries.
    6. Generates a 768-dimensional query embedding via embed_query().
    7. Queries ChromaDB using cosine distance and safe where-filter.
    8. Validates and formats matching chunk results, preserving timestamps, chunk IDs, and distances.

    Args:
        query (str): User natural language search query.
        recording_id (Optional[str]): Optional recording ID to scope results to a single media file.
        top_k (int): Maximum number of top matching chunks to retrieve (default: 5).
        db_path (Optional[Union[str, Path]]): Optional path to SQLite registry (for test isolation).
        collection (Optional[Collection]): Optional ChromaDB collection instance (for test isolation).

    Returns:
        List[Dict[str, Any]]: Ranked list of matching chunks sorted by cosine distance (lower is better),
        where each item contains:
            - chunk_id (str): Unique chunk identifier (e.g., '{recording_id}_v_{version}_{chunk_index:06d}').
            - recording_id (str): ID of the media recording this chunk belongs to.
            - chunk_index (int): Sequential index of the chunk within the recording.
            - start (float): Timestamp in seconds marking the start of the chunk.
            - end (float): Timestamp in seconds marking the end of the chunk.
            - text (str): Transcript text content of the chunk.
            - distance (float): Cosine distance between query and chunk embedding.

    Raises:
        TypeError: If query, recording_id, or top_k are of invalid types.
        ValueError: If query or recording_id are empty/whitespace, or top_k <= 0.
        RuntimeError: If embedding generation, registry resolution, or ChromaDB query fails.
    """
    # ---------------------------------------------------------
    # 1. Input Validation
    # ---------------------------------------------------------
    if isinstance(query, bool) or not isinstance(query, str):
        raise TypeError(f"Query must be a string, got {type(query).__name__}.")

    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Query must not be empty or whitespace-only.")

    clean_recording_id: Optional[str] = None
    if recording_id is not None:
        if isinstance(recording_id, bool) or not isinstance(recording_id, str):
            raise TypeError(
                f"recording_id must be a string or None, got {type(recording_id).__name__}."
            )
        clean_recording_id = recording_id.strip()
        if not clean_recording_id:
            raise ValueError(
                "recording_id must not be empty or whitespace-only if provided."
            )

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}.")

    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer greater than 0, got {top_k}.")

    # ---------------------------------------------------------
    # 2. Resolve Active Version from Registry
    # ---------------------------------------------------------
    where_filter: Optional[Dict[str, Any]] = None
    active_versions_map: Optional[Dict[str, str]] = None

    if clean_recording_id is not None:
        try:
            active_version = get_active_version(clean_recording_id, db_path=db_path)
        except RegistryError as reg_err:
            logger.error(
                "Registry error while resolving active version for recording '%s': %s",
                clean_recording_id,
                reg_err,
            )
            raise RuntimeError(
                f"Active-version registry error for recording '{clean_recording_id}': {type(reg_err).__name__}."
            ) from reg_err
        except Exception as exc:
            logger.error(
                "Unexpected error accessing active-version registry for recording '%s': %s",
                clean_recording_id,
                exc,
            )
            raise RuntimeError(
                f"Active-version registry unavailable for recording '{clean_recording_id}': {type(exc).__name__}."
            ) from exc

        if active_version is not None:
            # Versioned recording: query must match both recording_id and active version
            where_filter = {
                "$and": [
                    {"recording_id": clean_recording_id},
                    {"version": active_version},
                ]
            }
            logger.info(
                "Retrieval scoped to versioned recording '%s' (active version: '%s').",
                clean_recording_id,
                active_version,
            )
        else:
            # Unversioned legacy recording: registry confirmed no entry, use legacy filter
            where_filter = {"recording_id": clean_recording_id}
            logger.info(
                "Retrieval scoped to legacy unversioned recording '%s'.",
                clean_recording_id,
            )
    else:
        # Global search: load active versions mapping to filter out inactive chunks post-query
        try:
            active_versions_map = list_active_versions(db_path=db_path)
        except Exception as exc:
            logger.error("Failed to query active versions map for global retrieval: %s", exc)
            raise RuntimeError("Active-version registry unavailable for vector search.") from exc

    # ---------------------------------------------------------
    # 3. ChromaDB Collection Access & Empty Collection Check
    # ---------------------------------------------------------
    target_col: Collection
    if collection is not None:
        target_col = collection
    else:
        try:
            target_col = get_vector_collection()
        except Exception as exc:
            logger.error(
                "Failed to access ChromaDB collection: %s",
                type(exc).__name__,
            )
            raise RuntimeError(
                f"Failed to access vector collection: {type(exc).__name__}."
            ) from exc

    try:
        total_records = target_col.count()
    except Exception as exc:
        logger.error(
            "Failed to retrieve record count from ChromaDB: %s",
            type(exc).__name__,
        )
        raise RuntimeError(
            f"Failed to inspect vector store count: {type(exc).__name__}."
        ) from exc

    # Gracefully return empty results if collection has no indexed data
    if total_records == 0:
        logger.info(
            "Vector store collection is empty. Returning 0 results without querying."
        )
        return []

    # ---------------------------------------------------------
    # 4. Query Embedding Generation
    # ---------------------------------------------------------
    logger.info(
        "Generating query embedding (query_length=%d chars)...",
        len(clean_query),
    )

    try:
        query_embedding = embed_query(clean_query)
    except Exception as exc:
        logger.error(
            "Failed to generate embedding for query: %s",
            type(exc).__name__,
        )
        raise RuntimeError(
            f"Query embedding generation failed: {type(exc).__name__}."
        ) from exc

    # ---------------------------------------------------------
    # 5. ChromaDB Vector Query Execution
    # ---------------------------------------------------------
    logger.info(
        "Executing ChromaDB query (top_k=%d, scoped_recording_id=%s, filter=%s)...",
        top_k,
        clean_recording_id,
        where_filter,
    )

    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    try:
        raw_results = target_col.query(**query_kwargs)
    except Exception as exc:
        logger.error(
            "ChromaDB vector search query failed: %s",
            type(exc).__name__,
        )
        raise RuntimeError(
            f"Vector search query failed: {type(exc).__name__}."
        ) from exc

    # ---------------------------------------------------------
    # 6. Response Processing & Result Sanitization
    # ---------------------------------------------------------
    ids_batch = raw_results.get("ids") or []
    docs_batch = raw_results.get("documents") or []
    metas_batch = raw_results.get("metadatas") or []
    dists_batch = raw_results.get("distances") or []

    result_ids = ids_batch[0] if ids_batch else []
    result_docs = docs_batch[0] if docs_batch else []
    result_metas = metas_batch[0] if metas_batch else []
    result_dists = dists_batch[0] if dists_batch else []

    formatted_results: List[Dict[str, Any]] = []

    for idx, chunk_id in enumerate(result_ids):
        # 1. Validate chunk_id
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, str) or not chunk_id.strip():
            logger.warning(
                "Skipping search result at index %d: invalid or empty chunk_id (%s).",
                idx,
                chunk_id,
            )
            continue
        clean_chunk_id = chunk_id.strip()

        doc = result_docs[idx] if idx < len(result_docs) else None
        meta = result_metas[idx] if idx < len(result_metas) else None
        dist = result_dists[idx] if idx < len(result_dists) else None

        # Validate metadata dictionary presence
        if not isinstance(meta, dict):
            logger.warning(
                "Skipping search result %s at index %d: metadata is missing or not a dict (%s).",
                clean_chunk_id,
                idx,
                type(meta).__name__,
            )
            continue

        rec_id = meta.get("recording_id")
        chunk_idx = meta.get("chunk_index")
        start = meta.get("start")
        end = meta.get("end")
        chunk_ver = meta.get("version")

        # Validate recording_id
        if not isinstance(rec_id, str) or not rec_id.strip():
            logger.warning(
                "Skipping search result %s at index %d: invalid or empty recording_id in metadata.",
                clean_chunk_id,
                idx,
            )
            continue
        clean_rec = str(rec_id).strip()

        # Ensure inactive or abandoned version chunks are excluded (for both scoped and global queries)
        if clean_recording_id is not None:
            if active_version is not None:
                if chunk_ver != active_version:
                    logger.warning(
                        "Skipping chunk %s for recording %s: chunk version '%s' != active version '%s'.",
                        clean_chunk_id,
                        clean_rec,
                        chunk_ver,
                        active_version,
                    )
                    continue
            else:
                # Legacy recording: exclude chunks tagged with an unactivated version
                if chunk_ver is not None:
                    logger.warning(
                        "Skipping chunk %s for legacy recording %s: contains unactivated version tag '%s'.",
                        clean_chunk_id,
                        clean_rec,
                        chunk_ver,
                    )
                    continue
        elif active_versions_map is not None:
            if clean_rec in active_versions_map:
                active_ver = active_versions_map[clean_rec]
                if chunk_ver != active_ver:
                    logger.warning(
                        "Skipping chunk %s for recording %s: chunk version '%s' != active version '%s'.",
                        clean_chunk_id,
                        clean_rec,
                        chunk_ver,
                        active_ver,
                    )
                    continue
            else:
                # Legacy recording: exclude chunks tagged with an unactivated version
                if chunk_ver is not None:
                    logger.warning(
                        "Skipping chunk %s for legacy recording %s: contains unactivated version tag '%s'.",
                        clean_chunk_id,
                        clean_rec,
                        chunk_ver,
                    )
                    continue

        # Validate numeric fields with OverflowError/TypeError guards
        try:
            if isinstance(chunk_idx, bool) or not isinstance(chunk_idx, int) or chunk_idx < 0:
                logger.warning(
                    "Skipping search result %s at index %d: invalid chunk_index in metadata (%s).",
                    clean_chunk_id,
                    idx,
                    chunk_idx,
                )
                continue

            if not _is_safe_finite_number(start) or start < 0.0:
                logger.warning(
                    "Skipping search result %s at index %d: invalid start timestamp in metadata (%s).",
                    clean_chunk_id,
                    idx,
                    start,
                )
                continue

            if not _is_safe_finite_number(end) or end < start:
                logger.warning(
                    "Skipping search result %s at index %d: invalid end timestamp in metadata (%s).",
                    clean_chunk_id,
                    idx,
                    end,
                )
                continue

            if not _is_safe_finite_number(dist):
                logger.warning(
                    "Skipping search result %s at index %d: distance is non-numeric or non-finite (%s).",
                    clean_chunk_id,
                    idx,
                    dist,
                )
                continue

            f_start = float(start)
            f_end = float(end)
            f_dist = float(dist)
            i_chunk_idx = int(chunk_idx)
        except (TypeError, ValueError, OverflowError) as num_err:
            logger.warning(
                "Skipping search result %s at index %d: malformed numeric metadata (%s).",
                clean_chunk_id,
                idx,
                num_err,
            )
            continue

        # Validate document transcript text
        if not isinstance(doc, str) or not doc.strip():
            logger.warning(
                "Skipping search result %s at index %d: document text is missing or empty.",
                clean_chunk_id,
                idx,
            )
            continue

        formatted_results.append({
            "chunk_id": clean_chunk_id,
            "recording_id": clean_rec,
            "chunk_index": i_chunk_idx,
            "start": f_start,
            "end": f_end,
            "text": str(doc).strip(),
            "distance": f_dist,
        })

    logger.info(
        "Search complete. Returning %d matching chunk(s).",
        len(formatted_results),
    )

    return formatted_results
