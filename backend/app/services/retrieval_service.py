import logging
import math
from typing import Any, Dict, List, Optional

from app.services.embedding_service import embed_query
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
) -> List[Dict[str, Any]]:
    """
    Searches indexed transcript chunks using vector similarity (cosine distance).

    Flow:
    1. Validates input parameters (query, recording_id, top_k).
    2. Obtains the persistent ChromaDB collection.
    3. Handles an empty collection gracefully by returning an empty list without unnecessary queries.
    4. Generates a 768-dimensional query embedding via embed_query().
    5. Queries ChromaDB using cosine distance, optionally filtered by recording_id.
    6. Safely extracts, validates, and formats matching chunk results.

    Args:
        query (str): User natural language search query.
        recording_id (Optional[str]): Optional recording ID to scope results to a single media file.
        top_k (int): Maximum number of top matching chunks to retrieve (default: 5).

    Returns:
        List[Dict[str, Any]]: Ranked list of matching chunks sorted by cosine distance (lower is better),
        where each item contains:
            - chunk_id (str): Unique chunk identifier (e.g., '{recording_id}_{chunk_index:06d}').
            - recording_id (str): ID of the media recording this chunk belongs to.
            - chunk_index (int): Sequential index of the chunk within the recording.
            - start (float): Timestamp in seconds marking the start of the chunk.
            - end (float): Timestamp in seconds marking the end of the chunk.
            - text (str): Transcript text content of the chunk.
            - distance (float): Cosine distance between query and chunk embedding.

    Raises:
        TypeError: If query, recording_id, or top_k are of invalid types (or boolean where numeric/string expected).
        ValueError: If query or recording_id are empty/whitespace, or top_k <= 0.
        RuntimeError: If embedding generation or ChromaDB query fails.
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
    # 2. ChromaDB Collection Access & Empty Collection Check
    # ---------------------------------------------------------
    try:
        collection = get_vector_collection()
    except Exception as exc:
        logger.error(
            "Failed to access ChromaDB collection: %s",
            type(exc).__name__,
        )
        raise RuntimeError(
            f"Failed to access vector collection: {type(exc).__name__}."
        ) from exc

    try:
        total_records = collection.count()
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
    # 3. Query Embedding Generation
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
    # 4. ChromaDB Vector Query Execution
    # ---------------------------------------------------------
    where_filter: Optional[Dict[str, Any]] = (
        {"recording_id": clean_recording_id} if clean_recording_id else None
    )

    logger.info(
        "Executing ChromaDB query (top_k=%d, scoped_recording_id=%s)...",
        top_k,
        clean_recording_id,
    )

    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    try:
        raw_results = collection.query(**query_kwargs)
    except Exception as exc:
        logger.error(
            "ChromaDB vector search query failed: %s",
            type(exc).__name__,
        )
        raise RuntimeError(
            f"Vector search query failed: {type(exc).__name__}."
        ) from exc

    # ---------------------------------------------------------
    # 5. Response Processing & Result Sanitization
    # ---------------------------------------------------------
    # ChromaDB returns nested lists for batch queries; safely access first query results
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

        # Validate recording_id
        if not isinstance(rec_id, str) or not rec_id.strip():
            logger.warning(
                "Skipping search result %s at index %d: invalid or empty recording_id in metadata.",
                clean_chunk_id,
                idx,
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
            "recording_id": str(rec_id).strip(),
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
