import logging
import math
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Configure module logger for server-side diagnostics
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Environment & Path Configuration
# ---------------------------------------------------------
# Dynamic path resolution anchored to the project structure:
# __file__ -> backend/app/services/embedding_service.py
# .parent -> services/
# .parent.parent -> app/
# .parent.parent.parent -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BACKEND_DIR / ".env"

# Explicitly load .env from backend/.env so GEMINI_API_KEY is available
if ENV_PATH.is_file():
    load_dotenv(dotenv_path=ENV_PATH)

# ---------------------------------------------------------
# Model & Vector Dimension Constants
# ---------------------------------------------------------
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768

def get_gemini_client() -> genai.Client:
    """
    Initializes and returns an authenticated Google Gen AI client using GEMINI_API_KEY.

    Returns:
        genai.Client: Configured Google Gen AI client instance.

    Raises:
        ValueError: If GEMINI_API_KEY is missing from the environment or is empty.
        RuntimeError: If client initialization fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        logger.error("GEMINI_API_KEY is missing from environment or backend/.env.")
        raise ValueError("GEMINI_API_KEY is not configured or is empty.")

    try:
        # Initialize client securely without printing or logging the key
        return genai.Client(api_key=api_key.strip())
    except Exception as exc:
        logger.error(
            "Failed to initialize Google Gen AI client: %s",
            type(exc).__name__,
        )
        raise RuntimeError(
            f"Failed to initialize Google Gen AI client: {type(exc).__name__}."
        ) from exc


def _generate_embedding(
    text: str,
    task_type: str,
    client: Optional[genai.Client] = None,
) -> List[float]:
    """
    Internal helper to validate inputs, request an embedding from Gemini, and
    validate the returned vector.

    Args:
        text (str): Input text to be embedded.
        task_type (str): Gemini task type ('RETRIEVAL_DOCUMENT' or 'RETRIEVAL_QUERY').
        client (Optional[genai.Client]): Optional client instance (used for testing/dependency injection).

    Returns:
        List[float]: Validated 768-dimensional float embedding vector.

    Raises:
        TypeError: If text is not a string.
        ValueError: If text is empty/whitespace-only, or if Gemini returns an invalid vector.
        RuntimeError: If the Gemini API request fails.
    """
    # ---------------------------------------------------------
    # 1. Input Validation
    # ---------------------------------------------------------
    if isinstance(text, bool) or not isinstance(text, str):
        raise TypeError(
            f"Input text must be a string, got {type(text).__name__}."
        )

    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Input text must not be empty or whitespace-only.")

    # ---------------------------------------------------------
    # 2. Client Initialization
    # ---------------------------------------------------------
    if client is None:
        client = get_gemini_client()

    # ---------------------------------------------------------
    # 3. Request Embedding from Gemini API
    # ---------------------------------------------------------
    logger.info(
        "Requesting Gemini embedding (model: %s, dim: %d, task: %s, text_len: %d chars)...",
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSIONS,
        task_type,
        len(clean_text),
    )

    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=clean_text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
    except Exception as exc:
        logger.error(
            "Gemini embedding API request failed for task '%s': %s",
            task_type,
            type(exc).__name__,
        )
        raise RuntimeError(
            f"Gemini embedding API request failed: {type(exc).__name__}."
        ) from exc

    # ---------------------------------------------------------
    # 4. Response Validation
    # ---------------------------------------------------------
    if not response or not getattr(response, "embeddings", None):
        logger.error("Gemini API response did not contain an embeddings list.")
        raise ValueError("Gemini API response did not contain any embeddings.")

    # Validate that the embeddings collection contains at least one non-None item
    if not any(item is not None for item in response.embeddings) or response.embeddings[0] is None:
        logger.error("Gemini API response embeddings collection contains no non-None items.")
        raise ValueError(
            "Gemini API response embeddings collection contains no non-None items."
        )

    embedding_obj = response.embeddings[0]
    raw_values = getattr(embedding_obj, "values", None)

    if raw_values is None or not isinstance(raw_values, (list, tuple)):
        logger.error("Gemini API response embedding object contains no values sequence.")
        raise ValueError("Gemini API response did not contain valid embedding values.")

    if len(raw_values) != EMBEDDING_DIMENSIONS:
        logger.error(
            "Embedding dimension mismatch: expected %d, got %d.",
            EMBEDDING_DIMENSIONS,
            len(raw_values),
        )
        raise ValueError(
            f"Expected embedding to contain exactly {EMBEDDING_DIMENSIONS} dimensions, "
            f"got {len(raw_values)}."
        )

    # ---------------------------------------------------------
    # 5. Numerical Validation & Float Conversion
    # ---------------------------------------------------------
    validated_embedding: List[float] = []
    for idx, val in enumerate(raw_values):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(
                f"Embedding value at index {idx} must be numeric, got {type(val).__name__}."
            )
        f_val = float(val)
        if not math.isfinite(f_val):
            raise ValueError(
                f"Embedding value at index {idx} must be a finite number (not NaN or infinity)."
            )
        validated_embedding.append(f_val)

    logger.info(
        "Successfully generated and validated Gemini embedding (task: %s, dim: %d).",
        task_type,
        len(validated_embedding),
    )

    return validated_embedding


def embed_document(
    text: str,
    client: Optional[genai.Client] = None,
) -> List[float]:
    """
    Generates a 768-dimensional vector embedding for a transcript document chunk.
    Uses Gemini task type 'RETRIEVAL_DOCUMENT'.

    Args:
        text (str): Transcript text chunk to embed.
        client (Optional[genai.Client]): Optional Google Gen AI client instance.

    Returns:
        List[float]: 768-dimensional float embedding vector.

    Raises:
        TypeError: If text is not a string.
        ValueError: If text is empty/whitespace-only, or if validation fails.
        RuntimeError: If Gemini API request fails.
    """
    return _generate_embedding(
        text=text,
        task_type="RETRIEVAL_DOCUMENT",
        client=client,
    )


def embed_query(
    text: str,
    client: Optional[genai.Client] = None,
) -> List[float]:
    """
    Generates a 768-dimensional vector embedding for a user search query.
    Uses Gemini task type 'RETRIEVAL_QUERY'.

    Args:
        text (str): Search query string to embed.
        client (Optional[genai.Client]): Optional Google Gen AI client instance.

    Returns:
        List[float]: 768-dimensional float embedding vector.

    Raises:
        TypeError: If text is not a string.
        ValueError: If text is empty/whitespace-only, or if validation fails.
        RuntimeError: If Gemini API request fails.
    """
    return _generate_embedding(
        text=text,
        task_type="RETRIEVAL_QUERY",
        client=client,
    )
