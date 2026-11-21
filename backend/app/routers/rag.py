import logging
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.services.rag_service import answer_query

# Configure module logger for server-side diagnostics
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# APIRouter Configuration
# ---------------------------------------------------------
# Prefix: /api/rag -> Endpoints in this router start with /api/rag
# Tag: RAG -> Groups endpoints under "RAG" in OpenAPI/Swagger documentation
router = APIRouter(prefix="/api/rag", tags=["RAG"])


# ---------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------
class RAGQueryRequest(BaseModel):
    """
    Request payload schema for asking evidence-grounded questions over transcripts.
    """

    query: str = Field(
        ...,
        description="Natural language question to answer from transcript evidence.",
        examples=["What was discussed about the quarterly budget?"],
    )
    recording_id: Optional[str] = Field(
        default=None,
        description="Optional recording ID to scope retrieval to a specific media file.",
        examples=["5aa51179-2c02-4def-ab08-015caf455998"],
    )
    top_k: int = Field(
        default=5,
        gt=0,
        description="Number of candidate transcript chunks to retrieve for evidence grounding.",
        examples=[5],
    )

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, v: Any) -> str:
        if isinstance(v, bool) or not isinstance(v, str):
            raise ValueError("query must be a string.")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("query must not be empty or whitespace-only.")
        return cleaned

    @field_validator("recording_id", mode="before")
    @classmethod
    def validate_recording_id(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, str):
            raise ValueError("recording_id must be a string or None.")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError(
                "recording_id must not be empty or whitespace-only if provided."
            )
        return cleaned

    @field_validator("top_k", mode="before")
    @classmethod
    def validate_top_k(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("top_k must be an integer, not a boolean or float.")
        if v <= 0:
            raise ValueError("top_k must be a positive integer greater than 0.")
        return v


# ---------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------
class RAGSourceItem(BaseModel):
    """
    Schema representing a verified transcript evidence source chunk cited in the RAG response.
    """

    chunk_id: str = Field(
        description="Unique identifier of the transcript chunk in vector storage.",
    )
    recording_id: str = Field(
        description="Identifier of the recording to which this chunk belongs.",
    )
    chunk_index: int = Field(
        description="Sequential index of the chunk within the parent recording.",
    )
    start: float = Field(
        description="Start time of the chunk in seconds.",
    )
    end: float = Field(
        description="End time of the chunk in seconds.",
    )
    timestamp_range: str = Field(
        description="Formatted time range string (e.g., '00:00–00:11').",
    )
    text: str = Field(
        description="Transcript excerpt text supporting the answer.",
    )
    distance: float = Field(
        description="Vector similarity distance score (lower values represent higher similarity).",
    )


class RAGQueryResponse(BaseModel):
    """
    Response schema returned by the RAG query endpoint.
    """

    query: str = Field(
        description="The sanitized user query that was executed.",
    )
    answer: str = Field(
        description="Evidence-grounded answer or fixed fallback response.",
    )
    has_evidence: bool = Field(
        description="True if verified supporting evidence was found and cited; False otherwise.",
    )
    sources: List[RAGSourceItem] = Field(
        default_factory=list,
        description="List of verified transcript source chunks supporting the generated answer.",
    )


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@router.post(
    "/ask",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query recording transcripts via RAG",
    description="Retrieves relevant transcript chunks and generates an evidence-grounded answer using Groq LLM.",
)
def ask_question(request: RAGQueryRequest) -> RAGQueryResponse:
    """
    Processes a natural language query against indexed recording transcripts.

    Args:
        request (RAGQueryRequest): Validated query payload containing query,
            optional recording_id, and optional top_k.

    Returns:
        RAGQueryResponse: Grounded answer, evidence flag, and cited sources.

    Raises:
        HTTPException(422): If input parameters fail semantic validation.
        HTTPException(500): If an unexpected internal error or API failure occurs.
    """
    try:
        result = answer_query(
            query=request.query,
            recording_id=request.recording_id,
            top_k=request.top_k,
        )
        return RAGQueryResponse(**result)
    except (ValueError, TypeError) as val_err:
        logger.warning("RAG input validation failure: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error in RAG answer generation endpoint: %s",
            type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the RAG query. Please try again later.",
        )
