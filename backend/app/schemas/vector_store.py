import math
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TranscriptChunkRecord(BaseModel):
    """
    Schema representing a timestamp-aligned transcript chunk and its vector embedding
    for persistence and similarity search in ChromaDB.

    Fields:
        recording_id (str): Unique identifier of the uploaded media recording.
        chunk_index (int): Zero-based sequential order of this chunk within the recording.
        text (str): Combined text content of the transcript chunk.
        start (float): Timestamp in seconds marking the beginning of the chunk.
        end (float): Timestamp in seconds marking the end of the chunk (must be >= start).
        segment_count (int): Number of original speech segments consolidated into this chunk.
        embedding (list[float]): 768-dimensional embedding vector (Gemini embedding).
    """

    # Model configuration: enforce strict typing and forbid unexpected extra fields
    model_config = ConfigDict(strict=True, extra="forbid")

    recording_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the parent recording.",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Sequential index of the chunk (0, 1, 2, ...).",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Non-empty transcript text for this chunk.",
    )
    start: float = Field(
        ...,
        ge=0.0,
        description="Start time of the chunk in seconds.",
    )
    end: float = Field(
        ...,
        ge=0.0,
        description="End time of the chunk in seconds (must be >= start).",
    )
    segment_count: int = Field(
        ...,
        gt=0,
        description="Total number of speech segments merged in this chunk.",
    )
    embedding: list[float] = Field(
        ...,
        description="768-dimensional float embedding vector.",
    )

    # ---------------------------------------------------------
    # Field Validators (mode="before" intercepts raw inputs to reject booleans and convert types)
    # Raising ValueError allows Pydantic to wrap all validation failures into ValidationError
    # ---------------------------------------------------------
    @field_validator("recording_id", mode="before")
    @classmethod
    def validate_recording_id(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("recording_id must be a non-empty string.")
        return v.strip()

    @field_validator("chunk_index", mode="before")
    @classmethod
    def validate_chunk_index(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("chunk_index must be an integer, not a boolean or other type.")
        if v < 0:
            raise ValueError("chunk_index must be greater than or equal to 0.")
        return v

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("text must be a non-empty string.")
        return v.strip()

    @field_validator("start", mode="before")
    @classmethod
    def validate_start(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("start timestamp must be a numeric int or float, not boolean.")
        f_val = float(v)
        if not math.isfinite(f_val):
            raise ValueError("start timestamp must be a finite number (not NaN or infinity).")
        if f_val < 0.0:
            raise ValueError("start timestamp cannot be negative.")
        return f_val

    @field_validator("end", mode="before")
    @classmethod
    def validate_end(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("end timestamp must be a numeric int or float, not boolean.")
        f_val = float(v)
        if not math.isfinite(f_val):
            raise ValueError("end timestamp must be a finite number (not NaN or infinity).")
        if f_val < 0.0:
            raise ValueError("end timestamp cannot be negative.")
        return f_val

    @field_validator("segment_count", mode="before")
    @classmethod
    def validate_segment_count(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("segment_count must be an integer, not boolean.")
        if v <= 0:
            raise ValueError("segment_count must be a positive integer greater than 0.")
        return v

    @field_validator("embedding", mode="before")
    @classmethod
    def validate_embedding(cls, v: Any) -> list[float]:
        if isinstance(v, bool) or not isinstance(v, (list, tuple)):
            raise ValueError("embedding must be a list of floats.")

        expected_dim = 768
        if len(v) != expected_dim:
            raise ValueError(
                f"embedding vector must contain exactly {expected_dim} dimensions, got {len(v)}."
            )

        clean_embedding: list[float] = []
        for idx, item in enumerate(v):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(
                    f"Embedding item at index {idx} must be numeric, got {type(item).__name__}."
                )
            f_val = float(item)
            if not math.isfinite(f_val):
                raise ValueError(
                    f"Embedding item at index {idx} must be finite (not NaN or infinity)."
                )
            clean_embedding.append(f_val)

        return clean_embedding

    # ---------------------------------------------------------
    # Cross-Field Model Validators
    # ---------------------------------------------------------
    @model_validator(mode="after")
    def validate_start_end_order(self) -> "TranscriptChunkRecord":
        if self.start > self.end:
            raise ValueError(
                f"end timestamp ({self.end}) must be greater than or equal to start timestamp ({self.start})."
            )
        return self
