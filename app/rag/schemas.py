"""
app/rag/schemas.py
==================
Pydantic data models for the Enterprise HR AI RAG pipeline.
Defines Document and Chunk abstractions with structured metadata.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Normalized document representation loaded from raw/processed sources.
    One document represents a coherent logical entity (e.g., an occupation,
    a role mapping crosswalk record, or a markdown documentation section).
    """
    doc_id: str = Field(..., description="Deterministic unique document identifier.")
    source: str = Field(..., description="Relative filesystem path or provenance source.")
    file_type: str = Field(..., description="Source file format (e.g., 'csv', 'markdown').")
    title: str = Field(..., description="Document or entity title.")
    text: str = Field(..., description="Normalized document text content.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Structured document metadata.")

    model_config = {
        "frozen": False,
        "extra": "ignore"
    }


class Chunk(BaseModel):
    """
    Structure-aware chunk derived from a Document, enriched with
    contextual header metadata ready for dense embedding and sparse indexing.
    """
    chunk_id: str = Field(..., description="Deterministic unique chunk identifier.")
    doc_id: str = Field(..., description="Parent document identifier.")
    source: str = Field(..., description="Relative source path.")
    title: str = Field(..., description="Document title.")
    section: str = Field(..., description="Section or subsection heading.")
    document_type: str = Field(..., description="Semantic document classification (e.g., 'occupation', 'role_mapping', 'governance', 'architecture').")
    text: str = Field(..., description="Raw chunk body text.")
    contextual_text: str = Field(..., description="Text enriched with deterministic document/section prefix headers.")
    token_count: int = Field(default=0, description="Approximate token count of the chunk.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Preserved and inherited metadata.")

    model_config = {
        "frozen": False,
        "extra": "ignore"
    }
