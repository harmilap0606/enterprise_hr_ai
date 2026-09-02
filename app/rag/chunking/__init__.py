"""
app/rag/chunking/__init__.py
============================
Chunking and contextual metadata enrichment package.
"""

from app.rag.chunking.chunker import (
    ChunkConfig,
    DEFAULT_CHUNK_CONFIG,
    estimate_tokens,
    chunk_document,
    chunk_all_documents
)
from app.rag.chunking.enricher import create_contextual_text

__all__ = [
    "ChunkConfig",
    "DEFAULT_CHUNK_CONFIG",
    "estimate_tokens",
    "chunk_document",
    "chunk_all_documents",
    "create_contextual_text",
]
