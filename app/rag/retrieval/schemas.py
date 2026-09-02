"""
app/rag/retrieval/schemas.py
============================
Typed data models and configuration for hybrid dense + sparse retrieval and reranking.
Derived directly from the architecture and result structures of:
- 06_hybrid_search.ipynb
- 07_reranking.ipynb
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    """
    Configurable parameters for hybrid retrieval.
    Defaults match the production architecture established from 06_hybrid_search.ipynb:
    0.8 * Dense + 0.2 * Sparse.
    """
    dense_top_k: int = Field(default=20, ge=1, description="Number of candidate chunks to retrieve from dense vector store.")
    sparse_top_k: int = Field(default=20, ge=1, description="Number of candidate chunks to retrieve from BM25 sparse index.")
    final_top_k: int = Field(default=15, ge=1, description="Final number of fused results to return after reranking.")
    dense_weight: float = Field(default=0.8, ge=0.0, le=1.0, description="Weight (1 - alpha) assigned to normalized dense score.")
    sparse_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Weight (alpha) assigned to normalized sparse score.")

    model_config = {
        "frozen": False,
        "extra": "ignore"
    }


class HybridSearchResult(BaseModel):
    """
    Structured, fully traceable result item combining dense and sparse retrieval signals.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    doc_id: str = Field(..., description="Parent document identifier.")
    text: str = Field(..., description="Raw text of the chunk.")
    contextual_text: str = Field(..., description="Contextualized text with hierarchical metadata headers.")
    source: str = Field(..., description="Source file provenance.")
    title: str = Field(..., description="Document title.")
    section: str = Field(..., description="Section title or occupational title.")
    document_type: str = Field(..., description="Document semantic category.")
    dense_score: float = Field(default=0.0, description="Raw cosine similarity score from dense index (0 to 1).")
    sparse_score: float = Field(default=0.0, description="Raw BM25 score from sparse index.")
    normalized_dense_score: float = Field(default=0.0, description="Min-Max normalized dense score (0 to 1).")
    normalized_sparse_score: float = Field(default=0.0, description="Min-Max normalized sparse score (0 to 1).")
    hybrid_score: float = Field(default=0.0, description="Weighted fusion score: dense_weight*norm_dense + sparse_weight*norm_sparse.")
    rank: int = Field(default=1, ge=1, description="Final 1-based rank after hybrid score sorting.")
    is_exact_match: bool = Field(default=False, description="Whether chunk contains an exact identifier match for the query.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Original preserved metadata.")

    model_config = {
        "frozen": False,
        "extra": "ignore"
    }


class RerankerConfig(BaseModel):
    """
    Configuration parameters for Cross-Encoder reranking stage (07_reranking.ipynb).
    """
    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Pretrained HuggingFace cross-encoder repository ID."
    )
    final_top_k: int = Field(
        default=3,
        ge=1,
        description="Number of reranked results to return (matches Notebook 07 top-3 default)."
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Inference batch size for cross-encoder scoring (matches Notebook 07 default)."
    )
    device: Optional[str] = Field(
        default=None,
        description="Hardware execution device ('cpu', 'cuda'). None for automatic detection."
    )

    model_config = {
        "frozen": False,
        "extra": "ignore"
    }


class RerankedResult(BaseModel):
    """
    Output model preserving full provenance from Step 2, Step 4B, and Step 5B.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    doc_id: str = Field(..., description="Parent document identifier.")
    text: str = Field(..., description="Raw text of the chunk used for reranker evaluation.")
    contextual_text: str = Field(..., description="Contextualized text with hierarchical metadata headers.")
    source: str = Field(..., description="Source file provenance.")
    title: str = Field(..., description="Document title.")
    section: str = Field(..., description="Section title or occupational title.")
    document_type: str = Field(..., description="Document semantic category.")
    dense_score: float = Field(default=0.0, description="Raw cosine similarity score from dense index.")
    sparse_score: float = Field(default=0.0, description="Raw BM25 score from sparse index.")
    normalized_dense_score: float = Field(default=0.0, description="Min-Max normalized dense score.")
    normalized_sparse_score: float = Field(default=0.0, description="Min-Max normalized sparse score.")
    hybrid_score: float = Field(default=0.0, description="Weighted fusion score from Step 4B.")
    original_hybrid_rank: int = Field(..., ge=1, description="Rank from Step 4B hybrid retrieval.")
    rerank_score: float = Field(..., description="Raw cross-encoder logit score.")
    rerank_rank: int = Field(..., ge=1, description="Final 1-based rank after cross-encoder reranking.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Original preserved metadata.")

    model_config = {
        "frozen": False,
        "extra": "ignore"
    }
