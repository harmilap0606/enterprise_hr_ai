"""
app/api/rag.py
==============
Router for Grounded RAG Knowledge Base Retrieval.
Endpoint: POST /rag/ask

Connects the verified Step 4B (Hybrid BGE + BM25) and Step 5B (Cross-Encoder Reranker)
retrieval stack to the local grounded generator (google/flan-t5-base) with complete
source provenance and anti-hallucination guardrails.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.rag.pipeline import answer_grounded_question
from app.utils.logger import logger

router = APIRouter(prefix="/rag", tags=["RAG Knowledge Base"])


class QuestionRequest(BaseModel):
    question: str = Field(..., description="User question to answer from the platform's verified knowledge base.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is the production model's decision threshold and why was it chosen?"
            }
        }
    }


class SourceItem(BaseModel):
    source: str = Field(..., description="Source document and section provenance.")
    excerpt: str = Field(..., description="Excerpt from the source document.")
    score: float = Field(..., description="Relevance similarity score (cross-encoder raw logit or hybrid score).")
    chunk_id: Optional[str] = Field(None, description="Unique chunk identifier.")
    doc_id: Optional[str] = Field(None, description="Source document identifier.")
    title: Optional[str] = Field(None, description="Document or section title.")
    section: Optional[str] = Field(None, description="Section header.")
    document_type: Optional[str] = Field(None, description="Type of document (e.g. occupation, markdown_doc).")
    rank: Optional[int] = Field(None, description="Rank in reranked result list.")


class AskResponse(BaseModel):
    answer: str = Field(..., description="Grounded answer from the knowledge base, or refusal message if not present.")
    sources: List[SourceItem] = Field(..., description="List of top source chunks used for auditability.")


@router.post("/ask", response_model=AskResponse, summary="Query Verified Knowledge Base")
def ask_knowledge_base(payload: QuestionRequest):
    """
    POST /rag/ask
    Answers a question strictly from the project's verified documentation (model_card.md,
    data_relationships.md) and real O*NET occupational descriptions (occupation_master.csv).
    
    Retriever: Step 4B Hybrid Dense (BGE-small-en-v1.5) + Sparse (BM25Okapi) (Top 15).
    Reranker: Step 5B Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) (Top 3).
    Generator: Local Seq2Seq LM (google/flan-t5-base) with strict context grounding.
    Always returns sources alongside the answer for auditability.
    """
    try:
        q = payload.question.strip()
        if not q:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
            
        logger.info(f"RAG question received: '{q}'")
        result = answer_grounded_question(q)
        logger.info(f"RAG answer generated (sources: {len(result['sources'])}): '{result['answer'][:80]}...'")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing RAG question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
