"""
tests/test_rag_functional_pipeline.py
=====================================
Unit and integration tests for the Grounded RAG Pipeline:
A. Query reaches hybrid retriever
B. Hybrid results reach reranker
C. Reranked Top-3 become LLM context
D. Metadata is preserved (chunk_id, doc_id, title, section, source, doc_type)
E. Source attribution is returned with correct schema
F. Empty retrieval behavior / empty query
G. LLM receives correctly formatted context with explicit source boundaries
H. LLM does not receive unrelated documents (top-3 slice enforced)
I. Existing API behavior remains compatible (POST /rag/ask returns 200, source/excerpt/score)
J. No API key is hardcoded
K. No embeddings or internal model tensors are exposed through the API
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.rag.retrieval.schemas import HybridSearchResult, RerankedResult
from app.rag.pipeline import GroundedRAGPipeline, REFUSAL_MESSAGE

client = TestClient(app)


def make_hybrid_candidate(chunk_id: str, title: str, text: str, score: float, rank: int) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=text,
        contextual_text=f"[Doc: {title}] {text}",
        source="docs/model_card.md",
        title=title,
        section="Specs",
        document_type="markdown_doc",
        dense_score=0.9,
        sparse_score=8.5,
        normalized_dense_score=0.9,
        normalized_sparse_score=0.85,
        hybrid_score=score,
        rank=rank,
        metadata={"soc_code": "N/A"}
    )


def make_reranked_candidate(chunk_id: str, title: str, text: str, score: float, rank: int) -> RerankedResult:
    return RerankedResult(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=text,
        contextual_text=f"[Doc: {title}] {text}",
        source="docs/model_card.md",
        title=title,
        section="Specs",
        document_type="markdown_doc",
        dense_score=0.9,
        sparse_score=8.5,
        normalized_dense_score=0.9,
        normalized_sparse_score=0.85,
        hybrid_score=0.88,
        rank=rank,
        metadata={"soc_code": "N/A"},
        rerank_score=score,
        rerank_rank=rank,
        original_hybrid_rank=rank
    )


@pytest.fixture
def mocked_pipeline():
    """Returns a GroundedRAGPipeline with mocked retriever, reranker, and LLM."""
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    pipeline = GroundedRAGPipeline(
        retriever=mock_retriever,
        reranker=mock_reranker,
        tokenizer=mock_tokenizer,
        model=mock_model
    )
    return pipeline


# =======================================================
# A & B: Query reaches hybrid retriever and reranker
# =======================================================

def test_query_reaches_retriever_and_reranker(mocked_pipeline):
    """A, B: Verify query flows from input to hybrid retriever, and hybrid candidates reach reranker."""
    query = "What is the model threshold?"
    cand1 = make_hybrid_candidate("c1", "Title 1", "Text 1", 0.9, 1)
    cand2 = make_hybrid_candidate("c2", "Title 2", "Text 2", 0.8, 2)

    mocked_pipeline.retriever.retrieve.return_value = [cand1, cand2]
    
    r1 = make_reranked_candidate("c1", "Title 1", "Text 1", 4.2, 1)
    mocked_pipeline.reranker.rerank.return_value = [r1]

    with patch.object(mocked_pipeline, "generate_answer", return_value="The threshold is 0.40."):
        res = mocked_pipeline.run(query)

        mocked_pipeline.retriever.retrieve.assert_called_once_with(query)
        mocked_pipeline.reranker.rerank.assert_called_once_with(query, [cand1, cand2], top_k=3)
        assert res.query == query
        assert res.answer == "The threshold is 0.40."


# =======================================================
# C, D, E, G, H: Context construction and metadata
# =======================================================

def test_context_construction_and_metadata_preservation(mocked_pipeline):
    """C, D, E, G, H: Verify top-3 format explicit source boundaries and preserve full metadata."""
    r1 = make_reranked_candidate("c1", "Model Card", "The production threshold is 0.40.", 5.1, 1)
    r2 = make_reranked_candidate("c2", "Relationships", "Datasets are linked by employee_id.", 3.4, 2)
    r3 = make_reranked_candidate("c3", "Occupations", "Scientists analyze clinical data.", 1.2, 3)

    mocked_pipeline.retriever.retrieve.return_value = [
        make_hybrid_candidate("c1", "Model Card", "text", 0.9, 1),
        make_hybrid_candidate("c2", "Relationships", "text", 0.8, 2),
        make_hybrid_candidate("c3", "Occupations", "text", 0.7, 3),
    ]
    mocked_pipeline.reranker.rerank.return_value = [r1, r2, r3]

    # Test context builder directly
    context = mocked_pipeline.build_context([r1, r2, r3])
    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "[Source 3]" in context
    assert "Title: Model Card" in context
    assert "Title: Relationships" in context
    assert "The production threshold is 0.40." in context

    with patch.object(mocked_pipeline, "generate_answer", return_value="The production threshold is 0.40."):
        response = mocked_pipeline.run("Query")

        assert len(response.sources) == 3
        s1 = response.sources[0]
        assert s1.chunk_id == "c1"
        assert s1.doc_id == "doc_c1"
        assert s1.title == "Model Card"
        assert s1.section == "Specs"
        assert s1.document_type == "markdown_doc"
        assert s1.rank == 1
        assert s1.score == 5.1


# =======================================================
# F: Empty query and zero retrieval handling
# =======================================================

def test_empty_query_and_zero_retrieval(mocked_pipeline):
    """F: Blank queries and zero retrieval return safe refusal."""
    res_blank = mocked_pipeline.run("   ")
    assert res_blank.answer == REFUSAL_MESSAGE
    assert res_blank.sources == []

    mocked_pipeline.retriever.retrieve.return_value = []
    res_empty = mocked_pipeline.run("Unmatchable query")
    assert res_empty.answer == REFUSAL_MESSAGE
    assert res_empty.sources == []


# =======================================================
# I: Existing API backward compatibility
# =======================================================

def test_api_endpoint_backward_compatibility():
    """I: Verify POST /rag/ask returns 200 with required keys and no missing fields."""
    response = client.post("/rag/ask", json={"question": "What is the production model's decision threshold and why was it chosen?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
    for s in data["sources"]:
        assert "source" in s
        assert "excerpt" in s
        assert "score" in s
        # New enriched provenance fields
        assert "chunk_id" in s
        assert "title" in s
        assert "rank" in s


def test_api_empty_question_returns_400():
    """I: Empty question produces 400 Bad Request."""
    response = client.post("/rag/ask", json={"question": ""})
    assert response.status_code == 400


# =======================================================
# J: No API keys hardcoded
# =======================================================

def test_no_hardcoded_api_keys():
    """J: Inspect pipeline.py and qa_chain.py to ensure no hardcoded API keys exist."""
    from pathlib import Path
    pipeline_code = Path("app/rag/pipeline.py").read_text(encoding="utf-8")
    api_code = Path("app/api/rag.py").read_text(encoding="utf-8")

    forbidden = ["sk-proj", "AIzaSy", "ghp_", "bearer "]
    for k in forbidden:
        assert k not in pipeline_code.lower()
        assert k not in api_code.lower()


# =======================================================
# K: No embeddings or internal tensors exposed in API
# =======================================================

def test_no_embeddings_exposed_in_api():
    """K: Ensure API response contains only text, numbers, and provenance—no vector arrays."""
    response = client.post("/rag/ask", json={"question": "What does a Medical Scientist do?"})
    assert response.status_code == 200
    data = response.json()
    for s in data["sources"]:
        assert "embedding" not in s
        assert "vector" not in s
        assert "embeddings" not in s
        assert "tensors" not in s
        assert isinstance(s["score"], float)
