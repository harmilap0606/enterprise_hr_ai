"""
tests/test_rag_pipeline.py
==========================
Unit and integration tests for the Minimal Honest RAG module.
Covers:
1. Retrieval returns top-k chunks with provenance and similarity scores.
2. Grounded queries succeed with real sources.
3. Out-of-domain and ungrounded queries are strictly refused with the exact standard message.
4. FastAPI POST /rag/ask endpoint returns valid 200 response with auditable sources.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.rag.retriever import retrieve
from app.rag.qa_chain import answer_question, REFUSAL_MESSAGE

client = TestClient(app)


def test_retriever_returns_top_k():
    """Verify retriever returns exactly k chunks with source, excerpt, and score."""
    chunks = retrieve("What does a Research Scientist do?", k=3)
    assert len(chunks) == 3
    for c in chunks:
        assert "source" in c
        assert "excerpt" in c
        assert "score" in c
        assert isinstance(c["score"], float)
        assert len(c["excerpt"]) > 0


def test_rag_research_scientist_grounded():
    """Verify query grounded in O*NET retrieves occupation_master source."""
    res = answer_question("What does a Research Scientist do?")
    assert res["answer"] != REFUSAL_MESSAGE
    assert len(res["sources"]) > 0
    assert any("occupation_master.csv" in s["source"] for s in res["sources"])


def test_rag_manager_mapping_grounded():
    """Verify query on Manager mapping retrieves data_relationships.md Open Issue #1."""
    res = answer_question("Why is the Manager role's O*NET mapping unreliable?")
    assert res["answer"] != REFUSAL_MESSAGE
    assert len(res["sources"]) > 0
    assert any("data_relationships.md" in s["source"] for s in res["sources"])


def test_rag_threshold_grounded():
    """Verify query on production threshold retrieves model_card.md."""
    res = answer_question("What is the production model's decision threshold and why was it chosen?")
    assert res["answer"] != REFUSAL_MESSAGE
    assert len(res["sources"]) > 0
    assert any("model_card.md" in s["source"] for s in res["sources"])
    assert "0.40" in res["answer"]


def test_rag_parental_leave_refusal():
    """Verify ungrounded query on parental leave is strictly refused."""
    res = answer_question("What is the company's parental leave policy?")
    assert res["answer"] == REFUSAL_MESSAGE
    assert len(res["sources"]) == 3  # Still returns sources for auditability


def test_rag_capital_of_france_refusal():
    """Verify out-of-domain query on France's capital is strictly refused."""
    res = answer_question("What is the capital of France?")
    assert res["answer"] == REFUSAL_MESSAGE
    assert "Paris" not in res["answer"]
    assert len(res["sources"]) == 3


def test_api_rag_ask_endpoint():
    """Verify POST /rag/ask endpoint schema and response structure."""
    response = client.post("/rag/ask", json={"question": "What is the production model's decision threshold and why was it chosen?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert all("source" in s and "excerpt" in s and "score" in s for s in data["sources"])


def test_api_rag_ask_empty_question_returns_400():
    """Verify empty question raises 400."""
    response = client.post("/rag/ask", json={"question": "   "})
    assert response.status_code == 400
