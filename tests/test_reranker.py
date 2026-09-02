"""
tests/test_reranker.py
======================
Unit tests for CrossEncoderReranker (07_reranking.ipynb):
A. Model interface
B. Correct query/document pair construction
C. candidate.text is used (not contextual_text)
D. No query prefix is added
E. Raw logits are preserved
F. No score normalization applied
G. Descending ranking
H. Deterministic tie-breaking on (-rerank_score, chunk_id)
I. Top-k behavior
J. Empty candidates & empty query behavior
K. Single candidate behavior
L. Metadata preservation
M. Original hybrid rank preservation
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.rag.retrieval.schemas import HybridSearchResult, RerankerConfig, RerankedResult
from app.rag.retrieval.reranker import CrossEncoderReranker


def make_candidate(
    chunk_id: str,
    text: str,
    contextual_text: str,
    hybrid_score: float,
    rank: int,
    source: str = "occupation_master.csv",
    title: str = "Test Title",
    soc_code: str = "19-1042.00"
) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=text,
        contextual_text=contextual_text,
        source=source,
        title=title,
        section="Description",
        document_type="occupation",
        dense_score=0.8,
        sparse_score=10.0,
        normalized_dense_score=0.8,
        normalized_sparse_score=1.0,
        hybrid_score=hybrid_score,
        rank=rank,
        metadata={"soc_code": soc_code}
    )


@pytest.fixture
def mock_reranker():
    """Returns a CrossEncoderReranker with mocked model and tokenizer."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    config = RerankerConfig(final_top_k=3, batch_size=32, device="cpu")

    reranker = CrossEncoderReranker(
        config=config,
        model=mock_model,
        tokenizer=mock_tokenizer
    )
    return reranker


# =======================================================
# A, B, C, D: Pair Construction & Text Selection Tests
# =======================================================

def test_pair_construction_uses_contextual_text_and_no_prefix(mock_reranker):
    """B, C, D: Verify pairs use candidate.contextual_text (incorporating title/section) and have no query prefix."""
    cand1 = make_candidate(
        "pol_c1",
        "Raw policy rule text for skill gap severity.",
        "[Document: POL-SKILL-001: Skill Gap Identification Policy — Rule 2]\n[Section: Rule 2: Classification of Severity]\n\nRaw policy rule text for skill gap severity.",
        0.9,
        1,
        title="POL-SKILL-001: Skill Gap Identification Policy — Rule 2"
    )
    cand2 = make_candidate(
        "pol_c2",
        "Raw policy rule text for upskilling.",
        "[Document: POL-LEARN-001: Employee Upskilling Recommendation Policy — Rule 1]\n[Section: Rule 1: Course Catalog]\n\nRaw policy rule text for upskilling.",
        0.7,
        2,
        title="POL-LEARN-001: Employee Upskilling Recommendation Policy — Rule 1"
    )

    query = "What does POL-SKILL-001 state regarding the classification of skill gap severity?"

    # Mock predict method
    with patch.object(mock_reranker, "predict", return_value=np.array([4.5, -1.2])) as mock_predict:
        mock_reranker.rerank(query, [cand1, cand2])

        # Assert predict was called with contextual text pairs [[query, cand.contextual_text]]
        mock_predict.assert_called_once()
        passed_pairs = mock_predict.call_args[0][0]

        assert len(passed_pairs) == 2
        # Verify query has NO prefix
        assert passed_pairs[0][0] == query
        assert passed_pairs[1][0] == query

        # Verify candidate.contextual_text is used, containing policy title and section headers
        assert passed_pairs[0][1] == cand1.contextual_text
        assert "POL-SKILL-001" in passed_pairs[0][1]
        assert "Section: Rule 2: Classification of Severity" in passed_pairs[0][1]

        assert passed_pairs[1][1] == cand2.contextual_text
        assert "POL-LEARN-001" in passed_pairs[1][1]
        assert "Section: Rule 1: Course Catalog" in passed_pairs[1][1]


def test_pair_construction_fallback_to_raw_text_when_contextual_empty(mock_reranker):
    """Verify fallback to candidate.text when candidate.contextual_text is empty or None."""
    cand = make_candidate("c1", "Fallback raw text.", "", 0.9, 1)
    query = "Test query"

    with patch.object(mock_reranker, "predict", return_value=np.array([2.0])) as mock_predict:
        mock_reranker.rerank(query, [cand])
        passed_pairs = mock_predict.call_args[0][0]
        assert passed_pairs[0][1] == "Fallback raw text."



# =======================================================
# E, F, G: Raw Logits & Descending Ranking Tests
# =======================================================

def test_raw_logits_preserved_without_normalization(mock_reranker):
    """E, F, G: Verify raw logits (positive, negative, unbounded) are preserved without sigmoid/softmax/scaling."""
    cand1 = make_candidate("c1", "Text 1", "Ctx 1", 0.9, 1)
    cand2 = make_candidate("c2", "Text 2", "Ctx 2", 0.8, 2)
    cand3 = make_candidate("c3", "Text 3", "Ctx 3", 0.7, 3)

    # Cross-encoder raw logits: c1=7.82, c2=-4.15, c3=2.33
    mock_scores = np.array([7.82, -4.15, 2.33])

    with patch.object(mock_reranker, "predict", return_value=mock_scores):
        results = mock_reranker.rerank("test query", [cand1, cand2, cand3], top_k=3)

        assert len(results) == 3
        # Expected descending order: c1 (7.82), c3 (2.33), c2 (-4.15)
        assert results[0].chunk_id == "c1"
        assert results[0].rerank_score == 7.82
        assert results[0].rerank_rank == 1

        assert results[1].chunk_id == "c3"
        assert results[1].rerank_score == 2.33
        assert results[1].rerank_rank == 2

        assert results[2].chunk_id == "c2"
        # Negative logit must be preserved directly
        assert results[2].rerank_score == -4.15
        assert results[2].rerank_rank == 3


# =======================================================
# H: Deterministic Tie-Breaking Test
# =======================================================

def test_deterministic_tie_breaking(mock_reranker):
    """H: When rerank scores are identical, sort chunk_id ascending alphabetically."""
    cand_b = make_candidate("chunk_b", "Text B", "Ctx B", 0.5, 1)
    cand_a = make_candidate("chunk_a", "Text A", "Ctx A", 0.5, 2)

    # Identical score 3.5 for both
    with patch.object(mock_reranker, "predict", return_value=np.array([3.5, 3.5])):
        results = mock_reranker.rerank("query", [cand_b, cand_a], top_k=2)

        # chunk_a must precede chunk_b
        assert results[0].chunk_id == "chunk_a"
        assert results[1].chunk_id == "chunk_b"


# =======================================================
# I: Top-K Slicing Test
# =======================================================

def test_top_k_slicing_behavior(mock_reranker):
    """I: Verify reranker slices exactly top_k results."""
    candidates = [
        make_candidate(f"c{i}", f"Text {i}", f"Ctx {i}", 0.5, i)
        for i in range(1, 6)
    ]
    # Scores: 5, 4, 3, 2, 1
    with patch.object(mock_reranker, "predict", return_value=np.array([5.0, 4.0, 3.0, 2.0, 1.0])):
        # Request top 2
        results = mock_reranker.rerank("query", candidates, top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[1].chunk_id == "c2"

        # Default top_k is 3
        results_default = mock_reranker.rerank("query", candidates)
        assert len(results_default) == 3


# =======================================================
# J, K: Empty & Single Candidate Tests
# =======================================================

def test_empty_candidates_returns_empty(mock_reranker):
    """J: Verify empty candidate list or blank query returns empty list."""
    assert mock_reranker.rerank("query", []) == []
    assert mock_reranker.rerank("", [make_candidate("c1", "t", "ctx", 0.5, 1)]) == []
    assert mock_reranker.rerank("   ", [make_candidate("c1", "t", "ctx", 0.5, 1)]) == []


def test_single_candidate(mock_reranker):
    """K: Verify single candidate is handled properly."""
    cand = make_candidate("c1", "Single text", "Single ctx", 0.85, 1)
    with patch.object(mock_reranker, "predict", return_value=np.array([3.14])):
        results = mock_reranker.rerank("query", [cand])
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].rerank_score == 3.14
        assert results[0].rerank_rank == 1


# =======================================================
# L, M: Metadata & Original Rank Preservation Tests
# =======================================================

def test_metadata_and_original_rank_preservation(mock_reranker):
    """L, M: Verify all original metadata, contextual text, and hybrid ranks are preserved."""
    cand1 = make_candidate("c1", "Text 1", "Contextual Header 1", 0.95, 1, soc_code="19-1042.00")
    cand2 = make_candidate("c2", "Text 2", "Contextual Header 2", 0.85, 2, soc_code="13-1071.00")

    # Suppose cross-encoder ranks cand2 HIGHER than cand1
    # c1 = 1.0, c2 = 5.0
    with patch.object(mock_reranker, "predict", return_value=np.array([1.0, 5.0])):
        results = mock_reranker.rerank("query", [cand1, cand2], top_k=2)

        # Winner is c2
        r0 = results[0]
        assert r0.chunk_id == "c2"
        assert r0.rerank_rank == 1
        assert r0.original_hybrid_rank == 2  # Original rank 2 from hybrid
        assert r0.hybrid_score == 0.85
        assert r0.contextual_text == "Contextual Header 2"
        assert r0.metadata["soc_code"] == "13-1071.00"

        # Runner-up is c1
        r1 = results[1]
        assert r1.chunk_id == "c1"
        assert r1.rerank_rank == 2
        assert r1.original_hybrid_rank == 1  # Original rank 1 from hybrid
        assert r1.hybrid_score == 0.95
        assert r1.contextual_text == "Contextual Header 1"
        assert r1.metadata["soc_code"] == "19-1042.00"
