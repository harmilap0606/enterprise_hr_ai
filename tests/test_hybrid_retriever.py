"""
tests/test_hybrid_retriever.py
==============================
Unit tests for hybrid dense + sparse retrieval:
A. BM25 index creation and querying
B. Dense retrieval invocation and mock integration
C. Min-Max normalization correctness
D. Identical-score and zero-range edge cases
E. Weighted fusion (0.8 Dense + 0.2 Sparse)
F. Candidate union and duplicate chunk ID handling
G. Deterministic ranking and tie-breaking
H. Metadata preservation
I. Expected result schema (HybridSearchResult)
J. Empty and whitespace retrieval behavior
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.rag.retrieval.schemas import RetrievalConfig, HybridSearchResult
from app.rag.retrieval.hybrid_retriever import (
    HybridRetriever,
    min_max_normalize,
    tokenize_query,
)


# ==========================================
# C, D: Min-Max Normalization Tests
# ==========================================

def test_min_max_normalize_standard_range():
    """Verify min_max_normalize maps scores linearly into [0.0, 1.0]."""
    scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    normalized = min_max_normalize(scores)

    assert pytest.approx(normalized[0]) == 0.0
    assert pytest.approx(normalized[2]) == 0.5
    assert pytest.approx(normalized[4]) == 1.0
    assert len(normalized) == len(scores)


def test_min_max_normalize_identical_scores():
    """Verify identical scores return 1.0 without division by zero or NaN."""
    scores = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    normalized = min_max_normalize(scores)

    assert not np.isnan(normalized).any()
    assert (normalized == 1.0).all()


def test_min_max_normalize_all_zeros():
    """Verify all zeros return 0.0 safely."""
    scores = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    normalized = min_max_normalize(scores)

    assert not np.isnan(normalized).any()
    assert (normalized == 0.0).all()


def test_min_max_normalize_empty_array():
    """Verify empty input returns empty array."""
    scores = np.array([], dtype=np.float32)
    normalized = min_max_normalize(scores)
    assert len(normalized) == 0


# ==========================================
# J: Tokenization & Empty Query Tests
# ==========================================

def test_tokenize_query_preserves_codes():
    """Verify query tokenization preserves domain codes like 19-1042.00, SHAP, and 0.40."""
    q = "What is O*NET code 19-1042.00 and SHAP threshold 0.40?"
    tokens = tokenize_query(q)

    assert "19-1042.00" in tokens
    assert "shap" in tokens
    assert "0.40" in tokens
    assert "what" not in tokens  # Stopword removed
    assert "is" not in tokens    # Stopword removed


def test_tokenize_query_all_stopwords_fallback():
    """Verify all-stopword query falls back gracefully without returning empty."""
    q = "What is it"
    tokens = tokenize_query(q)
    assert len(tokens) >= 1


# ==========================================
# Mocked Retriever Fixture
# ==========================================

@pytest.fixture
def mock_retriever():
    """Creates a HybridRetriever with mocked Chroma collection and BM25 model."""
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = np.ones((384,), dtype=np.float32)

    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection

    config = RetrievalConfig(
        dense_top_k=5,
        sparse_top_k=5,
        final_top_k=4,
        dense_weight=0.8,
        sparse_weight=0.2
    )

    with patch("app.rag.retrieval.hybrid_retriever.HybridRetriever._load_sparse_index"):
        retriever = HybridRetriever(
            config=config,
            embedder=mock_embedder,
            chroma_client=mock_chroma_client,
            collection_name="test_collection",
            sparse_dir="dummy_path"
        )

    # Set up mock BM25 and metadata list
    retriever.chunk_metadata_list = [
        {
            "chunk_id": "chunk_01",
            "doc_id": "doc_01",
            "text": "Medical and clinical research scientist.",
            "contextual_text": "[Doc: O*NET] Medical scientist",
            "source": "occupation_master.csv",
            "title": "Medical Scientists",
            "section": "Description",
            "document_type": "occupation",
            "metadata": {"soc_code": "19-1042.00"}
        },
        {
            "chunk_id": "chunk_02",
            "doc_id": "doc_02",
            "text": "HR specialist managing benefits and recruiting.",
            "contextual_text": "[Doc: O*NET] HR specialist",
            "source": "occupation_master.csv",
            "title": "HR Specialists",
            "section": "Description",
            "document_type": "occupation",
            "metadata": {"soc_code": "13-1071.00"}
        },
        {
            "chunk_id": "chunk_03",
            "doc_id": "doc_03",
            "text": "Model card governance and 0.40 threshold.",
            "contextual_text": "[Doc: ModelCard] Threshold section",
            "source": "docs/model_card.md",
            "title": "Model Card",
            "section": "Thresholds",
            "document_type": "governance",
            "metadata": {"version": "1.0"}
        }
    ]
    retriever.chunk_lookup = {m["chunk_id"]: m for m in retriever.chunk_metadata_list}

    mock_bm25 = MagicMock()
    # Scores corresponding to chunk_01, chunk_02, chunk_03
    mock_bm25.get_scores.return_value = [12.5, 4.0, 0.0]
    retriever.bm25_model = mock_bm25

    return retriever


# ==========================================
# B, E, F, G, H, I: Retrieval Pipeline Tests
# ==========================================

def test_retrieve_empty_query_returns_empty_list(mock_retriever):
    """J. Verify empty and whitespace queries return empty list."""
    assert mock_retriever.retrieve("") == []
    assert mock_retriever.retrieve("   ") == []


def test_dense_retrieval_invocation_and_mapping(mock_retriever):
    """B. Verify dense retrieval invokes ChromaDB and maps distance to cosine similarity."""
    mock_retriever.dense_collection.query.return_value = {
        "ids": [["chunk_01", "chunk_02"]],
        "distances": [[0.1, 0.4]],  # cosine distances
        "documents": [["Doc 1 text", "Doc 2 text"]],
        "metadatas": [[{"title": "T1"}, {"title": "T2"}]]
    }

    dense_items = mock_retriever.retrieve_dense("test query", top_k=2)

    assert len(dense_items) == 2
    assert dense_items[0]["chunk_id"] == "chunk_01"
    # similarity = 1.0 - 0.1 = 0.9
    assert pytest.approx(dense_items[0]["dense_score"]) == 0.9
    # similarity = 1.0 - 0.4 = 0.6
    assert pytest.approx(dense_items[1]["dense_score"]) == 0.6


def test_sparse_retrieval_invocation(mock_retriever):
    """A. Verify sparse retrieval invokes BM25Okapi and ranks descending."""
    sparse_items = mock_retriever.retrieve_sparse("research scientist", top_k=2)

    assert len(sparse_items) == 2
    assert sparse_items[0]["chunk_id"] == "chunk_01"
    assert sparse_items[0]["sparse_score"] == 12.5
    assert sparse_items[1]["chunk_id"] == "chunk_02"
    assert sparse_items[1]["sparse_score"] == 4.0


def test_hybrid_weighted_fusion_and_ranking(mock_retriever):
    """
    E, F, G, H, I:
    Verify candidate union, Min-Max normalization, 0.8*Dense + 0.2*Sparse weighting,
    metadata preservation, and deterministic ranking.
    """
    # Dense results: chunk_01=0.8, chunk_03=0.4
    # Sparse results: chunk_01=10.0, chunk_02=5.0
    mock_retriever.dense_collection.query.return_value = {
        "ids": [["chunk_01", "chunk_03"]],
        "distances": [[0.2, 0.6]],  # sims: 0.8, 0.4
        "documents": [["Text 1", "Text 3"]],
        "metadatas": [[{}, {}]]
    }
    mock_retriever.bm25_model.get_scores.return_value = [10.0, 5.0, 0.0]

    results = mock_retriever.retrieve("medical scientist")

    # Union has 3 chunks: chunk_01, chunk_02, chunk_03
    assert len(results) == 3

    # Candidate chunk_01 has dense=0.8, sparse=10.0
    # Candidate chunk_02 has dense=0.0, sparse=5.0
    # Candidate chunk_03 has dense=0.4, sparse=0.0
    r0 = results[0]
    assert isinstance(r0, HybridSearchResult)
    assert r0.chunk_id == "chunk_01"
    assert r0.rank == 1

    # Dense scores: [0.8, 0.0, 0.4] -> min=0.0, max=0.8 -> norms: [1.0, 0.0, 0.5]
    # Sparse scores: [10.0, 5.0, 0.0] -> min=0.0, max=10.0 -> norms: [1.0, 0.5, 0.0]
    # Chunk 01: 0.8*1.0 + 0.2*1.0 = 1.0
    # Chunk 02: 0.8*0.0 + 0.2*0.5 = 0.1
    # Chunk 03: 0.8*0.5 + 0.2*0.0 = 0.4
    # Ranking order: chunk_01 (1.0), chunk_03 (0.4), chunk_02 (0.1)
    assert results[0].chunk_id == "chunk_01"
    assert pytest.approx(results[0].hybrid_score) == 1.0

    assert results[1].chunk_id == "chunk_03"
    assert pytest.approx(results[1].hybrid_score) == 0.4

    assert results[2].chunk_id == "chunk_02"
    assert pytest.approx(results[2].hybrid_score) == 0.1

    # Check metadata preservation
    assert results[0].title == "Medical Scientists"
    assert results[0].metadata["soc_code"] == "19-1042.00"


def test_deterministic_tie_breaking(mock_retriever):
    """G. Verify deterministic tie-breaking on chunk_id ascending when scores match."""
    mock_retriever.dense_collection.query.return_value = {
        "ids": [["chunk_02", "chunk_01"]],
        "distances": [[0.5, 0.5]],  # equal dense scores
        "documents": [["Text 2", "Text 1"]],
        "metadatas": [[{}, {}]]
    }
    mock_retriever.bm25_model.get_scores.return_value = [0.0, 0.0, 0.0]  # equal sparse scores

    results = mock_retriever.retrieve("query")

    # Identical hybrid scores -> chunk_01 must precede chunk_02 alphabetically
    assert results[0].chunk_id == "chunk_01"
    assert results[1].chunk_id == "chunk_02"


# ==========================================
# Regression: Exact-Match & Candidate Coverage
# ==========================================

def test_exact_identifier_match_prioritization(mock_retriever):
    """
    Regression Test:
    Verify that queries containing exact structured identifiers (e.g. 19-1042.00)
    promote the matching chunk to Rank 1 even if absent from dense top results.
    """
    # Dense returns only chunk_02 and chunk_03 (chunk_01 absent, dense_score=0.0)
    mock_retriever.dense_collection.query.return_value = {
        "ids": [["chunk_02", "chunk_03"]],
        "distances": [[0.1, 0.2]],  # high dense similarity for unrelated chunks
        "documents": [["Text 2", "Text 3"]],
        "metadatas": [[{}, {}]]
    }
    # BM25 finds chunk_01 containing the exact code 19-1042.00
    mock_retriever.bm25_model.get_scores.return_value = [15.0, 0.0, 0.0]

    results = mock_retriever.retrieve("What does O*NET code 19-1042.00 represent?")

    assert len(results) >= 1
    # chunk_01 must be ranked #1 due to exact identifier match protection
    assert results[0].chunk_id == "chunk_01"
    assert results[0].is_exact_match is True
    assert results[0].rank == 1


def test_sparse_candidate_coverage_guarantee(mock_retriever):
    """
    Regression Test:
    Verify that candidate pool coverage ensures top BM25 lexical results
    survive into the candidate pool even for non-identifier queries.
    """
    # Dense returns chunk_03 with high similarity
    mock_retriever.dense_collection.query.return_value = {
        "ids": [["chunk_03"]],
        "distances": [[0.05]],
        "documents": [["Text 3"]],
        "metadatas": [[{}]]
    }
    # BM25 returns chunk_02 with strong score, chunk_01 with moderate score
    mock_retriever.bm25_model.get_scores.return_value = [5.0, 10.0, 0.0]

    # final_top_k is 4 in mock_retriever
    results = mock_retriever.retrieve("general lexical query without identifier")

    result_cids = [r.chunk_id for r in results]
    # Both top sparse hits must be present in the candidate pool
    assert "chunk_02" in result_cids
    assert "chunk_01" in result_cids

