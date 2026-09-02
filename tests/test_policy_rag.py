"""
tests/test_policy_rag.py
========================
Comprehensive test suite for the Dedicated Synthetic HR Policy RAG Index:
1. Markdown policy documents load correctly
2. Policy chunking preserves headings, sections, and metadata
3. Chunk IDs and policy IDs are correctly formed
4. Chroma collection exists and has expected count
5. BM25 policy index exists and loads cleanly
6. Dense retrieval on policy collection works
7. Sparse retrieval on policy collection works
8. Hybrid retrieval on policy collection works
9. Exact policy ID query retrieves the correct document
10. Cross-encoder reranks policy results
11. General knowledge collection is untouched (1,042 chunks)
12. General BM25 index is untouched (1,042 chunks)
13. Policy ingestion is idempotent
14. Policy metadata includes source_type == "synthetic_hr_policy"
"""

import os
import json
import pickle
import pytest
import chromadb
from pathlib import Path

from app.rag.loaders.policy_loader import load_all_hr_policies, load_single_policy_file
from app.rag.chunking import chunk_all_documents
from app.rag.embeddings import BGEEmbedder
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.retrieval.reranker import CrossEncoderReranker
from app.rag.retrieval.schemas import RetrievalConfig
from app.utils.config import BASE_DIR

POLICY_COLLECTION_NAME = "enterprise_hr_policies_bge"
GENERAL_COLLECTION_NAME = "enterprise_hr_knowledge_bge"
POLICY_SPARSE_DIR = BASE_DIR / "data" / "rag" / "policy_sparse_index"
GENERAL_SPARSE_DIR = BASE_DIR / "data" / "rag" / "sparse_index"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vectorstore"
POLICY_DOCS_DIR = BASE_DIR / "data" / "knowledge_base" / "hr_policies"


@pytest.fixture(scope="module")
def policy_client():
    return chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))


@pytest.fixture(scope="module")
def policy_retriever():
    embedder = BGEEmbedder()
    config = RetrievalConfig(dense_top_k=10, sparse_top_k=10, final_top_k=5)
    return HybridRetriever(
        config=config,
        embedder=embedder,
        collection_name=POLICY_COLLECTION_NAME,
        sparse_dir=POLICY_SPARSE_DIR
    )


# Test 1: Markdown policy documents load correctly
def test_1_markdown_policy_documents_load_correctly():
    docs = load_all_hr_policies(POLICY_DOCS_DIR)
    assert len(docs) == 150, f"Expected 150 structured sections from 10 policy files, got {len(docs)}"
    policy_ids = {d.metadata.get("policy_id") for d in docs}
    expected_ids = {
        "POL-JOB-001", "POL-AI-001", "POL-MODEL-001", "POL-RISK-001", "POL-SKILL-001",
        "POL-LEARN-001", "POL-CAREER-001", "POL-DATA-001", "POL-REVIEW-001", "POL-MONITOR-001"
    }
    assert policy_ids == expected_ids, f"Mismatch in policy IDs: {policy_ids.symmetric_difference(expected_ids)}"


# Test 2: Policy chunking preserves headings, sections, and metadata
def test_2_policy_chunking_preserves_headings_and_sections():
    docs = load_all_hr_policies(POLICY_DOCS_DIR)
    chunks = chunk_all_documents(docs)
    assert len(chunks) == 150
    for c in chunks:
        assert c.title, "Chunk title must not be empty"
        assert c.section, "Chunk section must not be empty"
        assert c.metadata.get("policy_id"), "Chunk metadata must contain policy_id"
        assert c.metadata.get("policy_title"), "Chunk metadata must contain policy_title"
        assert c.metadata.get("policy_domain"), "Chunk metadata must contain policy_domain"
        assert c.metadata.get("policy_version"), "Chunk metadata must contain policy_version"
        assert c.metadata.get("policy_status"), "Chunk metadata must contain policy_status"


# Test 3: Chunk IDs and policy IDs are correctly formed
def test_3_chunk_ids_and_policy_ids_are_correctly_formed():
    docs = load_all_hr_policies(POLICY_DOCS_DIR)
    chunks = chunk_all_documents(docs)
    for c in chunks:
        assert c.chunk_id.startswith("pol_"), f"Chunk ID {c.chunk_id} should start with 'pol_'"
        assert "_c01" in c.chunk_id, f"Chunk ID {c.chunk_id} missing chunk index suffix '_c01'"
        assert c.metadata["policy_id"].startswith("POL-"), f"Policy ID {c.metadata['policy_id']} invalid format"


# Test 4: Chroma collection exists and has expected count
def test_4_chroma_collection_exists_and_has_expected_count(policy_client):
    col_names = [c.name for c in policy_client.list_collections()]
    assert POLICY_COLLECTION_NAME in col_names, f"Collection {POLICY_COLLECTION_NAME} missing from ChromaDB"
    col = policy_client.get_collection(POLICY_COLLECTION_NAME)
    assert col.count() == 150, f"Expected 150 chunks in {POLICY_COLLECTION_NAME}, got {col.count()}"


# Test 5: BM25 policy index exists and loads cleanly
def test_5_bm25_policy_index_exists_and_loads_cleanly():
    index_file = POLICY_SPARSE_DIR / "bm25_index.pkl"
    meta_file = POLICY_SPARSE_DIR / "chunk_metadata.json"
    assert index_file.exists(), f"Policy BM25 index missing at {index_file}"
    assert meta_file.exists(), f"Policy metadata missing at {meta_file}"

    with open(index_file, "rb") as f:
        bm25_model = pickle.load(f)
    assert hasattr(bm25_model, "corpus_size"), "Loaded object is not a valid BM25Okapi model"
    assert bm25_model.corpus_size == 150, f"Expected 150 documents in BM25 index, got {bm25_model.corpus_size}"

    with open(meta_file, "r", encoding="utf-8") as f:
        meta_list = json.load(f)
    assert len(meta_list) == 150, f"Expected 150 metadata entries, got {len(meta_list)}"


# Test 6: Dense retrieval on policy collection works
def test_6_dense_retrieval_on_policy_collection_works(policy_retriever):
    results = policy_retriever.retrieve_dense("decision threshold 0.40 attrition", top_k=5)
    assert len(results) > 0, "Dense retrieval returned empty results"
    top_pids = [r["metadata"].get("policy_id") for r in results]
    assert "POL-MODEL-001" in top_pids, "Expected POL-MODEL-001 in dense top results"


# Test 7: Sparse retrieval on policy collection works
def test_7_sparse_retrieval_on_policy_collection_works(policy_retriever):
    results = policy_retriever.retrieve_sparse("POL-JOB-001 O*NET SOC mapping", top_k=5)
    assert len(results) > 0, "Sparse retrieval returned empty results"
    top_pids = [r["metadata"].get("policy_id") for r in results]
    assert "POL-JOB-001" in top_pids, "Expected POL-JOB-001 in sparse top results"


# Test 8: Hybrid retrieval on policy collection works
def test_8_hybrid_retrieval_on_policy_collection_works(policy_retriever):
    results = policy_retriever.retrieve("What is the role of human review in AI recommendations?")
    assert len(results) > 0, "Hybrid retrieval returned empty results"
    top_pids = [r.metadata.get("policy_id") for r in results]
    assert any(pid in ["POL-REVIEW-001", "POL-AI-001"] for pid in top_pids), (
        f"Expected POL-REVIEW-001 or POL-AI-001 in hybrid results, got {top_pids[:3]}"
    )


# Test 9: Exact policy ID query retrieves the correct document
def test_9_exact_policy_id_retrieves_correct_document(policy_retriever):
    test_cases = [
        ("POL-MODEL-001", "What does POL-MODEL-001 specify?"),
        ("POL-JOB-001", "What is defined in POL-JOB-001?"),
        ("POL-REVIEW-001", "What are the rules of POL-REVIEW-001?"),
        ("POL-AI-001", "Explain the mandates in POL-AI-001")
    ]
    for expected_pid, query in test_cases:
        results = policy_retriever.retrieve(query)
        assert len(results) > 0, f"Query '{query}' returned no results"
        retrieved_pids = [r.metadata.get("policy_id") for r in results[:3]]
        assert expected_pid in retrieved_pids, (
            f"Expected {expected_pid} in top-3 for query '{query}', got {retrieved_pids}"
        )
        assert results[0].metadata.get("policy_id") == expected_pid, (
            f"Expected {expected_pid} at Rank 1 for query '{query}', got {results[0].metadata.get('policy_id')}"
        )


# Test 10: Cross-encoder reranks policy results
def test_10_cross_encoder_reranks_policy_results(policy_retriever):
    reranker = CrossEncoderReranker()
    query = "What is the operational decision threshold for attrition probability?"
    candidates = policy_retriever.retrieve(query)
    reranked = reranker.rerank(query, candidates, top_k=3)
    assert len(reranked) == 3, f"Expected exactly 3 reranked chunks, got {len(reranked)}"
    for rank, item in enumerate(reranked, 1):
        assert item.rerank_rank == rank
        assert isinstance(item.rerank_score, float)
    pids = [item.metadata.get("policy_id") for item in reranked]
    assert "POL-MODEL-001" in pids or "POL-RISK-001" in pids


# Test 11: General knowledge collection is untouched
def test_11_general_knowledge_collection_is_untouched(policy_client):
    col = policy_client.get_collection(GENERAL_COLLECTION_NAME)
    assert col.count() == 1042, f"Expected 1,042 chunks in {GENERAL_COLLECTION_NAME}, found {col.count()}"


# Test 12: General BM25 index is untouched
def test_12_general_bm25_index_is_untouched():
    meta_file = GENERAL_SPARSE_DIR / "chunk_metadata.json"
    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    assert len(meta_data) == 1042, f"Expected 1,042 chunks in general BM25 index, found {len(meta_data)}"


# Test 13: Policy ingestion is idempotent
def test_13_policy_ingestion_is_idempotent(policy_client):
    from scripts.build_rag_policy_index import build_policy_indices
    count_after_first = policy_client.get_collection(POLICY_COLLECTION_NAME).count()
    # Re-run build
    total_rebuilt = build_policy_indices()
    assert total_rebuilt == 150
    count_after_second = policy_client.get_collection(POLICY_COLLECTION_NAME).count()
    assert count_after_first == count_after_second == 150, (
        f"Idempotency violation: count changed from {count_after_first} to {count_after_second}"
    )


# Test 14: Policy metadata includes source_type == "synthetic_hr_policy"
def test_14_policy_metadata_source_type():
    with open(POLICY_SPARSE_DIR / "chunk_metadata.json", "r", encoding="utf-8") as f:
        meta_list = json.load(f)
    for item in meta_list:
        assert item.get("source_type") == "synthetic_hr_policy", (
            f"Chunk {item.get('chunk_id')} missing source_type='synthetic_hr_policy'"
        )
        assert item.get("policy_status") == "Synthetic Demo Policy", (
            f"Chunk {item.get('chunk_id')} missing policy_status='Synthetic Demo Policy'"
        )
