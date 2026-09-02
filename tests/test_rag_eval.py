"""
tests/test_rag_eval.py
======================
Unit tests for the RAG embedding evaluation dataset and metric calculation:
1. Evaluation dataset schema validation (tests/fixtures/rag_eval_queries.json)
2. All expected chunk IDs exist in the real Step 2 chunk corpus
3. All queries have at least one valid expected relevant chunk
4. Retrieval metric computation correctness (Hit@1, Hit@3, Hit@5, Hit@10, MRR)
5. Benchmark report output schema verification
"""

import json
import pytest
import numpy as np

from app.rag.loaders import load_all_knowledge_documents
from app.rag.chunking import chunk_all_documents
from app.rag.metrics import compute_metrics
from app.utils.config import BASE_DIR

EVAL_QUERIES_PATH = BASE_DIR / "tests" / "fixtures" / "rag_eval_queries.json"
REPORT_FILE = BASE_DIR / "reports" / "rag" / "embedding_benchmark.json"


def test_eval_queries_file_exists_and_valid():
    """Verify evaluation dataset exists and has at least 20 queries."""
    assert EVAL_QUERIES_PATH.exists()
    with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    assert isinstance(queries, list)
    assert len(queries) >= 20

    categories = set()
    for q in queries:
        assert "query_id" in q
        assert "category" in q
        assert "query" in q
        assert "expected_chunk_ids" in q
        assert isinstance(q["expected_chunk_ids"], list)
        assert len(q["expected_chunk_ids"]) >= 1
        categories.add(q["category"])

    # Must cover all 4 core knowledge themes
    assert "occupation" in categories
    assert "role_mapping" in categories
    assert "governance" in categories
    assert "architecture" in categories


def test_all_expected_chunk_ids_exist_in_corpus():
    """Verify that every chunk_id referenced in rag_eval_queries.json exists in real corpus."""
    with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    documents = load_all_knowledge_documents()
    chunks = chunk_all_documents(documents)
    valid_chunk_ids = {c.chunk_id for c in chunks}

    for q in queries:
        for cid in q["expected_chunk_ids"]:
            assert cid in valid_chunk_ids, f"Expected chunk_id '{cid}' in query '{q['query_id']}' does not exist in corpus!"


def test_metric_calculation_correctness():
    """Verify compute_metrics calculates Hit@k and MRR correctly on synthetic rank vectors."""
    # 2 queries, 4 chunks: [c0, c1, c2, c3]
    chunk_ids = ["c0", "c1", "c2", "c3"]
    eval_queries = [
        {"query_id": "q1", "expected_chunk_ids": ["c1"]},  # will be at rank 2
        {"query_id": "q2", "expected_chunk_ids": ["c0"]},  # will be at rank 1
    ]

    # Construct similarity matrix where:
    # q1 similarities: c2=0.9, c1=0.8, c0=0.5, c3=0.1 -> ranks: [c2, c1, c0, c3] -> c1 is rank 2
    # q2 similarities: c0=0.95, c1=0.4, c2=0.3, c3=0.2 -> ranks: [c0, c1, c2, c3] -> c0 is rank 1
    query_emb = np.array([
        [0.5, 0.8, 0.9, 0.1],
        [0.95, 0.4, 0.3, 0.2]
    ])
    corpus_emb = np.eye(4)

    metrics = compute_metrics(query_emb, corpus_emb, chunk_ids, eval_queries)

    # q1: first relevant c1 is rank 2 -> RR = 1/2 = 0.5
    # q2: first relevant c0 is rank 1 -> RR = 1/1 = 1.0
    # Mean RR = (0.5 + 1.0) / 2 = 0.75
    # Hit@1: q1=0, q2=1 -> 0.5
    # Hit@3: q1=1, q2=1 -> 1.0
    assert metrics["Hit@1"] == 0.5
    assert metrics["Hit@3"] == 1.0
    assert metrics["Hit@5"] == 1.0
    assert metrics["Hit@10"] == 1.0
    assert metrics["MRR"] == 0.75


def test_benchmark_output_schema_if_exists():
    """Verify that if the benchmark report exists, its schema matches specification."""
    if not REPORT_FILE.exists():
        pytest.skip("Benchmark report not yet generated.")

    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "total_corpus_chunks" in data
    assert "total_eval_queries" in data
    assert "candidates" in data
    assert isinstance(data["candidates"], list)

    for c in data["candidates"]:
        assert "model_name" in c
        if "metrics" in c:
            assert "Hit@1" in c["metrics"]
            assert "Hit@3" in c["metrics"]
            assert "Hit@5" in c["metrics"]
            assert "Hit@10" in c["metrics"]
            assert "MRR" in c["metrics"]
            assert "dimension" in c
            assert "timings" in c
