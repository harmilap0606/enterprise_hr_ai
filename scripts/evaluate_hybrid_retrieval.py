"""
scripts/evaluate_hybrid_retrieval.py
====================================
Empirical benchmark evaluating:
1. Dense Only (BAAI/bge-small-en-v1.5)
2. Sparse Only (BM25Okapi)
3. Hybrid Search (0.8 Dense + 0.2 Sparse via Min-Max Normalization)

Evaluates on the 24 ground-truth queries in tests/fixtures/rag_eval_queries.json.
Calculates: Hit@1, Hit@3, Hit@5, Hit@10, and Mean Reciprocal Rank (MRR).
Exports structured results to reports/rag/hybrid_evaluation.json.
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Set

import numpy as np

from app.rag.retrieval.schemas import RetrievalConfig
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.utils.config import BASE_DIR

sys.stdout.reconfigure(encoding="utf-8")

EVAL_QUERIES_FILE = BASE_DIR / "tests" / "fixtures" / "rag_eval_queries.json"
REPORT_FILE = BASE_DIR / "reports" / "rag" / "hybrid_evaluation.json"


def evaluate_retrieval():
    print("=" * 75)
    print("EMPIRICAL EVALUATION: DENSE vs SPARSE vs HYBRID RETRIEVAL (0.8 / 0.2)")
    print("=" * 75)

    # 1. Load evaluation queries
    with open(EVAL_QUERIES_FILE, "r", encoding="utf-8") as f:
        eval_queries: List[Dict[str, Any]] = json.load(f)

    total_queries = len(eval_queries)
    print(f"Loaded {total_queries} evaluation queries from: {EVAL_QUERIES_FILE}")

    # 2. Initialize HybridRetriever
    print("Initializing HybridRetriever (BGE-small-en-v1.5 + BM25Okapi)...")
    init_start = time.perf_counter()
    retriever = HybridRetriever(
        config=RetrievalConfig(
            dense_top_k=25,
            sparse_top_k=25,
            final_top_k=20,
            dense_weight=0.8,
            sparse_weight=0.2
        )
    )
    init_time = time.perf_counter() - init_start
    print(f"Retriever initialized in {init_time:.2f}s.")
    print(f"Dense collection count: {retriever.dense_collection.count()}")
    print(f"Sparse corpus count: {len(retriever.chunk_metadata_list)}")

    # Data structures to accumulate ranks
    dense_ranks: List[List[str]] = []
    sparse_ranks: List[List[str]] = []
    hybrid_ranks: List[List[str]] = []

    dense_latencies: List[float] = []
    sparse_latencies: List[float] = []
    hybrid_latencies: List[float] = []

    print("\nRunning evaluation on queries...")
    for idx, q_item in enumerate(eval_queries, 1):
        q_text = q_item["query"]

        # Dense Only
        t0 = time.perf_counter()
        dense_res = retriever.retrieve_dense(q_text, top_k=20)
        t_dense = (time.perf_counter() - t0) * 1000.0
        dense_latencies.append(t_dense)
        dense_ranked_ids = [d["chunk_id"] for d in dense_res]
        dense_ranks.append(dense_ranked_ids)

        # Sparse Only
        t0 = time.perf_counter()
        sparse_res = retriever.retrieve_sparse(q_text, top_k=20)
        t_sparse = (time.perf_counter() - t0) * 1000.0
        sparse_latencies.append(t_sparse)
        sparse_ranked_ids = [s["chunk_id"] for s in sparse_res]
        sparse_ranks.append(sparse_ranked_ids)

        # Hybrid (0.8 Dense + 0.2 Sparse)
        t0 = time.perf_counter()
        hybrid_res = retriever.retrieve(q_text)
        t_hybrid = (time.perf_counter() - t0) * 1000.0
        hybrid_latencies.append(t_hybrid)
        hybrid_ranked_ids = [h.chunk_id for h in hybrid_res]
        hybrid_ranks.append(hybrid_ranked_ids)

    # 3. Compute Metrics Function
    def calculate_metrics(ranked_lists: List[List[str]]) -> Dict[str, float]:
        hit_1 = 0
        hit_3 = 0
        hit_5 = 0
        hit_10 = 0
        reciprocal_ranks = []

        for i, q in enumerate(eval_queries):
            expected: Set[str] = set(q["expected_chunk_ids"])
            ranked = ranked_lists[i]

            top_1 = set(ranked[:1])
            top_3 = set(ranked[:3])
            top_5 = set(ranked[:5])
            top_10 = set(ranked[:10])

            if any(c in top_1 for c in expected):
                hit_1 += 1
            if any(c in top_3 for c in expected):
                hit_3 += 1
            if any(c in top_5 for c in expected):
                hit_5 += 1
            if any(c in top_10 for c in expected):
                hit_10 += 1

            # Reciprocal rank
            found_rank = None
            for r_idx, c_id in enumerate(ranked, 1):
                if c_id in expected:
                    found_rank = r_idx
                    break

            if found_rank is not None:
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)

        n = len(eval_queries)
        return {
            "Hit@1": round(hit_1 / n, 4),
            "Hit@3": round(hit_3 / n, 4),
            "Hit@5": round(hit_5 / n, 4),
            "Hit@10": round(hit_10 / n, 4),
            "MRR": round(float(np.mean(reciprocal_ranks)), 4),
        }

    dense_metrics = calculate_metrics(dense_ranks)
    sparse_metrics = calculate_metrics(sparse_ranks)
    hybrid_metrics = calculate_metrics(hybrid_ranks)

    dense_metrics["avg_latency_ms"] = round(float(np.mean(dense_latencies)), 2)
    sparse_metrics["avg_latency_ms"] = round(float(np.mean(sparse_latencies)), 2)
    hybrid_metrics["avg_latency_ms"] = round(float(np.mean(hybrid_latencies)), 2)

    # 4. Print Results Table
    print("\n" + "=" * 75)
    print("EMPIRICAL RETRIEVAL BENCHMARK RESULTS (24 Ground-Truth Queries)")
    print("=" * 75)
    header = f"{'Method':<24} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'Hit@10':<8} | {'MRR':<8} | {'Latency':<10}"
    print(header)
    print("-" * len(header))

    def fmt_row(name, m):
        return f"{name:<24} | {m['Hit@1']:<8.4f} | {m['Hit@3']:<8.4f} | {m['Hit@5']:<8.4f} | {m['Hit@10']:<8.4f} | {m['MRR']:<8.4f} | {m['avg_latency_ms']:<7.2f} ms"

    print(fmt_row("Dense Only (BGE-Small)", dense_metrics))
    print(fmt_row("Sparse Only (BM25)", sparse_metrics))
    print(fmt_row("Hybrid (0.8 Dense + 0.2)", hybrid_metrics))
    print("=" * 75)

    # 5. Export Report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": total_queries,
        "dense_corpus_size": retriever.dense_collection.count(),
        "sparse_corpus_size": len(retriever.chunk_metadata_list),
        "weights": {
            "dense_weight": retriever.config.dense_weight,
            "sparse_weight": retriever.config.sparse_weight
        },
        "methods": {
            "dense_only": dense_metrics,
            "sparse_only": sparse_metrics,
            "hybrid_0.8_0.2": hybrid_metrics
        }
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved empirical report to: {REPORT_FILE}")


if __name__ == "__main__":
    evaluate_retrieval()
