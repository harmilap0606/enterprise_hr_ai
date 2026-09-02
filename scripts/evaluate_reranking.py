"""
scripts/evaluate_reranking.py
=============================
Empirical evaluation comparing:
A. Step 4B Hybrid Retrieval (0.8 Dense + 0.2 Sparse)
B. Step 5B Hybrid Retrieval + Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)

Evaluates on the 24 ground-truth queries in tests/fixtures/rag_eval_queries.json.
Uses candidate pool size = 15.
Calculates: Hit@1, Hit@3, Hit@5, Hit@10, MRR, mean latency, p50 latency, p95 latency.
Exports structured benchmark results to reports/rag/reranking_evaluation.json.
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Set

import numpy as np

from app.rag.retrieval.schemas import RetrievalConfig, RerankerConfig
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.retrieval.reranker import CrossEncoderReranker
from app.utils.config import BASE_DIR

sys.stdout.reconfigure(encoding="utf-8")

EVAL_QUERIES_FILE = BASE_DIR / "tests" / "fixtures" / "rag_eval_queries.json"
REPORT_FILE = BASE_DIR / "reports" / "rag" / "reranking_evaluation.json"


def evaluate_reranking():
    print("=" * 80)
    print("EMPIRICAL EVALUATION: HYBRID RETRIEVAL vs HYBRID + CROSS-ENCODER RERANKING")
    print("=" * 80)

    # 1. Load evaluation queries
    with open(EVAL_QUERIES_FILE, "r", encoding="utf-8") as f:
        eval_queries: List[Dict[str, Any]] = json.load(f)

    total_queries = len(eval_queries)
    print(f"Loaded {total_queries} evaluation queries from: {EVAL_QUERIES_FILE}")

    # 2. Initialize Hybrid Retriever (candidate_count = 15)
    print("\nInitializing Step 4B HybridRetriever...")
    retriever = HybridRetriever(
        config=RetrievalConfig(
            dense_top_k=20,
            sparse_top_k=20,
            final_top_k=15,
            dense_weight=0.8,
            sparse_weight=0.2
        )
    )

    # 3. Initialize Cross-Encoder Reranker
    print(f"Initializing Step 5B CrossEncoderReranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    reranker = CrossEncoderReranker(
        config=RerankerConfig(
            final_top_k=15,  # Retain all 15 candidates for fair Hit@10 evaluation
            batch_size=32
        )
    )
    actual_device = str(reranker.device)
    print(f"Reranker initialized on device: '{actual_device}'")

    # Arrays to accumulate ranks and timing
    hybrid_ranks: List[List[str]] = []
    reranked_ranks: List[List[str]] = []

    hybrid_latencies: List[float] = []
    rerank_latencies: List[float] = []
    total_latencies: List[float] = []

    print("\nExecuting evaluation across all queries...")
    for idx, q_item in enumerate(eval_queries, 1):
        q_text = q_item["query"]

        # Step 1: Hybrid Retrieval (15 candidates)
        t0 = time.perf_counter()
        hybrid_candidates = retriever.retrieve(q_text)
        t_hybrid = (time.perf_counter() - t0) * 1000.0
        hybrid_latencies.append(t_hybrid)

        h_chunk_ids = [c.chunk_id for c in hybrid_candidates]
        hybrid_ranks.append(h_chunk_ids)

        # Step 2: Cross-Encoder Reranking
        t0_rerank = time.perf_counter()
        reranked_results = reranker.rerank(q_text, hybrid_candidates, top_k=15)
        t_rerank = (time.perf_counter() - t0_rerank) * 1000.0
        rerank_latencies.append(t_rerank)

        r_chunk_ids = [r.chunk_id for r in reranked_results]
        reranked_ranks.append(r_chunk_ids)

        total_latencies.append(t_hybrid + t_rerank)

    # 4. Metric computation
    def compute_ranking_metrics(ranked_lists: List[List[str]]) -> Dict[str, float]:
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
            "MRR": round(float(np.mean(reciprocal_ranks)), 4)
        }

    hybrid_metrics = compute_ranking_metrics(hybrid_ranks)
    reranked_metrics = compute_ranking_metrics(reranked_ranks)

    # Latency percentiles
    hybrid_metrics["avg_latency_ms"] = round(float(np.mean(hybrid_latencies)), 2)
    hybrid_metrics["p50_latency_ms"] = round(float(np.percentile(hybrid_latencies, 50)), 2)
    hybrid_metrics["p95_latency_ms"] = round(float(np.percentile(hybrid_latencies, 95)), 2)

    reranked_metrics["avg_hybrid_latency_ms"] = round(float(np.mean(hybrid_latencies)), 2)
    reranked_metrics["avg_rerank_latency_ms"] = round(float(np.mean(rerank_latencies)), 2)
    reranked_metrics["avg_total_latency_ms"] = round(float(np.mean(total_latencies)), 2)
    reranked_metrics["p50_total_latency_ms"] = round(float(np.percentile(total_latencies, 50)), 2)
    reranked_metrics["p95_total_latency_ms"] = round(float(np.percentile(total_latencies, 95)), 2)

    # 5. Print Results Table
    print("\n" + "=" * 85)
    print("EMPIRICAL BENCHMARK: STEP 4B HYBRID vs STEP 5B HYBRID + RERANKING")
    print("=" * 85)
    header = f"{'Metric':<18} | {'Hybrid (Step 4B)':<18} | {'Hybrid + Rerank (5B)':<20} | {'Delta':<12}"
    print(header)
    print("-" * len(header))

    for m in ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"]:
        val_h = hybrid_metrics[m]
        val_r = reranked_metrics[m]
        delta = val_r - val_h
        sign = "+" if delta >= 0 else ""
        print(f"{m:<18} | {val_h:<18.4f} | {val_r:<20.4f} | {sign}{delta:.4f}")

    print("-" * len(header))
    print(f"{'Mean Latency':<18} | {hybrid_metrics['avg_latency_ms']:<15.2f} ms | {reranked_metrics['avg_total_latency_ms']:<17.2f} ms | +{reranked_metrics['avg_rerank_latency_ms']:.2f} ms")
    print(f"{'p50 Latency':<18} | {hybrid_metrics['p50_latency_ms']:<15.2f} ms | {reranked_metrics['p50_total_latency_ms']:<17.2f} ms |")
    print(f"{'p95 Latency':<18} | {hybrid_metrics['p95_latency_ms']:<15.2f} ms | {reranked_metrics['p95_total_latency_ms']:<17.2f} ms |")
    print("=" * 85)

    # 6. Save JSON Report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": total_queries,
        "reranker_model": reranker.model_name,
        "device": actual_device,
        "candidate_count": 15,
        "evaluation_top_k": 10,
        "production_default_top_k": 3,
        "hybrid_baseline": hybrid_metrics,
        "hybrid_with_reranking": reranked_metrics,
        "deltas": {
            m: round(reranked_metrics[m] - hybrid_metrics[m], 4)
            for m in ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"]
        }
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved empirical report to: {REPORT_FILE}\n")


if __name__ == "__main__":
    evaluate_reranking()
