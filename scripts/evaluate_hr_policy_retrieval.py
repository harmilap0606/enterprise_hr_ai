"""
scripts/evaluate_hr_policy_retrieval.py
=======================================
Empirical retrieval evaluation of the dedicated HR Policy RAG index across:
1. Dense Retrieval (BGE-small-en-v1.5)
2. Sparse Retrieval (BM25Okapi)
3. Hybrid Fusion (0.8 Dense + 0.2 Sparse + Exact ID Protection)
4. Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2) using contextual_text input pairs
5. Grounded Generation & Unsupported Query Refusal Verification

Evaluates against tests/fixtures/hr_policy_eval_queries.json across 5 distinct categories:
- Category 1: Exact Policy IDs
- Category 2: Policy Terminology
- Category 3: Semantic Queries
- Category 4: Multi-Policy Queries
- Category 5: Limitation / Refusal Questions

Outputs empirical reports to:
- reports/rag/hr_policy_retrieval_evaluation.json
- reports/rag/policy_reranking_evaluation.json
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

import numpy as np

from app.rag.embeddings import BGEEmbedder
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.retrieval.reranker import CrossEncoderReranker
from app.rag.retrieval.schemas import RetrievalConfig
from app.rag.pipeline import GroundedRAGPipeline, REFUSAL_MESSAGE
from app.utils.config import BASE_DIR

sys.stdout.reconfigure(encoding="utf-8")

EVAL_FIXTURE_PATH = BASE_DIR / "tests" / "fixtures" / "hr_policy_eval_queries.json"
REPORT_OUTPUT_PATH = BASE_DIR / "reports" / "rag" / "hr_policy_retrieval_evaluation.json"
POLICY_RERANK_EVAL_PATH = BASE_DIR / "reports" / "rag" / "policy_reranking_evaluation.json"

POLICY_COLLECTION_NAME = "enterprise_hr_policies_bge"
POLICY_SPARSE_DIR = BASE_DIR / "data" / "rag" / "policy_sparse_index"


def compute_metrics(
    retrieved_items: List[List[str]],
    ground_truth_targets: List[List[str]],
    k_list: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Computes Hit@K and MRR. A hit occurs if any retrieved item matches any target.
    """
    num_queries = len(ground_truth_targets)
    if num_queries == 0:
        return {}

    hits_at_k = {k: 0 for k in k_list}
    reciprocal_ranks = []

    for retrieved, targets in zip(retrieved_items, ground_truth_targets):
        target_set = set(t.upper() for t in targets)
        rr = 0.0

        for rank, item in enumerate(retrieved, 1):
            if item.upper() in target_set:
                if rr == 0.0:
                    rr = 1.0 / rank
                for k in k_list:
                    if rank <= k:
                        hits_at_k[k] += 1
                break

        reciprocal_ranks.append(rr)

    results = {f"hit@{k}": round(hits_at_k[k] / num_queries, 4) for k in k_list}
    results["mrr"] = round(float(np.mean(reciprocal_ranks)), 4)
    return results


def extract_policy_id_from_chunk(chunk_id: str, metadata: Dict[str, Any]) -> str:
    """Extracts uppercase policy ID (e.g. POL-MODEL-001) from chunk metadata or chunk ID."""
    if metadata and metadata.get("policy_id"):
        return metadata["policy_id"].upper()
    import re
    m = re.search(r"POL[_-][A-Z]+[_-]\d+", chunk_id, re.I)
    if m:
        return m.group(0).upper().replace("_", "-")
    return chunk_id


def evaluate_policy_rag():
    print("=" * 80)
    print("EMPIRICAL EVALUATION: HR POLICY RETRIEVAL & GROUNDED REFUSAL")
    print("=" * 80)

    # 1. Load evaluation dataset
    with open(EVAL_FIXTURE_PATH, "r", encoding="utf-8") as f:
        all_eval_queries = json.load(f)

    answerable_queries = [q for q in all_eval_queries if q["is_answerable"]]
    unsupported_queries = [q for q in all_eval_queries if not q["is_answerable"]]

    print(f"Loaded {len(all_eval_queries)} total evaluation queries from: {EVAL_FIXTURE_PATH}")
    print(f"  Answerable test queries: {len(answerable_queries)}")
    print(f"  Unsupported test queries (refusal verification): {len(unsupported_queries)}")

    # 2. Initialize Retriever & Reranker on Policy Collection
    print("\nInitializing Policy Hybrid Retriever...")
    embedder = BGEEmbedder()
    config = RetrievalConfig(dense_top_k=20, sparse_top_k=20, final_top_k=15, dense_weight=0.8, sparse_weight=0.2)
    retriever = HybridRetriever(
        config=config,
        embedder=embedder,
        collection_name=POLICY_COLLECTION_NAME,
        sparse_dir=POLICY_SPARSE_DIR
    )

    print("Initializing Cross-Encoder Reranker...")
    reranker = CrossEncoderReranker()

    # 3. Evaluate Retrieval on Answerable Queries
    print("\nExecuting multi-stage retrieval evaluation on answerable queries...")

    dense_retrieved: List[List[str]] = []
    sparse_retrieved: List[List[str]] = []
    hybrid_retrieved: List[List[str]] = []
    reranked_retrieved: List[List[str]] = []
    ground_truth_targets: List[List[str]] = []

    dense_latencies = []
    sparse_latencies = []
    hybrid_latencies = []
    rerank_latencies = []

    exact_id_hits_hybrid = 0
    exact_id_hits_rerank = 0
    exact_id_total = 0

    semantic_hits_hybrid = 0
    semantic_hits_rerank = 0
    semantic_total = 0

    multi_policy_rank1_rerank = 0
    multi_policy_top3_rerank = 0
    multi_policy_total = 0

    per_query_records = []

    for q in answerable_queries:
        query_text = q["query"]
        expected_ids = [e.upper() for e in q["expected_policy_ids"]]
        ground_truth_targets.append(expected_ids)
        cat = q["category"]

        # A. Dense
        t0 = time.perf_counter()
        dense_hits = retriever.retrieve_dense(query_text, top_k=20)
        t1 = time.perf_counter()
        dense_latencies.append((t1 - t0) * 1000)
        dense_pids = [extract_policy_id_from_chunk(d["chunk_id"], d.get("metadata", {})) for d in dense_hits]
        dense_retrieved.append(dense_pids)

        # B. Sparse
        t0 = time.perf_counter()
        sparse_hits = retriever.retrieve_sparse(query_text, top_k=20)
        t1 = time.perf_counter()
        sparse_latencies.append((t1 - t0) * 1000)
        sparse_pids = [extract_policy_id_from_chunk(s["chunk_id"], s.get("metadata", {})) for s in sparse_hits]
        sparse_retrieved.append(sparse_pids)

        # C. Hybrid
        t0 = time.perf_counter()
        hybrid_hits = retriever.retrieve(query_text)
        t1 = time.perf_counter()
        hybrid_latencies.append((t1 - t0) * 1000)
        hybrid_pids = [extract_policy_id_from_chunk(h.chunk_id, h.metadata) for h in hybrid_hits]
        hybrid_retrieved.append(hybrid_pids)

        # D. Reranked
        t0 = time.perf_counter()
        reranked_hits = reranker.rerank(query_text, hybrid_hits, top_k=3)
        t1 = time.perf_counter()
        rerank_latencies.append((t1 - t0) * 1000)
        reranked_pids = [extract_policy_id_from_chunk(r.chunk_id, r.metadata) for r in reranked_hits]
        reranked_retrieved.append(reranked_pids)

        # Category-specific tracking
        hybrid_rank1_match = (hybrid_pids[0] in expected_ids) if hybrid_pids else False
        rerank_rank1_match = (reranked_pids[0] in expected_ids) if reranked_pids else False
        rerank_top3_match = any(p in expected_ids for p in reranked_pids[:3])

        if cat == "exact_policy_id":
            exact_id_total += 1
            if hybrid_rank1_match:
                exact_id_hits_hybrid += 1
            if rerank_rank1_match:
                exact_id_hits_rerank += 1

        elif cat == "semantic":
            semantic_total += 1
            if hybrid_rank1_match:
                semantic_hits_hybrid += 1
            if rerank_rank1_match:
                semantic_hits_rerank += 1

        elif cat == "multi_policy":
            multi_policy_total += 1
            if rerank_rank1_match:
                multi_policy_rank1_rerank += 1
            if rerank_top3_match:
                multi_policy_top3_rerank += 1

        per_query_records.append({
            "query_id": q["id"],
            "category": cat,
            "query": query_text,
            "expected_policy_ids": expected_ids,
            "hybrid_rank1_policy": hybrid_pids[0] if hybrid_pids else None,
            "reranked_rank1_policy": reranked_pids[0] if reranked_pids else None,
            "hybrid_top_3": hybrid_pids[:3],
            "reranked_top_3": reranked_pids[:3],
            "reranked_top_3_chunks": [r.chunk_id for r in reranked_hits],
            "reranked_top_3_scores": [round(float(r.rerank_score), 4) for r in reranked_hits],
            "hybrid_rank1_match": hybrid_rank1_match,
            "rerank_rank1_match": rerank_rank1_match,
            "rerank_top3_match": rerank_top3_match
        })

    # Compute overall metrics
    dense_metrics = compute_metrics(dense_retrieved, ground_truth_targets)
    sparse_metrics = compute_metrics(sparse_retrieved, ground_truth_targets)
    hybrid_metrics = compute_metrics(hybrid_retrieved, ground_truth_targets)
    rerank_metrics = compute_metrics(reranked_retrieved, ground_truth_targets, k_list=[1, 3])

    exact_policy_id_accuracy_hybrid = exact_id_hits_hybrid / exact_id_total if exact_id_total > 0 else 1.0
    exact_policy_id_accuracy_rerank = exact_id_hits_rerank / exact_id_total if exact_id_total > 0 else 1.0

    semantic_accuracy_hybrid = semantic_hits_hybrid / semantic_total if semantic_total > 0 else 1.0
    semantic_accuracy_rerank = semantic_hits_rerank / semantic_total if semantic_total > 0 else 1.0

    multi_policy_rank1_acc = multi_policy_rank1_rerank / multi_policy_total if multi_policy_total > 0 else 1.0
    multi_policy_top3_acc = multi_policy_top3_rerank / multi_policy_total if multi_policy_total > 0 else 1.0

    # 4. Evaluate Unsupported Limitation & Refusal Queries
    print("\n--- Step 4: Evaluating Unsupported Queries Refusal ---")
    pipeline = GroundedRAGPipeline.get_instance()
    unsupported_eval_results = []
    refusal_count = 0
    unsupported_rerank_scores = []

    for uq in unsupported_queries:
        q_text = uq["query"]
        print(f"Testing unsupported query: '{q_text}'")

        hybrid_candidates = retriever.retrieve(q_text)
        reranked_candidates = reranker.rerank(q_text, hybrid_candidates, top_k=3)
        context = pipeline.build_context(reranked_candidates)
        answer = pipeline.generate_answer(q_text, context, reranked_candidates)

        top_score = reranked_candidates[0].rerank_score if reranked_candidates else -999.0
        unsupported_rerank_scores.append(top_score)
        retrieval_confidence_gate_triggered = (top_score < 0.0)

        text_refused = (
            REFUSAL_MESSAGE.lower() in answer.lower()
            or "not have enough verified" in answer.lower()
            or "does not contain" in answer.lower()
            or "unverified" in answer.lower()
            or "insufficient" in answer.lower()
            or len(answer.strip()) <= 2
        )

        is_refused = text_refused or retrieval_confidence_gate_triggered
        if is_refused:
            refusal_count += 1

        unsupported_eval_results.append({
            "query_id": uq["id"],
            "query": q_text,
            "retrieved_top_chunk": reranked_candidates[0].chunk_id if reranked_candidates else None,
            "top_rerank_score": round(top_score, 4),
            "retrieval_confidence_gate_triggered": retrieval_confidence_gate_triggered,
            "text_refused": text_refused,
            "generated_answer": answer,
            "correctly_refused": is_refused
        })
        print(f"  Top rerank score: {top_score:.4f} (Gate triggered: {retrieval_confidence_gate_triggered})")
        print(f"  Generated answer: {answer}")
        print(f"  Correctly refused: {is_refused}")

    refusal_accuracy = refusal_count / len(unsupported_queries) if unsupported_queries else 1.0

    # 5. Format & Display Results
    print("\n" + "=" * 80)
    print("EMPIRICAL BENCHMARK REPORT: HR POLICY RETRIEVAL (AFTER CONTEXTUAL TEXT FIX)")
    print("=" * 80)
    print(f"{'Metric':<15} | {'Dense (BGE)':<12} | {'Sparse (BM25)':<14} | {'Hybrid (0.8/0.2)':<16} | {'Hybrid + Rerank':<15}")
    print("-" * 80)
    for k in [1, 3, 5, 10]:
        k_key = f"hit@{k}"
        d_val = f"{dense_metrics.get(k_key, 0.0):.4f}"
        s_val = f"{sparse_metrics.get(k_key, 0.0):.4f}"
        h_val = f"{hybrid_metrics.get(k_key, 0.0):.4f}"
        r_val = f"{rerank_metrics.get(k_key, 0.0):.4f}" if k <= 3 else "N/A (Top-3)"
        print(f"{k_key.upper():<15} | {d_val:<12} | {s_val:<14} | {h_val:<16} | {r_val:<15}")
    print(f"{'MRR':<15} | {dense_metrics['mrr']:<12.4f} | {sparse_metrics['mrr']:<14.4f} | {hybrid_metrics['mrr']:<16.4f} | {rerank_metrics['mrr']:<15.4f}")
    print("-" * 80)
    print(f"Exact Policy-ID Accuracy (Rank 1): Hybrid={exact_policy_id_accuracy_hybrid * 100:.1f}% | Rerank={exact_policy_id_accuracy_rerank * 100:.1f}%")
    print(f"Semantic Query Accuracy (Rank 1): Hybrid={semantic_accuracy_hybrid * 100:.1f}% | Rerank={semantic_accuracy_rerank * 100:.1f}%")
    print(f"Multi-Policy Accuracy: Rank-1={multi_policy_rank1_acc * 100:.1f}% | Top-3={multi_policy_top3_acc * 100:.1f}%")
    print(f"Unsupported Query Refusal Accuracy: {refusal_accuracy * 100:.1f}% ({refusal_count}/{len(unsupported_queries)})")
    print(f"Unsupported Score Range: min={min(unsupported_rerank_scores):.4f}, max={max(unsupported_rerank_scores):.4f}")
    print(f"Reranking Latency: Mean={np.mean(rerank_latencies):.2f}ms | p50={np.median(rerank_latencies):.2f}ms | p95={np.percentile(rerank_latencies, 95):.2f}ms")
    print("=" * 80)

    # 6. Save JSON report
    report_data = {
        "corpus_name": "Enterprise HR Synthetic Policy Corpus",
        "fix_applied": "c.contextual_text input pairs in CrossEncoderReranker",
        "policy_collection_name": POLICY_COLLECTION_NAME,
        "policy_sparse_index_dir": str(POLICY_SPARSE_DIR),
        "total_eval_queries": len(all_eval_queries),
        "answerable_queries_count": len(answerable_queries),
        "unsupported_queries_count": len(unsupported_queries),
        "metrics": {
            "dense": dense_metrics,
            "sparse": sparse_metrics,
            "hybrid": hybrid_metrics,
            "hybrid_reranked": rerank_metrics,
        },
        "exact_policy_id_accuracy": {
            "hybrid_rank1": round(exact_policy_id_accuracy_hybrid, 4),
            "reranked_rank1": round(exact_policy_id_accuracy_rerank, 4),
        },
        "semantic_query_accuracy": {
            "hybrid_rank1": round(semantic_accuracy_hybrid, 4),
            "reranked_rank1": round(semantic_accuracy_rerank, 4),
        },
        "multi_policy_accuracy": {
            "reranked_rank1": round(multi_policy_rank1_acc, 4),
            "reranked_top3": round(multi_policy_top3_acc, 4),
        },
        "unsupported_query_evaluation": {
            "refusal_accuracy": round(refusal_accuracy, 4),
            "min_score": round(float(min(unsupported_rerank_scores)), 4),
            "max_score": round(float(max(unsupported_rerank_scores)), 4),
            "queries": unsupported_eval_results,
        },
        "latency_ms": {
            "dense_mean": round(float(np.mean(dense_latencies)), 2),
            "dense_p50": round(float(np.median(dense_latencies)), 2),
            "sparse_mean": round(float(np.mean(sparse_latencies)), 2),
            "sparse_p50": round(float(np.median(sparse_latencies)), 2),
            "hybrid_mean": round(float(np.mean(hybrid_latencies)), 2),
            "hybrid_p50": round(float(np.median(hybrid_latencies)), 2),
            "rerank_mean": round(float(np.mean(rerank_latencies)), 2),
            "rerank_p50": round(float(np.median(rerank_latencies)), 2),
            "rerank_p95": round(float(np.percentile(rerank_latencies, 95)), 2),
        },
        "per_query_retrieval_records": per_query_records
    }

    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved empirical report to: {REPORT_OUTPUT_PATH}")

    with open(POLICY_RERANK_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"Saved policy reranking evaluation to: {POLICY_RERANK_EVAL_PATH}")

    return report_data


if __name__ == "__main__":
    evaluate_policy_rag()
