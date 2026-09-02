"""
scripts/diagnose_policy_reranking.py
====================================
Detailed diagnostic script to determine precisely why Cross-Encoder reranking
is altering or degrading ranking on the synthetic HR policy corpus.

Analyzes all 22 policy queries:
1. 18 answerable queries across 4 categories (Exact ID, Terminology, Semantic, Multi-Policy)
2. 4 unsupported refusal queries (Category 5)

Extracts:
- Exact text passed to Cross-Encoder (candidate.text vs candidate.contextual_text)
- Full candidate traces (Top-15 hybrid pool before reranking, Top-3 after reranking)
- Cross-encoder logits, hybrid scores, dense scores, sparse scores
- Presence of policy ID in candidate.text vs candidate.title vs candidate.contextual_text
- Cross-policy citation presence in retrieved chunks
- Movement direction: Improved, Neutral, Degraded
- Detailed failure analysis for every query

Outputs: reports/rag/policy_reranking_diagnostic.json
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

from app.rag.embeddings import BGEEmbedder
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.retrieval.reranker import CrossEncoderReranker
from app.rag.retrieval.schemas import RetrievalConfig
from app.utils.config import BASE_DIR

sys.stdout.reconfigure(encoding="utf-8")

EVAL_FIXTURE_PATH = BASE_DIR / "tests" / "fixtures" / "hr_policy_eval_queries.json"
REPORT_OUTPUT_PATH = BASE_DIR / "reports" / "rag" / "policy_reranking_diagnostic.json"

POLICY_COLLECTION_NAME = "enterprise_hr_policies_bge"
POLICY_SPARSE_DIR = BASE_DIR / "data" / "rag" / "policy_sparse_index"


def extract_policy_id(chunk_id: str, metadata: Dict[str, Any]) -> str:
    if metadata and metadata.get("policy_id"):
        return metadata["policy_id"].upper()
    import re
    m = re.search(r"POL[_-][A-Z]+[_-]\d+", chunk_id, re.I)
    if m:
        return m.group(0).upper().replace("_", "-")
    return chunk_id.upper()


def run_diagnostic():
    print("=" * 80)
    print("CROSS-ENCODER POLICY RERANKING DIAGNOSTIC TRACE")
    print("=" * 80)

    with open(EVAL_FIXTURE_PATH, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    embedder = BGEEmbedder()
    config = RetrievalConfig(dense_top_k=20, sparse_top_k=20, final_top_k=15, dense_weight=0.8, sparse_weight=0.2)
    retriever = HybridRetriever(
        config=config,
        embedder=embedder,
        collection_name=POLICY_COLLECTION_NAME,
        sparse_dir=POLICY_SPARSE_DIR
    )
    reranker = CrossEncoderReranker()

    query_diagnostics = []
    
    improved_count = 0
    neutral_count = 0
    degraded_count = 0

    category_stats = {
        "exact_policy_id": {"improved": 0, "neutral": 0, "degraded": 0, "total": 0},
        "policy_terminology": {"improved": 0, "neutral": 0, "degraded": 0, "total": 0},
        "semantic": {"improved": 0, "neutral": 0, "degraded": 0, "total": 0},
        "multi_policy": {"improved": 0, "neutral": 0, "degraded": 0, "total": 0},
        "limitation_refusal": {"improved": 0, "neutral": 0, "degraded": 0, "total": 0},
    }

    for item in eval_queries:
        qid = item["id"]
        cat = item["category"]
        query = item["query"]
        expected_pids = [p.upper() for p in item["expected_policy_ids"]]
        is_answerable = item["is_answerable"]

        category_stats[cat]["total"] += 1

        # Retrieve Top 15 hybrid candidates
        hybrid_candidates = retriever.retrieve(query)
        
        # Prepare pairs as reranker does: [[query, c.text] for c in hybrid_candidates]
        input_pairs = [[query, c.text] for c in hybrid_candidates]
        raw_logits = reranker.predict(input_pairs)

        # Build candidate full trace
        candidate_traces = []
        for rank_h, (cand, logit) in enumerate(zip(hybrid_candidates, raw_logits), 1):
            pid = extract_policy_id(cand.chunk_id, cand.metadata)
            pid_in_text = pid in cand.text.upper()
            pid_in_title = pid in cand.title.upper()
            
            # Check cross references to other policies
            import re
            cross_refs = list(set(re.findall(r"POL-[A-Z]+-\d+", cand.text, re.I)))

            candidate_traces.append({
                "hybrid_rank": rank_h,
                "chunk_id": cand.chunk_id,
                "policy_id": pid,
                "section": cand.section,
                "title": cand.title,
                "hybrid_score": round(float(cand.hybrid_score), 4),
                "dense_score": round(float(cand.dense_score), 4),
                "sparse_score": round(float(cand.sparse_score), 4),
                "cross_encoder_logit": round(float(logit), 4),
                "pid_in_text": pid_in_text,
                "pid_in_title": pid_in_title,
                "cross_references_in_text": cross_refs,
                "raw_text_snippet": cand.text[:200] + "..." if len(cand.text) > 200 else cand.text,
                "contextual_text_snippet": cand.contextual_text[:200] + "..." if len(cand.contextual_text) > 200 else cand.contextual_text,
            })

        # Sort descending by cross_encoder_logit
        reranked_traces = sorted(candidate_traces, key=lambda x: (-x["cross_encoder_logit"], x["chunk_id"]))
        for rank_r, ctrace in enumerate(reranked_traces, 1):
            ctrace["reranked_rank"] = rank_r

        # Evaluate rank before vs rank after
        best_rank_before = None
        best_rank_after = None

        if is_answerable and expected_pids:
            # Find best rank of an expected policy before and after
            for c in candidate_traces:
                if c["policy_id"] in expected_pids:
                    if best_rank_before is None or c["hybrid_rank"] < best_rank_before:
                        best_rank_before = c["hybrid_rank"]

            for c in reranked_traces:
                if c["policy_id"] in expected_pids:
                    if best_rank_after is None or c["reranked_rank"] < best_rank_after:
                        best_rank_after = c["reranked_rank"]

            # Movement classification
            if best_rank_after < best_rank_before:
                movement = "IMPROVED"
                improved_count += 1
                category_stats[cat]["improved"] += 1
            elif best_rank_after > best_rank_before:
                movement = "DEGRADED"
                degraded_count += 1
                category_stats[cat]["degraded"] += 1
            else:
                movement = "NEUTRAL"
                neutral_count += 1
                category_stats[cat]["neutral"] += 1
        else:
            # Unsupported query
            movement = "UNSUPPORTED_QUERY"
            top_logit = reranked_traces[0]["cross_encoder_logit"] if reranked_traces else -999.0
            if top_logit < 0.0:
                movement = "CORRECT_REFUSAL_LOGIT"
                improved_count += 1
                category_stats[cat]["improved"] += 1
            else:
                movement = "FALSE_POSITIVE_LOGIT"
                degraded_count += 1
                category_stats[cat]["degraded"] += 1

        # Failure diagnosis
        failure_reason = None
        if movement == "DEGRADED":
            rank1_after = reranked_traces[0]
            failure_reason = {
                "top_ranked_policy_after": rank1_after["policy_id"],
                "top_ranked_section_after": rank1_after["section"],
                "top_ranked_logit": rank1_after["cross_encoder_logit"],
                "expected_policies": expected_pids,
                "best_expected_rank_after": best_rank_after,
                "analysis": (
                    f"Chunk '{rank1_after['chunk_id']}' from '{rank1_after['policy_id']}' scored {rank1_after['cross_encoder_logit']} "
                    f"outranking expected policy '{expected_pids}' (best rank {best_rank_after}). "
                    f"Cross-references present: {rank1_after['cross_references_in_text']}."
                )
            }

        diag_record = {
            "query_id": qid,
            "category": cat,
            "query": query,
            "is_answerable": is_answerable,
            "expected_policy_ids": expected_pids,
            "movement": movement,
            "best_expected_rank_before": best_rank_before,
            "best_expected_rank_after": best_rank_after,
            "hybrid_rank1_policy": candidate_traces[0]["policy_id"] if candidate_traces else None,
            "reranked_rank1_policy": reranked_traces[0]["policy_id"] if reranked_traces else None,
            "reranked_top3_policies": [r["policy_id"] for r in reranked_traces[:3]],
            "failure_reason": failure_reason,
            "hybrid_top15_candidates": candidate_traces,
            "reranked_top15_candidates": reranked_traces,
        }
        query_diagnostics.append(diag_record)

    summary_data = {
        "total_queries_evaluated": len(eval_queries),
        "answerable_queries_evaluated": len([q for q in eval_queries if q["is_answerable"]]),
        "unsupported_queries_evaluated": len([q for q in eval_queries if not q["is_answerable"]]),
        "movement_summary_overall": {
            "improved": improved_count,
            "neutral": neutral_count,
            "degraded": degraded_count,
        },
        "movement_by_category": category_stats,
        "query_diagnostics": query_diagnostics
    }

    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\nDiagnostic analysis completed. Report saved to: {REPORT_OUTPUT_PATH}")
    print(f"Overall Movement: Improved={improved_count} | Neutral={neutral_count} | Degraded={degraded_count}")
    for cat, stats in category_stats.items():
        print(f"  Category '{cat}': Improved={stats['improved']}, Neutral={stats['neutral']}, Degraded={stats['degraded']} (Total: {stats['total']})")
    
    return summary_data


if __name__ == "__main__":
    run_diagnostic()
