"""
app/rag/metrics.py
==================
Deterministic retrieval evaluation metrics for RAG benchmark:
- Hit@1, Hit@3, Hit@5, Hit@10
- Mean Reciprocal Rank (MRR)
"""

from typing import List, Dict, Any, Set
import numpy as np


def compute_metrics(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    chunk_ids: List[str],
    eval_queries: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Computes Hit@1, Hit@3, Hit@5, Hit@10, and MRR.
    Uses cosine similarity (matrix dot product since embeddings are L2 normalized).
    """
    # (n_queries, n_chunks) similarity matrix
    similarity_matrix = np.dot(query_embeddings, corpus_embeddings.T)

    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    hit_10 = 0
    reciprocal_ranks = []

    for i, q in enumerate(eval_queries):
        expected: Set[str] = set(q["expected_chunk_ids"])
        sims = similarity_matrix[i]

        # Sort chunks descending by similarity
        ranked_indices = np.argsort(-sims)
        ranked_chunk_ids = [chunk_ids[idx] for idx in ranked_indices]

        # Check top-k
        top_1 = set(ranked_chunk_ids[:1])
        top_3 = set(ranked_chunk_ids[:3])
        top_5 = set(ranked_chunk_ids[:5])
        top_10 = set(ranked_chunk_ids[:10])

        if any(c in top_1 for c in expected):
            hit_1 += 1
        if any(c in top_3 for c in expected):
            hit_3 += 1
        if any(c in top_5 for c in expected):
            hit_5 += 1
        if any(c in top_10 for c in expected):
            hit_10 += 1

        # MRR calculation (find rank of first relevant chunk)
        found_rank = None
        for r_idx, c_id in enumerate(ranked_chunk_ids, 1):
            if c_id in expected:
                found_rank = r_idx
                break

        if found_rank is not None:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    total_q = len(eval_queries)
    return {
        "Hit@1": round(hit_1 / total_q, 4),
        "Hit@3": round(hit_3 / total_q, 4),
        "Hit@5": round(hit_5 / total_q, 4),
        "Hit@10": round(hit_10 / total_q, 4),
        "MRR": round(float(np.mean(reciprocal_ranks)), 4)
    }
