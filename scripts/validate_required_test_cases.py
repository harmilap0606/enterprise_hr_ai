"""
scripts/validate_required_test_cases.py
Executes the 5 required test cases:
A. 'What does O*NET code 19-1042.00 represent?'
B. 'What occupation is associated with the Scientist role?'
C. 'What is the purpose of jobrole_onet_mapping.csv and occupation_master.csv, and how are they related?'
D. 'What factors influence employee retention and job satisfaction?'
E. 'Where is StandardScaler used in the model training data?'

For each query records:
* query
* Dense Top-20
* Sparse Top-20
* candidate union
* hybrid Top-15
* final Cross-Encoder Top-3
* scores
* source/chunk IDs
* generated answer
"""

import json
from pathlib import Path
from app.rag.pipeline import GroundedRAGPipeline
from app.utils.config import BASE_DIR

TEST_CASES = [
    {
        "id": "A",
        "query": "What does O*NET code 19-1042.00 represent?",
        "expected_target": "occ_19-1042.00_c01",
        "type": "Exact Structured Identifier (O*NET Code)"
    },
    {
        "id": "B",
        "query": "What occupation is associated with the Scientist role?",
        "expected_target": "map_research_scientist_c01",
        "type": "Role Mapping Semantic Query"
    },
    {
        "id": "C",
        "query": "What is the purpose of jobrole_onet_mapping.csv and occupation_master.csv, and how are they related?",
        "expected_target": "data_relationships_open_issues_08_c01",
        "type": "Documentation & Schema Relationship Query"
    },
    {
        "id": "D",
        "query": "What factors influence employee retention and job satisfaction?",
        "expected_target": None,
        "type": "Normal Semantic NL Query"
    },
    {
        "id": "E",
        "query": "Where is StandardScaler used in the model training data?",
        "expected_target": "model_card_training_data_04_c01",
        "type": "Exact Lexical Term (Preprocessing / Technical Keyword)"
    }
]

def run_validation():
    pipeline = GroundedRAGPipeline.get_instance()
    retriever = pipeline.retriever
    reranker = pipeline.reranker
    
    results = []
    
    for tc in TEST_CASES:
        q = tc["query"]
        print(f"\n{'='*80}\nRUNNING TEST CASE {tc['id']}: {q}\n{'='*80}")
        
        # 1. Dense Top-20
        dense_hits = retriever.retrieve_dense(q, top_k=20)
        dense_list = [{"chunk_id": d["chunk_id"], "dense_score": round(d["dense_score"], 4)} for d in dense_hits]
        
        # 2. Sparse Top-20
        sparse_hits = retriever.retrieve_sparse(q, top_k=20)
        sparse_list = [{"chunk_id": s["chunk_id"], "sparse_score": round(s["sparse_score"], 4)} for s in sparse_hits]
        
        # 3. Candidate Union
        union_cids = list(set([d["chunk_id"] for d in dense_hits] + [s["chunk_id"] for s in sparse_hits]))
        
        # 4. Hybrid Top-15
        hybrid_hits = retriever.retrieve(q)
        hybrid_list = [
            {
                "rank": h.rank,
                "chunk_id": h.chunk_id,
                "hybrid_score": h.hybrid_score,
                "dense_score": h.dense_score,
                "sparse_score": h.sparse_score,
                "is_exact_match": h.is_exact_match,
                "source": h.source,
                "section": h.section
            }
            for h in hybrid_hits
        ]
        
        # 5. Cross-Encoder Top-3
        reranked_hits = reranker.rerank(q, hybrid_hits, top_k=3)
        reranked_list = [
            {
                "rerank_rank": r.rerank_rank,
                "chunk_id": r.chunk_id,
                "rerank_score": round(r.rerank_score, 4),
                "original_hybrid_rank": r.original_hybrid_rank,
                "source": r.source,
                "section": r.section
            }
            for r in reranked_hits
        ]
        
        # 6. Generated Answer
        context = pipeline.build_context(reranked_hits)
        generated_answer = pipeline.generate_answer(q, context, reranked_hits)
        
        print(f"Dense Top 3: {[d['chunk_id'] for d in dense_list[:3]]}")
        print(f"Sparse Top 3: {[s['chunk_id'] for s in sparse_list[:3]]}")
        print(f"Union Count: {len(union_cids)}")
        print(f"Hybrid Top 3: {[(h['chunk_id'], h['hybrid_score'], h['is_exact_match']) for h in hybrid_list[:3]]}")
        print(f"Cross-Encoder Top 3: {[(r['chunk_id'], r['rerank_score']) for r in reranked_list]}")
        print(f"Generated Answer: {generated_answer}")
        
        record = {
            "test_id": tc["id"],
            "type": tc["type"],
            "query": q,
            "expected_target": tc["expected_target"],
            "dense_top_20": dense_list,
            "sparse_top_20": sparse_list,
            "candidate_union_count": len(union_cids),
            "hybrid_top_15": hybrid_list,
            "cross_encoder_top_3": reranked_list,
            "target_in_hybrid_top_15": (tc["expected_target"] in [h["chunk_id"] for h in hybrid_list]) if tc["expected_target"] else None,
            "target_in_rerank_top_3": (tc["expected_target"] in [r["chunk_id"] for r in reranked_list]) if tc["expected_target"] else None,
            "generated_answer": generated_answer
        }
        results.append(record)
        
    out_path = BASE_DIR / "reports" / "rag" / "required_test_cases_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll 5 test cases successfully recorded to: {out_path}")

if __name__ == "__main__":
    run_validation()
