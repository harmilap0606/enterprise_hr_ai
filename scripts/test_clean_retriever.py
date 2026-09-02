import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
from app.rag.retriever import get_collection, _tokenize

col = get_collection()
total_docs = col.count()

queries = [
    "What does a Research Scientist do?",
    "Why is the Manager role's O*NET mapping unreliable?",
    "What is the production model's decision threshold and why was it chosen?",
    "What is the company's parental leave policy?",
    "What is the capital of France?",
    "What is our company's dress code policy?",
    "Who won the 2024 World Series?"
]

for query in queries:
    raw_docs, raw_metas, raw_dists = [], [], []
    
    # 1. Query markdown docs
    res_docs = col.query(query_texts=[query], where={"doc_type": "markdown_doc"}, n_results=5)
    if res_docs and res_docs["documents"]:
        raw_docs.extend(res_docs["documents"][0])
        raw_metas.extend(res_docs["metadatas"][0])
        raw_dists.extend(res_docs["distances"][0])
        
    # 2. Query occupation docs
    res_occ = col.query(query_texts=[query], where={"doc_type": "occupation"}, n_results=5)
    if res_occ and res_occ["documents"]:
        raw_docs.extend(res_occ["documents"][0])
        raw_metas.extend(res_occ["metadatas"][0])
        raw_dists.extend(res_occ["distances"][0])

    q_tokens = _tokenize(query)
    scored = []
    for doc, meta, dist in zip(raw_docs, raw_metas, raw_dists):
        cos_sim = max(0.0, 1.0 - dist)
        doc_tokens = _tokenize(doc)
        kw_overlap = len(q_tokens.intersection(doc_tokens)) / max(1, len(q_tokens))
        
        # Pure general hybrid score: 70% dense cosine similarity + 30% lexical keyword overlap
        score = 0.70 * cos_sim + 0.30 * kw_overlap
        scored.append((score, cos_sim, kw_overlap, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[0]
    print(f"QUERY: {query}")
    print(f"  Top Score: {top[0]:.4f} (cos: {top[1]:.4f}, kw: {top[2]:.4f})")
    print(f"  Source: {top[3].get('source')} (Section: {top[3].get('section', '')})")
    print("-" * 60)
