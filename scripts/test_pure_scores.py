import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
from app.rag.retriever import get_collection

col = get_collection()

queries = [
    "What does a Research Scientist do?",
    "Why is the Manager role's O*NET mapping unreliable?",
    "What is the production model's decision threshold and why was it chosen?",
    "What is the company's parental leave policy?",
    "What is the capital of France?",
    "What is our company's dress code policy?",
    "Who won the 2024 World Series?"
]

STOPWORDS = {
    "what", "why", "how", "when", "where", "who", "which", "is", "are", "was",
    "were", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "does", "do", "did", "about", "that", "this", "it", "its", "our", "their",
    "with", "from", "by"
}

for q in queries:
    res = col.query(query_texts=[q], n_results=5)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    print(f"=== QUERY: {q} ===")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        cos_sim = max(0.0, 1.0 - dist)
        
        # Content tokens in query
        q_tokens = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", q.lower()) if w not in STOPWORDS]
        # Check presence in retrieved doc
        matched = [w for w in q_tokens if w in doc.lower()]
        grounding_ratio = len(matched) / len(q_tokens) if q_tokens else 0.0
        
        # Composite score: cosine similarity + keyword overlap
        composite = 0.65 * cos_sim + 0.35 * grounding_ratio
        
        print(f"  [{i+1}] cos_sim: {cos_sim:.4f} | ground_ratio: {grounding_ratio:.2f} ({matched}/{q_tokens}) | composite: {composite:.4f}")
        print(f"      source: {meta.get('source')} (Section: {meta.get('section', '')})")
    print()
