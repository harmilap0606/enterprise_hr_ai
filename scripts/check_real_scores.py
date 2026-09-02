import sys
sys.stdout.reconfigure(encoding='utf-8')

from app.rag.retriever import retrieve

queries = [
    "What does a Research Scientist do?",
    "Why is the Manager role's O*NET mapping unreliable?",
    "What is the production model's decision threshold and why was it chosen?",
    "What is the company's parental leave policy?",
    "What is the capital of France?",
    "What is our company's dress code policy?",
    "Who won the 2024 World Series?"
]

for q in queries:
    chunks = retrieve(q, k=3)
    scores = [round(c['score'], 4) for c in chunks]
    top = chunks[0] if chunks else {}
    print(f"QUERY: {q}")
    print(f"  Top Score: {top.get('score'):.4f} | Source: {top.get('source')}")
    print(f"  All Scores: {scores}")
    print(f"  Excerpt: {top.get('excerpt')[:120]}...")
    print("-" * 60)
