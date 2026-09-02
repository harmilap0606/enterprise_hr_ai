from app.rag.retriever import retrieve

questions = [
    "What does a Research Scientist do?",
    "Why is the Manager role's O*NET mapping unreliable?",
    "What is the production model's decision threshold and why was it chosen?",
    "What is the company's parental leave policy?",
    "What is the capital of France?"
]

for q in questions:
    print("=" * 70)
    print("Q:", q)
    chunks = retrieve(q, k=3)
    for c in chunks:
        print(f"  [{c['score']:.4f}] {c['source']}")
        print(f"    Excerpt: {c['excerpt'][:120]}...")
