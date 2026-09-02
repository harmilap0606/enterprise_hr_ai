import sys
import json
sys.stdout.reconfigure(encoding='utf-8')

from app.rag.qa_chain import answer_question, RETRIEVAL_SCORE_THRESHOLD

test_queries = [
    # 5 Original Test Questions
    ("1 (In-Domain)", "What does a Research Scientist do?"),
    ("2 (In-Domain)", "Why is the Manager role's O*NET mapping unreliable?"),
    ("3 (In-Domain)", "What is the production model's decision threshold and why was it chosen?"),
    ("4 (Out-of-Domain)", "What is the company's parental leave policy?"),
    ("5 (Out-of-Domain)", "What is the capital of France?"),
    # 2 New Unplanned Test Questions
    ("6 (New Out-of-Domain)", "What is our company's dress code policy?"),
    ("7 (New Out-of-Domain)", "Who won the 2024 World Series?")
]

print(f"================================================================================")
print(f"RAG GROUNDING GATE CONFIGURATION:")
print(f"  Gate Mechanism: Pure Numerical Score-Threshold Gate")
print(f"  RETRIEVAL_SCORE_THRESHOLD = {RETRIEVAL_SCORE_THRESHOLD}")
print(f"  Rule: If top_chunk_score < {RETRIEVAL_SCORE_THRESHOLD} -> REFUSE without calling LLM")
print(f"  Hardcoded Keyword Lists: NONE (Completely removed)")
print(f"================================================================================\n")

for label, query in test_queries:
    res = answer_question(query)
    answer = res["answer"]
    sources = res["sources"]
    top_score = sources[0]["score"] if sources else 0.0
    top_source = sources[0]["source"] if sources else "None"
    
    status = "GROUNDED ANSWER" if answer != "I don't have information about that in the platform's knowledge base" else "REFUSED"
    gate_trigger = f"score {top_score:.4f} >= {RETRIEVAL_SCORE_THRESHOLD} (PASSED GATE)" if top_score >= RETRIEVAL_SCORE_THRESHOLD else f"score {top_score:.4f} < {RETRIEVAL_SCORE_THRESHOLD} (TRIGGERED REFUSAL)"

    print(f"--- TEST QUERY {label} ---")
    print(f"Question: {query}")
    print(f"Status: {status}")
    print(f"Gate Evaluation: {gate_trigger}")
    print(f"Answer:\n  \"{answer}\"")
    print(f"Top Retrieved Sources ({len(sources)} returned):")
    for i, s in enumerate(sources, 1):
        print(f"  [{i}] Score: {s['score']:.4f} | Source: {s['source']}")
        print(f"      Excerpt: \"{s['excerpt'][:110]}...\"")
    print()
