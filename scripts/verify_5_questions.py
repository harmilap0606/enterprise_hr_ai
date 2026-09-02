import sys
import codecs

# Ensure stdout handles unicode without crashing Windows cp1252
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from app.rag.qa_chain import answer_question

questions = [
    "What does a Research Scientist do?",
    "Why is the Manager role's O*NET mapping unreliable?",
    "What is the production model's decision threshold and why was it chosen?",
    "What is the company's parental leave policy?",
    "What is the capital of France?"
]

for idx, q in enumerate(questions, 1):
    print("=" * 80)
    print(f"TEST QUESTION {idx}: {q}")
    result = answer_question(q)
    print(f"ANSWER:\n{result['answer']}")
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  - [{s['score']:.4f}] {s['source']}")
        clean_excerpt = s['excerpt'][:110].replace("\n", " ")
        print(f"    Excerpt: {clean_excerpt}...")
