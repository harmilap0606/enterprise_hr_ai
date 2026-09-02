import sys
sys.stdout.reconfigure(encoding='utf-8')
from app.rag.qa_chain import _get_model
from app.rag.retriever import retrieve

tokenizer, model = _get_model()

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
    context = chunks[0]["content"] if chunks else ""
    prompt = (
        f"Answer the question based only on the provided context. If the context does not provide the answer, say \"I don't have information about that in the platform's knowledge base\".\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {q}\n"
        f"Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"QUERY: {q}")
    print(f"  TOP CHUNK: {chunks[0]['source']} (score: {chunks[0]['score']})")
    print(f"  T5 OUTPUT: {ans}")
    print("-" * 60)
