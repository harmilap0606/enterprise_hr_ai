from app.rag.retriever import get_collection

col = get_collection()
q = "Why is the Manager role's O*NET mapping unreliable?"
res = col.query(
    query_texts=[q],
    where={"doc_type": "markdown_doc"},
    n_results=3
)
print("Querying markdown_doc for Q2:")
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"  [dist={dist:.4f}] {meta['source']} | {meta['section']}")
    print(f"    {doc.splitlines()[:2]}")
