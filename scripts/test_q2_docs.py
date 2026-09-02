import chromadb
from pathlib import Path
import re

client = chromadb.Client()
col_docs = client.create_collection("docs_only", metadata={"hnsw:space": "cosine"})

def chunk_markdown_file(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    sections = re.split(r"\n(?=##\s+)", text)
    chunks = []
    for s in sections:
        s = s.strip()
        if not s:
            continue
        lines = s.splitlines()
        first_line = lines[0].strip()
        sec_title = first_line.replace("## ", "").strip() if first_line.startswith("## ") else "Overview"
        chunks.append({
            "source": str(file_path).replace("\\", "/"),
            "section": sec_title,
            "content": s
        })
    return chunks

docs = []
for doc_path in [Path("docs/model_card.md"), Path("docs/data_relationships.md")]:
    for i, c in enumerate(chunk_markdown_file(doc_path)):
        text = f"Document: {c['source']}\nSection: {c['section']}\nContent:\n{c['content']}"
        col_docs.add(
            documents=[text],
            metadatas=[{"source": c["source"], "section": c["section"]}],
            ids=[f"{doc_path.stem}_{i}"]
        )

q = "Why is the Manager role's O*NET mapping unreliable?"
res = col_docs.query(query_texts=[q], n_results=3)
print("Top doc results for Q2:")
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"  [dist={dist:.4f}] source={meta['source']} | section={meta['section']}")
