import re
from pathlib import Path
import pandas as pd
import chromadb
import numpy as np

client = chromadb.Client()
collection = client.create_collection("corpus", metadata={"hnsw:space": "cosine"})

# 1. Occupations
df_occ = pd.read_csv("data/processed/occupation_master.csv")
docs, metas, ids = [], [], []
for idx, row in df_occ.iterrows():
    code = str(row["O*NET-SOC Code"])
    title = str(row["Title"])
    desc = str(row["Description"])
    # Clean text representation
    text = f"O*NET Occupation Title: {title}\nO*NET-SOC Code: {code}\nDescription: {desc}"
    docs.append(text)
    metas.append({"source": "data/processed/occupation_master.csv", "section": title, "code": code})
    ids.append(f"occ_{code}")

# 2. Markdown docs
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

for doc_path in [Path("docs/model_card.md"), Path("docs/data_relationships.md")]:
    for i, c in enumerate(chunk_markdown_file(doc_path)):
        text = f"Source: {c['source']}\nSection: {c['section']}\nContent:\n{c['content']}"
        docs.append(text)
        metas.append({"source": c["source"], "section": c["section"], "code": ""})
        ids.append(f"{doc_path.stem}_{i}")

# Add in batches
batch_size = 200
for i in range(0, len(docs), batch_size):
    collection.add(
        documents=docs[i:i+batch_size],
        metadatas=metas[i:i+batch_size],
        ids=ids[i:i+batch_size]
    )

def tokenize(text):
    return set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", text.lower()))

def hybrid_retrieve(query: str, k: int = 3):
    # 1. Dense retrieval (top 15 candidates)
    res = collection.query(query_texts=[query], n_results=min(25, len(docs)))
    dense_docs = res["documents"][0]
    dense_metas = res["metadatas"][0]
    dense_dists = res["distances"][0]

    q_tokens = tokenize(query)
    
    scored_candidates = []
    for rank, (doc, meta, dist) in enumerate(zip(dense_docs, dense_metas, dense_dists)):
        # Cosine similarity is 1.0 - distance
        cos_sim = max(0.0, 1.0 - dist)
        
        # Keyword overlap
        doc_tokens = tokenize(doc)
        kw_overlap = len(q_tokens.intersection(doc_tokens)) / max(1, len(q_tokens))
        
        # Combined score (dense similarity + keyword boost)
        combined_score = 0.65 * cos_sim + 0.35 * kw_overlap
        
        scored_candidates.append({
            "source": f"{meta['source']} (Section: {meta['section']})" if meta['section'] else meta['source'],
            "section": meta["section"],
            "raw_source": meta["source"],
            "content": doc,
            "score": round(float(combined_score), 4),
            "cos_sim": round(float(cos_sim), 4),
            "kw_overlap": round(float(kw_overlap), 4),
            "distance": round(float(dist), 4)
        })

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    return scored_candidates[:k]

questions = [
    "What does a Research Scientist do?",
    "Why is the Manager role's O*NET mapping unreliable?",
    "What is the production model's decision threshold and why was it chosen?",
    "What is the company's parental leave policy?",
    "What is the capital of France?"
]

for q in questions:
    print("=" * 60)
    print("Q:", q)
    top_chunks = hybrid_retrieve(q, k=3)
    for c in top_chunks:
        print(f"  [score={c['score']:.4f} | cos={c['cos_sim']:.4f} | kw={c['kw_overlap']:.4f}] {c['source']}")
        print(f"    Excerpt: {c['content'].replace(chr(10), ' ')[:100]}...")
