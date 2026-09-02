import time
import re
from pathlib import Path
import pandas as pd
import chromadb

t0 = time.time()
client = chromadb.Client()
collection = client.create_collection("corpus")

# 1. Occupations
df_occ = pd.read_csv("data/processed/occupation_master.csv")
docs, metas, ids = [], [], []
for idx, row in df_occ.iterrows():
    code = str(row["O*NET-SOC Code"])
    title = str(row["Title"])
    desc = str(row["Description"])
    docs.append(f"O*NET Title: {title}\nO*NET-SOC Code: {code}\nDescription: {desc}")
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
        docs.append(f"Document: {c['source']}\nSection: {c['section']}\nContent:\n{c['content']}")
        metas.append({"source": c["source"], "section": c["section"], "code": ""})
        ids.append(f"{doc_path.stem}_{i}")

print(f"Total docs to embed: {len(docs)}")
batch_size = 200
for i in range(0, len(docs), batch_size):
    collection.add(
        documents=docs[i:i+batch_size],
        metadatas=metas[i:i+batch_size],
        ids=ids[i:i+batch_size]
    )
print(f"Embedded in {time.time()-t0:.2f}s")

questions = [
    "What does a Research Scientist do?",
    "Why is the Manager role's O*NET mapping unreliable?",
    "What is the production model's decision threshold and why was it chosen?",
    "What is the company's parental leave policy?",
    "What is the capital of France?"
]

for q in questions:
    res = collection.query(query_texts=[q], n_results=3)
    print("=" * 60)
    print("Q:", q)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"  [dist={dist:.4f}] source={meta['source']} | section={meta['section']}")
        first_line = doc.splitlines()[:2]
        print(f"    Excerpt: {' '.join(first_line)[:120]}...")
