"""
app/rag/embed_corpus.py
=======================
Corpus ingestion and embedding pipeline for the Enterprise HR AI platform.
Grounds the RAG system strictly on verified real text sources:
1. data/processed/occupation_master.csv (real O*NET occupational descriptions)
2. docs/model_card.md (production ML model specifications and limitations)
3. docs/data_relationships.md (dataset inventory, join rules, and open issues)

Stores embeddings in a local persistent Chroma vector store at data/vectorstore/.
Uses the lightweight sentence-transformer all-MiniLM-L6-v2 embedding model.
"""

import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

from app.utils.logger import logger
from app.utils.config import BASE_DIR

VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
OCCUPATION_MASTER_PATH = BASE_DIR / "data" / "processed" / "occupation_master.csv"
MODEL_CARD_PATH = BASE_DIR / "docs" / "model_card.md"
DATA_RELATIONSHIPS_PATH = BASE_DIR / "docs" / "data_relationships.md"

COLLECTION_NAME = "enterprise_hr_knowledge"


def chunk_markdown_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Splits a markdown document into clean section chunks based on '## ' headers.
    Tags each chunk with its source file path and section title.
    """
    if not file_path.exists():
        logger.warning(f"Markdown file not found: {file_path}")
        return []

    text = file_path.read_text(encoding="utf-8")
    raw_sections = re.split(r"\n(?=##\s+)", text)
    
    chunks = []
    rel_path = str(file_path.relative_to(BASE_DIR)).replace("\\", "/") if file_path.is_relative_to(BASE_DIR) else file_path.name
    
    for s in raw_sections:
        s_clean = s.strip()
        if not s_clean:
            continue
        lines = s_clean.splitlines()
        first_line = lines[0].strip()
        if first_line.startswith("## "):
            sec_title = first_line.replace("## ", "").strip()
        elif first_line.startswith("# "):
            sec_title = first_line.replace("# ", "").strip()
        else:
            sec_title = "Overview"
            
        chunks.append({
            "source": rel_path,
            "section": sec_title,
            "content": s_clean
        })
    return chunks


def load_corpus() -> List[Dict[str, Any]]:
    """
    Loads all approved real text sources:
    1. Occupation master records (O*NET titles and descriptions)
    2. Markdown documentation sections (model card and data relationships)
    """
    items = []

    # 1. O*NET Occupations
    if OCCUPATION_MASTER_PATH.exists():
        df_occ = pd.read_csv(OCCUPATION_MASTER_PATH)
        rel_occ_path = str(OCCUPATION_MASTER_PATH.relative_to(BASE_DIR)).replace("\\", "/")
        for _, row in df_occ.iterrows():
            code = str(row["O*NET-SOC Code"])
            title = str(row["Title"])
            desc = str(row["Description"])
            doc_text = (
                f"O*NET Occupation: {title}\n"
                f"O*NET-SOC Code: {code}\n"
                f"Description: {desc}"
            )
            items.append({
                "id": f"occ_{code}",
                "document": doc_text,
                "metadata": {
                    "source": rel_occ_path,
                    "section": title,
                    "code": code,
                    "doc_type": "occupation"
                }
            })
        logger.info(f"Loaded {len(df_occ)} occupation descriptions from {OCCUPATION_MASTER_PATH.name}")
    else:
        logger.error(f"Occupation master not found at {OCCUPATION_MASTER_PATH}")

    # 2. Markdown documentation chunks
    for doc_path in [MODEL_CARD_PATH, DATA_RELATIONSHIPS_PATH]:
        chunks = chunk_markdown_file(doc_path)
        for idx, chunk in enumerate(chunks):
            doc_text = (
                f"Document: {chunk['source']}\n"
                f"Section: {chunk['section']}\n\n"
                f"{chunk['content']}"
            )
            items.append({
                "id": f"{doc_path.stem}_{idx}",
                "document": doc_text,
                "metadata": {
                    "source": chunk["source"],
                    "section": chunk["section"],
                    "code": "",
                    "doc_type": "markdown_doc"
                }
            })
        logger.info(f"Loaded {len(chunks)} sections from {doc_path.name}")

    return items


def build_vectorstore(persist_dir: Path = VECTORSTORE_DIR) -> int:
    """
    Builds and persists the local Chroma vector store.
    Vector store choice: ChromaDB
    Rationale:
    1. Zero external server requirement; stores directly to local disk (PersistentClient).
    2. Ships with built-in all-MiniLM-L6-v2 ONNX embedding generation without compiling C++ FAISS wheels.
    3. Handles rich metadata filtering (doc_type, section, source) natively.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initializing Chroma persistent vector store at: {persist_dir}")
    
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    # Re-create collection for a clean, deterministic index
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    corpus_items = load_corpus()
    if not corpus_items:
        raise ValueError("Corpus is empty. Cannot build vector store.")

    documents = [item["document"] for item in corpus_items]
    metadatas = [item["metadata"] for item in corpus_items]
    ids = [item["id"] for item in corpus_items]

    batch_size = 200
    for i in range(0, len(corpus_items), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        logger.info(f"Indexed batch {i // batch_size + 1} / {(len(corpus_items) + batch_size - 1) // batch_size}")

    logger.info(f"Successfully embedded and saved {len(corpus_items)} items to {persist_dir}")
    return len(corpus_items)


if __name__ == "__main__":
    count = build_vectorstore()
    print(f"Index complete: {count} documents stored in {VECTORSTORE_DIR}")
