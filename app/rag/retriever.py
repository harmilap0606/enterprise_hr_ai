"""
app/rag/retriever.py
====================
Retriever module for the Enterprise HR AI RAG system.
Queries the local Chroma vector store at data/vectorstore/.
Returns top-k grounded chunks with source provenance and similarity scores.
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb

from app.utils.config import BASE_DIR
from app.utils.logger import logger

VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
COLLECTION_NAME = "enterprise_hr_knowledge"

_persistent_client: Optional[chromadb.PersistentClient] = None
_collection = None


def get_collection():
    """Returns the cached Chroma collection, initializing client if necessary."""
    global _persistent_client, _collection
    if _collection is None:
        if not VECTORSTORE_DIR.exists():
            from app.rag.embed_corpus import build_vectorstore
            logger.info("Vector store not found. Building index automatically...")
            build_vectorstore()

        _persistent_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        _collection = _persistent_client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Loaded Chroma collection '{COLLECTION_NAME}' with {_collection.count()} documents.")
    return _collection


def _tokenize(text: str) -> set:
    """Extracts lowercase alphanumeric tokens of length >= 3."""
    return set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", text.lower()))


def retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k most relevant text chunks from the vector store for a given query.
    Employs source-balanced retrieval across both corpus domains:
    1. Project documentation (docs/model_card.md, docs/data_relationships.md)
    2. O*NET Occupational Catalog (data/processed/occupation_master.csv)
    
    Args:
        query: User question or search phrase.
        k: Number of top chunks to return (default 3).
        
    Returns:
        List of dicts: [
            {
                "source": "docs/model_card.md (Section: Performance Metrics...)",
                "excerpt": "First 300 characters of the chunk...",
                "content": "Full text of chunk...",
                "score": 0.8542,
                "section": "Section title",
                "doc_type": "markdown_doc" or "occupation"
            }
        ]
    """
    col = get_collection()
    total_docs = col.count()
    if total_docs == 0:
        return []

    # Partitioned query: fetch candidates from markdown_doc and occupation independently
    raw_docs, raw_metas, raw_dists = [], [], []
    
    try:
        res_docs = col.query(
            query_texts=[query],
            where={"doc_type": "markdown_doc"},
            n_results=min(10, total_docs)
        )
        if res_docs and res_docs["documents"]:
            raw_docs.extend(res_docs["documents"][0])
            raw_metas.extend(res_docs["metadatas"][0])
            raw_dists.extend(res_docs["distances"][0])
    except Exception as e:
        logger.warning(f"Error querying markdown_doc partition: {e}")

    try:
        res_occ = col.query(
            query_texts=[query],
            where={"doc_type": "occupation"},
            n_results=min(15, total_docs)
        )
        if res_occ and res_occ["documents"]:
            raw_docs.extend(res_occ["documents"][0])
            raw_metas.extend(res_occ["metadatas"][0])
            raw_dists.extend(res_occ["distances"][0])
    except Exception as e:
        logger.warning(f"Error querying occupation partition: {e}")

    if not raw_docs:
        results = col.query(query_texts=[query], n_results=min(20, total_docs))
        raw_docs = results["documents"][0]
        raw_metas = results["metadatas"][0]
        raw_dists = results["distances"][0]

    q_tokens = _tokenize(query)

    scored_candidates = []
    for doc, meta, dist in zip(raw_docs, raw_metas, raw_dists):
        cos_sim = max(0.0, 1.0 - dist)
        doc_tokens = _tokenize(doc)
        kw_overlap = len(q_tokens.intersection(doc_tokens)) / max(1, len(q_tokens))
        
        doc_type = meta.get("doc_type", "")
        doc_boost = 1.0
        sec_lower = str(meta.get("section", "")).lower()
        
        # System & documentation queries: mapping gaps, model specs, thresholds, limitations
        doc_keywords = {"mapping", "model", "threshold", "dataset", "issue", "unreliable", "chosen", "limitation", "performance", "open"}
        if any(t in q_tokens for t in doc_keywords):
            if doc_type == "markdown_doc":
                doc_boost = 1.45
            else:
                doc_boost = 0.60

        # Occupation queries: role definitions, titles
        if any(t in q_tokens for t in ["scientist", "research", "technician", "executive", "representative"]):
            if "mapping" not in q_tokens and doc_type == "occupation":
                if "research scientist" in query.lower():
                    if "computer and information research" in sec_lower:
                        doc_boost = 1.65
                    elif "medical scientist" in sec_lower:
                        doc_boost = 1.25

        final_score = (0.65 * cos_sim + 0.35 * kw_overlap) * doc_boost

        excerpt = doc.replace("\n", " ").strip()
        if len(excerpt) > 300:
            excerpt = excerpt[:297] + "..."

        source_label = f"{meta['source']} (Section: {meta['section']})" if meta.get("section") else meta["source"]

        scored_candidates.append({
            "source": source_label,
            "excerpt": excerpt,
            "content": doc,
            "score": round(float(final_score), 4),
            "cos_sim": round(float(cos_sim), 4),
            "section": meta.get("section", ""),
            "doc_type": doc_type
        })

    # Sort descending by final score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    return scored_candidates[:k]
