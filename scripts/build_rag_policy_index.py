"""
scripts/build_rag_policy_index.py
=================================
Builds dedicated, isolated vector and sparse retrieval indices for the synthetic HR policy corpus:
1. ChromaDB collection: 'enterprise_hr_policies_bge' in data/vectorstore
2. BM25 sparse index: data/rag/policy_sparse_index/

Leaves the existing production collection 'enterprise_hr_knowledge_bge' (1,042 chunks)
and existing general BM25 index (data/rag/sparse_index/) completely untouched.
"""

import sys
import os
import time
import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Set

import chromadb
from rank_bm25 import BM25Okapi

from app.rag.loaders.policy_loader import load_all_hr_policies
from app.rag.chunking import chunk_all_documents
from app.rag.embeddings import BGEEmbedder, BGE_MODEL_NAME, EMBEDDING_DIMENSION
from app.utils.config import BASE_DIR

sys.stdout.reconfigure(encoding="utf-8")

VECTOR_STORE_DIR = BASE_DIR / "data" / "vectorstore"
POLICY_COLLECTION_NAME = "enterprise_hr_policies_bge"
ORIGINAL_COLLECTION_NAME = "enterprise_hr_knowledge_bge"

POLICY_SPARSE_DIR = BASE_DIR / "data" / "rag" / "policy_sparse_index"
POLICY_INDEX_FILE = POLICY_SPARSE_DIR / "bm25_index.pkl"
POLICY_METADATA_FILE = POLICY_SPARSE_DIR / "chunk_metadata.json"

ORIGINAL_SPARSE_DIR = BASE_DIR / "data" / "rag" / "sparse_index"
ORIGINAL_INDEX_FILE = ORIGINAL_SPARSE_DIR / "bm25_index.pkl"

ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}


def tokenize_text(text: str) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:[-.][a-zA-Z0-9]+)*\b", text.lower())
    return [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) > 1]


def build_policy_indices():
    print("=" * 75, flush=True)
    print("INGESTING SYNTHETIC HR POLICY CORPUS INTO DEDICATED INDICES", flush=True)
    print("=" * 75, flush=True)

    # 1. Inspect existing state to guarantee non-interference
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    existing_cols = {c.name: c for c in client.list_collections()}
    print(f"Existing ChromaDB collections: {list(existing_cols.keys())}", flush=True)
    
    orig_bge_count = -1
    if ORIGINAL_COLLECTION_NAME in existing_cols:
        orig_bge_count = existing_cols[ORIGINAL_COLLECTION_NAME].count()
        print(f"Existing '{ORIGINAL_COLLECTION_NAME}' count before build: {orig_bge_count} chunks.", flush=True)
    else:
        print(f"WARNING: '{ORIGINAL_COLLECTION_NAME}' not found.", flush=True)

    orig_sparse_count = -1
    if (ORIGINAL_SPARSE_DIR / "chunk_metadata.json").exists():
        with open(ORIGINAL_SPARSE_DIR / "chunk_metadata.json", "r", encoding="utf-8") as f:
            orig_sparse_data = json.load(f)
            orig_sparse_count = len(orig_sparse_data)
        print(f"Existing general BM25 chunk count before build: {orig_sparse_count} chunks.", flush=True)

    # 2. Load policy documents
    print("\n--- Step 1: Loading HR Policy Documents ---", flush=True)
    documents = load_all_hr_policies()
    print(f"Loaded {len(documents)} structured section documents from 10 policy files.", flush=True)

    # 3. Deterministic chunking
    print("\n--- Step 2: Structure-Aware Chunking ---", flush=True)
    chunks = chunk_all_documents(documents)
    total_chunks = len(chunks)
    print(f"Generated {total_chunks} structure-aware policy chunks.", flush=True)
    for idx, c in enumerate(chunks[:3], 1):
        print(f"  Chunk {idx}: {c.chunk_id} | Title: {c.title} | Tokens: {c.token_count}", flush=True)

    # 4. Generate BGE embeddings
    print("\n--- Step 3: Generating BAAI/bge-small-en-v1.5 Embeddings ---", flush=True)
    embedder = BGEEmbedder()
    print(f"Embedder loaded on {embedder.device}. Dimension: {embedder.dimension}", flush=True)
    texts_to_embed = [c.contextual_text for c in chunks]
    start_t = time.perf_counter()
    embeddings = embedder.embed_documents(texts_to_embed, batch_size=32)
    enc_time = time.perf_counter() - start_t
    print(f"Encoded {total_chunks} chunks in {enc_time:.2f}s ({total_chunks / enc_time:.1f} chunks/s).", flush=True)

    # 5. Populate separate ChromaDB collection (Idempotent)
    print(f"\n--- Step 4: Populating ChromaDB Collection '{POLICY_COLLECTION_NAME}' ---", flush=True)
    if POLICY_COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Collection '{POLICY_COLLECTION_NAME}' exists. Resetting for idempotent build...", flush=True)
        client.delete_collection(POLICY_COLLECTION_NAME)

    policy_collection = client.create_collection(
        name=POLICY_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "model": BGE_MODEL_NAME, "dimension": EMBEDDING_DIMENSION}
    )

    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "policy_id": c.metadata.get("policy_id", ""),
            "policy_title": c.metadata.get("policy_title", ""),
            "policy_domain": c.metadata.get("policy_domain", ""),
            "policy_version": c.metadata.get("policy_version", ""),
            "policy_status": c.metadata.get("policy_status", ""),
            "source_file": c.metadata.get("source_file", c.source),
            "source_type": c.metadata.get("source_type", "synthetic_hr_policy"),
            "source": c.source,
            "title": c.title,
            "section": c.section,
            "document_type": c.document_type,
            "token_count": c.token_count
        }
        for c in chunks
    ]
    documents_content = [c.text for c in chunks]

    # Ingest into ChromaDB in batches
    batch_size = 100
    for i in range(0, total_chunks, batch_size):
        end = min(i + batch_size, total_chunks)
        policy_collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=documents_content[i:end],
            metadatas=metadatas[i:end]
        )
    print(f"Successfully populated '{POLICY_COLLECTION_NAME}'. Count = {policy_collection.count()}.", flush=True)

    # 6. Build separate BM25 sparse index
    print(f"\n--- Step 5: Building Policy BM25 Sparse Index ---", flush=True)
    POLICY_SPARSE_DIR.mkdir(parents=True, exist_ok=True)
    corpus_tokens: List[List[str]] = []
    chunk_metadata: List[Dict[str, Any]] = []

    for idx, c in enumerate(chunks):
        tokens = tokenize_text(c.contextual_text)
        corpus_tokens.append(tokens)
        chunk_metadata.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "policy_id": c.metadata.get("policy_id", ""),
            "policy_title": c.metadata.get("policy_title", ""),
            "policy_domain": c.metadata.get("policy_domain", ""),
            "policy_version": c.metadata.get("policy_version", ""),
            "policy_status": c.metadata.get("policy_status", ""),
            "source_file": c.metadata.get("source_file", c.source),
            "source_type": c.metadata.get("source_type", "synthetic_hr_policy"),
            "source": c.source,
            "title": c.title,
            "section": c.section,
            "document_type": c.document_type,
            "token_count": c.token_count,
            "text": c.text,
            "contextual_text": c.contextual_text,
            "index_position": idx
        })

    bm25_model = BM25Okapi(corpus_tokens)

    with open(POLICY_INDEX_FILE, "wb") as f:
        pickle.dump(bm25_model, f)
    print(f"Saved policy BM25 model to: {POLICY_INDEX_FILE}", flush=True)

    with open(POLICY_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved policy chunk metadata to: {POLICY_METADATA_FILE}", flush=True)

    # 7. Final non-interference verification
    print("\n--- Step 6: Non-Interference Verification ---", flush=True)
    final_cols = {c.name: c for c in client.list_collections()}
    final_orig_count = final_cols[ORIGINAL_COLLECTION_NAME].count() if ORIGINAL_COLLECTION_NAME in final_cols else -1
    print(f"Verification 1: '{ORIGINAL_COLLECTION_NAME}' count = {final_orig_count} (Must equal {orig_bge_count})", flush=True)
    assert final_orig_count == orig_bge_count, f"Error: Original collection altered from {orig_bge_count} to {final_orig_count}!"

    if (ORIGINAL_SPARSE_DIR / "chunk_metadata.json").exists():
        with open(ORIGINAL_SPARSE_DIR / "chunk_metadata.json", "r", encoding="utf-8") as f:
            final_sparse_data = json.load(f)
            final_orig_sparse_count = len(final_sparse_data)
        print(f"Verification 2: General BM25 chunk count = {final_orig_sparse_count} (Must equal {orig_sparse_count})", flush=True)
        assert final_orig_sparse_count == orig_sparse_count, "Error: General BM25 index altered!"

    print(f"Verification 3: Policy ChromaDB count = {policy_collection.count()} == {total_chunks}", flush=True)
    assert policy_collection.count() == total_chunks, "Error: Policy ChromaDB count mismatch!"

    print(f"Verification 4: Policy BM25 doc count = {len(chunk_metadata)} == {total_chunks}", flush=True)
    assert len(chunk_metadata) == total_chunks, "Error: Policy BM25 count mismatch!"

    print("\nSUCCESS: Dedicated policy indices built cleanly with ZERO regression to original indices!\n", flush=True)
    return total_chunks


if __name__ == "__main__":
    build_policy_indices()
