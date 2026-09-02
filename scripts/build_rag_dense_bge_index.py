"""
scripts/build_rag_dense_bge_index.py
====================================
Builds the persistent ChromaDB collection 'enterprise_hr_knowledge_bge'
using BAAI/bge-small-en-v1.5 embeddings over the 1,042 Step 2 chunks.
Leaves existing collection 'enterprise_hr_knowledge' completely untouched.
"""

import sys
import time
from pathlib import Path
import chromadb

from app.rag.loaders import load_all_knowledge_documents
from app.rag.chunking import chunk_all_documents
from app.rag.embeddings import BGEEmbedder, BGE_MODEL_NAME, EMBEDDING_DIMENSION
from app.utils.config import BASE_DIR

sys.stdout.reconfigure(encoding="utf-8")

VECTOR_STORE_DIR = BASE_DIR / "data" / "vectorstore"
COLLECTION_NAME = "enterprise_hr_knowledge_bge"


def build_dense_bge_index():
    print("=" * 70)
    print(f"BUILDING DENSE BGE CHROMA COLLECTION: {COLLECTION_NAME}")
    print("=" * 70)

    # 1. Load Step 2 chunks
    print("Loading knowledge documents and chunking...")
    documents = load_all_knowledge_documents()
    chunks = chunk_all_documents(documents)
    total_chunks = len(chunks)
    print(f"Prepared {total_chunks} chunks.")

    # 2. Initialize BGE embedder
    print(f"Loading embedding model: {BGE_MODEL_NAME}...")
    embedder = BGEEmbedder()
    print(f"Model loaded on {embedder.device}. Embedding dimension: {embedder.dimension}")

    # 3. Generate embeddings (documents are embedded without instruction prefix)
    texts_to_embed = [c.contextual_text for c in chunks]
    print(f"Encoding {total_chunks} chunks on CPU (batch_size=32)...")
    start_t = time.perf_counter()
    embeddings = embedder.embed_documents(texts_to_embed, batch_size=32)
    encode_time = time.perf_counter() - start_t
    print(f"Encoding completed in {encode_time:.2f}s ({total_chunks / encode_time:.1f} chunks/s).")

    # 4. Ingest into ChromaDB persistent collection
    print(f"Connecting to ChromaDB at: {VECTOR_STORE_DIR}")
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    # Check existing collections
    existing_cols = [c.name for c in client.list_collections()]
    print(f"Collections before build: {existing_cols}")

    # Recreate or create collection
    if COLLECTION_NAME in existing_cols:
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting to rebuild fresh...")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "model": BGE_MODEL_NAME, "dimension": EMBEDDING_DIMENSION}
    )

    # Prepare batches for ChromaDB upsert
    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "source": c.source,
            "title": c.title,
            "section": c.section,
            "document_type": c.document_type,
            "token_count": c.token_count
        }
        for c in chunks
    ]
    documents_content = [c.text for c in chunks]

    # Ingest in batches of 250
    batch_size = 250
    for i in range(0, total_chunks, batch_size):
        end = min(i + batch_size, total_chunks)
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=documents_content[i:end],
            metadatas=metadatas[i:end]
        )

    print(f"\nSuccessfully populated collection '{COLLECTION_NAME}'. Count = {collection.count()}.")

    # Verify existing collection untouched
    legacy_count = client.get_collection("enterprise_hr_knowledge").count()
    print(f"VERIFICATION: Legacy collection 'enterprise_hr_knowledge' count = {legacy_count} (UNTOUCHED).")
    print("Dense BGE index build complete.\n")


if __name__ == "__main__":
    build_dense_bge_index()
