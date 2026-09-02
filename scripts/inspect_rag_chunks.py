"""
scripts/inspect_rag_chunks.py
=============================
Offline inspection script for Document Ingestion + Normalization + Structural Chunking.
Prints summary metrics, document/chunk counts, character lengths, and 5 representative chunks.
Zero LLM calls; zero database modifications.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from collections import Counter
from app.rag.loaders import load_all_knowledge_documents
from app.rag.chunking import chunk_all_documents


def run_inspection():
    print("=" * 80)
    print("OFFLINE RAG CHUNK INSPECTION (STEP 2 FOUNDATION)")
    print("=" * 80)

    # 1. Load documents
    print("\nLoading approved knowledge documents...")
    documents = load_all_knowledge_documents()
    num_docs = len(documents)
    print(f"-> Total Documents Loaded: {num_docs}")

    # Document counts by type
    doc_types = Counter(doc.metadata.get("document_type", "unknown") for doc in documents)
    print("\nDocuments by Type:")
    for dt, cnt in doc_types.most_common():
        print(f"   - {dt:15s}: {cnt:5d} documents")

    # 2. Chunk documents
    print("\nExecuting structure-aware offline chunking...")
    chunks = chunk_all_documents(documents)
    num_chunks = len(chunks)
    print(f"-> Total Chunks Produced: {num_chunks}")

    # Chunk counts by document_type
    chunk_types = Counter(c.document_type for c in chunks)
    print("\nChunks by Document Type:")
    for ct, cnt in chunk_types.most_common():
        print(f"   - {ct:15s}: {cnt:5d} chunks")

    # Character length statistics (on raw text)
    lengths = [len(c.text) for c in chunks]
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    token_counts = [c.token_count for c in chunks]
    min_tok = min(token_counts) if token_counts else 0
    max_tok = max(token_counts) if token_counts else 0
    avg_tok = sum(token_counts) / len(token_counts) if token_counts else 0

    print("\nChunk Statistics (Body Text):")
    print(f"   - Min character length : {min_len:,} chars")
    print(f"   - Max character length : {max_len:,} chars")
    print(f"   - Avg character length : {avg_len:.1f} chars")
    print(f"   - Min token count      : {min_tok} tokens")
    print(f"   - Max token count      : {max_tok} tokens")
    print(f"   - Avg token count      : {avg_tok:.1f} tokens")

    # 3. Print 5 representative chunks with metadata
    print("\n" + "=" * 80)
    print("5 REPRESENTATIVE CHUNKS WITH METADATA")
    print("=" * 80)

    # Select representative samples:
    # 1 from occupation
    # 1 from role_mapping
    # 2 from model_card (governance: metrics, limitations)
    # 1 from data_relationships (architecture: relationship / open issues)
    sample_indices = []
    
    # Sample 1: Occupation
    for i, c in enumerate(chunks):
        if c.document_type == "occupation" and "Research Scientists" in c.title:
            sample_indices.append(i)
            break
    if not sample_indices:
        sample_indices.append(0)

    # Sample 2: Role mapping
    for i, c in enumerate(chunks):
        if c.document_type == "role_mapping" and "Healthcare Representative" in c.title:
            sample_indices.append(i)
            break

    # Sample 3: Model Card (Performance Metrics or Decision Threshold)
    for i, c in enumerate(chunks):
        if c.document_type == "governance" and "Performance Metrics" in c.section:
            sample_indices.append(i)
            break

    # Sample 4: Model Card (Known Limitations)
    for i, c in enumerate(chunks):
        if c.document_type == "governance" and "Known Limitations" in c.section:
            sample_indices.append(i)
            break

    # Sample 5: Data Relationships (Open Issues)
    for i, c in enumerate(chunks):
        if c.document_type == "architecture" and "Open Issues" in c.section:
            sample_indices.append(i)
            break

    for rank, idx in enumerate(sample_indices, 1):
        c = chunks[idx]
        print(f"\n--- Representative Sample #{rank} ---")
        print(f"Chunk ID      : {c.chunk_id}")
        print(f"Doc ID        : {c.doc_id}")
        print(f"Source        : {c.source}")
        print(f"Document Type : {c.document_type}")
        print(f"Section       : {c.section}")
        print(f"Tokens        : {c.token_count} tokens ({len(c.text)} chars)")
        print(f"Metadata      : {c.metadata}")
        print(f"Contextual Text:")
        for line in c.contextual_text.splitlines():
            print(f"   | {line}")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_inspection()
