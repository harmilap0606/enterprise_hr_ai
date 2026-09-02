"""
scripts/build_rag_sparse_index.py
=================================
Builds a persistent BM25 sparse retrieval index over the verified Step 2 knowledge chunks.
Preserves exact chunk_id alignment with the dense vector store.
Adapts the sparse indexing concept and tokenization strategy from 06_hybrid_search.ipynb.
"""

import sys
import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Set

from rank_bm25 import BM25Okapi

from app.rag.loaders import load_all_knowledge_documents
from app.rag.chunking import chunk_all_documents
from app.utils.config import BASE_DIR

sys.stdout.reconfigure(encoding="utf-8")

SPARSE_INDEX_DIR = BASE_DIR / "data" / "rag" / "sparse_index"
INDEX_FILE = SPARSE_INDEX_DIR / "bm25_index.pkl"
METADATA_FILE = SPARSE_INDEX_DIR / "chunk_metadata.json"

# Common English stopwords derived from standard NLP stopword lists (matching notebook stopwords="english")
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
    """
    Tokenizes text for BM25 indexing and querying.
    - Converts to lowercase
    - Extracts word and domain tokens while preserving codes (e.g. '19-1042.00', '0.40', 'shap')
    - Filters stopwords
    """
    # Regex matching alphanumeric words, hyphenated codes, and dotted decimals
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:[-.][a-zA-Z0-9]+)*\b", text.lower())
    return [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) > 1]


def build_sparse_index():
    print("=" * 70)
    print("BUILDING RAG BM25 SPARSE INDEX")
    print("=" * 70)

    # 1. Load Step 2 documents and chunks
    print("Loading knowledge documents...")
    documents = load_all_knowledge_documents()
    chunks = chunk_all_documents(documents)
    total_chunks = len(chunks)
    print(f"Loaded {len(documents)} documents -> {total_chunks} chunks.")

    # 2. Prepare tokens and metadata
    corpus_tokens: List[List[str]] = []
    chunk_metadata: List[Dict[str, Any]] = []
    all_terms: Set[str] = set()

    for idx, c in enumerate(chunks):
        # We index chunk.text (and contextual headers enhance semantic scope)
        # To maximize retrieval alignment with contextual text, index the full contextual text:
        tokens = tokenize_text(c.contextual_text)
        corpus_tokens.append(tokens)
        all_terms.update(tokens)

        chunk_metadata.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "text": c.text,
            "contextual_text": c.contextual_text,
            "source": c.source,
            "title": c.title,
            "section": c.section,
            "document_type": c.document_type,
            "token_count": c.token_count,
            "metadata": c.metadata,
            "index_position": idx
        })

    # 3. Fit BM25 Okapi model
    print(f"Fitting BM25Okapi on {total_chunks} chunk documents...")
    bm25_model = BM25Okapi(corpus_tokens)

    # 4. Save persistent artifacts
    SPARSE_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving BM25 index to: {INDEX_FILE}")
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(bm25_model, f)

    print(f"Saving chunk metadata mapping to: {METADATA_FILE}")
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, indent=2)

    # 5. Print Index Statistics
    avg_tokens = sum(len(t) for t in corpus_tokens) / total_chunks if total_chunks > 0 else 0
    print("\n--- BM25 Sparse Index Statistics ---")
    print(f"Total Chunks Indexed: {total_chunks}")
    print(f"Total Unique Terms (Vocabulary): {len(all_terms):,}")
    print(f"Average Tokens per Chunk: {avg_tokens:.1f}")
    print(f"Index Artifact Size: {INDEX_FILE.stat().st_size / 1024:.1f} KB")
    print(f"Metadata Artifact Size: {METADATA_FILE.stat().st_size / 1024:.1f} KB")
    print("Sparse index build complete successfully.\n")


if __name__ == "__main__":
    build_sparse_index()
