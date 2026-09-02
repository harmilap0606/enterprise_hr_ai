"""
app/rag/retrieval/hybrid_retriever.py
====================================
Hybrid dense + sparse retriever implementing the dual retrieval and score
combination algorithm from 06_hybrid_search.ipynb:
1. Dense Retrieval via ChromaDB with BAAI/bge-small-en-v1.5
2. Sparse Retrieval via BM25Okapi
3. Candidate union of top-K results
4. Min-Max normalization on separate score spaces with edge-case protection
5. Weighted combination: 0.8 * Dense + 0.2 * Sparse (configurable)
6. Deterministic tie-breaking: hybrid_score descending, then chunk_id ascending
7. Top-K final result selection
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi

from app.rag.embeddings import BGEEmbedder
from app.rag.retrieval.schemas import RetrievalConfig, HybridSearchResult
from app.utils.config import BASE_DIR

DEFAULT_VECTOR_STORE_DIR = BASE_DIR / "data" / "vectorstore"
DEFAULT_SPARSE_DIR = BASE_DIR / "data" / "rag" / "sparse_index"
DEFAULT_COLLECTION_NAME = "enterprise_hr_knowledge_bge"

# Stopword list matching scripts/build_rag_sparse_index.py and notebook convention
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


def tokenize_query(query: str) -> List[str]:
    """Tokenizes incoming user search query for BM25 retrieval."""
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:[-.][a-zA-Z0-9]+)*\b", query.lower())
    filtered = [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) > 1]
    # Fallback to unfiltered tokens if all terms were stopwords
    return filtered if filtered else [t for t in tokens if len(t) > 1]


STRUCTURED_IDENTIFIER_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b\d{2}-\d{4}(?:\.\d{2})?\b"),               # O*NET / SOC codes (e.g. 19-1042.00, 15-1221.00)
    re.compile(r"\b[a-zA-Z0-9_-]+\.(?:csv|md|json)\b", re.I), # Datasets/docs (e.g. occupation_master.csv, model_card.md)
    re.compile(r"\b(?:SOC|DOC|POL|ID)[-_]?(?:[A-Z0-9]+[-_])?\d+\b", re.I)     # Policy/document/SOC codes (e.g. POL-JOB-001, POL-MODEL-001, SOC-101)
]


def extract_query_identifiers(text: str) -> List[str]:
    """Extracts structured identifiers from user query for exact match protection."""
    found: List[str] = []
    for pat in STRUCTURED_IDENTIFIER_PATTERNS:
        for m in pat.findall(text):
            if m.lower() not in [x.lower() for x in found]:
                found.append(m)
    return found


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Min-Max normalization as implemented in 06_hybrid_search.ipynb Cell 37:
    norm = (scores - min(scores)) / (max(scores) - min(scores))
    Safely handles edge cases (empty arrays, zero range, NaN/inf).
    """
    if len(scores) == 0:
        return np.array([], dtype=np.float32)

    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    span = s_max - s_min

    # Edge case: all scores identical or span is zero
    if span <= 1e-9:
        # If all scores are non-zero, normalize to 1.0; if all zero, normalize to 0.0
        return np.ones_like(scores, dtype=np.float32) if s_max > 1e-9 else np.zeros_like(scores, dtype=np.float32)

    norm = (scores - s_min) / span
    return np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


class HybridRetriever:
    """
    Production hybrid retriever combining BGE dense retrieval and BM25 sparse retrieval.
    """
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        embedder: Optional[BGEEmbedder] = None,
        chroma_client: Optional[chromadb.PersistentClient] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        sparse_dir: Path = DEFAULT_SPARSE_DIR,
    ):
        self.config = config or RetrievalConfig()
        self.collection_name = collection_name
        self.sparse_dir = Path(sparse_dir)

        # 1. Initialize or inject Dense Embedder
        self.embedder = embedder or BGEEmbedder()

        # 2. Connect to ChromaDB
        if chroma_client is None:
            self.chroma_client = chromadb.PersistentClient(path=str(DEFAULT_VECTOR_STORE_DIR))
        else:
            self.chroma_client = chroma_client

        self.dense_collection = self.chroma_client.get_collection(self.collection_name)

        # 3. Load persistent BM25 index & metadata mapping
        self._load_sparse_index()

    def _load_sparse_index(self):
        index_file = self.sparse_dir / "bm25_index.pkl"
        metadata_file = self.sparse_dir / "chunk_metadata.json"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                f"Sparse index files not found in {self.sparse_dir}. "
                f"Run 'python -m scripts.build_rag_sparse_index' first."
            )

        with open(index_file, "rb") as f:
            self.bm25_model: BM25Okapi = pickle.load(f)

        with open(metadata_file, "r", encoding="utf-8") as f:
            self.chunk_metadata_list: List[Dict[str, Any]] = json.load(f)

        # Build fast lookup dictionary by chunk_id
        self.chunk_lookup: Dict[str, Dict[str, Any]] = {
            m["chunk_id"]: m for m in self.chunk_metadata_list
        }

    def retrieve_dense(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Executes dense retrieval against ChromaDB.
        Notebook convention: Query must be prefixed with BGE query instruction:
        'Represent this sentence for searching relevant passages: {query}'
        """
        k = top_k or self.config.dense_top_k

        # Embed single query using query-conditioned instruction prefix
        query_vector = self.embedder.embed_query(query).tolist()

        results = self.dense_collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        dense_items = []
        if not results or not results["ids"] or not results["ids"][0]:
            return dense_items

        ids = results["ids"][0]
        distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)
        documents = results["documents"][0] if "documents" in results else [""] * len(ids)
        metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(ids)

        for cid, dist, doc, meta in zip(ids, distances, documents, metadatas):
            # ChromaDB cosine distance: distance = 1.0 - cosine_similarity
            # Similarity score = 1.0 - distance
            cosine_sim = max(0.0, min(1.0, 1.0 - float(dist)))
            dense_items.append({
                "chunk_id": cid,
                "dense_score": cosine_sim,
                "text": doc,
                "metadata": meta
            })

        return dense_items

    def retrieve_sparse(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Executes sparse retrieval using the BM25Okapi index.
        """
        k = top_k or self.config.sparse_top_k
        query_tokens = tokenize_query(query)

        if not query_tokens:
            return []

        # Get raw BM25 scores for all corpus documents
        scores = self.bm25_model.get_scores(query_tokens)
        scores_arr = np.array(scores)

        # Get top-k indices
        top_indices = np.argsort(-scores_arr)[:k]

        sparse_items = []
        for idx in top_indices:
            score = float(scores_arr[idx])
            # Filter out documents with zero BM25 match if desired, but keep top matches
            meta = self.chunk_metadata_list[idx]
            sparse_items.append({
                "chunk_id": meta["chunk_id"],
                "sparse_score": score,
                "text": meta["text"],
                "metadata": meta
            })

        return sparse_items

    def retrieve(
        self,
        query: str,
        config_override: Optional[RetrievalConfig] = None
    ) -> List[HybridSearchResult]:
        """
        Performs end-to-end hybrid retrieval:
        1. Dense retrieval + Sparse retrieval
        2. Candidate union
        3. Min-Max normalization
        4. Weighted combination (default: 0.8 * Dense + 0.2 * Sparse)
        5. Deterministic tie-breaking and top-K selection
        """
        cfg = config_override or self.config

        # Check for blank query
        if not query or not query.strip():
            return []

        # 1. Retrieve candidates from both sources
        dense_results = self.retrieve_dense(query, top_k=cfg.dense_top_k)
        sparse_results = self.retrieve_sparse(query, top_k=cfg.sparse_top_k)

        # If both empty, return empty list
        if not dense_results and not sparse_results:
            return []

        # 2. Build candidate union indexed by unique chunk_id
        candidates: Dict[str, Dict[str, Any]] = {}

        for d in dense_results:
            cid = d["chunk_id"]
            candidates[cid] = {
                "chunk_id": cid,
                "dense_score": d["dense_score"],
                "sparse_score": 0.0,
                "text": d["text"],
                "metadata": d.get("metadata", {})
            }

        for s in sparse_results:
            cid = s["chunk_id"]
            if cid in candidates:
                candidates[cid]["sparse_score"] = s["sparse_score"]
            else:
                candidates[cid] = {
                    "chunk_id": cid,
                    "dense_score": 0.0,
                    "sparse_score": s["sparse_score"],
                    "text": s["text"],
                    "metadata": s.get("metadata", {})
                }

        candidate_list = list(candidates.values())

        # 3. Min-Max normalize separate score spaces
        dense_scores = np.array([c["dense_score"] for c in candidate_list], dtype=np.float32)
        sparse_scores = np.array([c["sparse_score"] for c in candidate_list], dtype=np.float32)

        norm_dense = min_max_normalize(dense_scores)
        norm_sparse = min_max_normalize(sparse_scores)

        # 4. Weighted combination: (1 - alpha) * dense + alpha * sparse
        # Default: 0.8 * norm_dense + 0.2 * norm_sparse
        w_dense = cfg.dense_weight
        w_sparse = cfg.sparse_weight

        # Check for exact structured identifiers in query
        identifiers = extract_query_identifiers(query)

        scored_candidates: List[Tuple[float, str, Dict[str, Any], float, float, bool]] = []

        for i, cand in enumerate(candidate_list):
            nd = float(norm_dense[i])
            ns = float(norm_sparse[i])
            hybrid_score = (w_dense * nd) + (w_sparse * ns)

            # Check if chunk represents an exact identifier match
            is_exact_id = False
            if identifiers:
                cid_lower = cand["chunk_id"].lower()
                text_lower = cand["text"].lower()
                meta_str = str(cand.get("metadata", {})).lower()
                for ident in identifiers:
                    ident_lower = ident.lower()
                    ident_norm = ident_lower.replace("-", "_")
                    if (
                        ident_lower in cid_lower
                        or ident_norm in cid_lower
                        or ident_lower in text_lower
                        or ident_lower in meta_str
                    ):
                        is_exact_id = True
                        break

            # Record tuple: (hybrid_score, chunk_id, cand, nd, ns, is_exact_id)
            scored_candidates.append((hybrid_score, cand["chunk_id"], cand, nd, ns, is_exact_id))

        # 5. Deterministic tie-breaking sort: exact identifier matches first, then -hybrid_score, then chunk_id
        scored_candidates.sort(key=lambda x: (not x[5], -x[0], x[1]))

        # 6. Balanced Candidate Pool Coverage:
        # Guarantee that top BM25 candidates with positive keyword relevance are not crowded out
        final_k = cfg.final_top_k
        sparse_reserved = min(3, max(1, final_k // 5))
        hybrid_quota = max(1, final_k - sparse_reserved)

        top_candidates = []
        seen_ids = set()

        # Step A: Add top candidates by scored priority up to hybrid_quota
        for item in scored_candidates:
            if len(top_candidates) >= hybrid_quota:
                break
            top_candidates.append(item)
            seen_ids.add(item[1])

        # Step B: Ensure top sparse candidates with positive BM25 scores are represented in candidate pool
        for s in sparse_results:
            if s["sparse_score"] <= 0.0:
                break
            cid = s["chunk_id"]
            if cid not in seen_ids:
                matching = next((sc for sc in scored_candidates if sc[1] == cid), None)
                if matching:
                    top_candidates.append(matching)
                    seen_ids.add(cid)
                    if len(top_candidates) >= final_k:
                        break

        # Step C: Fill remaining quota from scored_candidates
        if len(top_candidates) < final_k:
            for item in scored_candidates:
                if item[1] not in seen_ids:
                    top_candidates.append(item)
                    seen_ids.add(item[1])
                    if len(top_candidates) >= final_k:
                        break

        # 7. Build typed HybridSearchResult models
        final_results: List[HybridSearchResult] = []
        for rank_idx, (h_score, cid, cand, nd, ns, is_exact) in enumerate(top_candidates, 1):
            # Fetch complete metadata from stored lookup if available
            stored_meta = self.chunk_lookup.get(cid, {})

            doc_id = stored_meta.get("doc_id", cand.get("metadata", {}).get("doc_id", "unknown"))
            text = stored_meta.get("text", cand.get("text", ""))
            contextual_text = stored_meta.get("contextual_text", text)
            source = stored_meta.get("source", cand.get("metadata", {}).get("source", ""))
            title = stored_meta.get("title", cand.get("metadata", {}).get("title", ""))
            section = stored_meta.get("section", cand.get("metadata", {}).get("section", ""))
            document_type = stored_meta.get("document_type", cand.get("metadata", {}).get("document_type", "knowledge"))

            result_item = HybridSearchResult(
                chunk_id=cid,
                doc_id=doc_id,
                text=text,
                contextual_text=contextual_text,
                source=source,
                title=title,
                section=section,
                document_type=document_type,
                dense_score=round(cand["dense_score"], 4),
                sparse_score=round(cand["sparse_score"], 4),
                normalized_dense_score=round(nd, 4),
                normalized_sparse_score=round(ns, 4),
                hybrid_score=round(h_score, 4),
                rank=rank_idx,
                is_exact_match=is_exact,
                metadata=stored_meta.get("metadata", cand.get("metadata", {}))
            )
            final_results.append(result_item)

        return final_results
