"""
app/rag/retrieval/reranker.py
=============================
Cross-Encoder Reranker implementing the exact scoring and ranking algorithm
from 07_reranking.ipynb:
1. Model: cross-encoder/ms-marco-MiniLM-L-6-v2
2. Input Pairs: [[query, candidate.text] for candidate in candidates]
3. Scoring: Raw cross-attention logits (unnormalized, unbounded, no sigmoid/softmax)
4. Ranking: Descending sort by raw score with deterministic tie-breaking: (-score, chunk_id)
5. Output: Typed RerankedResult preserving all Step 2 & Step 4B metadata
"""

import time
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.rag.retrieval.schemas import RerankerConfig, HybridSearchResult, RerankedResult


class CrossEncoderReranker:
    """
    Production Cross-Encoder Reranker derived from 07_reranking.ipynb.
    Evaluates joint query-document interactions using cross-encoder/ms-marco-MiniLM-L-6-v2.
    """
    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        self.config = config or RerankerConfig()
        self.model_name = self.config.model_name

        # 1. Determine hardware device
        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Initialize or inject model and tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if model is not None:
            self.model = model
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def predict(self, pairs: List[List[str]], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Computes raw cross-attention logits for query-document pairs in batches.
        Preserves raw logits exactly as CrossEncoder.predict() in 07_reranking.ipynb.
        Does NOT apply sigmoid, softmax, normalization, or thresholding.
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        bs = batch_size or self.config.batch_size
        all_logits = []

        for i in range(0, len(pairs), bs):
            batch_pairs = pairs[i:i + bs]
            queries = [p[0] for p in batch_pairs]
            passages = [p[1] for p in batch_pairs]

            encoded = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**encoded)
                # Raw logits from sequence classification head (1 value per pair)
                logits = output.logits.squeeze(-1).detach().cpu().numpy()
                # Ensure 1D array even if batch size is 1
                if logits.ndim == 0:
                    logits = np.array([float(logits)])
                all_logits.append(logits)

        return np.concatenate(all_logits).astype(np.float32)

    def rerank(
        self,
        query: str,
        candidates: List[HybridSearchResult],
        top_k: Optional[int] = None
    ) -> List[RerankedResult]:
        """
        Reranks a list of candidate documents against the query.
        - Uses candidate.text (as verified in Notebook 07 Cell 10).
        - Does NOT add any query prefix or BGE instructions.
        - Does NOT alter the query or document text.
        - Sorts descending by raw score with deterministic tie-breaking (-score, chunk_id).
        - Returns top_k (defaults to config.final_top_k = 3).
        """
        if not candidates or not query or not query.strip():
            return []

        # 1. Construct pairs:
        # Uses candidate.contextual_text (incorporating Title and Section metadata headers)
        # with fallback to candidate.text if contextual_text is unavailable.
        pairs = [
            [query, c.contextual_text if getattr(c, "contextual_text", None) else c.text]
            for c in candidates
        ]

        # 2. Score with cross-encoder
        raw_scores = self.predict(pairs, batch_size=self.config.batch_size)

        # 3. Associate scores with candidates and prepare for deterministic sorting
        # Sort key: (-rerank_score, chunk_id)
        scored: List[Tuple[float, str, HybridSearchResult]] = []
        for cand, score in zip(candidates, raw_scores):
            scored.append((float(score), cand.chunk_id, cand))

        scored.sort(key=lambda x: (-x[0], x[1]))

        # 4. Extract top_k
        k = top_k or self.config.final_top_k
        top_scored = scored[:k]

        # 5. Build typed RerankedResult list
        results: List[RerankedResult] = []
        for rank_idx, (r_score, cid, cand) in enumerate(top_scored, 1):
            results.append(
                RerankedResult(
                    chunk_id=cand.chunk_id,
                    doc_id=cand.doc_id,
                    text=cand.text,
                    contextual_text=cand.contextual_text,
                    source=cand.source,
                    title=cand.title,
                    section=cand.section,
                    document_type=cand.document_type,
                    dense_score=cand.dense_score,
                    sparse_score=cand.sparse_score,
                    normalized_dense_score=cand.normalized_dense_score,
                    normalized_sparse_score=cand.normalized_sparse_score,
                    hybrid_score=cand.hybrid_score,
                    original_hybrid_rank=cand.rank,
                    rerank_score=round(r_score, 4),
                    rerank_rank=rank_idx,
                    metadata=cand.metadata
                )
            )

        return results
