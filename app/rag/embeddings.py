"""
app/rag/embeddings.py
=====================
Production embedding layer for BAAI/bge-small-en-v1.5.
Implements the asymmetric query/document embedding conventions from 02_embedding_model.ipynb
and verified empirically in Step 3 benchmark:
- Query prefix: "Represent this sentence for searching relevant passages: {query}"
- Document prefix: None (documents are encoded as raw/contextual text)
- Pooling: CLS pooling ([CLS] token representation)
- Normalization: L2 normalization
"""

from typing import List, Union
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIMENSION = 384


class BGEEmbedder:
    """
    Singleton-friendly embedding wrapper for BAAI/bge-small-en-v1.5 using PyTorch & HuggingFace Transformers.
    """
    def __init__(self, model_name: str = BGE_MODEL_NAME, device: str = None):
        self.model_name = model_name
        self.query_prefix = BGE_QUERY_PREFIX
        self.dimension = EMBEDDING_DIMENSION

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embeds document passages without instruction prefix.
        Applies CLS pooling and L2 normalization.
        """
        return self._encode_batch(texts, batch_size=batch_size)

    def embed_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embeds search queries conditioned with the BGE retrieval instruction prefix:
        'Represent this sentence for searching relevant passages: {query}'
        """
        conditioned_queries = [f"{self.query_prefix}{q}" for q in queries]
        return self._encode_batch(conditioned_queries, batch_size=batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string, returning a 1D float vector of shape (384,)."""
        return self.embed_queries([query])[0]

    def _encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**encoded)
                # BGE uses CLS pooling
                cls_token = output[0][:, 0]
                normalized = F.normalize(cls_token, p=2, dim=1)
                all_embeddings.append(normalized.cpu().numpy())

        if not all_embeddings:
            return np.empty((0, self.dimension), dtype=np.float32)

        return np.vstack(all_embeddings)
