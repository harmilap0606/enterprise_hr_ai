"""
scripts/evaluate_embedding_models.py
====================================
Offline benchmark evaluating embedding models on the Enterprise HR knowledge corpus.
Evaluates:
1. sentence-transformers/all-MiniLM-L6-v2
2. BAAI/bge-small-en-v1.5
3. thenlper/gte-base

Metrics:
- Hit@1, Hit@3, Hit@5, Hit@10, MRR
- Embedding dimension
- Ingestion time (1,042 chunks)
- Query embedding time (24 queries)
- Retrieval latency
- Memory / resource requirements

Saves structured results to reports/rag/embedding_benchmark.json.
Zero modifications to production Chroma database or API layer.
"""

import sys
import json
import time
import psutil
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Ensure utf-8 encoding on Windows console
sys.stdout.reconfigure(encoding="utf-8")

from app.rag.loaders import load_all_knowledge_documents
from app.rag.chunking import chunk_all_documents
from app.utils.config import BASE_DIR

EVAL_QUERIES_PATH = BASE_DIR / "tests" / "fixtures" / "rag_eval_queries.json"
REPORT_DIR = BASE_DIR / "reports" / "rag"
REPORT_FILE = REPORT_DIR / "embedding_benchmark.json"

CANDIDATE_MODELS = [
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "short_name": "all-MiniLM-L6-v2",
        "pooling": "mean",
        "query_prefix": "",
        "notes": "Lightweight 6-layer MiniLM, current project baseline"
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "short_name": "bge-small-en-v1.5",
        "pooling": "cls",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "notes": "BAAI English Small v1.5, optimized for passage retrieval with query prompt"
    },
    {
        "name": "thenlper/gte-base",
        "short_name": "gte-base",
        "pooling": "mean",
        "query_prefix": "",
        "notes": "Alibaba General Text Embeddings base model, 768-dim"
    }
]


def mean_pooling(model_output, attention_mask):
    """Mean pooling taking attention mask into account."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def cls_pooling(model_output):
    """First token [CLS] pooling."""
    return model_output[0][:, 0]


class EmbeddingEvaluator:
    def __init__(self, model_info: Dict[str, Any], device: torch.device):
        self.model_info = model_info
        self.device = device
        self.model_name = model_info["name"]
        self.pooling = model_info["pooling"]
        self.query_prefix = model_info.get("query_prefix", "")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.dimension = self.model.config.hidden_size

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Computes L2-normalized embeddings for a list of texts in batches."""
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
                if self.pooling == "cls":
                    pooled = cls_pooling(output)
                else:
                    pooled = mean_pooling(output, encoded["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
                all_embeddings.append(normalized.cpu().numpy())

        return np.vstack(all_embeddings)


def compute_metrics(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    chunk_ids: List[str],
    eval_queries: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Computes Hit@1, Hit@3, Hit@5, Hit@10, and MRR.
    Uses cosine similarity (matrix dot product since embeddings are L2 normalized).
    """
    # (n_queries, n_chunks) similarity matrix
    similarity_matrix = np.dot(query_embeddings, corpus_embeddings.T)

    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    hit_10 = 0
    reciprocal_ranks = []

    for i, q in enumerate(eval_queries):
        expected = set(q["expected_chunk_ids"])
        sims = similarity_matrix[i]
        
        # Sort chunks descending by similarity
        ranked_indices = np.argsort(-sims)
        ranked_chunk_ids = [chunk_ids[idx] for idx in ranked_indices]

        # Check top-k
        top_1 = set(ranked_chunk_ids[:1])
        top_3 = set(ranked_chunk_ids[:3])
        top_5 = set(ranked_chunk_ids[:5])
        top_10 = set(ranked_chunk_ids[:10])

        if any(c in top_1 for c in expected):
            hit_1 += 1
        if any(c in top_3 for c in expected):
            hit_3 += 1
        if any(c in top_5 for c in expected):
            hit_5 += 1
        if any(c in top_10 for c in expected):
            hit_10 += 1

        # MRR calculation (find rank of first relevant chunk)
        found_rank = None
        for r_idx, c_id in enumerate(ranked_chunk_ids, 1):
            if c_id in expected:
                found_rank = r_idx
                break

        if found_rank is not None:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    total_q = len(eval_queries)
    return {
        "Hit@1": round(hit_1 / total_q, 4),
        "Hit@3": round(hit_3 / total_q, 4),
        "Hit@5": round(hit_5 / total_q, 4),
        "Hit@10": round(hit_10 / total_q, 4),
        "MRR": round(float(np.mean(reciprocal_ranks)), 4)
    }


def run_benchmark():
    print("=" * 85)
    print("STEP 3: OFFLINE EMBEDDING MODEL BENCHMARK")
    print("=" * 85)

    # 1. Load Step 2 Chunks
    print("\n1. Ingesting Step 2 Corpus Chunks...")
    documents = load_all_knowledge_documents()
    chunks = chunk_all_documents(documents)
    chunk_texts = [c.contextual_text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    print(f"   -> Loaded {len(chunks)} contextualized chunks across {len(documents)} documents.")

    # 2. Load Evaluation Queries
    if not EVAL_QUERIES_PATH.exists():
        raise FileNotFoundError(f"Evaluation queries not found at: {EVAL_QUERIES_PATH}")
    with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)
    print(f"   -> Loaded {len(eval_queries)} deterministic evaluation queries from {EVAL_QUERIES_PATH.name}")

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   -> Inference device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    benchmark_results = []

    # 3. Evaluate Each Candidate Model
    for model_info in CANDIDATE_MODELS:
        model_name = model_info["name"]
        short_name = model_info["short_name"]
        print(f"\n--- Evaluating Candidate: {model_name} ---")

        try:
            mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
            load_start = time.perf_counter()
            evaluator = EmbeddingEvaluator(model_info, device)
            load_time = time.perf_counter() - load_start
            mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
            mem_used_mb = round(max(0.0, mem_after - mem_before), 1)

            # A. Embed Corpus
            t0 = time.perf_counter()
            corpus_embeddings = evaluator.embed_texts(chunk_texts, batch_size=32)
            corpus_time = time.perf_counter() - t0
            print(f"   - Corpus embedded ({len(chunks)} chunks): {corpus_time:.2f}s ({len(chunks)/corpus_time:.1f} chunks/sec)")

            # B. Embed Queries
            query_texts = [model_info.get("query_prefix", "") + q["query"] for q in eval_queries]
            t1 = time.perf_counter()
            query_embeddings = evaluator.embed_texts(query_texts, batch_size=len(query_texts))
            query_time = time.perf_counter() - t1
            print(f"   - Queries embedded ({len(query_texts)} queries): {query_time*1000:.1f}ms ({query_time/len(query_texts)*1000:.2f}ms/query)")

            # C. Top-K Retrieval & Metrics
            t2 = time.perf_counter()
            metrics = compute_metrics(query_embeddings, corpus_embeddings, chunk_ids, eval_queries)
            retrieval_time = time.perf_counter() - t2
            print(f"   - Retrieval & scoring time: {retrieval_time*1000:.2f}ms")

            result = {
                "model_name": model_name,
                "short_name": short_name,
                "dimension": evaluator.dimension,
                "metrics": metrics,
                "timings": {
                    "corpus_embedding_sec": round(corpus_time, 3),
                    "query_embedding_ms": round(query_time * 1000, 2),
                    "avg_query_latency_ms": round((query_time / len(query_texts)) * 1000, 2),
                    "retrieval_scoring_ms": round(retrieval_time * 1000, 2),
                    "model_load_sec": round(load_time, 2)
                },
                "resources": {
                    "device": str(device),
                    "process_ram_mb": round(mem_after, 1),
                    "model_ram_delta_mb": mem_used_mb
                },
                "notes": model_info["notes"]
            }
            benchmark_results.append(result)

            print(f"   -> Hit@1: {metrics['Hit@1']:.3f} | Hit@3: {metrics['Hit@3']:.3f} | Hit@5: {metrics['Hit@5']:.3f} | Hit@10: {metrics['Hit@10']:.3f} | MRR: {metrics['MRR']:.4f}")

            # Clean up GPU memory between models
            del evaluator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"   [ERROR] Failed evaluating {model_name}: {e}")
            benchmark_results.append({
                "model_name": model_name,
                "status": "FAILED",
                "error": str(e)
            })

    # 4. Print Comparison Table
    print("\n" + "=" * 85)
    print("EMBEDDING BENCHMARK COMPARISON TABLE")
    print("=" * 85)
    print(f"{'Model Name':<35} | {'Dim':<5} | {'Hit@1':<7} | {'Hit@3':<7} | {'Hit@5':<7} | {'Hit@10':<7} | {'MRR':<7} | {'Query Lat':<9}")
    print("-" * 85)
    for r in benchmark_results:
        if "metrics" in r:
            m = r["metrics"]
            t = r["timings"]
            print(f"{r['short_name']:<35} | {r['dimension']:<5} | {m['Hit@1']:<7.3f} | {m['Hit@3']:<7.3f} | {m['Hit@5']:<7.3f} | {m['Hit@10']:<7.3f} | {m['MRR']:<7.4f} | {t['avg_query_latency_ms']:>6.2f} ms")
        else:
            print(f"{r.get('model_name', 'Unknown'):<35} | FAILED ({r.get('error', '')[:30]})")
    print("=" * 85)

    # 5. Save to reports/rag/embedding_benchmark.json
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_corpus_chunks": len(chunks),
        "total_eval_queries": len(eval_queries),
        "device": str(device),
        "candidates": benchmark_results
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved benchmark results to: {REPORT_FILE.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    run_benchmark()
