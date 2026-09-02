"""
scripts/dump_notebook_details.py
================================
Extracts functional implementation details from all 8 notebooks.
"""

import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
nb_dir = Path("notebooks/rag")


def inspect_notebook(name: str):
    path = nb_dir / name
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print("=" * 80)
    print(f"NOTEBOOK: {name}")
    print("=" * 80)
    for idx, cell in enumerate(nb["cells"]):
        ctype = cell.get("cell_type")
        src = "".join(cell.get("source", []))
        # Print non-trivial code cells
        if ctype == "code":
            lines = [l for l in src.splitlines() if not l.strip().startswith("#")]
            # Filter out pure rich styling or pip installs
            if any(k in src for k in [
                "SentenceTransformer", "CrossEncoder", "QdrantClient", "BM25", "bm25s",
                "cosine_similarity", "fitz", "colpali", "OpenAI", "anthropic",
                "ReverseHyde", "chunker", "StatisticalChunker", "situate_context",
                "bge", "MiniLM", "reciprocal", "rrf", "dense", "sparse", "fusion"
            ]):
                print(f"\n--- Code Cell {idx} ---")
                print(src.strip())
        elif ctype == "markdown" and any(h in src for h in ["##", "###"]):
            print(f"\n[MD {idx}]: {src.strip()[:100]}")


if __name__ == "__main__":
    for n in [
        "01_simple_rag.ipynb",
        "02_embedding_model.ipynb",
        "03_semantic_chunking.ipynb",
        "04_contextual_retrieval.ipynb",
        "05_reverse_hyde.ipynb",
        "06_hybrid_search.ipynb",
        "07_reranking.ipynb",
        "08_multimodal_pdf.ipynb",
    ]:
        inspect_notebook(n)
