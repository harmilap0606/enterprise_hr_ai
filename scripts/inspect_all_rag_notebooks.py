"""
scripts/inspect_all_rag_notebooks.py
====================================
Detailed inspection script for all 8 uploaded RAG reference notebooks:
01_simple_rag.ipynb
02_embedding_model.ipynb
03_semantic_chunking.ipynb
04_contextual_retrieval.ipynb
05_reverse_hyde.ipynb
06_hybrid_search.ipynb
07_reranking.ipynb
08_multimodal_pdf.ipynb
"""

import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

notebooks_dir = Path("notebooks/rag")

for nb_path in sorted(notebooks_dir.glob("*.ipynb")):
    print("=" * 80)
    print(f"NOTEBOOK: {nb_path.name}")
    print("=" * 80)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    print(f"Total cells: {len(cells)}")

    markdown_headers = []
    imports = []
    key_lines = []
    code_cells = []

    for idx, c in enumerate(cells):
        src = "".join(c.get("source", []))
        cell_type = c.get("cell_type")
        if cell_type == "markdown":
            for line in src.splitlines():
                if line.startswith("#"):
                    markdown_headers.append(f"  [Cell {idx:02d}] {line}")
        elif cell_type == "code":
            code_cells.append((idx, src))
            lines = src.splitlines()
            for l in lines:
                l_str = l.strip()
                if l_str.startswith("import ") or l_str.startswith("from "):
                    imports.append(l_str)
                if any(k in l_str for k in [
                    "SentenceTransformer", "CrossEncoder", "AutoModel", "AutoTokenizer",
                    "QdrantClient", "chromadb", "BM25Okapi", "fitz", "pymupdf",
                    "OpenAI", "ChatOpenAI", "pipeline", "bge", "MiniLM", "gte",
                    "chunk", "semantic", "hyde", "fusion", "rrf"
                ]):
                    key_lines.append(f"[C{idx:02d}] {l_str}")

    print("\n--- Structure / Markdown Headings ---")
    for h in markdown_headers[:15]:
        print(h)
    if len(markdown_headers) > 15:
        print(f"  ... and {len(markdown_headers) - 15} more headings")

    print("\n--- Unique Imports ---")
    for imp in sorted(set(imports)):
        print(f"  {imp}")

    print("\n--- Key Model / Pipeline / Algorithm Lines ---")
    for kl in sorted(set(key_lines))[:20]:
        print(f"  {kl}")
    if len(set(key_lines)) > 20:
        print(f"  ... and {len(set(key_lines)) - 20} more key lines")
    print()
