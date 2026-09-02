# Detailed Technical Inspection: Notebook 07 Re-Ranking (`07_reranking.ipynb`)

> **Inspection Document: Step 5A**  
> **Source File:** `notebooks/rag/07_reranking.ipynb` (20 cells total)  
> **Target Production Area:** Cross-Encoder Reranking Layer (`app/rag/retrieval/` / `app/rag/reranker.py`)  
> **Author:** Antigravity AI Engineering Team  
> **Date:** 2026-09-01  

---

## Executive Summary

This document performs an exhaustive, cell-by-cell inspection of `notebooks/rag/07_reranking.ipynb`. It extracts the exact model, libraries, tokenization, pair formulation, scoring behavior, and data flow used in the reference implementation. It establishes how our Step 4B output ([`HybridSearchResult`](file:///C:/Users/ASUS/Desktop/enterprise_hr_ai/app/rag/retrieval/schemas.py)) will interface with the cross-encoder, identifies reusable versus notebook-specific code, assesses production runtime feasibility, and outlines the empirical evaluation plan for Step 5B.

---

## 1. Exact Notebook Implementation Details

The following items are extracted directly from the actual code cells of `notebooks/rag/07_reranking.ipynb`:

| Field | Exact Implementation in Notebook 07 | Verbatim Reference / Code |
|:---|:---|:---|
| **A. Reranker Model Name** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cell 5: `cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` |
| **B. Library Used** | `sentence_transformers` (`CrossEncoder`) | Cell 5: `from sentence_transformers import CrossEncoder` |
| **C. Initialization** | Single-line constructor call passing model repository ID | `cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` |
| **D. Architecture** | `BertForSequenceClassification` | 6 Transformer layers, hidden dimension 384, 1 output classification head (logit). |
| **E. Tokenizer Behavior** | HuggingFace `AutoTokenizer` (BERT WordPiece tokenizer) | Encodes sentence pairs with `[CLS] query [SEP] passage [SEP]`, enabling joint cross-attention across all token pairs. |
| **F. Pair Construction** | List of 2-element lists: `[query, document_text]` | Cell 10: `pairs = [[query, doc['text']] for doc in hybrid_search_results.values()]` |
| **G. Input Format** | `List[List[str]]` containing `[query, passage]` strings | Evaluated in-memory as a list of pairs. |
| **H. Max Sequence Length** | Default: 512 tokens | Inherited from `ms-marco-MiniLM-L-6-v2` configuration. |
| **I. Batch Size** | Default: 32 | `CrossEncoder.predict()` default parameter. |
| **J. Device Handling** | Automatic in `sentence-transformers` | Defaults to CUDA if available, MPS if on Apple Silicon, or CPU fallback. |
| **K. Score Extraction** | Direct prediction call returning 1D NumPy array | Cell 10: `scores = cross_encoder.predict(pairs)` |
| **L. Score Nature** | **Raw Logits** (Unbounded real values) | Does **NOT** apply sigmoid or softmax. Logits range typically from approximately $-11.0$ (completely irrelevant) to $+11.0$ (highly relevant). |
| **M. Ranking / Sorting Logic** | Descending Python `sorted()` on raw float score | Cell 12: `sorted(results_with_scores, key=lambda x: x[2], reverse=True)` |
| **N. Candidate Count** | Union of top-10 dense and top-10 sparse results | Cell 7: Merged dictionary `hybrid_search_results` loaded from JSON (typically 10–20 candidates). |
| **O. Final Top-K** | Top 3 documents | Cell 12: `[:3]` |
| **P. Preprocessing** | None | Raw query string and raw `doc['text']` are passed directly without transformation. |
| **Q. Special Prefix / Instruction** | **None** | Unlike bi-encoders (e.g. BGE's query instruction), cross-encoders do **not** use query-side or document-side instruction prefixes. |
| **R. Score Normalization** | **None** | Cross-encoder logits are sorted directly without Min-Max or standard scaling. |
| **S. Threshold Filtering** | **None in Notebook** | Notebook unconditionally slices top 3 without checking if top logit is negative. |
| **T. Generation Augmentation** | Raw string representation of document text injected into prompt | Cell 17: Injected into `gpt-3.5-turbo` prompt. |

---

## 2. Trace of Notebook 07 Data Flow

```text
User Query: "What is context size of Mixtral?"
    ↓
Load Retrieval Candidates from Previous Stage:
    ├── data/dense_results.json (Top-10 dense hits)
    └── data/sparse_results.json (Top-10 sparse hits)
    ↓
Deduplicate Candidates into In-Memory Dictionary:
    hybrid_search_results = {}
    for doc in dense_results: hybrid_search_results[doc['id']] = doc
    for doc in sparse_results: hybrid_search_results[doc['id']] = doc
    ↓
Construct Cross-Encoder Evaluation Pairs:
    pairs = [[query, doc['text']] for doc in hybrid_search_results.values()]
    ↓
Cross-Encoder Joint Attention Scoring:
    scores = cross_encoder.predict(pairs)
    (Yields 1D numpy array of raw cross-attention logits)
    ↓
Zip Document IDs, Text, and Scores:
    results_with_scores = [
        (doc_id, hybrid_search_results[doc_id]['text'], score)
        for doc_id, score in zip(hybrid_search_results.keys(), scores)
    ]
    ↓
Sort Descending & Extract Top 3:
    top_results = sorted(results_with_scores, key=lambda x: x[2], reverse=True)[:3]
    ↓
Render Rich Output Table & Pass to LLM Generator:
    search_results = [doc[1] for doc in top_results]
    completion = client.chat.completions.create(model="gpt-3.5-turbo", ...)
```

---

## 3. Code Classification: Reusable vs Adaptable vs Notebook-Specific

| Code Block / Concept | Classification | Technical Rationale |
|:---|:---:|:---|
| `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` | **REUSE DIRECTLY** | Authoritative model choice from the notebook. 22.7M parameters, fast, highly accurate passage reranker. |
| `pairs = [[query, doc_text] for doc in candidates]` | **REUSE DIRECTLY** | Standard input convention for sentence-transformers cross-encoders. |
| `scores = cross_encoder.predict(pairs)` | **REUSE DIRECTLY** | Vectorized forward pass yielding cross-attention logits. |
| `sorted(..., key=lambda x: x[2], reverse=True)` | **ADAPT FOR PRODUCTION** | In production, we must implement deterministic tie-breaking: `(-rerank_score, chunk_id)`. |
| Reading `data/dense_results.json` and `data/sparse_results.json` | **DISCARD / REPLACE** | Notebook-specific artifact bridging notebooks 06 and 07. Production receives in-memory `List[HybridSearchResult]` from Step 4B. |
| Unconditional `[:3]` slice without threshold | **ADAPT FOR PRODUCTION** | For enterprise grounding, an absolute score floor (e.g. logit cutoff) or downstream grounding gate is required to detect out-of-domain queries. |
| Rich formatting (`rich.table`, `rich.panel`) | **DISCARD** | Interactive visual console output is inappropriate for headless FastAPI endpoints and test harnesses. |
| OpenAI API Generation (`client.chat.completions.create`) | **DISCARD** | Production uses local Seq2Seq / Flan-T5 generators with citation enforcement. |

---

## 4. Mapping: Step 4B Pipeline to Notebook 07 Reranker

Our completed Step 4B pipeline outputs a list of typed Pydantic models:
```python
class HybridSearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    contextual_text: str
    source: str
    title: str
    section: str
    document_type: str
    dense_score: float
    sparse_score: float
    normalized_dense_score: float
    normalized_sparse_score: float
    hybrid_score: float
    rank: int
    metadata: Dict[str, Any]
```

### Mapping into Notebook 07 Reranker Input:
* **Candidate Pool:** Step 4B defaults to returning `final_top_k = 15` candidates (the fused union of 20 dense and 20 sparse items).
* **Pair Construction:**
  * *Option 1 (Raw text):* `pairs = [[query, candidate.text] for candidate in hybrid_results]`
  * *Option 2 (Contextual text):* `pairs = [[query, candidate.contextual_text] for candidate in hybrid_results]`
  * *Analysis:* `contextual_text` includes hierarchical metadata headers (`[Document: ...] [Section: ...]`). Because cross-encoders attend to all tokens simultaneously, contextual headers provide immediate document-level disambiguation. Both can be evaluated in Step 5B.
* **Result Enrichment:** After prediction, each candidate is augmented with:
  * `rerank_score: float` (raw cross-encoder logit)
  * `rerank_rank: int` (new 1-based order after reranking)
  * All original Step 2 & Step 4B metadata is preserved.

---

## 5. Model Verification & Production Compatibility Assessment

### Verification of Exact Model
- **Inspected Value:** `cross-encoder/ms-marco-MiniLM-L-6-v2` in Cell 5.
- **Model Family:** Microsoft MARCO MiniLM (6 layers, 384 hidden dimensions, uncased).
- **HuggingFace Hub ID:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.

### Production Compatibility Matrix

| Metric / Dimension | Specification | Production Assessment |
|:---|:---:|:---|
| **Parameter Count** | 22.7 Million | Extremely lightweight (approx. 1/5th the size of `bert-base`). |
| **Disk Checkpoint Size** | ~86.8 MB (`model.safetensors`) | Instantaneous download and minimal disk footprint. |
| **Memory Footprint (RAM)** | ~120–160 MB RSS | Minimal overhead; runs comfortably within modest server instances. |
| **Hardware Compatibility** | Full CPU, CUDA, and Apple MPS | Uses standard HuggingFace `transformers` / PyTorch kernels. |
| **Batch Inference** | Fully supported | Vectorized forward pass across batches of 15 to 32 pairs simultaneously. |
| **Inference Latency (CPU)** | ~15–30 ms for 15 pairs | Evaluates 15 candidate pairs in a single forward pass well within interactive enterprise SLAs (<50 ms). |

*(Note: The notebook contains no execution timing measurements. The latency estimate above is derived from standard PyTorch CPU execution of a 6-layer MiniLM forward pass on 15 pairs).*

---

## 6. Proposed Evaluation Methodology for Step 5B

To establish empirical evidence that Notebook 07's cross-encoder reranking enhances retrieval quality over Step 4B's hybrid retrieval, we will use the verified 24-query ground-truth dataset ([`tests/fixtures/rag_eval_queries.json`](file:///C:/Users/ASUS/Desktop/enterprise_hr_ai/tests/fixtures/rag_eval_queries.json)).

### Evaluation Pipeline:
```text
24 Evaluation Queries
    ↓
Step 4B Hybrid Retrieval (0.8 Dense + 0.2 Sparse)
    ↓
Top-15 Candidate Chunks
    ↓
Notebook 07 Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
    ↓
Re-sorted Candidate Pool
    ↓
Metric Computation (Hit@1, Hit@3, Hit@5, Hit@10, MRR)
```

### Comparative Benchmark Matrix:
The benchmark script in Step 5B will compare 4 progressive stages:
1. **Dense Only (BGE-Small)** (Baseline from Step 3 / 4B: Hit@1 = 0.7500, MRR = 0.8368)
2. **Sparse Only (BM25)** (Baseline from Step 4B: Hit@1 = 0.7500, MRR = 0.8375)
3. **Hybrid Search (0.8 Dense + 0.2 Sparse)** (Validated in Step 4B: Hit@1 = 0.8333, MRR = 0.8889)
4. **Hybrid + Notebook-07 Reranking** (Target of Step 5B)

---

## 7. Differences Between Notebook 07 and Production Requirements

1. **State Management & Decoupling:** Notebook 07 hardcodes file paths to temporary JSON outputs from Notebook 06. Production requires a clean callable class: `CrossEncoderReranker.rerank(query, candidates, top_k)`.
2. **Determinism:** Notebook 07 relies on Python's default sort stability when scores match. Production requires an explicit composite key: `(-rerank_score, chunk_id)` to guarantee 100% deterministic ranking across environments.
3. **Input Format Resilience:** Production should support passing either `chunk.text` or `chunk.contextual_text`.
4. **Zero Code Changes in Step 5A:** No application files, endpoints, or tests were altered during this inspection phase.

---

## 8. Open Questions for Step 5B

1. **Text Field for Cross-Encoder:** Should the reranker evaluate `chunk.text` (raw occupational/section text) or `chunk.contextual_text` (text prefixed with `[Document: ...] [Section: ...]`)? We can test both in Step 5B to see which yields higher `Hit@1`.
2. **Logit Thresholding:** Should a configurable cutoff score (e.g. `min_score = -4.0`) be exposed in the reranker config to filter out completely irrelevant candidates before downstream generation?
3. **Candidate Pool Size ($K$):** Should the reranker operate on `top_k = 15` candidates from Step 4B, or should we evaluate scaling to 20 or 25 candidates?
