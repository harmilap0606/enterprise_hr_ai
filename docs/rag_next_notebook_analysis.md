# Next RAG Notebook Analysis: Notebook 08 — Multimodal PDF Retrieval (`08_multimodal_pdf.ipynb`)

> **Specification Document: Step 7A**  
> **Source File:** `notebooks/rag/08_multimodal_pdf.ipynb` (42 cells total)  
> **Target Production Area:** Multimodal Document Ingestion & Retrieval Layer (`app/rag/multimodal/`)  
> **Author:** Antigravity AI Engineering Team  
> **Date:** 2026-09-01  

---

## 1. Notebook Identification

* **Notebook Filename:** `notebooks/rag/08_multimodal_pdf.ipynb`
* **Notebook Number:** `08`
* **Notebook Title / Purpose:** *"Indexing and searching image based documents (using ColPali with Qdrant)"*
* **Total Cells:** 42 cells (17 Markdown cells, 25 Code cells)
* **Status in RAG Pipeline Progression:** Represents the 8th and final notebook in the reference RAG series, demonstrating visual multi-vector document retrieval for complex layout-heavy PDF documents.

---

## 2. Notebook Purpose

Traditional text-based RAG relies heavily on text extractors (`pypdf`, `pymupdf`, OCR) which discard visual layouts, column hierarchies, figures, charts, and table alignments. 

Notebook 08 implements an end-to-end **Vision-Language RAG pipeline**:
1. It eliminates text extraction entirely by rendering PDF pages directly as raster images.
2. It generates multi-vector patch embeddings using **ColPali** (a vision model based on Google's PaliGemma-3B architecture).
3. It indexes multi-vectors in **Qdrant** using INT8 scalar quantization and MaxSim (late interaction) similarity.
4. It executes user queries directly against visual page representations without intermediate text conversion.
5. It feeds retrieved high-resolution page images to a vision-language generation model (**Claude 3.5 Sonnet**) to answer user queries with direct visual grounding.

---

## 3. Actual Implementation Details

The implementation across Cells 4–41 exhibits the following technical parameters:

| Parameter / Dimension | Notebook 08 Implementation | Code Reference |
|:---|:---|:---|
| **A. Problem Solved** | Document retrieval on layout-dense, diagrammatic PDFs where text extraction fails | Cell 5 markdown |
| **B. Inputs** | PDF files in local folders (`data/shokz/` product manuals) + raw text queries | Cell 6, 30 |
| **C. Outputs** | Top-3 retrieved page images (PNG) + multimodal Claude synthesis | Cell 36, 40 |
| **D. Models Used** | 1. Vision Retriever: `vidore/colpali-v1.2`<br>2. Vision Processor: `vidore/colpaligemma-3b-pt-448-base`<br>3. Multimodal Generator: `anthropic(model="claude-3-5-sonnet-20241022")` | Cell 10, 40 |
| **E. Libraries Used** | `pdf2image`, `colpali_engine`, `qdrant_client`, `torch`, `stamina`, `anthropic`, `matplotlib`, `base64`, `tqdm`, `rich` | Cells 6, 10, 15, 22, 40 |
| **F. Prompts Used** | Standard Anthropic multimodal message containing base64 image + user query text | Cell 40 |
| **G. Preprocessing** | `convert_from_path(pdf_path, poppler_path=...)` converting PDF pages to PIL images | Cell 6 |
| **H. Retrieval Mechanism** | Multi-Vector Late Interaction (ColBERT-style MaxSim on visual patch tokens) | Cell 17 |
| **I. Ingestion Batch Size** | `batch_size = 2` | Cell 24 |
| **J. Quantization** | Qdrant INT8 scalar quantization (`quantile=0.99`, `always_ram=False`) | Cell 19 |
| **K. Vector Dimensions** | Multi-vector: 1,024 token vectors per page, each vector having 128 dimensions | Cells 12–14 |
| **L. Comparator** | `models.MultiVectorComparator.MAX_SIM` | Cell 17 |
| **M. Search Top-K** | `limit = 3` | Cell 33 |
| **N. Retries / Resilience** | `stamina.retry(on=Exception, attempts=3)` wrapping upserts | Cell 22 |
| **O. Determinism** | Deterministic indexing given constant image resolution (448x448) and precision | PyTorch standard |

---

## 4. Models and Libraries

### 1. ColPali Vision-Language Model
* **Model ID:** `vidore/colpali-v1.2`
* **Base Architecture:** Google PaliGemma 3B (3 billion parameters: SigLIP vision encoder + Gemma-2B language backbone).
* **Processor ID:** `vidore/colpaligemma-3b-pt-448-base`.
* **Output:** For an input image, it outputs a matrix of shape `(1, 1024, 128)`—representing 1,024 visual tokens embedded in a 128-dimensional late interaction space.
* **Precision:** `torch.bfloat16` mapped to `"mps"` (Apple Silicon) or `"cuda"`.

### 2. Qdrant Vector Database
* **Instance:** Ephemeral `:memory:` in notebook (`QdrantClient(":memory:")`).
* **Vector Configuration:** Multi-vector configuration with MaxSim similarity:
  $$\text{Score}(Q, D) = \sum_{q \in Q} \max_{d \in D} (q \cdot d)$$
* **Storage Optimization:** INT8 scalar quantization reducing vector RAM consumption by ~75%.

### 3. Claude 3.5 Sonnet
* **API:** `anthropic.Anthropic()`.
* **Model:** `claude-3-5-sonnet-20241022`.
* **Input Payload:** Base64 encoded PNG of the top-1 retrieved page image alongside the user question.

---

## 5. Exact Data Flow

```text
INDEX-TIME (OFFLINE):
PDF Documents (*.pdf in folder)
    ↓
Convert pages to images: pdf2image.convert_from_path(pdf, poppler_path=...)
    ↓
Batch Images (batch_size=2):
    ↓
ColPali Processor: colpali_processor.process_images(images)
    ↓
ColPali Forward Pass: image_embeddings = colpali_model(**batch_images)
(Yields multivector of shape: [batch, 1024, 128])
    ↓
Prepare Qdrant PointStructs:
    id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}.{page_idx}")
    vector = multivector (list of 1,024 128-dim float lists)
    payload = {"doc": pdf_file, "page": page_number}
    ↓
Upsert into Qdrant Collection ("user-guides") with INT8 Scalar Quantization
(Index size = Total Pages * 1,024 vectors)

QUERY-TIME (ONLINE):
User Query String: "Why the led is flashing red and blue?"
    ↓
ColPali Processor: batch_query = colpali_processor.process_queries([query_text])
    ↓
ColPali Query Forward Pass: query_embedding = colpali_model(**batch_query)
(Yields multivector of shape: [1, query_tokens, 128])
    ↓
Qdrant MultiVector MaxSim Search:
    search_result = qdrant_client.query_points(query=multivector_query, limit=3)
    ↓
Extract Top Page Image:
    point = search_result.points[0]
    image = all_images[point.payload["doc"]][point.payload["page"] - 1]
    ↓
Encode Image to Base64 (PNG format)
    ↓
Call Anthropic API (Claude 3.5 Sonnet):
    Messages: [{"role": "user", "content": [{"type": "image", ...}, {"type": "text", "text": query}]}]
    ↓
Return Generated Grounded Answer
```

---

## 6. Parameters and Configuration

* **Image Input Resolution:** Resized by processor to 448x448 pixels (PaliGemma standard).
* **Visual Patch Count:** $32 \times 32 = 1,024$ image tokens per page.
* **Vector Dimension:** 128 dimensions per token.
* **Vector Distance Metric:** Cosine similarity with MaxSim late interaction comparator.
* **Quantization:** `ScalarQuantizationConfig(type=INT8, quantile=0.99, always_ram=False)`.
* **Search Limit:** `limit = 3` pages.
* **Generation Token Limit:** `max_tokens = 1024`.

---

## 7. Index-Time vs. Query-Time Operations

| Phase | Operations | Resource Footprint |
|:---|:---|:---|
| **Index-Time** | 1. PDF rasterization via `pdf2image` (requires OS-level Poppler binary).<br>2. Forward pass through 3B PaliGemma vision model.<br>3. Ingestion of 1,024 multi-vectors per page into Qdrant. | **Extremely Heavy:** Requires 6–8 GB GPU VRAM (or Apple MPS). On CPU, encoding a single page takes 5–10 seconds. |
| **Query-Time** | 1. Forward pass of query text through ColPali language backbone.<br>2. MaxSim multi-vector search over all indexed page patches.<br>3. Base64 encoding of top page image.<br>4. Cloud API roundtrip to Anthropic Claude 3.5 Sonnet. | **Moderate to Heavy:** Query encoding takes ~100–300 ms on GPU (1–2s on CPU). MaxSim search across thousands of patches requires specialized vector indexing. API call adds 1–3s. |

---

## 8. Code Reuse Classification

| Code Segment | Classification | Technical Rationale |
|:---|:---:|:---|
| `pdf2image.convert_from_path` | **ADAPT LATER** | Reliable PDF rasterization, but requires host `poppler` binaries that are not installed on standard production environments. |
| `ColPali.from_pretrained("vidore/colpali-v1.2")` | **ADAPT LATER** | State-of-the-art visual retrieval, but 3B parameter weights cannot execute efficiently within our CPU environment. |
| `models.MultiVectorConfig(comparator=MAX_SIM)` | **ADAPT LATER** | Standard late-interaction vector configuration for Qdrant. Not natively supported in ChromaDB. |
| Matplotlib visualization (`plt.subplots`) | **DISCARD** | Interactive notebook plotting code; unusable in automated API endpoints. |
| Hardcoded IKEA/Shokz folder paths | **DISCARD** | Notebook-specific demo paths. |
| Claude 3.5 Sonnet image base64 prompt | **REUSE / ADAPT** | Standard multimodal prompt formulation; adaptable for any vision-language LLM. |

---

## 9. Relationship to Current Pipeline

Our current validated production pipeline is:
```text
User Query
    ↓
BGE Dense + BM25
    ↓
Min-Max normalization (0.8 Dense + 0.2 Sparse)
    ↓
Top 15 Candidates
    ↓
Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    ↓
Top 3 Candidates
```

### Architectural Fit:
* Notebook 08 is **independent of our current text retrieval pipeline**.
* It does **NOT** fit between hybrid retrieval and reranking, nor does it fit before/after BGE.
* It represents a **separate, parallel multimodal retrieval branch** that activates only when querying unstructured, visually formatted documents (such as PDF employee handbooks, scanned insurance claim forms, or benefits booklets).
* For our current structured corpus (O\*NET occupation descriptions, role mapping tables, model card, and relational architecture markdown), text-based hybrid retrieval + cross-encoder reranking is fundamentally more accurate, faster, and cheaper.

---

## 10. Redundancy / Complementarity Analysis

* **Against BGE Dense Retrieval:** **COMPLEMENTARY (Different Modality).** BGE is optimized for dense passage text semantics; ColPali is optimized for document page images containing layouts and figures.
* **Against BM25 Sparse Search:** **COMPLEMENTARY.** BM25 matches exact lexical tokens; ColPali matches visual-spatial representations.
* **Against Cross-Encoder Reranking:** **ALTERNATIVE.** ColPali performs late interaction (MaxSim) at retrieval time; Cross-Encoders perform full cross-attention at rerank time.
* **Against Current Project Corpus:** **NOT USEFUL CURRENTLY.** Our verified corpus consists strictly of CSV tables (`occupation_master.csv`, `jobrole_onet_mapping.csv`) and Markdown files (`model_card.md`, `data_relationships.md`). Zero PDF manuals exist in the project knowledge base.

---

## 11. Production Cost Analysis

| Dimension | Notebook 08 Profile | Current Step 5B Production Profile |
|:---|:---|:---|
| **Model Size** | 3.0 Billion parameters (`colpaligemma-3b`) | 22.7 Million parameters (`MiniLM`) + 33M (`BGE-small`) |
| **Disk Storage** | ~6.5 GB model weights + Poppler binaries | ~220 MB total for all models |
| **RAM / VRAM** | 8 GB+ VRAM required (CUDA/MPS); unviable on CPU | ~250 MB RAM on CPU |
| **Vector Multiplier** | **1,024 vectors per single page** | **1 vector per chunk** |
| **Query Latency** | 1,500–3,500 ms (including vision LLM API call) | 21.7 ms (Hybrid) + 516 ms (Rerank on CPU) |
| **API Dependencies** | Anthropic Claude 3.5 Sonnet (paid external API) | 100% offline self-contained (local inference) |

---

## 12. Evaluation Feasibility against `rag_eval_queries.json`

Can Notebook 08 be evaluated against our current 24 ground-truth queries?
- **NO.**
- **Reason:** All 24 evaluation queries in `tests/fixtures/rag_eval_queries.json` target specific SOC codes (`19-1042.00`), attrition decision thresholds (`0.40`), SHAP drivers, and role mappings residing in CSV and Markdown files.
- There are no PDF pages in the repository for ColPali to index, nor would visual page embeddings add value to clean relational CSV tables.

---

## 13. Production Recommendation

### Classification: **OPTIONAL EXPERIMENT / ADAPT LATER (Category C: Optional Multimodal Branch)**

### Evidence-Based Rationale:
1. **Corpus Incompatibility:** The Enterprise HR AI platform knowledge corpus contains structured tabular data and technical markdown documentation. It contains zero PDF files.
2. **Resource Prohibitive:** Running a 3-billion-parameter vision-language model (`ColPali`) on a standard CPU server environment introduces massive memory overhead and unacceptable latency (~5–10s per page ingestion).
3. **External Dependencies:** Requires host-level `poppler` binaries and external Anthropic API tokens.
4. **Current Pipeline Superiority for Text:** Our current Step 5B pipeline delivers **0.9514 MRR, 0.9167 Hit@1, and 100% Hit@3/5/10** with 100% offline self-containment and zero external costs.
5. **Future Strategic Value:** ColPali should be archived as a dedicated blueprint in the project documentation. If scanned employee benefit handbooks or insurance guidebooks (with complex visual plan comparison tables) are uploaded in future enterprise phases, the ColPali branch can be activated specifically for those PDF collections.

---

## 14. Open Questions

1. **Enterprise PDF Ingestion Policy:** If clients upload 100-page HR benefit booklets in PDF format in the future, should we convert them via visual ColPali, or parse them into structured Markdown tables using document intelligence tools?
2. **Local Vision LLM Alternative:** Can Anthropic's Claude 3.5 Sonnet be replaced by a local lightweight vision model (e.g. `Qwen2-VL-2B-Instruct` or `PaliGemma-3B`) if visual PDF querying becomes a mandatory offline requirement?
3. **Vector Store Compatibility:** Since ChromaDB does not natively support multi-vector MaxSim indexing, would activating ColPali require deploying a separate Qdrant service container in production?
