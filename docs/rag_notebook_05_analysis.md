# Notebook 05 — Reverse HyDE Analysis

> **Specification Document: Step 6A**  
> **Source File:** `notebooks/rag/05_reverse_hyde.ipynb` (26 cells total)  
> **Target Production Area:** Document Enrichment & Indexing Architecture (`app/rag/`)  
> **Author:** Antigravity AI Engineering Team  
> **Date:** 2026-09-01  

---

## 1. Notebook Overview

Notebook `05_reverse_hyde.ipynb` explores an alternative strategy to improve semantic similarity between short user queries and longer document passages. 

Traditional **HyDE** (Hypothetical Document Embeddings, [Gao et al., 2022](https://arxiv.org/abs/2212.10496)) operates at **query time**: an incoming user query is expanded by prompting an LLM to generate a hypothetical *document*, which is then embedded to retrieve real documents.

In contrast, **Reverse HyDE** as implemented in Notebook 05 operates exclusively at **index time**:
- Rather than waiting for a user query, each document chunk in the corpus is processed offline by an LLM.
- The LLM generates $N$ hypothetical *questions* that the chunk effectively answers.
- The generated questions are embedded and indexed into the vector database, pointing back to the parent chunk.
- At query time, the user's query is searched against the indexed *questions*, transforming an asymmetric query-to-passage search into a symmetric query-to-question similarity search.

---

## 2. Actual Notebook Implementation

The core implementation in Cell 6 and Cells 14–24 consists of:

### A. The `ReverseHyde` Class
```python
class ReverseHyde:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.model = "text-embedding-ada-002"

    def get_embedding(self, text: str) -> List[float]:
        client = openai.OpenAI()
        response = client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    def generate_reverse_hyde(self, chunk: str, n: int = 3) -> List[str]:
        prompt = f"""
Given the following text chunk, generate {n} different questions that this chunk would be a good answer to:

Chunk: {chunk}

Questions (enumarate the questions with 1. 2., etc.):
"""
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7,
        )
        questions = response.choices[0].message.content.strip().split('\n')
        return [q.split('. ', 1)[1] for q in questions if '. ' in q]

    def process_chunks(self, chunks: List[str], n: int = 3) -> Dict[str, List[str]]:
        processed_chunks = {}
        for chunk in chunks:
            processed_chunks[chunk] = self.generate_reverse_hyde(chunk, n)
        return processed_chunks
```

### B. Vector Indexing of Hypothetical Questions (Cell 16)
```python
qdrant.upload_points(
    collection_name=hyde_collection_name,
    points=[
        models.PointStruct(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"{d_idx}-{q_idx}").hex,
            vector=encoder.encode(question).tolist(),
            payload={ 
                "document": document, 
                "doc_id": d_idx
            }
        ) for d_idx, (document, questions) 
            in enumerate(processed_chunks.items()) 
                for q_idx, question in enumerate(questions)
    ]
)
```

### C. Querying the Enriched Index (Cell 19)
```python
hits = qdrant.search(
    collection_name=collection_name,
    query_vector=encoder.encode(query).tolist(),
    limit=limit
)
# Returns hit.payload['document']
```

---

## 3. Models and Libraries

| Component | Technology in Notebook 05 | Configuration / Parameters |
|:---|:---|:---|
| **Question Generation LLM** | `openai.OpenAI()` (`gpt-3.5-turbo`) | `temperature=0.7`, `max_tokens=100`, `n=1` |
| **Embedding Model** | `SentenceTransformer('all-MiniLM-L6-v2')` | 384 dimensions, Cosine distance (`text-embedding-ada-002` defined in helper but unused in Qdrant) |
| **Vector Database** | `qdrant_client.QdrantClient(":memory:")` | In-memory ephemeral collection `reverse_hyde` |
| **Unique Point Generation** | Python standard `uuid.uuid5` | `uuid.uuid5(uuid.NAMESPACE_URL, f"{d_idx}-{q_idx}").hex` |
| **Supporting Libraries** | `rich`, `dotenv`, `numpy`, `sklearn` | Used for output formatting and warning suppression |

---

## 4. Exact Data Flow

```text
INDEX-TIME (Offline Generation & Storage):
Document Chunks: [chunk_0, chunk_1, ..., chunk_M]
    ↓
Iterate each chunk:
    ↓
Call OpenAI API (gpt-3.5-turbo):
    Prompt: "Given the following text chunk, generate {n} different questions..."
    ↓
Parse text response into list of strings: [q_i_1, q_i_2, ..., q_i_n]
    ↓
Iterate each generated question:
    vector = encoder.encode(q_i_j)
    PointStruct(id=uuid, vector=vector, payload={"document": chunk_text, "doc_id": i})
    ↓
Upsert into Qdrant collection: "reverse_hyde"
(Index size = M * N vectors)

QUERY-TIME (Online Retrieval):
User Query: "What generates energy in a cell?"
    ↓
Embed user query: query_vector = encoder.encode(query)
    ↓
Vector Similarity Search against Question Index:
    hits = qdrant.search(collection_name="reverse_hyde", query_vector=query_vector, limit=K)
    ↓
Extract parent document text from payload:
    result = hit.payload["document"]
```

---

## 5. Prompt Architecture

### System Prompt
```text
You are a helpful assistant.
```

### User Prompt
```text
Given the following text chunk, generate {n} different questions that this chunk would be a good answer to:

Chunk: {chunk}

Questions (enumarate the questions with 1. 2., etc.):
```

### Analysis of Prompt Characteristics:
* **Variables:** `{n}` (integer count of questions, default 3, set to 5 in Cell 10); `{chunk}` (raw chunk text).
* **Format Expectation:** Numbered list (`1. `, `2. `) parsed via `q.split('. ', 1)[1]`.
* **Determinism:** Non-deterministic (`temperature=0.7`). Re-running ingestion produces different synthetic questions.

---

## 6. Index-Time vs Query-Time Operations

| Operation Phase | Notebook 05 Actions | Cost / Latency Profile |
|:---|:---|:---|
| **Index-Time (Offline)** | 1. Prompt LLM for each chunk ($M$ calls).<br>2. Parse questions ($M \times N$ questions).<br>3. Embed all questions ($M \times N$ embeddings).<br>4. Upsert vectors into database. | **High computation & API cost.** Scales linearly with corpus size ($M$) and multiplier ($N$). |
| **Query-Time (Online)** | 1. Embed single user query string.<br>2. Perform single vector similarity search.<br>3. Extract parent document payload. | **Zero additional LLM calls.** Latency is identical to standard single-vector search (~2–5 ms). |

---

## 7. Relationship to Current Hybrid Retrieval

In our verified Step 4B architecture, retrieval consists of:
```text
User Query
    ├── Dense: ChromaDB with BGE-small-en-v1.5 (Top 20)
    └── Sparse: BM25Okapi over Step 2 chunks (Top 20)
    ↓
Candidate Union (Top 15 unique chunks)
    ↓
Min-Max Normalization & Weighted Fusion (0.8 Dense + 0.2 Sparse)
```

### Potential Integration Points for Reverse HyDE:
1. **Replacement for Dense Document Index:** Replace the 1,042 chunk vectors with 5,210 question vectors in ChromaDB.
2. **Supplemental Dense Collection:** Query both the chunk collection (`enterprise_hr_knowledge_bge`) AND a question collection (`enterprise_hr_questions_bge`), pooling candidates.
3. **Deduplication Requirement:** Because multiple questions originate from the same chunk, a top-20 search across 5,210 questions may retrieve 4–5 variations of the *same* document, reducing candidate diversity unless collapsed by `chunk_id` before hybrid fusion with BM25.

---

## 8. Relationship to Cross-Encoder Reranking

In Step 5B, we validated:
```text
Top 15 Hybrid Candidates
    ↓
Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
    ↓
Top 3 Reranked Results
```

### Cross-Encoder Interaction:
* Cross-encoders take `[query, document_text]` and perform joint cross-attention over all tokens simultaneously.
* The Cross-Encoder evaluated in Step 5B already achieved **Hit@1 = 0.9167, Hit@3 = 1.0000, and MRR = 0.9514** when evaluated directly on `candidate.text`.
* Because the Cross-Encoder directly solves the query-to-document vocabulary mismatch through deep cross-attention, the need for index-time hypothetical question generation is largely superseded.

---

## 9. Production Adaptation Requirements

If Reverse HyDE were to be adapted for production:
1. **Remove External OpenAI Dependency:** Must replace `OpenAI(model="gpt-3.5-turbo")` with a local open-weight model (e.g., local Flan-T5, Mistral, or Qwen) or pre-generated deterministic question artifacts to avoid external cloud API dependencies.
2. **Chunk ID Mapping & Deduplication:** Point structs in ChromaDB must store `chunk_id` as metadata, and dense retrieval must deduplicate hits by `chunk_id` (taking the max similarity score across questions for that chunk) before returning the top 20 to the hybrid merger.
3. **Deterministic Generation:** Set `temperature=0.0` or persist generated question JSON artifacts so the index is 100% reproducible.
4. **Preserve Metadata:** The parent chunk metadata (`soc_code`, `source`, `title`, `section`, `document_type`) must remain accessible when a question vector matches.

---

## 10. Cost and Latency Considerations

### Computational & Storage Analysis for Our Corpus (1,042 Chunks):
* **LLM Calls during Ingestion:** Exactly **1,042 LLM calls**.
* **Synthetic Questions Generated:** At $N=5$, produces **5,210 questions**.
* **Vector Store Expansion:** 
  * Current ChromaDB collection: 1,042 vectors.
  * Reverse HyDE ChromaDB collection: 5,210 vectors (**5x storage multiplier**).
* **Token Costs (if using external API):**
  * Ingestion tokens: ~150 prompt tokens + ~80 completion tokens = ~230 tokens per chunk.
  * Total tokens: $\approx 240,000$ tokens per full corpus build.
* **Query Latency Impact:**
  * **0 ms additional LLM latency** (since questions are pre-indexed).
  * Vector search over 5,210 items vs 1,042 items adds $< 2\text{ ms}$ on CPU.

---

## 11. Proposed Evaluation Methodology

To isolate and measure the empirical effect of Reverse HyDE:

### Comparative Benchmark:
1. **Pipeline A (Step 5B Validated Baseline):**
   `Query -> BGE Dense (1,042 chunks) + BM25 -> Hybrid 0.8/0.2 (Top 15) -> Cross-Encoder -> Top 10`
   * Baseline Metrics: Hit@1 = 0.9167, Hit@3 = 1.0000, MRR = 0.9514.
2. **Pipeline B (Hybrid with Reverse HyDE Dense):**
   `Query -> Reverse HyDE Dense (5,210 question vectors, collapsed by chunk_id) + BM25 -> Hybrid 0.8/0.2 (Top 15) -> Cross-Encoder -> Top 10`

### Evaluation Dataset:
- [`tests/fixtures/rag_eval_queries.json`](file:///C:/Users/ASUS/Desktop/enterprise_hr_ai/tests/fixtures/rag_eval_queries.json) (24 queries across O\*NET occupations, role crosswalk, model governance, and data architecture).

### Key Metrics:
- `Hit@1`, `Hit@3`, `Hit@5`, `Hit@10`, `MRR`
- Total indexing time delta
- Storage footprint delta
- Query retrieval latency delta

---

## 12. Architectural Recommendation

### Classification: **OPTIONAL EXPERIMENT / ADAPT LATER**

### Technical Rationale:
1. **Near-Ceiling Baseline Accuracy:** Our Step 5B pipeline already achieves an outstanding **MRR of 0.9514 and Hit@1 of 0.9167**, with a perfect **Hit@3, Hit@5, and Hit@10 of 1.0000 (100%)**. There are only 2 queries out of 24 that are not at Rank 1.
2. **Index-Time Complexity & Cost:** Generating 5,210 questions requires 1,042 sequential LLM calls. If run locally on CPU, this would take 30–60 minutes of heavy generation time during every corpus build.
3. **Cross-Encoder Already Solves Vocabulary Mismatch:** The primary benefit of Reverse HyDE—bridging the phrasing gap between user queries and descriptive text—is already achieved with higher fidelity by our Cross-Encoder (`ms-marco-MiniLM-L-6-v2`), which directly compares token interactions with full cross-attention.
4. **Non-Determinism Risk:** Introducing LLM-generated synthetic questions into what is currently a strictly audited, deterministic corpus introduces potential hallucinated queries.

---

## 13. Open Questions

1. **Local LLM for Offline Ingestion:** If we decide to benchmark Reverse HyDE, should we generate the questions using a small local model (e.g. `flan-t5-base` / `Qwen2.5-0.5B`) to maintain 100% offline self-containment?
2. **Dense Fusion vs Replacement:** Should question embeddings *replace* chunk embeddings, or should we maintain dual dense collections (`chunks` + `questions`) and merge them?
3. **Query Instructions:** When searching against hypothetical questions, should we omit BGE's asymmetric passage prefix (`"Represent this sentence for searching relevant passages: "`) since question-to-question similarity is symmetric?
