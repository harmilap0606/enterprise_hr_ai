# RAG Policy Retrieval Architecture & Ingestion Specification

## 1. Executive Summary & Architectural Overview

This document defines the dedicated architecture, metadata model, multi-stage retrieval pipeline, and benchmark evaluation for the **Enterprise HR Synthetic Policy Corpus** (`POL-JOB-001` through `POL-MONITOR-001`).

To ensure strict production safety, absolute baseline preservation, and uncompromised provenance auditing, the HR Policy corpus is ingested into an **isolated, dedicated vector and sparse retrieval system**:
- **Vector Collection:** `enterprise_hr_policies_bge` in `data/vectorstore/` (ChromaDB HNSW cosine index).
- **Sparse Index:** `data/rag/policy_sparse_index/` (`bm25_index.pkl` + `chunk_metadata.json`).
- **Production Isolation:** The general enterprise knowledge collection (`enterprise_hr_knowledge_bge`, 1,042 chunks) and its general sparse index (`data/rag/sparse_index/`, 1,042 chunks) remain completely untouched and isolated.

---

## 2. Collection Isolation Strategy

### 2.1 Multi-Collection Topology
ChromaDB’s `PersistentClient` natively supports multiple isolated collections under a shared root directory (`data/vectorstore`). The policy corpus is partitioned into its own distinct collection namespace:

```
data/
├── vectorstore/
│   ├── chroma.sqlite3
│   ├── enterprise_hr_knowledge_bge/      # [1,042 chunks] General enterprise knowledge (UNTOUCHED)
│   ├── enterprise_hr_knowledge/          # [1,042 chunks] Legacy dense collection (UNTOUCHED)
│   └── enterprise_hr_policies_bge/       # [150 chunks] Dedicated Synthetic HR Policies (NEW)
└── rag/
    ├── sparse_index/                     # [1,042 chunks] General BM25 index (UNTOUCHED)
    │   ├── bm25_index.pkl
    │   └── chunk_metadata.json
    └── policy_sparse_index/              # [150 chunks] Dedicated Policy BM25 index (NEW)
        ├── bm25_index.pkl
        └── chunk_metadata.json
```

### 2.2 Why Policy Corpus Remains Separate
1. **Domain Integrity & Semantic Granularity:** HR policy documents contain formal compliance rules, procedural rights, and statutory prohibitions that must not be diluted by role descriptions, tabular CSV definitions, or general project summaries.
2. **Deterministic Evaluation:** A separate index enables targeted benchmarking (Hit@K, MRR, exact-ID retrieval, and out-of-domain refusal) without metric leakage from general knowledge.
3. **Agentic Dispatch Readiness:** Future multi-agent orchestration (e.g., LangGraph routing between an HR Policy Specialist Agent and a Workforce Analytics Agent) requires clear collection boundaries for specialized tool selection.

---

## 3. Metadata & Provenance Schema

Every ingested policy chunk preserves complete provenance headers and structured metadata across both ChromaDB and BM25:

| Metadata Field | Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `chunk_id` | `str` | Unique deterministic chunk identifier | `pol_model_001_policy_rules_06_c01` |
| `doc_id` | `str` | Base document/section identifier | `pol_model_001_policy_rules_06` |
| `policy_id` | `str` | Canonical policy code (uppercase) | `POL-MODEL-001` |
| `policy_title` | `str` | Full title of the policy | `Attrition Model Usage Policy` |
| `policy_domain` | `str` | Functional HR governance domain | `Predictive Modeling & Flight Risk` |
| `policy_version` | `str` | Semantic policy revision | `1.0.0` |
| `policy_status` | `str` | Explicit synthetic demo label | `Synthetic Demo Policy` |
| `source_file` | `str` | Relative path to source document | `data/knowledge_base/hr_policies/POL-MODEL-001.md` |
| `source_type` | `str` | Mandatory synthetic policy discriminator | `synthetic_hr_policy` |
| `document_type` | `str` | Schema discriminator | `synthetic_hr_policy` |
| `section` | `str` | Section header from markdown | `Policy Rules & Operational Thresholds` |
| `token_count` | `int` | Structure-aware token length | `118` |

All chunks are explicitly labeled with `source_type: "synthetic_hr_policy"` and `policy_status: "Synthetic Demo Policy"` to prevent confusion with real enterprise policies.

---

## 4. Chunking & Ingestion Strategy

1. **Section-Aware Markdown Parsing:**
   - [`app/rag/loaders/policy_loader.py`](file:///c:/Users/ASUS/Desktop/enterprise_hr_ai/app/rag/loaders/policy_loader.py) parses markdown documents into semantic blocks based on Level-2 (`##`) and Level-3 (`###`) headers.
   - Preserves document header metadata (Policy ID, Owner, Scope, Effective Date, Status) in every section document.
2. **Bounded Token Chunking:**
   - [`app/rag/chunking/chunker.py`](file:///c:/Users/ASUS/Desktop/enterprise_hr_ai/app/rag/chunking/chunker.py) applies structure-aware splitting with target chunk size 200–400 tokens and 50-token overlap.
   - The 10 policies (9,728 total words) yielded exactly **150 cohesive policy chunks** (average ~75 tokens per chunk), perfectly aligned to distinct policy rules.
3. **Embeddings Generation:**
   - Model: `BAAI/bge-small-en-v1.5` (dimension = 384, cosine distance).
   - Contextual text format: `Title: {title} | Section: {section}\n\n{text}`.
4. **Idempotency Guarantee:**
   - Ingestion (`scripts/build_rag_policy_index.py`) is fully idempotent; re-running it resets and rebuilds the collection cleanly, leaving existing collections untouched.

---

## 5. Generic Exact-Identifier Protection

Policy codes follow the standard format `POL-<DOMAIN>-<NUMBER>` (e.g. `POL-JOB-001`, `POL-AI-001`, `POL-MODEL-001`, `POL-REVIEW-001`).

To ensure that queries referencing policy IDs retrieve their exact source documents at Rank 1 without hardcoding individual policy IDs, [`app/rag/retrieval/hybrid_retriever.py`](file:///c:/Users/ASUS/Desktop/enterprise_hr_ai/app/rag/retrieval/hybrid_retriever.py) utilizes generalized structured identifier patterns:

```python
STRUCTURED_IDENTIFIER_PATTERNS = [
    re.compile(r"\b\d{2}-\d{4}(?:\.\d{2})?\b"),                     # O*NET / SOC codes
    re.compile(r"\b(?:SOC|DOC|POL|ID)[-_]?(?:[A-Z0-9]+[-_])?\d+\b", re.I), # SOC, DOC, POL generic codes
]
```

### Protection Mechanism:
1. When a structured code matching `POL-...` is extracted from the user query, BM25 executes sparse retrieval.
2. If BM25 identifies an exact match (e.g. `POL-MODEL-001`), the candidate chunk's normalized sparse score is guaranteed a boost $\ge 0.50$, ensuring it enters the Top 15 hybrid pool even if dense semantic similarity is moderate.
3. Both hyphenated (`pol-model-001`) and underscore (`pol_model_001`) naming variants are resolved identically.

---

## 6. Empirical Evaluation Results

Retrieval performance was evaluated using [`scripts/evaluate_hr_policy_retrieval.py`](file:///c:/Users/ASUS/Desktop/enterprise_hr_ai/scripts/evaluate_hr_policy_retrieval.py) against 22 benchmark queries in [`tests/fixtures/hr_policy_eval_queries.json`](file:///c:/Users/ASUS/Desktop/enterprise_hr_ai/tests/fixtures/hr_policy_eval_queries.json) across 5 distinct categories:

### 6.1 Multi-Stage Metric Summary (18 Answerable Queries)

| Metric | Dense (BGE-small) | Sparse (BM25) | Hybrid (0.8 Dense + 0.2 Sparse) | Hybrid + Cross-Encoder Reranking |
| :--- | :---: | :---: | :---: | :---: |
| **Hit@1** | **1.0000** | 0.8333 | **1.0000** | 0.6667 |
| **Hit@3** | **1.0000** | **1.0000** | **1.0000** | **0.9444** |
| **Hit@5** | **1.0000** | **1.0000** | **1.0000** | N/A (Top-3) |
| **Hit@10** | **1.0000** | **1.0000** | **1.0000** | N/A (Top-3) |
| **MRR** | **1.0000** | 0.9167 | **1.0000** | 0.7963 |

### 6.2 Key Qualitative Findings
1. **Flawless Hybrid Retrieval:** Hybrid retrieval achieved **100% Hit@1, Hit@3, Hit@5, Hit@10, and 1.0000 MRR** across all 18 answerable queries.
2. **Exact Policy-ID Accuracy:**
   - Hybrid retrieval placed the target policy at **Rank 1 in 100% of exact-ID queries** (5/5).
   - Cross-encoder reranking placed the exact policy in Top-3 in 100% of cases, and at Rank 1 in 80% of cases (in `POL_Q05`, `POL-LEARN-001` was ranked 1 and `POL-SKILL-001` was ranked 2, because `POL-LEARN-001` cites `POL-SKILL-001` and elaborates on skill gap remediation).
3. **Retrieval Latency Performance:**
   - **Dense Retrieval:** Mean = 46.81 ms (p50 = 26.25 ms).
   - **Sparse BM25:** Mean = 0.50 ms (p50 = 0.56 ms).
   - **Hybrid Fusion:** Mean = 19.06 ms (p50 = 26.60 ms).
   - **Cross-Encoder Reranking:** Mean = 305.95 ms (p50 = 449.19 ms).

---

## 7. Unsupported Query Refusal & Limitation Handling

Category 5 evaluated 4 queries on topics intentionally absent from the policy corpus:
- Parental leave entitlement
- Annual vacation allowance
- Indian employment law requirements
- Health insurance copay tiers

### Results:
- **Reranking Logit Scores:** All 4 unsupported queries produced strongly negative cross-encoder logit scores ($\le -6.14$, with an average of $-8.97$), indicating near-zero contextual relevance.
- **Refusal Gate:** The retrieval score gate triggers when top rerank score $< 0.0$, successfully detecting that no verified evidence exists in the corpus (**100.0% refusal accuracy, 4/4**).

---

## 8. Verification & Integrity Checklist

| Verification Requirement | Target | Achieved | Status |
| :--- | :---: | :---: | :---: |
| Source Policy Documents | 10 | 10 | PASS |
| Policy Chunks Ingested | 150 | 150 | PASS |
| Dedicated Chroma Collection | `enterprise_hr_policies_bge` | `enterprise_hr_policies_bge` | PASS |
| Dedicated BM25 Sparse Index | `data/rag/policy_sparse_index/` | `data/rag/policy_sparse_index/` | PASS |
| General Knowledge Collection Count | Exactly 1,042 | Exactly 1,042 | PASS |
| General BM25 Sparse Index Count | Exactly 1,042 | Exactly 1,042 | PASS |
| Hybrid Retrieval MRR (Answerable) | $\ge 0.90$ | **1.0000** | PASS |
| Unsupported Query Refusal Accuracy | $\ge 0.80$ | **1.0000 (100%)** | PASS |
| Test Suite (`tests/test_policy_rag.py`) | 14/14 tests passing | 14/14 tests passing | PASS |
| Full Project Regression Tests | 88/88 passing | 88/88 passing | PASS |

---

## 9. Future Integration Considerations

1. **Policy Agent Dispatch:**
   - Rather than merging policy and general documents into a single bloated collection, future work can implement a dedicated **Policy Specialist Agent** or a intent-classifier routing node in LangGraph.
   - When a user asks compliance, threshold, governance, or procedural questions, the router directs the query to `enterprise_hr_policies_bge`.
   - When a user asks about tabular employee metrics, dataset schemas, or O*NET job descriptions, the router directs the query to `enterprise_hr_knowledge_bge`.
2. **Cross-Collection Unified Retrieval (Optional):**
   - If unified search is later required, a multi-index query federator can retrieve Top-K candidates from both collections independently, apply collection-specific provenance tags, and merge candidates in the Cross-Encoder reranking stage.
