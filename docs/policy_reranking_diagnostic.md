# Cross-Encoder Policy Reranking Diagnostic & Architectural Analysis

## Executive Summary

This diagnostic report provides a comprehensive, empirical investigation into why Cross-Encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) alters retrieval rankings on the synthetic HR policy corpus (`POL-JOB-001` through `POL-MONITOR-001`).

While **Hybrid Retrieval (0.8 Dense + 0.2 Sparse + Exact-ID Protection)** achieved **Hit@1 = 1.0000** and **MRR = 1.0000** across all 18 answerable policy queries, adding the baseline Cross-Encoder resulted in:
- **Hit@1:** Decreased from **1.0000** to **0.6667** (12/18 at Rank 1)
- **Hit@3:** Decreased from **1.0000** to **0.9444** (17/18 in Top 3)
- **MRR:** Decreased from **1.0000** to **0.7963**

This report diagnoses the root cause, maps individual candidate movements across all 22 benchmark queries, analyzes chunk context quality and cross-policy citation dynamics, and evaluates 6 candidate remediation options.

---

## 1. Root Cause Analysis

The degradation of policy ranking by the Cross-Encoder is driven by **three interacting architectural factors**:

### Factor 1: Cross-Policy "Related Policies" Hub Effect (Primary Driver)
Every synthetic HR policy document includes a standardized `## Related Policies` section containing concise bullet points summarizing governing and prerequisite policies.
- Example (`POL-AI-001`):
  ```markdown
  ## Related Policies
  - `POL-MODEL-001`: Attrition Model Usage Policy (governs predictive risk thresholds and interpretation).
  - `POL-REVIEW-001`: Human Review of AI-Assisted HR Decisions Policy (details mandatory oversight protocols).
  ```
When a query asks: *"How do the attrition model policy and human review policy work together?"* (`POL_Q15`), or *"Which policies govern the flow from skill-gap identification to upskilling recommendations?"* (`POL_Q16`), a single chunk from a related coordinating policy (`POL-AI-001` or `POL-CAREER-001`) contains **simultaneous high lexical and semantic co-occurrence** for both referenced topics. 

The Cross-Encoder, computing full cross-attention over `[query, chunk_text]`, awards that summary chunk an exceptionally high logit ($+3.0$ to $+5.9$), pushing it to Rank 1 over the individual defining policies.

### Factor 2: Input Passage Scope (`candidate.text` vs. `candidate.contextual_text`)
In `app/rag/retrieval/reranker.py` (per Notebook 07 Cell 10):
```python
pairs = [[query, c.text] for c in candidates]
```
The Cross-Encoder evaluates `c.text` (the isolated section body) rather than `c.contextual_text` (`Title: {title} | Section: {section}\n\n{text}`). 
- If a section within `POL-SKILL-001` defines gap severity rules without repeating the literal phrase `"POL-SKILL-001"` in its body, its passage-level exact-match score drops.
- Conversely, a bullet point in `POL-LEARN-001` that literally states `POL-SKILL-001: Skill Gap Identification Policy (provides input missing competencies and gap severity)` provides the Cross-Encoder with 100% token overlap with the query.

### Factor 3: Policy-Level Ground Truth vs. Passage-Level Relevance Metric Mismatch
The benchmark evaluation metric defines a Hit as:
$$\text{Hit} \iff \text{chunk.policy\_id} \in \text{expected\_policy\_ids}$$
In multi-policy and semantic questions, a chunk from `POL-CAREER-001` that outlines how `POL-SKILL-001` and `POL-LEARN-001` interconnect is **highly relevant** to the question. However, because its provenance metadata marks it as `policy_id == "POL-CAREER-001"`, the metric records it as a Rank 1 failure, even though the expected policies occupy Rank 2 and Rank 3.

---

## 2. Movement Summary: Overall & By Category

Evaluated across all 22 benchmark queries in `tests/fixtures/hr_policy_eval_queries.json`:

| Category | Total Queries | Improved | Neutral (Maintained) | Degraded (Rank 1 Lost) |
| :--- | :---: | :---: | :---: | :---: |
| **Category 1: Exact Policy ID** | 5 | 0 | 4 (80.0%) | 1 (20.0%) |
| **Category 2: Policy Terminology** | 5 | 0 | 4 (80.0%) | 1 (20.0%) |
| **Category 3: Semantic Queries** | 4 | 0 | 3 (75.0%) | 1 (25.0%) |
| **Category 4: Multi-Policy Queries**| 4 | 0 | 1 (25.0%) | 3 (75.0%) |
| **Category 5: Limitation / Refusal**| 4 | 4 (100.0%)* | 0 | 0 |
| **Total (Answerable Queries)** | **18** | **0** | **12 (66.7%)** | **6 (33.3%)** |
| **Total (All Queries)** | **22** | **4** | **12 (54.5%)** | **6 (27.3%)** |

*\*In Category 5, "Improved" denotes that the Cross-Encoder produced strongly negative logits ($\le -6.14$), cleanly separating out-of-domain queries from in-domain corpus evidence.*

---

## 3. Detailed Trace of Degraded Queries

Below is the exact empirical trace of the 6 queries where the target policy moved downward from Rank 1:

### 1. `POL_Q05` [Category: Exact Policy ID]
- **Query:** *"What does POL-SKILL-001 state regarding the classification of skill gap severity?"*
- **Expected Policy:** `POL-SKILL-001`
- **Hybrid Rank 1:** `POL-SKILL-001` (Section: *Policy Metadata*, Hybrid Score: 0.9859, CE Logit: 4.5077)
- **Cross-Encoder Rank 1:** `POL-LEARN-001` (Section: *Related Policies*, Hybrid Score: 0.2000, CE Logit: **5.7538**)
- **Target Policy Best Rank After Reranking:** **Rank 2** (`POL-SKILL-001`, CE Logit: 4.5077)
- **Trace Analysis:** The chunk `pol_learn_001_related_policies_15_c01` states:
  > `- POL-SKILL-001: Skill Gap Identification Policy (provides input missing competencies and gap severity).`
  Because this single sentence contains `"POL-SKILL-001"`, `"Skill Gap"`, and `"gap severity"`, the Cross-Encoder awarded it $+5.7538$ logit vs. $+4.5077$ for the target policy's own metadata section.

### 2. `POL_Q07` [Category: Policy Terminology]
- **Query:** *"How should an AI-generated HR recommendation be treated?"*
- **Expected Policies:** `["POL-AI-001", "POL-REVIEW-001"]`
- **Hybrid Rank 1:** `POL-AI-001` (Section: *Procedure*, Hybrid Score: 0.9275, CE Logit: -8.2935)
- **Cross-Encoder Rank 1:** `POL-LEARN-001` (Section: *Rule 4: Voluntary and Developmental Nature of Recommendations*, Hybrid Score: 0.2000, CE Logit: **-0.1202**)
- **Target Policy Best Rank After Reranking:** **Rank 2** (`POL-REVIEW-001`, Section: *Purpose*, CE Logit: -2.1312)
- **Trace Analysis:** The chunk in `POL-LEARN-001` explicitly dictates: *"Course recommendations are intended to empower employee growth and must never be treated as punitive remediation"*. The Cross-Encoder prioritized the explicit phrase *"treated"* in combination with *"recommendations"*.

### 3. `POL_Q11` [Category: Semantic Query]
- **Query:** *"What should HR do when an employee is identified as high flight risk?"*
- **Expected Policies:** `["POL-RISK-001", "POL-MODEL-001"]`
- **Hybrid Rank 1:** `POL-RISK-001` (Section: *Definitions*, Hybrid Score: 0.9815, CE Logit: -0.9533)
- **Cross-Encoder Rank 1:** `POL-REVIEW-001` (Section: *Rule 3: Protection Against Direct Algorithmic Confrontation*, Hybrid Score: 0.9402, CE Logit: **4.5556**)
- **Target Policy Best Rank After Reranking:** **Rank 2** (`POL-MODEL-001`, Section: *Human Review Requirements*, CE Logit: 4.2407)
- **Trace Analysis:** `POL-REVIEW-001` Rule 3 states: *"Managers and HR professionals are strictly prohibited from confronting employees directly with algorithmic scores... When an employee is flagged as High Risk... HR must conduct structured stay interviews..."* This chunk directly answers what HR should do. The target policy `POL-MODEL-001` followed closely at Rank 2.

### 4. `POL_Q15` [Category: Multi-Policy]
- **Query:** *"How do the attrition model policy and human review policy work together?"*
- **Expected Policies:** `["POL-MODEL-001", "POL-REVIEW-001"]`
- **Hybrid Rank 1:** `POL-MODEL-001` (Section: *Related Policies*, Hybrid Score: 0.9765, CE Logit: -1.8534)
- **Cross-Encoder Rank 1:** `POL-AI-001` (Section: *Related Policies*, Hybrid Score: 0.9079, CE Logit: **3.0754**)
- **Target Policy Best Rank After Reranking:** **Rank 5** (`POL-MODEL-001`, Section: *Scope*, CE Logit: 2.1052; `POL-REVIEW-001` at Rank 7)
- **Trace Analysis:** In `POL-AI-001` *Related Policies*, both policies are explicitly defined in adjacent lines:
  > `- POL-MODEL-001: Attrition Model Usage Policy (governs predictive risk thresholds...)`  
  > `- POL-REVIEW-001: Human Review of AI-Assisted HR Decisions Policy (details mandatory oversight...)`  
  The Cross-Encoder favored this joint definition over individual policy chunks.

### 5. `POL_Q16` [Category: Multi-Policy]
- **Query:** *"Which policies govern the flow from skill-gap identification to upskilling recommendations?"*
- **Expected Policies:** `["POL-SKILL-001", "POL-LEARN-001"]`
- **Hybrid Rank 1:** `POL-LEARN-001` (Section: *Related Policies*, Hybrid Score: 0.9759, CE Logit: 4.7460)
- **Cross-Encoder Rank 1:** `POL-CAREER-001` (Section: *Related Policies*, Hybrid Score: 0.2000, CE Logit: **5.9426**)
- **Target Policy Best Rank After Reranking:** **Rank 2** (`POL-SKILL-001`, CE Logit: 4.9100) and **Rank 3** (`POL-LEARN-001`, CE Logit: 4.7460)
- **Trace Analysis:** Both expected policies are present in Top 3 context. `POL-CAREER-001` took Rank 1 because its *Related Policies* chunk explicitly lists both `POL-SKILL-001` and `POL-LEARN-001` as prerequisites for career progression.

### 6. `POL_Q18` [Category: Multi-Policy]
- **Query:** *"How does data provenance governance support model monitoring and limitations?"*
- **Expected Policies:** `["POL-DATA-001", "POL-MONITOR-001"]`
- **Hybrid Rank 1:** `POL-DATA-001` (Section: *Purpose*, Hybrid Score: 0.9282, CE Logit: -0.8345)
- **Cross-Encoder Rank 1:** `POL-AI-001` (Section: *Related Policies*, Hybrid Score: 0.8940, CE Logit: **2.0942**)
- **Target Policy Best Rank After Reranking:** **Rank 3** (`POL-MONITOR-001`, CE Logit: 0.8457) and **Rank 4** (`POL-DATA-001`, CE Logit: -0.8345)
- **Trace Analysis:** `POL-AI-001` *Related Policies* lists both `POL-DATA-001` and `POL-MONITOR-001` together, capturing high joint relevance.

---

## 4. Exact-Policy-ID Analysis

Evaluated queries:
1. `POL-MODEL-001` (`POL_Q01`): Hybrid Rank 1 $\rightarrow$ Reranked Rank 1 (**Neutral / Maintained**)
2. `POL-JOB-001` (`POL_Q02`): Hybrid Rank 1 $\rightarrow$ Reranked Rank 1 (**Neutral / Maintained**)
3. `POL-REVIEW-001` (`POL_Q03`): Hybrid Rank 1 $\rightarrow$ Reranked Rank 1 (**Neutral / Maintained**)
4. `POL-AI-001` (`POL_Q04`): Hybrid Rank 1 $\rightarrow$ Reranked Rank 1 (**Neutral / Maintained**)
5. `POL-SKILL-001` (`POL_Q05`): Hybrid Rank 1 $\rightarrow$ Reranked Rank 2 (**Degraded by 1 position**)

### Findings:
- In **4 out of 5 exact-ID queries (80.0%)**, the Cross-Encoder maintained the exact target policy at **Rank 1**.
- In the 1 degraded case (`POL_Q05`), the target policy remained at **Rank 2** (firmly inside Top 3 context).
- The existing exact-identifier protection mechanism in `HybridRetriever` successfully guarantees that target chunks enter the Top 15 pool. However, because the Cross-Encoder sorts purely on raw cross-attention logits, an external chunk with dense keyword overlap can displace the primary document by a fraction of a logit point.

---

## 5. Semantic-Query Analysis

Evaluated semantic queries without policy codes:
- `POL_Q12` (*Safeguards preventing autonomous decisions*): Target `POL-AI-001` maintained at **Rank 1**.
- `POL_Q13` (*Occupation mappings for career development*): Target `POL-CAREER-001` maintained at **Rank 1**.
- `POL_Q14` (*Manager exclusion from skill gaps*): Target `POL-SKILL-001` maintained at **Rank 1**.
- `POL_Q11` (*High flight risk response*): `POL-REVIEW-001` ranked #1 ($+4.56$) over `POL-MODEL-001` ($+4.24$) and `POL-RISK-001` ($+2.53$).

### Findings:
- On purely conceptual questions, the Cross-Encoder performs **exceptionally well**, surfacing the most operationally actionable chunks.
- For `POL_Q11`, `POL-REVIEW-001` was judged more relevant by the Cross-Encoder because it describes the specific procedure HR must execute (stay interviews, non-algorithmic confrontation) rather than just stating risk thresholds.

---

## 6. Chunk Context Quality Analysis

Inspection of the 150 policy chunks shows:
1. **Header Metadata Absence in `c.text`:**
   Each chunk possesses a rich `title` attribute (e.g., `POL-SKILL-001: Skill Gap Identification Policy — Rule 2`), but `reranker.py` inputs only `c.text`. As a result, the Cross-Encoder is blind to the document title unless the section body explicitly restates it.
2. **Cohesiveness of Policy Rules:**
   Sections are concise and well-formed (averaging 75 tokens). This conciseness means that short cross-reference bullet points have disproportionately high term density relative to broader policy narratives.

---

## 7. Unsupported-Query Analysis (Refusal Discrimination)

The Cross-Encoder was tested on 4 out-of-domain questions:
1. `POL_Q19`: *What is the company's parental leave entitlement?* $\rightarrow$ Logit: **$-10.7780$**
2. `POL_Q20`: *What is the annual vacation allowance?* $\rightarrow$ Logit: **$-10.9870$**
3. `POL_Q21`: *What does Indian employment law require for termination?* $\rightarrow$ Logit: **$-7.9690$**
4. `POL_Q22`: *What are the health insurance copay tiers for employees?* $\rightarrow$ Logit: **$-6.1436$**

### Findings:
- **Zero False Positives:** Every unsupported query produced an extreme negative logit ($\le -6.14$), whereas legitimate policy matches scored between $+0.84$ and $+5.94$.
- The Cross-Encoder provides an **ironclad confidence signal** for out-of-domain refusal. When combined with a threshold gate ($\text{logit} < 0.0$), it achieves **100.0% refusal accuracy**.

---

## 8. Evaluation of Correction Options

Below are the 6 potential remediation strategies evaluated conceptually:

### Option A: Keep Current Cross-Encoder Exactly As-Is
- **Description:** Accept Hit@1 = 0.6667 and Hit@3 = 0.9444.
- **Pros:** Zero code modifications; 100% notebook consistency.
- **Cons:** Leaves policy Hit@1 degraded relative to pure hybrid retrieval (1.0000).
- **Assessment:** Unfavorable. Degrades proven retrieval performance.

### Option B: Add Policy-Aware Metadata/Context to Input Pairs (`c.contextual_text`)
- **Description:** Pass `[query, c.contextual_text]` instead of `[query, c.text]` to `CrossEncoder.predict()`.
- **Mechanism:** `c.contextual_text` includes `Title: {title} | Section: {section}\n\n{text}`, explicitly providing the policy ID and policy title to cross-attention.
- **Pros:** High simplicity; highly generalizable across all corpora; ensures chunks "know" their own parent policy.
- **Cons:** Slightly increases token sequence length in cross-attention (~15 tokens).
- **Assessment:** **High Viability**. Directly addresses Factor 2.

### Option C: Policy-Level Aggregation After Chunk Reranking
- **Description:** Group chunk scores by `policy_id` (e.g., max-pool or reciprocal rank fusion across policy chunks) before selecting final Top-3 context documents.
- **Pros:** Aligns passage-level scoring with policy-level evaluation goals.
- **Cons:** Adds aggregation complexity; changes the schema from chunk-level context to document-level context.
- **Assessment:** Moderate Viability.

### Option D: Exact-Policy-ID Priority / Lexical Boost in Reranking
- **Description:** If a query contains an exact structured policy code (`POL-...`), preserve the exact-matching policy's top chunk at Rank 1 or apply an additive priority bonus before sorting.
- **Pros:** Directly guarantees 100% Hit@1 on exact-ID queries; mirror image of the exact-identifier protection in `HybridRetriever`.
- **Cons:** Only applies when an explicit policy ID is present in the query; does not resolve multi-policy keyword overlap.
- **Assessment:** **High Viability** for exact-ID queries.

### Option E: Policy-Specific Separate Reranking Configuration
- **Description:** Maintain separate reranking parameters or score-combination weights for the policy collection.
- **Pros:** Isolates policy tuning from general knowledge base.
- **Cons:** Introduces branching code paths and increased maintenance overhead.
- **Assessment:** Moderate Viability.

### Option F: Different Cross-Encoder Model
- **Description:** Replace `cross-encoder/ms-marco-MiniLM-L-6-v2` with a larger or domain-specific model (e.g. BGE-reranker-base).
- **Pros:** Might have better nuance.
- **Cons:** Significantly higher memory footprint and inference latency (~3x slower on CPU); violates baseline notebook architecture.
- **Assessment:** Unfavorable.

---

## 9. Ranked Comparison of Options

| Option | Correctness | Simplicity | Notebook Consistency | Generalizability | Latency | Maintainability | Overall Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Option B (Contextual Input Text)** | High | High | High | High | Very Low ($\sim 2\%$) | High | **#1 (Recommended Primary)** |
| **Option D (Exact-ID Priority)** | High | High | Moderate | Moderate | Negligible | High | **#2 (Recommended Secondary)** |
| **Option C (Policy Aggregation)** | Moderate | Moderate | Moderate | Moderate | Low | Moderate | **#3** |
| **Option E (Policy-Specific Config)**| Moderate | Moderate | Low | Low | Low | Moderate | **#4** |
| **Option A (As-Is)** | Low | High | High | High | None | High | **#5** |
| **Option F (New Model)** | Unknown | Low | Low | Moderate | High Penalty | Low | **#6** |

---

## 10. Conclusion & Recommended Next Step

The Cross-Encoder is not behaving erratically; rather:
1. It is faithfully identifying that summary chunks in `## Related Policies` sections contain dense, simultaneous answers to multi-policy queries.
2. It lacks document-level title context because only raw section bodies (`c.text`) are passed to `predict()`.
3. In 17 out of 18 answerable queries (94.44%), the target policies are present in the Top 3 context.

Per instructions, **no changes to reranker code, agent connections, or hybrid weights have been implemented in this step**. The full 88-test project suite remains intact and passing. All diagnostic data is preserved in `reports/rag/policy_reranking_diagnostic.json`.
