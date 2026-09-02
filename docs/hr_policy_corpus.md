# Enterprise HR AI Synthetic Demo Policy Corpus Documentation

**Corpus Version:** 1.0  
**Release Date:** 2026-09-02  
**Status:** Synthetic Demo Policy Corpus  
**Maintained By:** Enterprise HR AI Governance & Architecture Team  
**Location:** `data/knowledge_base/hr_policies/`  
**Manifest:** `data/knowledge_base/hr_policies/manifest.json`  

---

## 1. Purpose and Rationale

The **Synthetic Demo Policy Corpus** was created to evaluate, benchmark, and stress-test the Enterprise HR AI Retrieval-Augmented Generation (RAG) system on realistic, multi-hop human resources governance questions. 

Prior to this corpus, the RAG knowledge base was limited to technical machine learning documentation (`docs/model_card.md`, `docs/data_relationships.md`) and occupational taxonomy catalogs (`data/processed/occupation_master.csv`, `data/external/jobrole_onet_mapping.csv`). While sufficient for testing raw schema lookups and model statistics, it lacked the procedural depth, cross-policy references, governance boundaries, and managerial guidelines necessary to evaluate complex enterprise talent queries.

This corpus provides a controlled, self-consistent knowledge corpus designed to test:
- **Exact Lexical Retrieval (BM25):** Precise identification of policy codes (e.g., `POL-MODEL-001`, `POL-JOB-001`), numerical thresholds (`0.40`), and specific course titles.
- **Dense Semantic Retrieval (BGE):** Conceptual matching of broad talent inquiries (e.g., *"Can an employee be terminated based on an AI score?"* or *"How are managerial competencies evaluated?"*).
- **Cross-Encoder Reranking:** Ranking of nuanced policy clauses and exceptions where lexical overlap is high across multiple related documents.
- **Multi-Document Synthesis:** Answering questions that require synthesizing rules across related policies (e.g., linking `POL-MODEL-001` risk thresholds to `POL-RISK-001` retention procedures and `POL-REVIEW-001` human override requirements).

---

## 2. Synthetic Demarcation and Legal Notice

> [!IMPORTANT]
> **SYNTHETIC DEMONSTRATION CORPUS NOTICE**  
> All documents in `data/knowledge_base/hr_policies/` are explicitly designated as **Synthetic Demo Policy** documents (`Status: Synthetic Demo Policy`).
> - These policies **do NOT** represent the real-world employment policies, legal commitments, or HR entitlements of any real corporate entity or institution.
> - They do not create legal obligations, statutory rights, or contractual terms.
> - No factual claims about any specific employer or live workforce are made or implied.

In accordance with project constraints, the corpus **intentionally excludes** invented real-world employment entitlements and statutory benefits, such as:
- Vacation day accrual and statutory holidays.
- Parental, maternity, or paternity leave durations.
- Base salary bands, minimum wage rules, or severance formulas.
- Statutory termination notice periods or labor union collective bargaining provisions.
- Country-specific labor codes or medical leave entitlements.

Instead, the corpus governs the **actual capabilities, models, schemas, and governance constraints** of this specific Enterprise HR AI project.

---

## 3. Source Basis and Factual Foundation

The policies are grounded in the empirical artifacts, schemas, notebooks, and architectural decisions established across the enterprise project repository:

| Project Source Artifact | Core Factual Elements Extracted | Consuming Policies |
|:---|:---|:---|
| `docs/model_card.md` | Model name (`logreg_balanced_threshold_0.40`), decision threshold (`0.40`), test performance metrics (Recall 0.7872, Precision 0.3426, ROC-AUC 0.8060, F1 0.4774), synthetic IBM dataset origin (1,470 rows, 84/16 class imbalance), top 5 SHAP drivers (`OverTime`, `YearsSinceLastPromotion`, `TotalWorkingYears`, `BusinessTravel_Travel_Frequently`, `JobLevel`), non-autonomous decision-support intended use. | `POL-AI-001`, `POL-MODEL-001`, `POL-RISK-001`, `POL-REVIEW-001`, `POL-MONITOR-001` |
| `data/processed/occupation_master.csv` | 1,016 canonical O*NET-SOC codes, standardized occupational titles, and detailed occupational duty descriptions. | `POL-JOB-001`, `POL-CAREER-001`, `POL-SKILL-001`, `POL-DATA-001` |
| `data/external/jobrole_onet_mapping.csv` | 9 internal IBM job roles mapped to 8 unique SOC codes, match confidences (`low`, `medium`, `very_low`), dual mapping of Healthcare Rep & Sales Rep to `41-3091.00`, Research Scientist mapped to `15-1221.00` (with noted `19-1042.00` life science alternative), Manager mapped to `11-9199.00` placeholder. | `POL-JOB-001`, `POL-SKILL-001`, `POL-CAREER-001`, `POL-DATA-001` |
| `docs/data_relationships.md` | Anchor table primacy (`employee_attrition_processed.csv`, 1,470 rows), engagement survey join coverage (49.7%, 731 matched rows, 50.3% nulls), cleaning decisions (Age=17 to Age=21 correction for IDs 1743/2038, whitespace stripping in `DepartmentType`), type enforcement. | `POL-DATA-001`, `POL-RISK-001`, `POL-JOB-001` |
| `notebooks/10_skill_gap_analysis.ipynb` & `12_skill_gap_engine.ipynb` | Essential skills benchmark (18,200 rows), software skills benchmark (31,821 rows), exclusion of 102 generic managers, workforce gap severity distributions (LOW: 870, MEDIUM: 427, HIGH: 71, N/A - Manager: 102). | `POL-SKILL-001`, `POL-LEARN-001`, `POL-RISK-001` |
| `notebooks/14_recommendation_engine.ipynb` | 33 configured catalog courses (e.g., Speaking Masterclass, Reading Comprehension Workshop, Scientific Methodology, Google Workspace, Bentley MicroStation), Top-3 selection heuristic, manager department-level referral. | `POL-LEARN-001`, `POL-CAREER-001` |
| `notebooks/15_employee_intelligence.ipynb` | Capstone schema (13 columns, 1,470 rows), risk distributions (Low Risk: 885 / 60.2%, High Risk: 585 / 39.8%), multi-dimensional profile integration. | `POL-RISK-001`, `POL-REVIEW-001`, `POL-DATA-001` |
| `models/model_config.json` | Dynamic decision threshold (0.40), feature names, preprocessing scaling rules (`StandardScaler`). | `POL-MODEL-001`, `POL-MONITOR-001` |

---

## 4. Policy Inventory

The corpus comprises 10 comprehensive policy documents totaling 9,728 words (averaging ~973 words per policy):

| Policy ID | Policy Title | Domain | Word Count | Status | Key Cross-References |
|:---|:---|:---|:---:|:---:|:---|
| **POL-JOB-001** | Job Role Classification and O*NET Mapping Policy | Occupational Architecture | 997 | Synthetic Demo Policy | `POL-CAREER-001`, `POL-DATA-001`, `POL-REVIEW-001`, `POL-SKILL-001` |
| **POL-AI-001** | HR AI Decision-Support Governance Policy | AI Governance & Ethics | 939 | Synthetic Demo Policy | `POL-DATA-001`, `POL-MODEL-001`, `POL-MONITOR-001`, `POL-REVIEW-001` |
| **POL-MODEL-001** | Attrition Model Usage Policy | Predictive Analytics & Operations | 1,019 | Synthetic Demo Policy | `POL-AI-001`, `POL-MONITOR-001`, `POL-REVIEW-001`, `POL-RISK-001` |
| **POL-RISK-001** | Workforce Risk Review Policy | Talent Retention & Risk Mitigation | 962 | Synthetic Demo Policy | `POL-AI-001`, `POL-CAREER-001`, `POL-JOB-001`, `POL-LEARN-001`, `POL-MODEL-001`, `POL-REVIEW-001`, `POL-SKILL-001` |
| **POL-SKILL-001** | Skill Gap Identification Policy | Skills Architecture & Gap Analysis | 1,003 | Synthetic Demo Policy | `POL-JOB-001`, `POL-LEARN-001`, `POL-REVIEW-001`, `POL-RISK-001` |
| **POL-LEARN-001** | Employee Upskilling Recommendation Policy | Learning & Development | 940 | Synthetic Demo Policy | `POL-CAREER-001`, `POL-JOB-001`, `POL-REVIEW-001`, `POL-SKILL-001` |
| **POL-CAREER-001** | Career and Occupation Mapping Policy | Internal Mobility & Pathways | 909 | Synthetic Demo Policy | `POL-JOB-001`, `POL-LEARN-001`, `POL-REVIEW-001`, `POL-RISK-001`, `POL-SKILL-001` |
| **POL-DATA-001** | HR Data Usage and Source Provenance Policy | Data Governance & Provenance | 1,007 | Synthetic Demo Policy | `POL-JOB-001`, `POL-MODEL-001`, `POL-MONITOR-001`, `POL-REVIEW-001` |
| **POL-REVIEW-001** | Human Review of AI-Assisted HR Decisions Policy | Human Oversight & Safeguards | 1,005 | Synthetic Demo Policy | `POL-AI-001`, `POL-CAREER-001`, `POL-LEARN-001`, `POL-MODEL-001`, `POL-MONITOR-001`, `POL-RISK-001`, `POL-SKILL-001` |
| **POL-MONITOR-001** | HR AI Monitoring and Model Limitations Policy | Model Lifecycle & Monitoring | 947 | Synthetic Demo Policy | `POL-AI-001`, `POL-DATA-001`, `POL-MODEL-001`, `POL-REVIEW-001` |

---

## 5. Distinction Between Project Facts and Synthetic Policy Rules

To ensure total transparency, the table below delineates the empirical project facts versus the governance and procedural rules synthesized for this demonstration corpus:

### A. Empirically Grounded Project Facts
1. **Model Specifications:** Logistic Regression (`class_weight='balanced'`), fitted with `StandardScaler` on training split, evaluated on 294-row test partition.
2. **Metrics:** Decision threshold is exactly `0.40`; test Recall is `0.7872`, Precision is `0.3426`, ROC-AUC is `0.8060`.
3. **SHAP Feature Ranking:** Top 5 drivers are `OverTime` (0.6558), `YearsSinceLastPromotion` (0.5663), `TotalWorkingYears` (0.5573), `BusinessTravel_Travel_Frequently` (0.5177), and `JobLevel` (0.4585).
4. **Data Dimensions:** Anchor dataset contains exactly 1,470 records; engagement survey matches 731 records (49.7%); occupation master contains 1,016 SOC codes.
5. **Crosswalk Gaps:** 9 IBM roles map to 8 SOC codes; `Manager` maps to `11-9199.00` (`very_low` confidence); `Healthcare Representative` and `Sales Representative` both map to `41-3091.00`.
6. **Skill Distributions:** 102 managers excluded from role-level gap analysis; remaining 1,368 employees distributed into LOW (870), MEDIUM (427), HIGH (71).
7. **Course Catalog:** Exactly 33 configured courses in the recommendation engine.

### B. Synthetic Governance Rules Introduced
1. **Decision-Support Classification (`POL-AI-001`):** Mandate classifying the software strictly as an advisory decision-support system, formally banning autonomous algorithmic terminations, demotions, or compensation cuts.
2. **Operational Retention Review Triggers (`POL-RISK-001`):** Establishing that employees scoring $\ge 0.40$ (585 employees) trigger quarterly managerial "Stay Interviews".
3. **Human Review and Override Rights (`POL-REVIEW-001`):** Establishing that HR professionals hold absolute authority to override risk flags and skill gaps without disciplinary consequence, and mandating that managers never confront employees with numerical AI scores.
4. **Managerial Competency Pathway (`POL-SKILL-001`, `POL-LEARN-001`):** Formally routing the 102 excluded managers to department-level leadership development frameworks rather than leaving them unaddressed.
5. **Model Health Floors and Recalibration Triggers (`POL-MONITOR-001`):** Establishing statistical monitoring floors (Recall $\ge 0.65$, Precision $\ge 0.25$, ROC-AUC $\ge 0.75$) and a 40% departmental override threshold triggering formal model retraining.
6. **Grounding Refusal Protocol (`POL-AI-001`):** Mandating that the RAG assistant emit a standardized refusal when enterprise context is insufficient.

---

## 6. Known Limitations of the Corpus

1. **Synthetic Data Lineage:** Because the underlying data originates from IBM's synthetic demonstration dataset, the operational distributions (such as 39.8% of the workforce being flagged as flight risk) reflect synthetic parameters rather than an actual enterprise distribution.
2. **Catalog Scope:** The learning catalog contains 33 courses. In a live enterprise environment, course catalogs typically encompass thousands of modular offerings.
3. **Single Point-in-Time Policy Set:** The 10 policies represent Version 1.0 (effective date 2026-09-01). They do not yet incorporate historical revision diffs or sunset policies.
4. **Unindexed State:** This corpus has been created and validated in the filesystem, but **has NOT yet been indexed** into ChromaDB or the BM25 sparse index. Ingestion into the RAG pipeline is reserved for subsequent planned steps.

---

## 7. Verification and Audit Trail

The corpus was validated using `scripts/validate_and_build_policy_manifest.py`:
- All 10 Markdown files verified present in `data/knowledge_base/hr_policies/`.
- Every policy confirmed to contain `**Status:** Synthetic Demo Policy`.
- All 10 required structural sections confirmed present in every document.
- Zero contradictory statements detected across cross-policy references.
- Word count confirmed in the 909–1,019 range across all documents.
- Full manifest compiled to `data/knowledge_base/hr_policies/manifest.json`.
