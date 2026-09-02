# HR Data Usage and Source Provenance Policy

## Policy Metadata
- **Policy ID:** POL-DATA-001
- **Policy Title:** HR Data Usage and Source Provenance Policy
- **Policy Domain:** Data Governance & Source Verification
- **Version:** 1.0
- **Status:** Synthetic Demo Policy
- **Effective Date:** 2026-09-01
- **Owner:** Enterprise Data Governance Office & HR Analytics Architecture Team
- **Scope:** Table inventory standards, data cleaning mandates, schema relationships, join cardinality, and source provenance
- **Classification:** Internal Demonstration Standard
- **Source Basis:** `docs/data_relationships.md`, `data/processed/`, `notebooks/03_data_cleaning.ipynb`, `notebooks/04_data_relationships.ipynb`

---

## Purpose
`POL-DATA-001` establishes enterprise-wide data governance standards, lineage auditing rules, and data integrity protocols for all tables, schemas, and analytical pipelines powering the Enterprise HR AI system. Reliable predictive models and grounded RAG responses depend completely on the cleanliness, consistency, and verifiable provenance of underlying enterprise datasets. By defining single-source-of-truth anchor tables, documenting join cardinalities, mandating cleaning rules, and requiring strict provenance auditing, this policy guarantees that all AI outputs are grounded in verified enterprise data.

---

## Scope
This policy applies to all raw, external, processed, and capstone datasets within the Enterprise HR AI repository, specifically governing the five core processed tables and one external crosswalk:
1. `employee_attrition_processed.csv` (1,470 rows — Primary Anchor)
2. `engagement_processed.csv` (2,845 rows — Survey Enrichment)
3. `occupation_master.csv` (1,016 rows — Taxonomic Master)
4. `essential_skills_processed.csv` (18,200 rows — Skill Requirements)
5. `software_skills_processed.csv` (31,821 rows — Tool Requirements)
6. `jobrole_onet_mapping.csv` (11 rows / 9 roles — Role Crosswalk)
7. `employee_intelligence.csv` (1,470 rows — Capstone Integrated Table)

---

## Definitions
- **Anchor Table:** The authoritative master table (`employee_attrition_processed.csv`) that defines the unique entity population (1,470 employees); all subsequent downstream tables must join to this anchor without reducing or duplicating rows.
- **Source Provenance:** The auditable metadata trace recording the origin file, raw lineage, cleaning transformations, join keys, and timestamp of every data point surfaced by the platform.
- **Left Join Enrichment:** A relational database operation preserving 100% of rows from the anchor table while selectively attaching attributes from secondary sources where matching keys exist.
- **Coverage Disparity:** The phenomenon where secondary enrichment datasets do not cover the complete population of the anchor table (e.g., 49.7% engagement survey coverage).

---

## Policy Rules

### Rule 1: Anchor Table Primacy and Row Invariance
`employee_attrition_processed.csv` is designated as the sole ANCHOR table for all enterprise workforce modeling. Downstream data pipelines, feature engineering notebooks, and capstone consolidations must maintain strict row invariance:
- The total workforce population is fixed at exactly **1,470 unique employee records** keyed by `EmployeeNumber`.
- Inner joins that drop unmatched employees, or cartesian joins that duplicate records, are strictly prohibited.
- Any merged analytical dataset (such as `employee_intelligence.csv`) must verify exactly 1,470 rows upon output generation.

### Rule 2: Handling Survey Enrichment Coverage (49.7%)
As formally documented in `docs/data_relationships.md` (Relationship 1), joining `engagement_processed.csv` to the anchor table produces a verified coverage of **49.7%** (731 matched records / 739 unmatched records):
- Pipelines must execute a `LEFT JOIN` on `employee_attrition_processed.EmployeeNumber == engagement_processed.Employee ID`.
- Data consumers and machine learning models must handle the resulting 50.3% null values gracefully without dropping records or assigning imputed default values that could bias downstream analyses.
- Dashboard views must clearly distinguish between verified survey scores and missing data states (`NaN`).

### Rule 3: Enforcing Standard Data Cleaning Decisions
All processed datasets must adhere to the standardized cleaning decisions established in `notebooks/03_data_cleaning.ipynb`:
- **Age Corrections:** Engagement records for IDs 1743 and 2038 must be corrected from Age=17 to Age=21 based on verified Date of Birth (2001) and Survey Date (2023).
- **Whitespace Stripping:** All string and object fields must undergo whitespace stripping (notably resolving trailing whitespace in `DepartmentType` values like `"Production       "`).
- **Type Casting:** Integer identifiers must be cast to nullable `Int64`, dates parsed to standard `datetime64`, and categorical columns standardized.
- **Duplicate Zero-Tolerance:** Zero duplicate rows are permitted in processed storage.

### Rule 4: Mandatory Provenance in RAG Generation
All retrieval systems and generative assistants (specifically `/rag/ask`) must preserve and surface end-to-end data provenance:
- Retrieved context presented to language models must include source file paths (e.g., `data/processed/occupation_master.csv`, `docs/model_card.md`), section headings, and chunk identifiers.
- Generated responses must cite verified source documents so that HR users can audit factual claims back to authoritative files.
- Generating responses based on untracked, unindexed external internet sources is strictly prohibited.

---

## Procedure
1. **Raw Ingestion & Validation:** Incoming datasets are staged in `data/raw/` and validated against structural schemas.
2. **Deterministic Preprocessing:** Cleaning scripts apply whitespace stripping, age corrections, and type enforcement, writing results to `data/processed/`.
3. **Integrity Auditing:** Automated tests verify row counts (1,470 anchor rows, 1,016 occupation rows) and key uniqueness.
4. **Relational Consolidation:** Consolidation notebooks (`notebooks/15_employee_intelligence.ipynb`) join anchor records with crosswalks and enrichment tables, verifying that output dimensions match $(1470, 13)$.
5. **RAG Chunk Indexing:** Markdown documentation and processed CSV tables are chunked, tagged with metadata, and embedded into the ChromaDB and BM25 knowledge bases.

---

## Exceptions and Limitations
- **Synthetic Data Demarcation:** All processed employee data originates from IBM's synthetic HR demonstration dataset. It contains no personally identifiable information (PII) or real employee health records.
- **Static Ingestion:** The current data pipeline operates on batch-processed file snapshots. Continuous streaming updates are not currently supported.

---

## Human Review Requirements
In accordance with `POL-REVIEW-001`:
1. Any structural modification to primary data keys or join logic must be reviewed and approved by the Lead Data Architect.
2. Data lineage anomalies discovered during RAG auditing must be investigated and remediated within 5 business days.

---

## Data and Source References
- `docs/data_relationships.md`: Authoritative data relationship specification and table inventory.
- `data/processed/employee_attrition_processed.csv`: Primary anchor dataset (1,470 rows).
- `data/processed/engagement_processed.csv`: Survey enrichment dataset (2,845 rows, 49.7% anchor match).
- `data/processed/occupation_master.csv`: Canonical O*NET occupational repository (1,016 rows).

---

## Related Policies
- `POL-JOB-001`: Job Role Classification & O*NET Mapping Policy (governs role crosswalk table integrity).
- `POL-MODEL-001`: Attrition Model Usage Policy (governs data consumption by predictive algorithms).
- `POL-MONITOR-001`: HR AI Monitoring & Model Limitations Policy (governs data drift and quality tracking).
- `POL-REVIEW-001`: Human Review of AI-Assisted HR Decisions Policy (governs human verification of data sources).
