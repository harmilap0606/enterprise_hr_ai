# Job Role Classification and O*NET Mapping Policy

## Policy Metadata
- **Policy ID:** POL-JOB-001
- **Policy Title:** Job Role Classification and O*NET Mapping Policy
- **Policy Domain:** Occupational Architecture & Role Classification
- **Version:** 1.0
- **Status:** Synthetic Demo Policy
- **Effective Date:** 2026-09-01
- **Owner:** Enterprise HR AI Architecture & Talent Governance Team
- **Scope:** Enterprise-wide job architecture, role crosswalking, and standardized skill baseline ingestion
- **Classification:** Internal Demonstration Standard
- **Source Basis:** `data/external/jobrole_onet_mapping.csv`, `data/processed/occupation_master.csv`, `docs/data_relationships.md`

---

## Purpose
The purpose of `POL-JOB-001` is to establish formal rules, technical standards, and governance procedures for mapping internal organizational job roles to standardized occupational codes defined in the Occupational Information Network (O*NET) taxonomy. Internal job titles frequently exhibit organizational idiosyncrasies that prevent direct cross-company benchmarking. By standardizing internal roles against verified O*NET Standard Occupational Classification (SOC) codes, the Enterprise HR AI platform establishes a common denominator for skills gap analysis, career pathway development, and external talent market alignment.

---

## Scope
This policy applies to all 9 standardized internal job role classifications utilized in the Enterprise HR AI platform anchor dataset:
1. Healthcare Representative
2. Human Resources
3. Laboratory Technician
4. Manager
5. Manufacturing Director
6. Research Director
7. Research Scientist
8. Sales Executive
9. Sales Representative

This policy governs the creation, auditing, maintenance, and consumption of crosswalk tables between internal records (`employee_attrition_processed.csv`) and the occupational master catalog (`occupation_master.csv`).

---

## Definitions
- **Internal Job Role:** The operational title assigned to an employee within enterprise business units (e.g., `Research Scientist`, `Sales Executive`).
- **O*NET-SOC Code:** The eight-digit hierarchical taxonomy identifier established by the U.S. Department of Labor (e.g., `15-1221.00`, `19-1042.00`).
- **Match Confidence:** A qualitative tier (`high`, `medium`, `low`, `very_low`) reflecting taxonomic alignment between internal duties and O*NET standard definitions.
- **Crosswalk Table:** The authoritative reference table (`data/external/jobrole_onet_mapping.csv`) maintaining bidirectional mappings and rationale.
- **Placeholder Code:** A broad, non-specific SOC code (specifically `11-9199.00`, *Managers, All Other*) assigned when internal titles lack operational specificity.

---

## Policy Rules

### Rule 1: Mandatory Crosswalk Reference
All downstream talent intelligence modules—including skill gap extraction, upskilling recommendations, and career mobility modeling—must resolve job roles exclusively through the verified crosswalk table (`jobrole_onet_mapping.csv`). Direct string matching between internal `JobRole` strings and `occupation_master.csv` titles is strictly prohibited because exact text matching fails for the majority of enterprise job roles.

### Rule 2: Treatment of Dual Mappings
Where multiple internal roles resolve to the same O*NET code, systems and practitioners must treat O*NET-derived baseline skills as shared baselines, while preserving internal job titles for all reporting:
- `Healthcare Representative` and `Sales Representative` both resolve to O*NET code `41-3091.00` (*Sales Representatives of Services*).
- Because `41-3091.00` represents the closest service-sales category in `occupation_master.csv`, both roles share identical baseline skill profiles at the O*NET layer.
- Talent professionals must acknowledge that `41-3091.00` does not reflect the specialized regulatory, compliance, or clinical context inherent to healthcare liaison work.

### Rule 3: Placeholder Classification for General Managers
Internal employees holding the generic job role `Manager` are mapped to `11-9199.00` (*Managers, All Other*) with a match confidence rating of `very_low`. Because `occupation_master.csv` contains over 52 functional management codes (ranging from Human Resources Managers to Sales Managers), the generic title `Manager` cannot be mapped to a specific functional discipline without introducing false assumptions. Consequently:
- Role-level O*NET skill analysis is disabled for the 102 employees in the `Manager` classification.
- Competency evaluation for managers must proceed through department-level functional analysis (`Department` attribute), as mandated by `POL-SKILL-001`.

### Rule 4: Handling Educational and Disciplinary Ambiguity
When mapping roles spanning computational and life science boundaries—specifically `Research Scientist` mapped to `15-1221.00` (*Computer and Information Research Scientists*, `medium` confidence)—HR analysts must account for the employee's `EducationField`:
- If `EducationField` is `Technical Degree` or computational, `15-1221.00` serves as a valid proxy.
- If `EducationField` is `Life Sciences` or `Medical`, analysts must record that `19-1042.00` (*Medical Scientists, Except Epidemiologists*) represents a superior functional analog, and apply human oversight per `POL-REVIEW-001`.

---

## Procedure
1. **Role Ingestion:** When an employee record is ingested from `employee_attrition_processed.csv`, the platform reads the `JobRole` attribute.
2. **Crosswalk Resolution:** The system performs a key lookup against `jobrole_onet_mapping.csv` matching `JobRole` to `ibm_job_role`.
3. **Confidence Assessment:** The system extracts `onet_soc_code`, `onet_title`, and `match_confidence`.
4. **Flagging Low-Confidence Alignments:** If `match_confidence` is `low` or `very_low`, the platform flags the record with an audit banner requiring supervisory confirmation before deploying automated career pathways.
5. **Enrichment Join:** The resolved `onet_soc_code` is joined to `occupation_master.csv` to retrieve canonical descriptions, and subsequently to `essential_skills_processed.csv` and `software_skills_processed.csv`.

---

## Exceptions and Limitations
- **Synthetic Demonstration Context:** This policy governs synthetic demonstrator data derived from the IBM HR Analytics schema and public O*NET 2026 extracts. It does not reflect union contracts, statutory job grading, or real corporate organizational structures.
- **Crosswalk Immutability:** Operational pipelines must not modify crosswalk mappings dynamically during batch inference. Any update to `jobrole_onet_mapping.csv` requires a formal version increment under `POL-DATA-001`.
- **Exclusion of Manager Cohort:** Automated role-level skill profiles cannot be generated for the 102 employees classified under `Manager`.

---

## Human Review Requirements
In accordance with `POL-REVIEW-001`:
1. Any workforce mobility recommendation affecting an employee whose role has `match_confidence: low` or `very_low` (specifically `Manager`, `Sales Executive`, `Manufacturing Director`, and `Research Director`) requires mandatory manual review by an HR Business Partner before career counseling.
2. Mappings must be re-evaluated whenever an employee's department transfers or job level increases.

---

## Data and Source References
- `data/external/jobrole_onet_mapping.csv`: Authoritative crosswalk containing 9 role definitions, confidence tiers, and architectural notes.
- `data/processed/occupation_master.csv`: Catalog of 1,016 canonical O*NET occupations with standard titles and official descriptions.
- `docs/data_relationships.md`: Architectural documentation detailing Relationship 2 (`employee_attrition` to `occupation_master`) and cleaning constraints.

---

## Related Policies
- `POL-CAREER-001`: Career & Occupation Mapping Policy (governs internal mobility pathways using resolved O*NET titles).
- `POL-SKILL-001`: Skill Gap Identification Policy (governs the extraction of skills based on resolved O*NET-SOC codes).
- `POL-DATA-001`: HR Data Usage & Source Provenance Policy (establishes data integrity standards for role crosswalks).
- `POL-REVIEW-001`: Human Review of AI-Assisted HR Decisions Policy (governs oversight of low-confidence mappings).
