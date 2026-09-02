# Skill Gap Identification Policy

## Policy Metadata
- **Policy ID:** POL-SKILL-001
- **Policy Title:** Skill Gap Identification Policy
- **Policy Domain:** Skills Architecture & Gap Analysis
- **Version:** 1.0
- **Status:** Synthetic Demo Policy
- **Effective Date:** 2026-09-01
- **Owner:** Enterprise Learning & Workforce Capability Steering Group
- **Scope:** Technical and behavioral skill requirement benchmarking, individual gap identification, and severity classification
- **Classification:** Internal Demonstration Standard
- **Source Basis:** `data/processed/essential_skills_processed.csv`, `data/processed/software_skills_processed.csv`, `data/processed/occupation_master.csv`, `data/external/jobrole_onet_mapping.csv`, `notebooks/10_skill_gap_analysis.ipynb`, `notebooks/12_skill_gap_engine.ipynb`

---

## Purpose
`POL-SKILL-001` governs the methodology, computational standards, and operational workflows for identifying competency gaps across the enterprise workforce. Rapid technological evolution necessitates that organizations continuously analyze the delta between current workforce capabilities and the evolving competency profiles demanded by standardized occupational standards. By mapping individual employee profiles against validated O*NET competency benchmarks (`essential_skills_processed.csv` and `software_skills_processed.csv`), the Enterprise HR AI platform provides an objective foundation for targeted professional development and capability planning.

---

## Scope
This policy applies to 1,368 individual-contributor and director-level employees across the enterprise. It specifically governs the taxonomy, calculation logic, and severity tiering for:
- Essential behavioral, foundational, and functional competencies (18,200 benchmark rows).
- Specialized technical tools and software applications (31,821 benchmark rows).

The policy explicitly defines the procedural exclusion and alternative governance path for the 102 employees holding the generic internal title `Manager`.

---

## Definitions
- **Competency Benchmark:** The curated list of essential skills and software tools associated with an occupation's canonical O*NET-SOC code.
- **Skill Gap:** A required technical tool, behavioral ability, or knowledge domain specified in the O*NET occupational benchmark that is absent from an employee's recorded capability profile.
- **Gap Count:** The integer count of missing competencies identified for an employee relative to their occupational benchmark.
- **Gap Severity Tier:** A standardized classification reflecting capability risk:
  - `LOW Severity`: Minor deficiency (1 to 2 missing competencies; 870 employees in baseline).
  - `MEDIUM Severity`: Moderate deficiency (3 to 5 missing competencies; 427 employees in baseline).
  - `HIGH Severity`: Significant deficiency (> 5 missing competencies; 71 employees in baseline).
  - `N/A - Manager`: Exclusion classification applied to generic managers (102 employees in baseline).

---

## Policy Rules

### Rule 1: Dual-Taxonomy Benchmark Ingestion
Skill requirements for all analyzed roles must be derived from the two authoritative processed tables:
1. `essential_skills_processed.csv` (18,200 rows) — covering cognitive abilities, scientific methodologies, communication competencies, and operational skills.
2. `software_skills_processed.csv` (31,821 rows) — covering domain-specific software platforms, database technologies, enterprise suites, and engineering tools.
Both tables join to `occupation_master.csv` via the canonical `O*NET-SOC Code` key.

### Rule 2: Mandatory Exclusion of Generic Managers
In alignment with `docs/data_relationships.md` and `POL-JOB-001`, employees holding the title `Manager` (102 employees) must be excluded from automated role-level skill gap calculation:
- Because the generic internal title `Manager` maps to placeholder code `11-9199.00` (*Managers, All Other*) with `very_low` confidence, running automated skill comparisons against that catch-all profile produces invalid gap metrics.
- The platform must systematically assign `SkillGapSeverity = 'N/A - Manager'` and `SkillGapCount = NaN` to these records.
- Managerial capabilities must be evaluated through department-level leadership frameworks rather than automated O*NET role-level comparisons.

### Rule 3: Objective Gap Severity Classification
Individual skill gap severity must be calculated strictly according to verified numerical thresholds:
- Employees missing 1 to 2 benchmark skills are categorized as `LOW Severity` (870 employees / 59.18% of workforce).
- Employees missing 3 to 5 benchmark skills are categorized as `MEDIUM Severity` (427 employees / 29.05% of workforce).
- Employees missing more than 5 benchmark skills are categorized as `HIGH Severity` (71 employees / 4.83% of workforce).
Customizing severity thresholds on an ad-hoc basis is prohibited to maintain enterprise reporting consistency.

### Rule 4: Decoupling Skill Gaps from Disciplinary Processes
Skill gaps identified through this automated process reflect developmental opportunities and learning needs, not employee fault or performance deficiencies. Skill gap metrics:
- Must feed directly into the upskilling recommendation engine (`POL-LEARN-001`).
- May be reviewed during workforce risk assessments (`POL-RISK-001`) to provide developmental support to flight-risk employees.
- Must never be used as criteria for formal performance improvement plans or disciplinary warnings.

---

## Procedure
1. **Profile Resolution:** The system resolves the employee's `JobRole` to its corresponding `O*NET-SOC Code` via `jobrole_onet_mapping.csv`.
2. **Exclusion Check:** If `JobRole == 'Manager'`, the system bypasses calculation and records `SkillGapSeverity = 'N/A - Manager'`.
3. **Requirement Extraction:** For eligible roles, the system queries `essential_skills_processed.csv` and `software_skills_processed.csv` to compile the required competency list.
4. **Delta Evaluation:** The engine compares the employee's documented skill inventory against the required competency list, generating a deduplicated `missing_skills` array.
5. **Severity Assignment:** The platform computes `gap_count` and applies Rule 3 to assign `LOW`, `MEDIUM`, or `HIGH` severity.
6. **Persistence:** The computed metrics (`SkillGapCount`, `SkillGapSeverity`) are stored in `employee_intelligence.csv` and transmitted to the recommendation engine (`POL-LEARN-001`).

---

## Exceptions and Limitations
- **O*NET Taxonomy Granularity:** O*NET skills describe broad occupational archetypes. They may not encompass proprietary internal tools, proprietary codebase architectures, or bespoke operational procedures unique to an enterprise.
- **Identical Profiles for Dual Mappings:** Per `POL-JOB-001`, `Healthcare Representative` and `Sales Representative` both map to `41-3091.00`. Their O*NET skill benchmarks are identical; internal differentiation must be applied during human review.

---

## Human Review Requirements
In accordance with `POL-REVIEW-001`:
1. Employees categorized under `HIGH Severity` (71 employees) must have their profiles audited by an HR Learning Specialist before automated course enrollment.
2. If an employee possesses equivalent practical experience not captured in the digital inventory, the HR reviewer has full authority to adjust the effective gap count.

---

## Data and Source References
- `data/processed/essential_skills_processed.csv`: Catalog of 18,200 essential occupational skill rows.
- `data/processed/software_skills_processed.csv`: Catalog of 31,821 technical and software tool rows.
- `data/external/jobrole_onet_mapping.csv`: Authoritative role crosswalk governing O*NET resolution.
- `data/processed/employee_intelligence.csv`: Repository containing finalized gap counts and severity tiers.

---

## Related Policies
- `POL-JOB-001`: Job Role Classification & O*NET Mapping Policy (governs role-to-SOC code resolution).
- `POL-LEARN-001`: Employee Upskilling Recommendation Policy (governs course recommendation generation).
- `POL-RISK-001`: Workforce Risk Review Policy (integrates skill gap severity into retention reviews).
- `POL-REVIEW-001`: Human Review of AI-Assisted HR Decisions Policy (governs manual overrides of gap severities).
