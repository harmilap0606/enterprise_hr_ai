# Employee Upskilling Recommendation Policy

## Policy Metadata
- **Policy ID:** POL-LEARN-001
- **Policy Title:** Employee Upskilling Recommendation Policy
- **Policy Domain:** Learning & Professional Development
- **Version:** 1.0
- **Status:** Synthetic Demo Policy
- **Effective Date:** 2026-09-01
- **Owner:** Global Learning & Organizational Development Council
- **Scope:** Automated course curriculum mapping, personalized upskilling generation, and learning engagement
- **Classification:** Internal Demonstration Standard
- **Source Basis:** `data/processed/employee_recommendations.csv`, `notebooks/14_recommendation_engine.ipynb`, `data/processed/employee_intelligence.csv`

---

## Purpose
`POL-LEARN-001` establishes standards and procedures for translating algorithmic skill gap diagnoses into actionable, personalized professional development pathways. Modern enterprise capability development requires moving beyond generic training catalogs toward precise, needs-based learning recommendations. By algorithmically matching missing O*NET competencies to an accredited 33-course learning catalog, the Enterprise HR AI platform delivers targeted upskilling recommendations that foster internal career mobility, enhance operational productivity, and mitigate employee flight risk.

---

## Scope
This policy applies to learning pathway generation for 1,368 active individual contributors and directors across all operational business units. It defines the mapping rules connecting missing skills to the 33 approved catalog courses, sets ranking heuristics for the Top-3 recommendations, and outlines the alternative developmental protocol for the 102 employees in the `Manager` cohort.

---

## Definitions
- **Learning Catalog:** The enterprise-curated portfolio of 33 accredited professional courses spanning executive communication, technical documentation, scientific rigor, cloud collaboration, and specialized engineering software.
- **Top-3 Recommendations:** The ranked list of three specific course interventions generated for an employee based on prioritized skill deficiencies.
- **Competency Priority Heuristic:** The deterministic ordering algorithm that prioritizes foundational cognitive and communication skills before technical software training.
- **Department-Level Learning Path:** The non-automated developmental pathway designed for managerial personnel focusing on organizational leadership, financial stewardship, and strategic coaching.

---

## Policy Rules

### Rule 1: Authoritative Catalog Constraints
All automated upskilling recommendations must be drawn strictly from the 33 pre-approved enterprise catalog courses configured in `notebooks/14_recommendation_engine.ipynb`. Recommending unaccredited third-party programs or unmapped courses through automated pipelines is prohibited. Key catalog offerings include:
- *Executive Presentation & Public Speaking Masterclass (Toastmasters / Internal Workshop)* — maps to `Speaking` competencies.
- *Technical & Regulatory Documentation Analysis Workshop* — maps to `Reading Comprehension` competencies.
- *Scientific Methodology, Evidence-Based Rigor & Laboratory Standards Training* — maps to `Science` competencies.
- *Google Workspace Collaboration: Document Co-Authoring & Cloud Governance* — maps to `Google Docs` competencies.
- *Bentley MicroStation CAD: 2D/3D Infrastructure Drafting & Asset Modeling* — maps to specialized drafting tool requirements.

### Rule 2: Deterministic Top-3 Selection Heuristic
The recommendation engine must generate exactly three distinct course recommendations for each eligible employee profile, serialized as a semicolon-delimited string in `Top3Recommendations`:
1. **First Priority:** Missing core communication or analytical competencies (e.g., `Speaking`, `Reading Comprehension`).
2. **Second Priority:** Missing functional domain competencies (e.g., `Science`, `Active Learning`).
3. **Third Priority:** Missing specialized software or tool proficiencies (e.g., `Bentley MicroStation`, `Google Docs`).
If an employee has fewer than three distinct missing competencies (as observed in `LOW Severity` cohorts), the engine backfills recommendations with advanced foundational mastery courses.

### Rule 3: Managerial Protocol Differentiation
In conformance with `POL-JOB-001` and `POL-SKILL-001`, the 102 employees classified as `Manager` do not receive automated course triples derived from O*NET role comparisons:
- Their recommendation field must explicitly state:
  `"N/A - Manager (use Department-level analysis)"`.
- Learning and development for managers must be coordinated directly through departmental HR partners using functional competency matrices tailored to their department (e.g., Sales Leadership vs. R&D Management).

### Rule 4: Voluntary and Developmental Nature of Recommendations
Course recommendations are intended to empower employee growth and must never be treated as punitive remediation:
- Completion of recommended courses should be integrated into annual professional development goals.
- Employees and their managers maintain discretion to substitute alternative accredited courses that align better with immediate project requirements.
- Lack of immediate course completion cannot be used as justification for negative performance appraisals.

---

## Procedure
1. **Diagnosis Ingestion:** The recommendation pipeline ingests the `missing_skills` list and `SkillGapSeverity` from the skill gap engine (`POL-SKILL-001`).
2. **Catalog Mapping:** Each missing competency is mapped to its associated catalog course ID using the verified lookup dictionary.
3. **Priority Ranking:** Courses are sorted according to the priority heuristic (Rule 2), filtering out duplicate course assignments.
4. **Output Generation:** The Top-3 recommended titles are formatted and appended to `employee_recommendations.csv`.
5. **Intelligence Integration:** The recommendations are merged into `employee_intelligence.csv` and surfaced on the Streamlit dashboard (`frontend/dashboard.py`) under the "Employee View" and "HR/Manager View".

---

## Exceptions and Limitations
- **Catalog Breadth:** The catalog contains exactly 33 configured courses. Certain highly specialized O*NET software skills may map to broader software competency equivalents rather than bespoke standalone training modules.
- **Dual-Mapped Roles:** `Healthcare Representative` and `Sales Representative` share O*NET skill baselines under `41-3091.00`, resulting in similar baseline upskilling recommendations. Managers should supplement recommendations with healthcare-specific compliance training.

---

## Human Review Requirements
In accordance with `POL-REVIEW-001`:
1. Managers must review recommended courses during semi-annual development discussions to ensure alignment with team operational priorities.
2. If an employee with `HIGH Severity` skill gaps (71 employees) is assigned training, the HR Learning Partner must ensure appropriate time allocation is provided during work hours to complete the curriculum.

---

## Data and Source References
- `data/processed/employee_recommendations.csv`: Production dataset containing generated recommendation profiles for 1,368 employees.
- `notebooks/14_recommendation_engine.ipynb`: Production logic implementing the 33-course catalog and selection heuristic.
- `data/processed/employee_intelligence.csv`: Capstone dataset containing merged `Top3Recommendations`.

---

## Related Policies
- `POL-SKILL-001`: Skill Gap Identification Policy (provides input missing competencies and gap severity).
- `POL-JOB-001`: Job Role Classification & O*NET Mapping Policy (governs baseline role crosswalking).
- `POL-CAREER-001`: Career & Occupation Mapping Policy (aligns upskilling with aspirational career paths).
- `POL-REVIEW-001`: Human Review of AI-Assisted HR Decisions Policy (governs manager-employee developmental reviews).
