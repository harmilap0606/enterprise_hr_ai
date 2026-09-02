# Career and Occupation Mapping Policy

## Policy Metadata
- **Policy ID:** POL-CAREER-001
- **Policy Title:** Career and Occupation Mapping Policy
- **Policy Domain:** Internal Mobility & Career Pathways
- **Version:** 1.0
- **Status:** Synthetic Demo Policy
- **Effective Date:** 2026-09-01
- **Owner:** Enterprise Talent Mobility & Career Architecture Directorate
- **Scope:** Internal career lattice modeling, cross-functional transition mapping, and occupational progression
- **Classification:** Internal Demonstration Standard
- **Source Basis:** `data/processed/occupation_master.csv`, `data/external/jobrole_onet_mapping.csv`, `docs/model_card.md`, `notebooks/10_onet_role_mapping.ipynb`

---

## Purpose
`POL-CAREER-001` defines the standards and governance mechanisms for structuring internal career progression, cross-functional job mobility, and occupational pathway modeling across the organization. Long-term talent retention depends on transparent, attainable career trajectories. As documented in the enterprise attrition model (`docs/model_card.md`), stagnation—quantified by `YearsSinceLastPromotion` (mean |SHAP|: 0.5663) and entry-level status reflected in `JobLevel` (mean |SHAP|: 0.4585)—represents two of the top five statistical drivers of employee turnover. By leveraging the standardized occupational master catalog (`occupation_master.csv`), this policy establishes clear competency ladders that facilitate equitable upward and lateral mobility.

---

## Scope
This policy applies to all 1,470 employees across all departments and job levels (Levels 1 through 5). It governs how internal job roles, formal O*NET occupational descriptions, career tenures, and skill overlaps are utilized to construct internal career pathways.

---

## Definitions
- **Occupational Catalog:** The repository of 1,016 canonical O*NET occupations maintained in `occupation_master.csv` detailing official SOC codes, standardized titles, and comprehensive functional descriptions.
- **Career Pathway:** A structured sequence of progressive or lateral job roles characterized by cumulative competency requirements, increasing organizational responsibility, and ascending `JobLevel` tiers (1 to 5).
- **Competency Transferability:** The degree of overlap between the essential and software skills of an employee's current O*NET profile and those required by an aspirational target role.
- **Promotion Velocity Stagnation:** An operational condition where an employee's `YearsSinceLastPromotion` exceeds organizational norms (e.g., $\ge 4$ years without advancement), significantly elevating flight risk.

---

## Policy Rules

### Rule 1: Canonical Description Authority
All internal career counseling, role exploration dashboards, and transition mapping tools must source occupational descriptions exclusively from `occupation_master.csv`. Inventing arbitrary job descriptions or non-standard occupational requirements is prohibited:
- For example, when an employee in a `Laboratory Technician` role (mapped to `29-2012.00`) explores progression toward scientific research, the target profile must be grounded in canonical O*NET profiles such as `19-1042.00` (*Medical Scientists, Except Epidemiologists*) or `15-1221.00` (*Computer and Information Research Scientists*).

### Rule 2: Career Path Progression along JobLevel Tiers
Internal career pathways must define clear milestone criteria anchored to the standardized `JobLevel` hierarchy (1 to 5) present in the employee schema:
- **Level 1 (Entry / Junior Contributor):** Focus on foundational task mastery and essential skill acquisition (`POL-SKILL-001`).
- **Level 2 (Intermediate Contributor):** Autonomous task execution and broadening technical proficiencies.
- **Level 3 (Senior Contributor):** Complex problem-solving, project management, and cross-functional leadership.
- **Level 4 (Principal / Lead / Director):** Domain strategy, high-impact operational leadership, and mentorship.
- **Level 5 (Executive / Functional Head):** Enterprise strategy, organizational governance, and resource allocation.

### Rule 3: Mitigating Turnover Stagnation
HR business partners and managers must actively monitor employees exhibiting high promotion stagnation (`YearsSinceLastPromotion \ge 4`) or tenure concentration (`YearsInCurrentRole \ge 5`):
- When an employee reaches Level 2 or Level 3 with more than 3 years without a promotion, an active Career Pathing Review must be initiated.
- The review must evaluate whether the employee can advance vertically or embark on a lateral cross-functional pathway (e.g., from `Sales Representative` to `Human Resources Specialist`).

### Rule 4: Managing Dual-Mapped and Catch-All Role Transitions
When employees in dual-mapped roles (`Healthcare Representative` or `Sales Representative` under `41-3091.00`) or placeholder roles (`Manager` under `11-9199.00`) explore career pathways:
- Career mobility advisors cannot rely solely on the O*NET baseline.
- Advisors must conduct a granular assessment of individual achievements, domain certifications, and departmental specializations before validating transition feasibility.

---

## Procedure
1. **Aspirational Role Identification:** An employee or manager identifies a target aspirational role in the enterprise portal.
2. **Taxonomic Benchmark Retrieval:** The platform queries `occupation_master.csv` for the target role's `O*NET-SOC Code` and retrieves canonical job requirements.
3. **Skill Gap Delta Evaluation:** The system compares the employee's current inventory against the target profile, outputting missing competencies (`POL-SKILL-001`).
4. **Curriculum Alignment:** The platform suggests specific catalog courses to bridge the delta (`POL-LEARN-001`).
5. **Career Milestone Agreement:** The employee, manager, and talent partner draft an agreed Development Action Plan establishing realistic timelines (e.g., 12–24 months) for progression to the next `JobLevel`.

---

## Exceptions and Limitations
- **Demographic Model Baseline:** Tenures and promotion intervals documented in the baseline reflect the synthetic IBM dataset.
- **Cross-Functional Feasibility:** High competency overlap does not guarantee immediate job opening availability. Pathway modeling defines capability readiness, not guaranteed staffing placement.

---

## Human Review Requirements
In accordance with `POL-REVIEW-001`:
1. All cross-functional career transfers require written approval from both the releasing and receiving department leaders.
2. Managers are prohibited from blocking the lateral or upward career progression of an employee whose promotion stagnation has placed them in the `HIGH` attrition risk tier.

---

## Data and Source References
- `data/processed/occupation_master.csv`: Authoritative repository of 1,016 occupational definitions and descriptions.
- `data/external/jobrole_onet_mapping.csv`: Crosswalk mapping internal roles to O*NET standards.
- `docs/model_card.md`: Documents `YearsSinceLastPromotion` and `JobLevel` as primary attrition drivers.

---

## Related Policies
- `POL-JOB-001`: Job Role Classification & O*NET Mapping Policy (governs role crosswalking).
- `POL-SKILL-001`: Skill Gap Identification Policy (governs prerequisite competency identification).
- `POL-LEARN-001`: Employee Upskilling Recommendation Policy (governs preparatory training).
- `POL-RISK-001`: Workforce Risk Review Policy (identifies career-stagnated employees requiring mobility reviews).
