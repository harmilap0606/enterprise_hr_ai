# Workforce Risk Review Policy

## Policy Metadata
- **Policy ID:** POL-RISK-001
- **Policy Title:** Workforce Risk Review Policy
- **Policy Domain:** Talent Retention & Risk Mitigation
- **Version:** 1.0
- **Status:** Synthetic Demo Policy
- **Effective Date:** 2026-09-01
- **Owner:** Global Talent Management & HR Operations Committee
- **Scope:** Operational retention reviews, employee flight-risk tiering, and managerial intervention protocols
- **Classification:** Internal Demonstration Standard
- **Source Basis:** `docs/model_card.md`, `data/processed/employee_intelligence.csv`, `notebooks/15_employee_intelligence.ipynb`, `data/processed/engagement_processed.csv`

---

## Purpose
`POL-RISK-001` establishes standard operating procedures, timeline requirements, and cross-functional protocols for reviewing employee attrition risk signals surfaced by the Enterprise HR AI platform. Proactive talent retention requires a structured, multi-dimensional methodology that integrates predictive model probabilities, individual explainability factors, engagement survey context, and professional human judgment. This policy standardizes how HR business partners and people leaders conduct empathetic, constructive retention reviews across all organizational units.

---

## Scope
This policy applies to all 1,470 active employee profiles tracked within the enterprise intelligence repository (`employee_intelligence.csv`), spanning:
- Research & Development (961 employees)
- Sales (446 employees)
- Human Resources (63 employees)

It directly governs the review process for the 585 employees (39.80% of workforce) classified as `HIGH` risk under the operational threshold established by `POL-MODEL-001`.

---

## Definitions
- **Workforce Risk Tier:** The risk segmentation reflecting predicted attrition probability:
  - `HIGH Risk`: Risk score $\ge 0.40$ (585 employees / 39.80% in baseline).
  - `LOW Risk`: Risk score $< 0.40$ (885 employees / 60.20% in baseline).
- **Multi-Dimensional Review:** An assessment that evaluates predictive model probability alongside employee engagement scores, satisfaction metrics, work-life balance scores, and tenure context.
- **Engagement Coverage Gap:** The documented architectural reality whereby 731 employees (49.7%) possess engagement survey records (`engagement_processed.csv`), while 739 employees (50.3%) have null engagement data.
- **Retention Intervention:** A supportive developmental, compensatory, or workload modification negotiated between manager, HR, and employee to mitigate flight risk drivers.

---

## Policy Rules

### Rule 1: Prioritization of High-Risk Talent Cohorts
HR business partners and department heads must conduct quarterly workforce risk reviews prioritizing employees in the `HIGH` risk tier (score $\ge 0.40$). Reviews must evaluate individual risk concentrations within departments:
- Research & Development and Sales departments exhibit specific operational risk clusters that require distinct retention approaches.
- Within the high-risk cohort, priority must be given to employees who also exhibit high skill-gap severity (`POL-SKILL-001`) or critical organizational tenure.

### Rule 2: Multi-Dimensional Evaluation Protocol
A flight risk flag cannot be reviewed in isolation. The reviewer must evaluate the four verified dimensions surfaced in `employee_intelligence.csv`:
1. **Model Probability & Top SHAP Driver:** Review whether the primary flight risk driver is structural (e.g., `OverTime = Yes`, `BusinessTravel = Travel_Frequently`) or career-based (`YearsSinceLastPromotion \ge 5`).
2. **Engagement & Satisfaction Context:** Review available `EngagementScore`, `SatisfactionScore`, and `WorkLifeBalanceScore`.
3. **Role Architecture & Confidence:** Cross-reference `ONET_Title` and `ONET_Confidence` (`POL-JOB-001`).
4. **Competency Alignment:** Review `SkillGapSeverity` and corresponding recommended course curricula (`POL-LEARN-001`).

### Rule 3: Equitable Handling of Missing Engagement Records
In accordance with `docs/data_relationships.md`, engagement survey records cover only 49.7% of the workforce:
- Reviewers are strictly prohibited from dismissing, penalizing, or deprioritizing an employee simply because their engagement or satisfaction fields contain `NaN` or null values.
- For employees lacking survey records, reviewers must place higher diagnostic weight on objective operational variables (e.g., overtime hours logged, time since last promotion, total tenure) and schedule an exploratory 1-on-1 discussion.

### Rule 4: Constructive Retention Interventions Only
All interventions resulting from a risk review must be supportive and developmental. Permissible retention actions include:
- Workload rebalancing and overtime reduction agreements.
- Career path clarification, internal mobility exploration (`POL-CAREER-001`), or mentorship pairing.
- Enrolling the employee in personalized upskilling courses identified in `employee_intelligence.csv` (`POL-LEARN-001`).
Adverse, punitive, or surveillance-oriented actions are strictly prohibited per `POL-AI-001`.

---

## Procedure
1. **Quarterly Batch Review Ingestion:** At the start of each fiscal quarter, HR business partners extract department-specific rosters from `employee_intelligence.csv` filtered by `RiskLevel == 'HIGH'`.
2. **Factor Synthesis:** The partner inspects the individual record, noting `RiskScore`, `JobRole`, `OverTime` status, `YearsSinceLastPromotion`, and `SkillGapSeverity`.
3. **Manager Alignment Session:** The HR partner convenes a confidential alignment session with the employee's direct manager to share context, review qualitative team dynamics, and establish whether observed drivers match workplace realities.
4. **Stay Interview Scheduling:** A structured, supportive "Stay Interview" is scheduled with the employee to explore career aspirations, workload sustainability, and development goals without disclosing algorithmic scores.
5. **Action Plan Recording:** Agreed-upon development or workload adjustments are documented in the HR talent system and cross-referenced with `POL-REVIEW-001`.

---

## Exceptions and Limitations
- **Demonstration Dataset Caveat:** Baseline metrics (585 High Risk / 885 Low Risk) are generated from synthetic demo records.
- **Precision Ceiling Awareness:** As established in `POL-MODEL-001`, model precision is 0.3426. Reviewers must anticipate that approximately 65% of flagged employees are stable, highly committed contributors whose high scores reflect demographic or workload attributes rather than active departure intent.

---

## Human Review Requirements
In accordance with `POL-REVIEW-001`:
1. Every retention intervention plan must be co-signed by the designated HR Business Partner and Department Director.
2. Direct disclosure of numerical attrition probabilities to employees is prohibited to prevent undue stress or misunderstanding.
3. Review status must be updated quarterly in People Analytics records.

---

## Data and Source References
- `data/processed/employee_intelligence.csv`: Capstone dataset containing comprehensive risk tiers, survey scores, and skill gap severities for 1,470 employees.
- `data/processed/engagement_processed.csv`: Survey enrichment source providing engagement metrics for 731 matching employees.
- `docs/model_card.md`: Technical documentation detailing the 0.40 threshold and precision/recall trade-offs.

---

## Related Policies
- `POL-MODEL-001`: Attrition Model Usage Policy (governs model scoring mechanics and threshold definitions).
- `POL-AI-001`: HR AI Decision-Support Governance Policy (governs ethical constraints and non-autonomous requirements).
- `POL-SKILL-001`: Skill Gap Identification Policy (governs integration of skill deficiency data with risk reviews).
- `POL-REVIEW-001`: Human Review of AI-Assisted HR Decisions Policy (governs formal review logging and oversight).
