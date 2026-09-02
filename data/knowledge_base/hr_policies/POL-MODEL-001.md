# Attrition Model Usage Policy

## Policy Metadata
- **Policy ID:** POL-MODEL-001
- **Policy Title:** Attrition Model Usage Policy
- **Policy Domain:** Predictive Analytics & Model Operations
- **Version:** 1.0
- **Status:** Synthetic Demo Policy
- **Effective Date:** 2026-09-01
- **Owner:** People Analytics Modeling Group & HR Operations
- **Scope:** Operational utilization, score interpretation, and risk thresholding for employee attrition modeling
- **Classification:** Internal Demonstration Standard
- **Source Basis:** `docs/model_card.md`, `models/model_config.json`, `notebooks/07b_model_comparison_balanced.ipynb`, `notebooks/08_shap_explainability.ipynb`

---

## Purpose
`POL-MODEL-001` defines the operational standards, analytical boundaries, and interpretation criteria for deploying the supervised employee attrition risk model (`logreg_balanced_threshold_0.40`). The objective of predictive attrition modeling is to enable early, compassionate intervention for talent flight risk before voluntary resignations occur. By defining standardized decision thresholds, establishing precision/recall expectations, and mandating explainability requirements, this policy ensures model scores are utilized responsibly, effectively, and equitably.

---

## Scope
This policy applies to all personnel and automated workflows utilizing predictions from the attrition model artifact (`models/attrition_model.pkl`) and its operational configuration (`models/model_config.json`). It directly guides HR business partners, workforce planners, and People Analytics practitioners conducting employee risk tiering across the 1,470 employees in the enterprise baseline.

---

## Definitions
- **Attrition Risk Score:** A calibrated continuous probability between 0.0000 and 1.0000 indicating the likelihood of an employee voluntarily separating from the organization within a projected evaluation cycle.
- **Decision Threshold (0.40):** The operational cutoff probability stored in `models/model_config.json`. Employees scoring $\ge 0.40$ are categorized as `HIGH` risk; employees scoring $< 0.40$ are categorized as `LOW` risk.
- **Balanced Class Weighting:** The mathematical adjustment (`class_weight='balanced'`) applied during logistic regression training to compensate for the natural 84/16 class imbalance (83.9% retained vs 16.1% departed).
- **SHAP Feature Attribution:** Shapley Additive Explanations computed via `shap.LinearExplainer` indicating the directional contribution of each demographic, compensatory, and environmental feature to an individual employee's risk score.

---

## Policy Rules

### Rule 1: Mandatory Decision Threshold Standard (0.40)
The platform establishes **0.40** as the enterprise-standard decision threshold for binary flight-risk classification. This threshold is established based on empirical evaluation documented in `docs/model_card.md`:
- At threshold 0.40, the model achieves a **Recall of 0.7872**, successfully capturing 37 out of 47 true leavers in the held-out test cohort (294 rows).
- Hardcoding custom thresholds in application logic or analytical scripts is strictly prohibited; all services must dynamically load the threshold parameter from `models/model_config.json`.

### Rule 2: Recognition of the Class Imbalance Precision Trade-Off
HR practitioners must understand and communicate the mathematical trade-off inherent to prioritizing leaver recall under severe class imbalance (84% stayers / 16% leavers):
- At threshold 0.40, the model achieves a **Precision of 0.3426** (37 true leavers out of 108 flagged individuals).
- Approximately two out of every three employees flagged as `HIGH` risk will remain with the organization.
- Consequently, a `HIGH` risk flag must **never** be interpreted as a certain resignation, an act of disloyalty, or a lack of engagement. It serves exclusively as a risk indicator warranting exploratory management dialogue.

### Rule 3: Interpretability via Top 5 SHAP Feature Drivers
All risk scores presented in managerial or HR interfaces must be accompanied by the top individualized SHAP feature drivers explaining the statistical score. Practitioners must interpret these features in accordance with the documented enterprise rankings:
1. `OverTime` (Mean |SHAP|: 0.6558) — Frequent overtime is the primary empirical indicator of departure risk.
2. `YearsSinceLastPromotion` (Mean |SHAP|: 0.5663) — Career stagnation without formal advancement elevates mobility risk.
3. `TotalWorkingYears` (Mean |SHAP|: 0.5573) — Early-career employees exhibit higher natural market mobility.
4. `BusinessTravel_Travel_Frequently` (Mean |SHAP|: 0.5177) — High travel frequency correlates with burnout and attrition.
5. `JobLevel` (Mean |SHAP|: 0.4585) — Lower organizational job tiers exhibit higher turnover propensity.

### Rule 4: Prohibition of Adverse Action
Under no circumstances may an attrition risk prediction be cited or utilized as justification for:
- Preemptive restructuring or withholding compensation, bonuses, or equity grants.
- Withholding project assignments, professional training, or promotion opportunities.
- Questioning an employee's organizational loyalty or initiating performance management.
Any such use violates `POL-AI-001` and triggers formal ethics escalation.

---

## Procedure
1. **Feature Vector Assembly:** Pipeline ingests 48 engineered features adhering to `app/validation/employee_schema.py`. Continuous variables are scaled using `StandardScaler` fitted strictly on the training partition.
2. **Probability Computation:** The logistic regression estimator computes the raw class probability $P(\text{Attrition} = 1)$.
3. **Threshold Application:** The system compares probability against `threshold = 0.40`.
   - If score $\ge 0.40$: Record assigned `RiskLevel = "HIGH"`.
   - If score $< 0.40$: Record assigned `RiskLevel = "LOW"`.
4. **SHAP Decomposition:** `shap.LinearExplainer` computes local additive contributions for all features.
5. **Surfacing Intelligence:** The intelligence record is written to `employee_intelligence.csv` and presented in `frontend/dashboard.py` under the "HR/Manager View" for review per `POL-RISK-001`.

---

## Exceptions and Limitations
- **Synthetic Data Foundation:** The model is trained on IBM's synthetic demonstration data. Real-world workforce dynamics will differ significantly.
- **Static Point-in-Time Inference:** Predictions reflect a static snapshot of historical variables. They do not automatically adjust for real-time external macroeconomic shifts, regional labor market demand, or sudden organizational re-organizations.
- **Absence of Causal Proof:** SHAP feature attributions reflect statistical associations. Reducing overtime will not mechanically guarantee retention if compensation or leadership issues predominate.

---

## Human Review Requirements
In accordance with `POL-REVIEW-001`:
1. Every employee flagged as `HIGH` risk must be reviewed by their designated HR Business Partner within 30 days of monthly scoring runs.
2. Retention discussions must be conducted constructively, focusing on workload balance, career trajectory, and professional recognition.
3. HR reviewers must document whether the risk flag was operationally validated by qualitative context.

---

## Data and Source References
- `models/attrition_model.pkl`: Serialized production logistic regression model artifact.
- `models/model_config.json`: Master configuration storing feature names, preprocessing parameters, and `threshold = 0.40`.
- `docs/model_card.md`: Model card detailing test set performance (Recall 0.7872, Precision 0.3426, ROC-AUC 0.8060).
- `data/processed/employee_attrition_processed.csv`: Cleaned anchor dataset containing 1,470 records.

---

## Related Policies
- `POL-AI-001`: HR AI Decision-Support Governance Policy (governs overarching ethics and non-autonomous boundaries).
- `POL-RISK-001`: Workforce Risk Review Policy (governs operational retention review cadences).
- `POL-REVIEW-001`: Human Review of AI-Assisted HR Decisions Policy (governs human oversight and override protocols).
- `POL-MONITOR-001`: HR AI Monitoring & Model Limitations Policy (governs performance tracking and threshold recalibration).
