# HR AI Monitoring and Model Limitations Policy

## Policy Metadata
- **Policy ID:** POL-MONITOR-001
- **Policy Title:** HR AI Monitoring and Model Limitations Policy
- **Policy Domain:** Model Lifecycle & Limitation Management
- **Version:** 1.0
- **Status:** Synthetic Demo Policy
- **Effective Date:** 2026-09-01
- **Owner:** Model Risk Management & AI Quality Assurance Directorate
- **Scope:** Ongoing performance tracking, statistical drift monitoring, limitation disclosure, and model recalibration
- **Classification:** Internal Demonstration Standard
- **Source Basis:** `docs/model_card.md`, `models/model_config.json`, `notebooks/09_model_versioning.ipynb`, `notebooks/07b_model_comparison_balanced.ipynb`

---

## Purpose
`POL-MONITOR-001` establishes operational controls, audit cadences, and continuous monitoring procedures for detecting model degradation, data drift, and performance drift in all machine learning models deployed within the Enterprise HR AI platform. Machine learning models deployed in human capital environments inevitably degrade over time as organizational demographics shift, compensation structures evolve, and workplace conditions change. By establishing performance monitoring boundaries, formalizing model limitations, and defining mandatory recalibration triggers, this policy guarantees the enduring safety, validity, and fairness of enterprise AI services.

---

## Scope
This policy governs the operational lifecycle, performance auditing, and limitation disclosures for:
1. The supervised attrition classification model (`logreg_balanced_threshold_0.40`, Version v3).
2. The SHAP explainability pipeline.
3. The hybrid knowledge retrieval and cross-encoder reranking infrastructure.
4. All future model candidates evaluated in the model versioning registry (`notebooks/09_model_versioning.ipynb`).

---

## Definitions
- **Model Drift:** The gradual degradation in predictive performance (Recall, Precision, ROC-AUC) over time caused by evolving workplace patterns and shifting employee demographics.
- **Concept Drift:** A fundamental alteration in the statistical relationship between employee input features (e.g., `OverTime`, `BusinessTravel`) and actual voluntary attrition.
- **Performance Floor:** The minimum acceptable statistical performance metric below which model inference must be suspended pending retraining.
- **Model Card:** The authoritative documentation artifact (`docs/model_card.md`) cataloging technical specifications, training provenance, and operational boundaries.

---

## Policy Rules

### Rule 1: Mandatory Model Limitations Disclosure
All stakeholders, users, and platform administrators must be explicitly informed of the five foundational model limitations documented in `docs/model_card.md`:
1. **Synthetic Training Data:** The model was trained exclusively on the synthetic IBM HR Analytics dataset (1,470 records). Performance in actual enterprise environments is unverified and requires local validation.
2. **Class Imbalance Precision Ceiling:** The 84/16 class distribution inherently caps model precision at 0.3426 at the operational threshold of 0.40; approximately 2 out of 3 flagged employees will not leave.
3. **JobRole to O*NET Taxonomy Gap:** Direct string matching fails across enterprise roles; automated skill recommendations rely on the crosswalk (`jobrole_onet_mapping.csv`) and are restricted for generic managers.
4. **Static Point-in-Time Nature:** The model captures historical cross-sectional relationships and cannot autonomously incorporate macroeconomic or organizational shifts without formal retraining.
5. **Absence of Causal Inference:** SHAP feature rankings identify statistical associations, not proven causal drivers.

### Rule 2: Minimum Acceptable Performance Floors
The production attrition model must maintain performance above established statistical floors when evaluated against held-out validation or operational ground-truth cohorts:
- **Recall Floor:** $\ge 0.6500$ (Production benchmark: 0.7872).
- **Precision Floor:** $\ge 0.2500$ (Production benchmark: 0.3426).
- **ROC-AUC Floor:** $\ge 0.7500$ (Production benchmark: 0.8060).
If measured recall falls below 0.6500 or precision drops below 0.2500 during quarterly evaluations, production scoring must be paused for retraining.

### Rule 3: Quarterly Audit and Recalibration Cadence
The Model Risk Management committee must execute formal quarterly performance audits comparing predicted risk flags against actual voluntary turnover events:
- Reviewers must assess whether the top 5 SHAP feature drivers (OverTime, YearsSinceLastPromotion, TotalWorkingYears, Travel_Frequently, JobLevel) retain consistent rankings.
- Reviewers must calculate human override rates recorded under `POL-REVIEW-001`. A departmental override rate exceeding 40% triggers immediate feature drift analysis.

### Rule 4: Controlled Threshold Adjustment Governance
Adjustments to the decision threshold (currently 0.40) may only be executed through formal governance:
- Any proposed shift in threshold (e.g., raising to 0.50 to increase precision, or lowering to 0.30 to increase recall) must be evaluated across the complete confusion matrix.
- Threshold adjustments must be approved by the AI Quality Assurance Directorate and committed to `models/model_config.json`.
- Modifying threshold parameters via unversioned script arguments is strictly prohibited.

---

## Procedure
1. **Telemetry Collection:** Ground-truth employment status updates (separations, transfers) are logged at the close of each quarter.
2. **Metric Evaluation:** The audit pipeline evaluates historical predictions against observed outcomes, computing updated confusion matrices, Recall, Precision, and ROC-AUC.
3. **Drift Detection:** The system compares quarterly feature distributions against the baseline distributions established during model training.
4. **Audit Reporting:** A formal Model Health Report is generated, detailing current metrics, drift indices, and override statistics.
5. **Remediation Trigger:** If any metric violates Rule 2, the committee initiates model re-training and version advancement following the procedures in `notebooks/09_model_versioning.ipynb`.

---

## Exceptions and Limitations
- **Evaluation Lags:** In low-turnover business units, voluntary separation events may be too sparse to compute statistically robust quarterly precision metrics. In such cases, rolling 12-month metrics must be utilized.
- **Synthetic Demonstration Context:** The operational metrics described here reflect the synthetic baseline dataset.

---

## Human Review Requirements
In accordance with `POL-REVIEW-001`:
1. Quarterly Model Health Reports must be signed off by both the Lead Data Scientist and the VP of People Analytics.
2. Any model retraining or parameter modification requires documented sign-off before artifact deployment to `models/attrition_model.pkl`.

---

## Data and Source References
- `docs/model_card.md`: Definitive model documentation detailing performance metrics, SHAP rankings, and limitations.
- `models/model_config.json`: Production configuration controlling the operational decision threshold.
- `notebooks/09_model_versioning.ipynb`: Master notebook governing model evaluation, versioning, and registry management.

---

## Related Policies
- `POL-MODEL-001`: Attrition Model Usage Policy (governs operational utilization of the monitored model).
- `POL-AI-001`: HR AI Decision-Support Governance Policy (governs overarching ethical mandates).
- `POL-DATA-001`: HR Data Usage & Source Provenance Policy (governs dataset integrity and lineage).
- `POL-REVIEW-001`: Human Review of AI-Assisted HR Decisions Policy (provides override feedback data).
