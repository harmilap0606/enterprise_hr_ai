# Model Card — Enterprise HR Attrition Risk Model

**Version:** v3  
**Date:** 2026-09-01  
**Status:** Production  
**Maintained by:** Enterprise HR AI Project Team  

---

## Model Name

`logreg_balanced_threshold_0.40`  
Algorithm: Logistic Regression (`class_weight='balanced'`, `max_iter=1000`, `solver='lbfgs'`)

---

## Intended Use

**Primary use:** Attrition risk scoring for HR business partners and People Analytics teams.  
The model assigns a probability score (0–1) to each active employee indicating their likelihood
of voluntarily leaving the organization. Employees with a score ≥ 0.40 are flagged for
proactive retention review.

**Intended users:** HR managers, People Analytics teams, department heads conducting retention reviews.  
**Not intended for:** Automated employment decisions, performance management, compensation determination,
or any legally consequential HR action without human review.

---

## Training Data

- **Source:** IBM HR Analytics Employee Attrition & Performance dataset (publicly available on Kaggle).
  This is a **synthetic dataset** created by IBM data scientists for demonstration purposes.
- **Size:** 1,470 employees (1,176 train / 294 test, 80/20 stratified split, `random_state=42`)
- **Features:** 48 engineered features derived from demographics, job characteristics, compensation,
  satisfaction scores, and career history. Encoded with `pd.get_dummies(drop_first=True)` and
  standardized with `StandardScaler` (fitted on training set only).
- **Target:** `Attrition` (binary: 1=Yes/Left, 0=No/Stayed)
- **Class balance:** 83.9% stayed / 16.1% left (84/16 imbalance)

---

## Performance Metrics (Test Set, Threshold = 0.40)

| Metric | Value | Notes |
|:---|:---:|:---|
| **Recall** | **0.7872** | Caught 37 of 47 true leavers (10 missed) |
| **Precision** | **0.3426** | 37 true positives out of 108 flagged employees |
| **F1 Score** | **0.4774** | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.8060** | Ranking quality across all thresholds |
| **Confusion Matrix** | TN=176, FP=71, FN=10, TP=37 | At threshold=0.40 on 294-row test set |

**Decision Threshold:** `0.40` (stored in `models/model_config.json` for use by the API layer;
not hardcoded in application code)

---

## Top 5 SHAP Feature Drivers

Computed using `shap.LinearExplainer` on the held-out test set (Notebook 08).

| Rank | Feature | Mean |SHAP| | HR Plain-Language Meaning |
|:---:|:---|:---:|:---|
| 1 | `OverTime` | 0.6558 | Employees who regularly work overtime are significantly more likely to leave. |
| 2 | `YearsSinceLastPromotion` | 0.5663 | Long stretches without career advancement signal stagnation and flight risk. |
| 3 | `TotalWorkingYears` | 0.5573 | Earlier-career employees have higher mobility; invest in retention from day one. |
| 4 | `BusinessTravel_Travel_Frequently` | 0.5177 | Frequent business travel is a major burnout and attrition driver. |
| 5 | `JobLevel` | 0.4585 | Junior-level employees are more likely to leave; clarify promotion paths. |

---

## Known Limitations

1. **Synthetic training data:** This model was trained on IBM's synthetic Kaggle dataset.
   Real-world performance on actual company employee data is **unverified**. Before deploying
   in production, the model should be re-trained or at minimum validated on real organizational data.

2. **Class imbalance ceiling on precision:** The 84/16 class imbalance means that even at
   high recall (0.7872), precision is inherently limited (0.3426 — roughly 1 in 3 flagged
   employees is a true leaver). This is an expected trade-off given the business priority
   of catching leavers (Recall primary). HR teams should be briefed that not every flagged
   employee will leave — flags are risk indicators, not certainties.

3. **JobRole ↔ O*NET taxonomy gap:** The `JobRole` categories in this dataset (e.g.
   'Laboratory Technician', 'Sales Representative') do not map directly to the O*NET
   occupational taxonomy used in the reference dataset (`occupation_master.csv`).
   Role-based skill recommendations planned for Day 3 will require the separate
   `data/external/jobrole_onet_mapping.csv` mapping table, which has not yet been finalized.
   Until that mapping exists, O*NET-derived recommendations cannot be reliably attributed
   to specific job roles.

4. **Static model:** The model captures a point-in-time snapshot of historical attrition patterns.
   It does not update automatically as organizational conditions change. A retraining schedule
   (e.g. quarterly) should be established once deployed on real data.

5. **No causal inference:** SHAP values identify statistical associations, not causal drivers.
   For example, 'OverTime' being the top driver does not prove that reducing overtime will
   reduce attrition — it means employees who work overtime tend to leave more often.
   Interventions should be designed with HR domain expertise, not derived mechanically from
   model outputs.

6. **O*NET taxonomy artifact in skill gap rollups:** The top organization-wide skill gaps
   (Speaking, Reading Comprehension, Active Listening, Critical Thinking) are partly an artifact
   of O*NET's essential-skills taxonomy, where these general skills appear in nearly every
   occupation's requirement list -- their high missing-counts reflect breadth of appearance across
   roles combined with random synthetic assignment, not necessarily the most business-critical
   real-world gap.

---

## Artifact Locations

| Artifact | Path |
|:---|:---|
| Production model | `models/attrition_pipeline.joblib` |
| Model config (threshold) | `models/model_config.json` |
| Feature scaler | `models/scaler.joblib` |
| Model registry | `models/model_registry.json` |
| Archived v1 baseline | `models/baseline_logreg.joblib` |
| Archived v2 candidate | `models/archive/attrition_pipeline_candidate_v1.joblib` |
| SHAP plots | `reports/shap/` (4 PNG files) |

---

## Version History

| Version | Model | Threshold | Recall | F1 | Status |
|:---:|:---|:---:|:---:|:---:|:---|
| v1 | LogReg (unweighted) | 0.50 | 0.3617 | 0.4658 | Superseded |
| v2 | XGBoost (default) | 0.50 | 0.2766 | 0.3714 | Rejected |
| **v3** | **LogReg (balanced)** | **0.40** | **0.7872** | **0.4774** | **Production** |

---

*This model card follows the Model Cards for Model Reporting standard (Mitchell et al., 2019).*
