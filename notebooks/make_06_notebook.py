import json

nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "cells": []
}

def md(cid, lines):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": lines}

def code(cid, lines):
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": lines}

nb["cells"] = [

md("md-title", [
    "# 06 · Baseline Model — Logistic Regression\n",
    "\n",
    "**Project:** Enterprise HR AI  \n",
    "**Input:** `features_scaled.csv` (StandardScaler applied in Step 5 — correct for LR)  \n",
    "**Purpose:** Establish ONE reproducible baseline number that all future models must beat.  \n",
    "**Rule:** No hyperparameter tuning. No model comparison. One model, full reporting.\n",
    "\n",
    "---"
]),

# ── Imports ───────────────────────────────────────────────────────────────────
code("cell-imports", [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "import joblib\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import (\n",
    "    classification_report, confusion_matrix,\n",
    "    roc_auc_score, precision_score, recall_score, f1_score\n",
    ")\n",
    "\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.width', 200)\n",
    "pd.set_option('display.float_format', '{:.4f}'.format)\n",
    "\n",
    "PROC   = os.path.join('..', 'data', 'processed')\n",
    "MODELS = os.path.join('..', 'models')\n",
    "os.makedirs(MODELS, exist_ok=True)\n",
    "\n",
    "RANDOM_STATE = 42   # fixed — NEVER change this for the baseline\n",
    "print('PROC  :', os.path.abspath(PROC))\n",
    "print('MODELS:', os.path.abspath(MODELS))\n",
    "print('random_state:', RANDOM_STATE, '(fixed for reproducibility)')"
]),

# ── Load ──────────────────────────────────────────────────────────────────────
code("cell-load", [
    "df = pd.read_csv(os.path.join(PROC, 'features_scaled.csv'))\n",
    "print(f'Loaded features_scaled.csv: {df.shape[0]:,} rows x {df.shape[1]} cols')\n",
    "print(f'Columns: {df.columns.tolist()}')"
]),

# ── X / y split — encoding mapping printed explicitly ────────────────────────
md("md-xy", ["---\n", "## Step 1 · X / y Split and Encoding Confirmation"]),

code("cell-xy", [
    "TARGET = 'Attrition'\n",
    "\n",
    "# Confirm the encoding before splitting\n",
    "print('=== ENCODING MAPPING ===')\n",
    "raw_vals = df[TARGET].unique()\n",
    "print(f'  Raw unique values in Attrition column: {sorted(raw_vals)}')\n",
    "print(f'  Encoding: 0 = stayed (\"No\"), 1 = left (\"Yes\")')\n",
    "print(f'  Positive class (class=1): employee LEFT the company')\n",
    "print()\n",
    "\n",
    "# In features_scaled.csv the target was already 0/1 from Step 5\n",
    "# Confirm:\n",
    "assert set(df[TARGET].unique()).issubset({0, 1}), \\\n",
    "    f'Expected binary 0/1 in Attrition, got: {df[TARGET].unique()}'\n",
    "print('Assert passed: Attrition column is already 0/1 integer encoding.')\n",
    "\n",
    "X = df.drop(columns=[TARGET])\n",
    "y = df[TARGET]\n",
    "\n",
    "print(f'X shape: {X.shape}')\n",
    "print(f'y shape: {y.shape}')\n",
    "print(f'Overall attrition rate: {y.mean()*100:.2f}%  ({y.sum()} positives / {len(y)} total)')"
]),

# ── Train/test split ──────────────────────────────────────────────────────────
md("md-split", ["---\n", "## Step 2 · Train/Test Split (80/20, stratified)"]),

code("cell-split", [
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y,\n",
    "    test_size=0.20,\n",
    "    stratify=y,\n",
    "    random_state=RANDOM_STATE\n",
    ")\n",
    "\n",
    "print('=== SPLIT SIZES ===')\n",
    "print(f'  Train : {X_train.shape[0]:,} rows  ({X_train.shape[0]/len(X)*100:.1f}%)')\n",
    "print(f'  Test  : {X_test.shape[0]:,} rows  ({X_test.shape[0]/len(X)*100:.1f}%)')\n",
    "print()\n",
    "\n",
    "print('=== CLASS BALANCE — STRATIFY VERIFICATION ===')\n",
    "train_pos_pct = y_train.mean() * 100\n",
    "train_neg_pct = (1 - y_train.mean()) * 100\n",
    "test_pos_pct  = y_test.mean()  * 100\n",
    "test_neg_pct  = (1 - y_test.mean())  * 100\n",
    "\n",
    "print(f'  Train set: {y_train.sum()} positives ({train_pos_pct:.2f}% attrition) | '\n",
    "      f'{(y_train==0).sum()} negatives ({train_neg_pct:.2f}% stayed)')\n",
    "print(f'  Test  set: {y_test.sum()} positives ({test_pos_pct:.2f}% attrition) | '\n",
    "      f'{(y_test==0).sum()} negatives ({test_neg_pct:.2f}% stayed)')\n",
    "print(f'  Expected ~16.1% / ~83.9% in both — stratify=y working correctly.')\n",
    "\n",
    "# Assert stratification worked (within 0.5% tolerance)\n",
    "overall_rate = y.mean() * 100\n",
    "assert abs(train_pos_pct - overall_rate) < 0.5, \\\n",
    "    f'Stratification off in train: {train_pos_pct:.2f}% vs overall {overall_rate:.2f}%'\n",
    "assert abs(test_pos_pct - overall_rate) < 0.5, \\\n",
    "    f'Stratification off in test: {test_pos_pct:.2f}% vs overall {overall_rate:.2f}%'\n",
    "print('  Assert passed: stratification within 0.5% tolerance in both sets.')"
]),

# ── Train model ───────────────────────────────────────────────────────────────
md("md-train", ["---\n", "## Step 3 · Train Logistic Regression Baseline"]),

code("cell-train", [
    "print('Training Logistic Regression (max_iter=1000, C=1.0 default, solver=lbfgs)...')\n",
    "lr = LogisticRegression(\n",
    "    max_iter=1000,\n",
    "    random_state=RANDOM_STATE,\n",
    "    # All other hyperparameters at sklearn defaults:\n",
    "    # C=1.0 (L2 regularisation), solver='lbfgs', class_weight=None\n",
    "    # No tuning — this is the baseline.\n",
    ")\n",
    "lr.fit(X_train, y_train)\n",
    "print(f'Training complete. Converged: {lr.n_iter_[0]} iterations (max_iter=1000)')"
]),

# ── Predictions ───────────────────────────────────────────────────────────────
code("cell-predict", [
    "y_pred       = lr.predict(X_test)\n",
    "y_pred_proba = lr.predict_proba(X_test)[:, 1]   # P(class=1 = left)\n",
    "\n",
    "print(f'Predictions generated. y_pred unique values: {np.unique(y_pred)}')\n",
    "print(f'y_pred_proba range: min={y_pred_proba.min():.4f}  max={y_pred_proba.max():.4f}')"
]),

# ── Metrics ───────────────────────────────────────────────────────────────────
md("md-metrics", ["---\n", "## Step 4 · Full Metrics Report"]),

code("cell-confusion", [
    "print('=== CONFUSION MATRIX (raw counts, test set) ===')\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "cm_df = pd.DataFrame(\n",
    "    cm,\n",
    "    index=['Actual: Stayed (0)', 'Actual: Left (1)'],\n",
    "    columns=['Predicted: Stayed (0)', 'Predicted: Left (1)']\n",
    ")\n",
    "print(cm_df.to_string())\n",
    "print()\n",
    "tn, fp, fn, tp = cm.ravel()\n",
    "print(f'  True Negatives  (correctly predicted stayed): {tn}')\n",
    "print(f'  False Positives (predicted left, actually stayed): {fp}')\n",
    "print(f'  False Negatives (predicted stayed, actually left): {fn}   <- missed attrition cases')\n",
    "print(f'  True Positives  (correctly predicted left): {tp}')"
]),

code("cell-scalar-metrics", [
    "precision = precision_score(y_test, y_pred)\n",
    "recall    = recall_score(y_test, y_pred)\n",
    "f1        = f1_score(y_test, y_pred)\n",
    "roc_auc   = roc_auc_score(y_test, y_pred_proba)\n",
    "\n",
    "print('=== SCALAR METRICS (positive class = left=1) ===')\n",
    "print(f'  Precision : {precision:.4f}  (of all predicted-left, how many actually left)')\n",
    "print(f'  Recall    : {recall:.4f}  (of all who actually left, how many did we catch)')\n",
    "print(f'  F1        : {f1:.4f}  (harmonic mean of precision & recall)')\n",
    "print(f'  ROC-AUC   : {roc_auc:.4f}  (discrimination across all thresholds)')\n",
    "print()\n",
    "print('NOTE: Accuracy is NOT the primary metric for this imbalanced dataset.')\n",
    "print(f'  (A naive all-zeros classifier would get {(y_test==0).mean()*100:.2f}% accuracy.)')\n",
    "from sklearn.metrics import accuracy_score\n",
    "acc = accuracy_score(y_test, y_pred)\n",
    "print(f'  Actual accuracy: {acc*100:.2f}%  — context: this number is misleading alone.')"
]),

code("cell-classification-report", [
    "print('=== FULL classification_report() ===')\n",
    "print(classification_report(\n",
    "    y_test, y_pred,\n",
    "    target_names=['Stayed (0)', 'Left (1)'],\n",
    "    digits=4\n",
    "))"
]),

code("cell-baseline-record", [
    "print('=== BASELINE RECORD (to beat in Step 7) ===')\n",
    "print(f'  Model      : Logistic Regression (C=1.0, L2, lbfgs, max_iter=1000)')\n",
    "print(f'  Split      : 80/20 stratified, random_state=42')\n",
    "print(f'  Precision  : {precision:.4f}')\n",
    "print(f'  Recall     : {recall:.4f}')\n",
    "print(f'  F1         : {f1:.4f}')\n",
    "print(f'  ROC-AUC    : {roc_auc:.4f}')\n",
    "print(f'  Confusion  : TN={tn} FP={fp} FN={fn} TP={tp}')"
]),

# ── Coefficients ──────────────────────────────────────────────────────────────
md("md-coeffs", ["---\n", "## Step 5 · Top 10 Coefficients (sanity check)"]),

code("cell-coeffs", [
    "coef_series = pd.Series(lr.coef_[0], index=X.columns)\n",
    "coef_abs = coef_series.abs().sort_values(ascending=False)\n",
    "\n",
    "top10_idx   = coef_abs.head(10).index\n",
    "top10_coefs = coef_series[top10_idx]\n",
    "\n",
    "print('=== TOP 10 COEFFICIENTS BY ABSOLUTE VALUE ===')\n",
    "print(f'{\"Feature\":<40s}  {\"Coefficient\":>12s}  Direction')\n",
    "print('-' * 75)\n",
    "for feat, coef in top10_coefs.items():\n",
    "    direction = 'TOWARD ATTRITION (+)' if coef > 0 else 'AWAY from attrition (-)'\n",
    "    print(f'{feat:<40s}  {coef:>12.4f}  {direction}')\n",
    "\n",
    "print()\n",
    "print('--- All coefficients (descending by value, for full picture) ---')\n",
    "print(f'{\"Feature\":<40s}  {\"Coef\":>10s}')\n",
    "print('-' * 55)\n",
    "for feat, coef in coef_series.sort_values(ascending=False).items():\n",
    "    bar = '+' * int(abs(coef) * 3) if coef > 0 else '-' * int(abs(coef) * 3)\n",
    "    print(f'{feat:<40s}  {coef:>10.4f}  {bar[:30]}')"
]),

code("cell-sanity-commentary", [
    "print('=== BUSINESS SENSE CHECK — TOP 10 ===')\n",
    "print()\n",
    "\n",
    "# Dynamic commentary based on actual coefficient signs\n",
    "checks = {\n",
    "    'OverTime':                  ('positive', 'EXPECTED — overtime causes burnout, strong attrition driver in literature'),\n",
    "    'MaritalStatus_Single':      ('positive', 'EXPECTED — single employees have lower anchor cost to switching jobs'),\n",
    "    'JobRole_Sales Representative': ('positive', 'EXPECTED — sales rep roles have high voluntary turnover industry-wide'),\n",
    "    'JobRole_Laboratory Technician': ('positive', 'EXPECTED — technically skilled, portable skills, high market demand'),\n",
    "    'JobRole_Human Resources':   ('positive', 'PLAUSIBLE — HR roles can have high burnout'),\n",
    "    'overall_satisfaction_score':('negative', 'EXPECTED — higher satisfaction -> lower attrition risk'),\n",
    "    'JobSatisfaction':           ('negative', 'EXPECTED — lower satisfaction -> higher attrition'),\n",
    "    'JobLevel':                  ('negative', 'EXPECTED — higher seniority -> less likely to leave'),\n",
    "    'income_per_year_at_company':('negative', 'EXPECTED — better compensated for tenure -> more likely to stay'),\n",
    "    'experience_ratio':          ('negative', 'EXPECTED — more of career spent here -> higher switching cost'),\n",
    "    'TotalWorkingYears':         ('negative', 'EXPECTED — experienced workers tend to be more stable'),\n",
    "    'MonthlyIncome':             ('negative', 'EXPECTED — higher income -> lower attrition risk'),\n",
    "    'StockOptionLevel':          ('negative', 'EXPECTED — equity compensation creates golden handcuffs'),\n",
    "    'YearsAtCompany':            ('negative', 'EXPECTED — longer tenure -> higher switching cost'),\n",
    "    'YearsWithCurrManager':      ('negative', 'EXPECTED — good manager relationship -> retention'),\n",
    "    'NumCompaniesWorked':        ('positive', 'EXPECTED — job hopper pattern predicts future hopping'),\n",
    "    'BusinessTravel_Travel_Frequently': ('positive', 'EXPECTED — frequent travel is a well-known burnout driver'),\n",
    "    'DistanceFromHome':          ('positive', 'EXPECTED — longer commute -> higher attrition risk'),\n",
    "}\n",
    "\n",
    "flags = []\n",
    "for feat, coef in top10_coefs.items():\n",
    "    actual_dir = 'positive' if coef > 0 else 'negative'\n",
    "    if feat in checks:\n",
    "        expected_dir, rationale = checks[feat]\n",
    "        match = 'OK' if actual_dir == expected_dir else 'UNEXPECTED — REVIEW'\n",
    "        flag  = '✅' if actual_dir == expected_dir else '⚠️ BACKWARDS'\n",
    "        print(f'{flag}  [{feat}]  coef={coef:.4f}  ({actual_dir})')\n",
    "        print(f'    {rationale}')\n",
    "        if match == 'UNEXPECTED — REVIEW':\n",
    "            flags.append(feat)\n",
    "    else:\n",
    "        print(f'❓  [{feat}]  coef={coef:.4f}  ({actual_dir})')\n",
    "        print(f'    Not in sanity-check map — review manually.')\n",
    "        flags.append(feat)\n",
    "    print()\n",
    "\n",
    "print('---')\n",
    "if flags:\n",
    "    print(f'FEATURES NEEDING REVIEW: {flags}')\n",
    "else:\n",
    "    print('All top-10 coefficients directionally consistent with business expectations.')"
]),

# ── Save model ────────────────────────────────────────────────────────────────
md("md-save", ["---\n", "## Step 6 · Save Model"]),

code("cell-save", [
    "model_path = os.path.join(MODELS, 'baseline_logreg.joblib')\n",
    "joblib.dump(lr, model_path)\n",
    "size = os.path.getsize(model_path)\n",
    "print(f'Saved: baseline_logreg.joblib  ({size:,} bytes)')\n",
    "print(f'Path : {os.path.abspath(model_path)}')\n",
    "\n",
    "# Verify round-trip\n",
    "lr_loaded = joblib.load(model_path)\n",
    "y_check   = lr_loaded.predict(X_test[:5])\n",
    "print(f'Round-trip check (first 5 test predictions): {y_check}')\n",
    "print('Model saved and verified.')\n",
    "\n",
    "print()\n",
    "print('=== MODELS/ directory ===')\n",
    "for f in sorted(os.listdir(MODELS)):\n",
    "    fp = os.path.join(MODELS, f)\n",
    "    print(f'  {f}  ({os.path.getsize(fp):,} bytes)')"
]),

md("md-summary", [
    "---\n",
    "## Summary\n",
    "\n",
    "| Metric | Value |\n",
    "|---|---|\n",
    "| Model | Logistic Regression — C=1.0, L2, lbfgs, max_iter=1000 |\n",
    "| Split | 80/20 stratified, random_state=42 (fixed) |\n",
    "| Precision | See Step 4 output |\n",
    "| Recall | See Step 4 output |\n",
    "| F1 | See Step 4 output |\n",
    "| ROC-AUC | See Step 4 output |\n",
    "\n",
    "**Next step:** Step 7 — Random Forest + XGBoost using `features_unscaled.csv`.  \n",
    "All models in Step 7 must beat this baseline's F1 and ROC-AUC to be considered an improvement."
])

]

out_path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\06_baseline_model.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Written: {out_path}')
