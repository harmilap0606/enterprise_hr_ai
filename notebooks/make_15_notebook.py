import json

nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.12.0"}
 },
 "cells": []
}

def md(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}

def code(cid, src):
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": src}

nb["cells"] = [

# ── Title & Verbatim Documentation Header ────────────────────────────────────
md("md-title", [
    "# 15 · Unified Employee Intelligence (Capstone Table)\n",
    "\n",
    "**Project:** Enterprise HR AI — Capstone Workforce Analytics Table  \n",
    "\n",
    "> ### ⚠️ DATA INTEGRITY & PROVENANCE HEADER\n",
    ">\n",
    "> **This table combines: (1) a validated ML risk model (see model_card.md), (2) real engagement survey data covering 49.7% of employees (Step 13 -- do not generalize to the rest), (3) O*NET role mappings with confidence levels ranging from very_low to medium -- none are exact matches (see data_relationships.md Open Issue #1), and (4) SYNTHETIC skill-gap and recommendation data (Steps 15-18) that has NOT been validated against real employee skill records. Sections 3 and 4 are illustrative of MVP capability, not production-ready HR guidance.**\n",
    "\n",
    "---"
]),

# ── Step 1: Load Anchor Table ────────────────────────────────────────────────
md("md-step1", [
    "---\n",
    "## Step 1 · Anchor Table: Load Full 1,470-Employee Workforce\n",
    "\n",
    "Anchor table: `data/processed/employee_attrition_processed.csv`.  \n",
    "Rule: All 1,470 employees are preserved across all subsequent left joins. Never subset."
]),

code("cell-anchor", [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "import json\n",
    "import os\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "PROC   = os.path.join('..', 'data', 'processed')\n",
    "MODELS = os.path.join('..', 'models')\n",
    "\n",
    "# Load anchor table\n",
    "df_anchor = pd.read_csv(os.path.join(PROC, 'employee_attrition_processed.csv'))\n",
    "print(f'Anchor table loaded: {len(df_anchor):,} employees')\n",
    "assert len(df_anchor) == 1470, f'Expected 1,470 anchor rows, got {len(df_anchor)}'\n",
    "\n",
    "# Base unified table starting from anchor\n",
    "unified_df = df_anchor[['EmployeeNumber', 'Department', 'JobRole']].copy()\n",
    "print(f'Initial anchor row count: {len(unified_df):,}')"
]),

# ── Step 2: Join 1 — Risk Scoring ────────────────────────────────────────────
md("md-join1", [
    "---\n",
    "## Step 2 · Join 1: Attrition Risk Model Scoring\n",
    "\n",
    "Load production model `attrition_pipeline.joblib` and `model_config.json`.  \n",
    "Score all 1,470 employees using `features_scaled.csv`. Threshold = 0.40."
]),

code("cell-join1", [
    "# Load production model & config\n",
    "model = joblib.load(os.path.join(MODELS, 'attrition_pipeline.joblib'))\n",
    "with open(os.path.join(MODELS, 'model_config.json'), 'r') as f:\n",
    "    model_cfg = json.load(f)\n",
    "\n",
    "threshold = model_cfg['threshold']\n",
    "print(f'Model algorithm : {model_cfg[\"model\"]}')\n",
    "print(f'Decision threshold: {threshold}')\n",
    "\n",
    "# Load scaled features matching model input\n",
    "fs = pd.read_csv(os.path.join(PROC, 'features_scaled.csv'))\n",
    "X = fs.drop(columns=['Attrition'])\n",
    "probs = model.predict_proba(X)[:, 1]\n",
    "\n",
    "df_risk = pd.DataFrame({\n",
    "    'EmployeeNumber': df_anchor['EmployeeNumber'],\n",
    "    'RiskScore': probs.round(4),\n",
    "    'RiskLevel': ['HIGH' if p >= threshold else 'LOW' for p in probs]\n",
    "})\n",
    "\n",
    "# Left Join 1\n",
    "unified_df = unified_df.merge(df_risk, on='EmployeeNumber', how='left')\n",
    "print(f'Row count after Join 1 (Risk Scoring): {len(unified_df):,}')\n",
    "assert len(unified_df) == 1470, f'Join 1 altered row count! {len(unified_df)}'"
]),

# ── Step 3: Join 2 — Engagement Survey Data ──────────────────────────────────
md("md-join2", [
    "---\n",
    "## Step 3 · Join 2: Engagement Survey Overlay\n",
    "\n",
    "Load `data/processed/employee_intelligence_partial.csv` to overlay `Engagement Score`, `Satisfaction Score`, and `Work-Life Balance Score`.  \n",
    "Retain null values for the 739 unmapped employees (do not impute)."
]),

code("cell-join2", [
    "df_eng = pd.read_csv(os.path.join(PROC, 'employee_intelligence_partial.csv'))\n",
    "eng_subset = df_eng[['EmployeeNumber', 'Engagement Score', 'Satisfaction Score', 'Work-Life Balance Score']].copy()\n",
    "eng_subset.rename(columns={\n",
    "    'Engagement Score': 'EngagementScore',\n",
    "    'Satisfaction Score': 'SatisfactionScore',\n",
    "    'Work-Life Balance Score': 'WorkLifeBalanceScore'\n",
    "}, inplace=True)\n",
    "\n",
    "# Left Join 2\n",
    "unified_df = unified_df.merge(eng_subset, on='EmployeeNumber', how='left')\n",
    "print(f'Row count after Join 2 (Engagement Survey): {len(unified_df):,}')\n",
    "assert len(unified_df) == 1470, f'Join 2 altered row count! {len(unified_df)}'\n",
    "print(f'  Non-null EngagementScore: {unified_df[\"EngagementScore\"].notnull().sum():,}')\n",
    "print(f'  Null EngagementScore    : {unified_df[\"EngagementScore\"].isnull().sum():,}')"
]),

# ── Step 4: Join 3 — O*NET Role Mappings ─────────────────────────────────────
md("md-join3", [
    "---\n",
    "## Step 4 · Join 3: O*NET Role Intelligence Profiles\n",
    "\n",
    "Load `role_skill_profiles.csv` to attach mapped `ONET_Title` and `ONET_Confidence` based on `JobRole`."
]),

code("cell-join3", [
    "df_rsp = pd.read_csv(os.path.join(PROC, 'role_skill_profiles.csv'))\n",
    "onet_subset = df_rsp[['ibm_job_role', 'onet_title', 'match_confidence']].copy()\n",
    "onet_subset.rename(columns={\n",
    "    'ibm_job_role': 'JobRole',\n",
    "    'onet_title': 'ONET_Title',\n",
    "    'match_confidence': 'ONET_Confidence'\n",
    "}, inplace=True)\n",
    "\n",
    "# Left Join 3\n",
    "unified_df = unified_df.merge(onet_subset, on='JobRole', how='left')\n",
    "print(f'Row count after Join 3 (O*NET Profiles): {len(unified_df):,}')\n",
    "assert len(unified_df) == 1470, f'Join 3 altered row count! {len(unified_df)}'"
]),

# ── Step 5: Join 4 — Employee Skill Gaps ─────────────────────────────────────
md("md-join4", [
    "---\n",
    "## Step 5 · Join 4: Individual Skill Gaps & Severity\n",
    "\n",
    "Load `data/processed/employee_skill_gaps.csv` for `SkillGapCount` and `SkillGapSeverity`.  \n",
    "Manager employees (102 records) explicitly assigned `'N/A - Manager'` for severity and `NaN` for count."
]),

code("cell-join4", [
    "df_gaps = pd.read_csv(os.path.join(PROC, 'employee_skill_gaps.csv'), comment='#')\n",
    "gaps_subset = df_gaps[['EmployeeNumber', 'gap_count', 'severity']].copy()\n",
    "gaps_subset.rename(columns={\n",
    "    'gap_count': 'SkillGapCount',\n",
    "    'severity': 'SkillGapSeverity'\n",
    "}, inplace=True)\n",
    "\n",
    "# Left Join 4\n",
    "unified_df = unified_df.merge(gaps_subset, on='EmployeeNumber', how='left')\n",
    "\n",
    "# Handle Manager exclusion explicitly\n",
    "unified_df.loc[unified_df['JobRole'] == 'Manager', 'SkillGapSeverity'] = 'N/A - Manager'\n",
    "\n",
    "print(f'Row count after Join 4 (Skill Gaps): {len(unified_df):,}')\n",
    "assert len(unified_df) == 1470, f'Join 4 altered row count! {len(unified_df)}'"
]),

# ── Step 6: Join 5 — Training Recommendations ────────────────────────────────
md("md-join5", [
    "---\n",
    "## Step 6 · Join 5: Top 3 Training Recommendations\n",
    "\n",
    "Load `data/processed/employee_recommendations.csv` for `Top3Recommendations`.  \n",
    "Manager role assigned `'N/A - Manager (use Department-level analysis)'`."
]),

code("cell-join5", [
    "df_recs = pd.read_csv(os.path.join(PROC, 'employee_recommendations.csv'), comment='#')\n",
    "recs_subset = df_recs[['EmployeeNumber', 'top_3_recommendations']].copy()\n",
    "recs_subset.rename(columns={\n",
    "    'top_3_recommendations': 'Top3Recommendations'\n",
    "}, inplace=True)\n",
    "\n",
    "# Left Join 5\n",
    "unified_df = unified_df.merge(recs_subset, on='EmployeeNumber', how='left')\n",
    "\n",
    "# Handle Manager exclusion explicitly\n",
    "unified_df.loc[unified_df['JobRole'] == 'Manager', 'Top3Recommendations'] = 'N/A - Manager (use Department-level analysis)'\n",
    "\n",
    "print(f'Row count after Join 5 (Recommendations): {len(unified_df):,}')\n",
    "assert len(unified_df) == 1470, f'Join 5 altered row count! {len(unified_df)}'"
]),

# ── Step 7: Final Column Ordering & Verification ─────────────────────────────
md("md-finalize", [
    "---\n",
    "## Step 7 · Finalize Columns & Complete Integrity Verification\n",
    "\n",
    "Ordering exactly as requested:\n",
    "`EmployeeNumber, Department, JobRole, RiskScore, RiskLevel, EngagementScore, SatisfactionScore, WorkLifeBalanceScore, ONET_Title, ONET_Confidence, SkillGapCount, SkillGapSeverity, Top3Recommendations`"
]),

code("cell-finalize", [
    "FINAL_COLUMNS = [\n",
    "    'EmployeeNumber', 'Department', 'JobRole', 'RiskScore', 'RiskLevel',\n",
    "    'EngagementScore', 'SatisfactionScore', 'WorkLifeBalanceScore',\n",
    "    'ONET_Title', 'ONET_Confidence', 'SkillGapCount', 'SkillGapSeverity',\n",
    "    'Top3Recommendations'\n",
    "]\n",
    "\n",
    "df_final = unified_df[FINAL_COLUMNS].copy()\n",
    "\n",
    "print('=== FINAL DATA INTEGRITY CONFIRMATION ===')\n",
    "print(f'Total workforce rows : {len(df_final):,} (Confirmed: exactly 1,470)')\n",
    "print(f'Total columns        : {len(df_final.columns)}')\n",
    "print(f'Column names         : {list(df_final.columns)}')\n",
    "assert len(df_final) == 1470, 'Final row count must be 1,470!'\n",
    "\n",
    "print('\\nFirst 3 Records:')\n",
    "print(df_final.head(3).to_string(index=False))\n",
    "\n",
    "print('\\nManager Records (Validation of N/A Exclusion):')\n",
    "print(df_final[df_final['JobRole'] == 'Manager'][['EmployeeNumber', 'JobRole', 'ONET_Title', 'ONET_Confidence', 'SkillGapSeverity', 'Top3Recommendations']].head(2).to_string(index=False))"
]),

# ── Step 8: Save Output CSV ──────────────────────────────────────────────────
md("md-save", [
    "---\n",
    "## Step 8 · Save Unified Dataset (`employee_intelligence.csv`)\n",
    "\n",
    "Saving the capstone intelligence table to `data/processed/employee_intelligence.csv`.\n",
    "The provenance warning header is permanently embedded in line 1 of the file."
]),

code("cell-save", [
    "out_path = os.path.join(PROC, 'employee_intelligence.csv')\n",
    "\n",
    "header_comment = (\n",
    "    '# This table combines: (1) a validated ML risk model (see model_card.md), '\n",
    "    '(2) real engagement survey data covering 49.7% of employees (Step 13 -- do not generalize to the rest), '\n",
    "    '(3) O*NET role mappings with confidence levels ranging from very_low to medium -- none are exact matches '\n",
    "    '(see data_relationships.md Open Issue #1), and (4) SYNTHETIC skill-gap and recommendation data (Steps 15-18) '\n",
    "    'that has NOT been validated against real employee skill records. Sections 3 and 4 are illustrative '\n",
    "    'of MVP capability, not production-ready HR guidance.\\n'\n",
    ")\n",
    "\n",
    "with open(out_path, 'w', encoding='utf-8') as f:\n",
    "    f.write(header_comment)\n",
    "    df_final.to_csv(f, index=False)\n",
    "\n",
    "file_size = os.path.getsize(out_path)\n",
    "print(f'Saved capstone table to : {out_path}')\n",
    "print(f'File size               : {file_size:,} bytes')\n",
    "print(f'Total rows              : {len(df_final):,}')\n",
    "\n",
    "# Round-trip reload verification\n",
    "df_reload = pd.read_csv(out_path, comment='#')\n",
    "assert len(df_reload) == 1470, 'Row count mismatch on reload!'\n",
    "assert list(df_reload.columns) == FINAL_COLUMNS, 'Columns mismatch on reload!'\n",
    "print('CONFIRMED: Round-trip verification passed cleanly.')"
]),

# ── Step 9: Summary Statistics ───────────────────────────────────────────────
md("md-summary", [
    "---\n",
    "## Step 9 · Summary Statistics & Completeness Breakdown"
]),

code("cell-summary-stats", [
    "print('=== FINAL SUMMARY STATISTICS (N=1,470) ===\\n')\n",
    "\n",
    "# 1. RiskLevel Distribution\n",
    "risk_dist = df_final['RiskLevel'].value_counts()\n",
    "risk_pcts = (df_final['RiskLevel'].value_counts(normalize=True) * 100).round(2)\n",
    "print('1. RiskLevel Distribution:')\n",
    "for level, count in risk_dist.items():\n",
    "    print(f'   {level:<5}: {count:>4} employees ({risk_pcts[level]:>5.2f}%)')\n",
    "\n",
    "# 2. SkillGapSeverity Distribution\n",
    "gap_dist = df_final['SkillGapSeverity'].value_counts()\n",
    "gap_pcts = (df_final['SkillGapSeverity'].value_counts(normalize=True) * 100).round(2)\n",
    "print('\\n2. SkillGapSeverity Distribution:')\n",
    "for sev, count in gap_dist.items():\n",
    "    print(f'   {sev:<15}: {count:>4} employees ({gap_pcts[sev]:>5.2f}%)')\n",
    "\n",
    "# 3. Data Completeness Across All 5 Sources\n",
    "complete_mask = (\n",
    "    df_final['RiskScore'].notnull() &\n",
    "    df_final['EngagementScore'].notnull() &\n",
    "    df_final['ONET_Title'].notnull() &\n",
    "    df_final['SkillGapCount'].notnull() &\n",
    "    (df_final['JobRole'] != 'Manager')\n",
    ")\n",
    "n_complete = complete_mask.sum()\n",
    "n_partial = len(df_final) - n_complete\n",
    "\n",
    "print('\\n3. Workforce Data Completeness Across All 5 Sources:')\n",
    "print(f'   Complete Data (all 5 sources present, non-manager) : {n_complete:>4} ({n_complete/1470*100:.2f}%)')\n",
    "print(f'   Partial Data (missing survey and/or Manager role)  : {n_partial:>4} ({n_partial/1470*100:.2f}%)')\n",
    "print(f'     - Missing engagement survey (known 49.7% sample) : {df_final[\"EngagementScore\"].isnull().sum():>4} employees')\n",
    "print(f'     - Manager role (O*NET skill gap exclusion)       : {(df_final[\"JobRole\"] == \"Manager\").sum():>4} employees')\n",
    "print(f'     - Overlap (both missing survey & Manager)        : {((df_final[\"EngagementScore\"].isnull()) & (df_final[\"JobRole\"] == \"Manager\")).sum():>4} employees')"
])

]

out_path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\15_employee_intelligence.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Written: {out_path}')
