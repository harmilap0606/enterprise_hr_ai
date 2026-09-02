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

# ── Title & Prominent Warning ────────────────────────────────────────────────
md("md-title", [
    "# 12 · Skill Gap Engine (Severity Classification & Gap Inventory)\n",
    "\n",
    "**Project:** Enterprise HR AI  \n",
    "\n",
    "> ### ⚠️ PROMINENT DATA INTEGRITY WARNING\n",
    "> **SYNTHETIC DATA — employee current-skill possession was not present in any source file and has been simulated using a tenure/training-based heuristic for MVP demonstration purposes only. This must NOT be presented to stakeholders as real observed skill data. Real deployment requires an actual skills inventory (HRIS export, LMS completion records, or self-assessment survey).**\n",
    "\n",
    "---"
]),

# ── Step 1: Load Data & Sanity Check ─────────────────────────────────────────
md("md-step1", [
    "---\n",
    "## Step 1 · Load Synthetic Skills & Perform 30-Employee Fidelity Sanity Check\n",
    "\n",
    "Before utilizing the synthetic skill inventory, we evaluate statistical fidelity by sampling 30 random employees (`random_state=42`), computing their realized skill possession rate (possessed skills / 10 required skills), and comparing it against their theoretical possession probability from Step 15's formula:\n",
    "$$\\text{possession\\_probability} = \\min(0.30 + 0.05 \\times \\text{YearsAtCompany} + 0.05 \\times \\text{TrainingTimesLastYear}, 0.95)$$\n",
    "\n",
    "If Pearson correlation $r < 0.50$, execution must halt with an explicit data quality warning."
]),

code("cell-sanity-check", [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "\n",
    "PROC = os.path.join('..', 'data', 'processed')\n",
    "skills_file = os.path.join(PROC, 'employee_skills_synthetic.csv')\n",
    "att_file = os.path.join(PROC, 'employee_attrition_processed.csv')\n",
    "profiles_file = os.path.join(PROC, 'role_skill_profiles.csv')\n",
    "\n",
    "# Load synthetic skills ignoring comment header\n",
    "df_skills = pd.read_csv(skills_file, comment='#')\n",
    "df_att = pd.read_csv(att_file)\n",
    "df_profiles = pd.read_csv(profiles_file)\n",
    "\n",
    "print(f'Loaded synthetic skills records : {len(df_skills):,}')\n",
    "print(f'Loaded employee anchor records   : {len(df_att):,}')\n",
    "print(f'Loaded role benchmark profiles   : {len(df_profiles):,}')\n",
    "\n",
    "# Identify and isolate Manager employees (102 records)\n",
    "mgr_ids = set(df_att[df_att['JobRole'] == 'Manager']['EmployeeNumber'])\n",
    "non_mgr_att = df_att[~df_att['EmployeeNumber'].isin(mgr_ids)].copy()\n",
    "non_mgr_skills = df_skills[~df_skills['EmployeeNumber'].isin(mgr_ids)].copy()\n",
    "\n",
    "print(f'\\nNon-Manager employees to evaluate: {len(non_mgr_att):,} (102 Managers excluded)')\n",
    "print(f'Non-Manager skill records        : {len(non_mgr_skills):,}')\n",
    "\n",
    "# Sample 30 non-manager employees\n",
    "sample_ids = non_mgr_att['EmployeeNumber'].sample(n=30, random_state=42).tolist()\n",
    "sample_att = non_mgr_att[non_mgr_att['EmployeeNumber'].isin(sample_ids)].copy()\n",
    "\n",
    "# Recompute theoretical probability\n",
    "sample_att['theoretical_prob'] = sample_att.apply(\n",
    "    lambda r: min(0.30 + 0.05 * r['YearsAtCompany'] + 0.05 * r['TrainingTimesLastYear'], 0.95),\n",
    "    axis=1\n",
    ")\n",
    "\n",
    "# Compute realized possession rate\n",
    "sample_realized = (\n",
    "    non_mgr_skills[non_mgr_skills['EmployeeNumber'].isin(sample_ids)]\n",
    "    .groupby('EmployeeNumber')\n",
    "    .agg(total_req=('has_skill', 'count'), possessed=('has_skill', 'sum'))\n",
    "    .reset_index()\n",
    ")\n",
    "sample_realized['realized_rate'] = sample_realized['possessed'] / sample_realized['total_req']\n",
    "\n",
    "check_df = sample_att.merge(sample_realized, on='EmployeeNumber')\n",
    "corr_val = check_df['theoretical_prob'].corr(check_df['realized_rate'])\n",
    "\n",
    "print('=' * 80)\n",
    "print(f'30-EMPLOYEE SANITY CHECK CORRELATION (Theoretical vs Realized): r = {corr_val:.4f}')\n",
    "print('=' * 80)\n",
    "\n",
    "if corr_val < 0.50:\n",
    "    raise ValueError(f'DATA QUALITY CONCERN: Realized rate correlation ({corr_val:.4f}) is below 0.50 threshold!')\n",
    "else:\n",
    "    print('CONFIRMED: Strong statistical fidelity (r >= 0.50). Synthetic data generation is sound.')\n",
    "\n",
    "print('\\nFirst 5 Sample Records:')\n",
    "for _, r in check_df.head(5).iterrows():\n",
    "    print(f'  Emp #{r[\"EmployeeNumber\"]} ({r[\"JobRole\"]}): Tenure={r[\"YearsAtCompany\"]}y, Training={r[\"TrainingTimesLastYear\"]}x | Theory={r[\"theoretical_prob\"]:.2f}, Realized={r[\"realized_rate\"]:.2f} ({r[\"possessed\"]}/{r[\"total_req\"]})')"
]),

# ── Step 2: Individual Skill Gap Computation ─────────────────────────────────
md("md-step2", [
    "---\n",
    "## Step 2 · Compute Individual Skill Gaps & Severity Classification\n",
    "\n",
    "For each of the 1,368 non-manager employees:\n",
    "1. **Missing Skills:** List all benchmark skills where `has_skill == 0`.\n",
    "2. **Gap Metrics:** Compute `gap_count` (missing count) and `gap_percentage` ($gap\\_count / total\\_required$).\n",
    "3. **Severity Thresholds:**\n",
    "   - **HIGH Severity:** $\\text{gap\\_percentage} \\ge 70\\%$ (missing 7+ of 10 skills)\n",
    "   - **MEDIUM Severity:** $40\\% \\le \\text{gap\\_percentage} < 70\\%$ (missing 4 to 6 skills)\n",
    "   - **LOW Severity:** $\\text{gap\\_percentage} < 40\\%$ (missing 0 to 3 skills)"
]),

code("cell-compute-gaps", [
    "gap_records = []\n",
    "emp_role_map = dict(zip(df_att['EmployeeNumber'], df_att['JobRole']))\n",
    "\n",
    "for emp_id, group in non_mgr_skills.groupby('EmployeeNumber'):\n",
    "    role = emp_role_map[emp_id]\n",
    "    total_req = len(group)\n",
    "    missing_list = group[group['has_skill'] == 0]['skill_name'].tolist()\n",
    "    gap_cnt = len(missing_list)\n",
    "    gap_pct = gap_cnt / total_req if total_req > 0 else 0.0\n",
    "    \n",
    "    # Explicit severity threshold classification\n",
    "    if gap_pct >= 0.70:\n",
    "        severity = 'HIGH'\n",
    "    elif gap_pct >= 0.40:\n",
    "        severity = 'MEDIUM'\n",
    "    else:\n",
    "        severity = 'LOW'\n",
    "        \n",
    "    gap_records.append({\n",
    "        'EmployeeNumber': emp_id,\n",
    "        'JobRole': role,\n",
    "        'missing_skills': '; '.join(missing_list) if missing_list else 'None',\n",
    "        'gap_count': gap_cnt,\n",
    "        'total_required': total_req,\n",
    "        'gap_percentage': round(gap_pct, 4),\n",
    "        'severity': severity\n",
    "    })\n",
    "\n",
    "df_gaps = pd.DataFrame(gap_records)\n",
    "print(f'Total skill gap profiles generated: {len(df_gaps):,} employees')\n",
    "assert len(df_gaps) == 1368, f'Expected 1,368 non-manager gap profiles, got {len(df_gaps)}'\n",
    "\n",
    "# Severity distribution\n",
    "sev_counts = df_gaps['severity'].value_counts()\n",
    "sev_pcts = (df_gaps['severity'].value_counts(normalize=True) * 100).round(2)\n",
    "\n",
    "sev_summary = pd.DataFrame({\n",
    "    'Headcount': sev_counts,\n",
    "    'Percentage (%)': sev_pcts\n",
    "})\n",
    "sev_summary.index.name = 'Severity Band'\n",
    "\n",
    "print('\\n=== SKILL GAP SEVERITY DISTRIBUTION (n=1,368) ===')\n",
    "print(sev_summary.to_string())\n",
    "\n",
    "print('\\nSample Profiles Across Severity Bands:')\n",
    "for band in ['HIGH', 'MEDIUM', 'LOW']:\n",
    "    ex_row = df_gaps[df_gaps['severity'] == band].iloc[0]\n",
    "    print(f'\\n[{band} SEVERITY EXAMPLE] Employee #{ex_row[\"EmployeeNumber\"]} ({ex_row[\"JobRole\"]})')\n",
    "    print(f'  Missing Skills Count : {ex_row[\"gap_count\"]} / {ex_row[\"total_required\"]} ({ex_row[\"gap_percentage\"]*100:.1f}%)')\n",
    "    print(f'  Missing Skills List  : {ex_row[\"missing_skills\"]}')"
]),

# ── Step 3: Save Output File with Warning Comment ────────────────────────────
md("md-step3", [
    "---\n",
    "## Step 3 · Save Skill Gap Inventory with Warning Comment Header\n",
    "\n",
    "Target file: `data/processed/employee_skill_gaps.csv`  \n",
    "The synthetic data caveat is explicitly preserved in line 1 of the export."
]),

code("cell-save", [
    "out_path = os.path.join(PROC, 'employee_skill_gaps.csv')\n",
    "\n",
    "warning_comment = (\n",
    "    '# SYNTHETIC DATA — employee current-skill possession was not present in any source file '\n",
    "    'and has been simulated using a tenure/training-based heuristic for MVP demonstration purposes only. '\n",
    "    'This must NOT be presented to stakeholders as real observed skill data. Real deployment requires '\n",
    "    'an actual skills inventory (HRIS export, LMS completion records, or self-assessment survey).\\n'\n",
    ")\n",
    "\n",
    "with open(out_path, 'w', encoding='utf-8') as f:\n",
    "    f.write(warning_comment)\n",
    "    df_gaps.to_csv(f, index=False)\n",
    "\n",
    "file_size = os.path.getsize(out_path)\n",
    "print(f'Saved skill gaps file to : {out_path}')\n",
    "print(f'File size                : {file_size:,} bytes')\n",
    "print(f'Total rows               : {len(df_gaps):,} records')\n",
    "\n",
    "# Round-trip reload validation\n",
    "df_check = pd.read_csv(out_path, comment='#')\n",
    "assert len(df_check) == 1368, 'Row count mismatch on reload!'\n",
    "assert list(df_check.columns) == ['EmployeeNumber', 'JobRole', 'missing_skills', 'gap_count', 'total_required', 'gap_percentage', 'severity']\n",
    "print('CONFIRMED: Round-trip read verified cleanly with comment handling.')"
])

]

out_path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\12_skill_gap_engine.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Written: {out_path}')
