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
    "# 13 · Organization-Wide Skill Gap Rollup & Concentration Analysis\n",
    "\n",
    "**Project:** Enterprise HR AI  \n",
    "\n",
    "> ### ⚠️ PROMINENT DATA INTEGRITY WARNING\n",
    "> **SYNTHETIC DATA — employee current-skill possession was not present in any source file and has been simulated using a tenure/training-based heuristic for MVP demonstration purposes only. This must NOT be presented to stakeholders as real observed skill data. Real deployment requires an actual skills inventory (HRIS export, LMS completion records, or self-assessment survey).**\n",
    "\n",
    "---"
]),

# ── Step 1: Load Employee Skill Gaps ─────────────────────────────────────────
md("md-step1", [
    "---\n",
    "## Step 1 · Load Employee Skill Gaps & Explode Missing Skills\n",
    "\n",
    "Loading `data/processed/employee_skill_gaps.csv` (1,368 non-manager employees) and exploding semicolon-separated `missing_skills` into individual `(EmployeeNumber, JobRole, skill_name)` tuples."
]),

code("cell-load-explode", [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "\n",
    "PROC = os.path.join('..', 'data', 'processed')\n",
    "gaps_path = os.path.join(PROC, 'employee_skill_gaps.csv')\n",
    "\n",
    "# Load employee skill gaps skipping the header comment\n",
    "df_gaps = pd.read_csv(gaps_path, comment='#')\n",
    "print(f'Loaded employee gap profiles: {len(df_gaps):,} employees')\n",
    "\n",
    "# Explode missing_skills\n",
    "exploded_records = []\n",
    "for _, row in df_gaps.iterrows():\n",
    "    if pd.isna(row['missing_skills']):\n",
    "        continue\n",
    "    missing_str = str(row['missing_skills']).strip()\n",
    "    if missing_str in ('', 'None', 'nan'):\n",
    "        continue\n",
    "        \n",
    "    skills = [s.strip() for s in missing_str.split(';') if s.strip()]\n",
    "    for s in skills:\n",
    "        exploded_records.append({\n",
    "            'EmployeeNumber': row['EmployeeNumber'],\n",
    "            'JobRole': row['JobRole'],\n",
    "            'skill_name': s\n",
    "        })\n",
    "\n",
    "df_exploded = pd.DataFrame(exploded_records)\n",
    "print(f'Total exploded (employee, missing_skill) pairs: {len(df_exploded):,}')\n",
    "print(f'Unique skills represented across all gaps     : {df_exploded[\"skill_name\"].nunique():,}')"
]),

# ── Step 2: Rollup & Severity Classification ─────────────────────────────────
md("md-step2", [
    "---\n",
    "## Step 2 · Roll Up Organization-Wide & Apply Severity Rule\n",
    "\n",
    "**Explicit Severity Threshold Rule:**\n",
    "- **HIGH Severity:** $\\ge 100$ employees missing the skill across the organization\n",
    "- **MEDIUM Severity:** $50$ to $99$ employees missing the skill\n",
    "- **LOW Severity:** $< 50$ employees missing the skill"
]),

code("cell-rollup", [
    "# Group and count total missing employees per skill\n",
    "skill_summary = (\n",
    "    df_exploded.groupby('skill_name')\n",
    "    .size()\n",
    "    .reset_index(name='total_missing_count')\n",
    "    .sort_values('total_missing_count', ascending=False)\n",
    "    .reset_index(drop=True)\n",
    ")\n",
    "\n",
    "# Apply explicit severity rule\n",
    "def classify_severity(count):\n",
    "    if count >= 100:\n",
    "        return 'HIGH'\n",
    "    elif count >= 50:\n",
    "        return 'MEDIUM'\n",
    "    else:\n",
    "        return 'LOW'\n",
    "\n",
    "skill_summary['severity'] = skill_summary['total_missing_count'].apply(classify_severity)\n",
    "\n",
    "# Print severity band breakdown\n",
    "sev_dist = skill_summary['severity'].value_counts()[['HIGH', 'MEDIUM', 'LOW']]\n",
    "print('=== SEVERITY BAND DISTRIBUTION ===')\n",
    "for band, count in sev_dist.items():\n",
    "    print(f'  {band:<8}: {count:2d} skills ({count/len(skill_summary)*100:.1f}%)')\n",
    "print(f'  Total   : {len(skill_summary):2d} unique skills\\n')\n",
    "\n",
    "print('=== FULL RANKED ORGANIZATION SKILL GAP INVENTORY (33 skills) ===')\n",
    "print(f'{\"Rank\":<5} {\"Skill Name\":<42} {\"Missing Count\":<15} {\"Severity\":<10}')\n",
    "print('-' * 75)\n",
    "for idx, r in skill_summary.iterrows():\n",
    "    print(f'{idx+1:<5} {r[\"skill_name\"]:42} {r[\"total_missing_count\"]:>13}   {r[\"severity\"]:10}')"
]),

# ── Step 3: Top 10 Role Breakdown & Concentration Analysis ───────────────────
md("md-step3", [
    "---\n",
    "## Step 3 · Cross-Reference Top Affected Roles & Concentration Flag\n",
    "\n",
    "**Role-Concentration Heuristic:**  \n",
    "A skill is flagged as **Role-Concentrated** (`is_role_concentrated = True`) if **$\\ge 80\\%$** of its organization-wide missing count originates from a **single job role**.  \n",
    "Otherwise, it is designated as **Cross-Cutting** (`is_role_concentrated = False`).\n",
    "\n",
    "> **Training Budget & Strategic Implications:**\n",
    "> - **Cross-Cutting Skills** (e.g., *Speaking, Reading Comprehension, Critical Thinking, MS Office, Excel*): Warrant company-wide L&D initiatives, centralized asynchronous learning platforms, or general onboarding modules.\n",
    "> - **Role-Concentrated Skills** (e.g., *AWS CloudFormation, EC2, DynamoDB, MEDITECH, AutoCAD*): Require targeted departmental budget allocation — funding specialized external certifications or domain bootcamps for specific teams rather than diluting funds across broad enterprise training."
]),

code("cell-concentration", [
    "CONCENTRATION_THRESHOLD = 0.80\n",
    "\n",
    "top_roles_list = []\n",
    "is_conc_list = []\n",
    "max_role_pct_list = []\n",
    "\n",
    "for _, row in skill_summary.iterrows():\n",
    "    s_name = row['skill_name']\n",
    "    sub = df_exploded[df_exploded['skill_name'] == s_name]\n",
    "    role_counts = sub['JobRole'].value_counts()\n",
    "    \n",
    "    # Format role string\n",
    "    role_str = ', '.join([f'{role} ({cnt})' for role, cnt in role_counts.items()])\n",
    "    top_roles_list.append(role_str)\n",
    "    \n",
    "    # Max share\n",
    "    top_share = role_counts.iloc[0] / len(sub)\n",
    "    max_role_pct_list.append(round(top_share * 100, 1))\n",
    "    is_conc_list.append(top_share >= CONCENTRATION_THRESHOLD)\n",
    "\n",
    "skill_summary['top_affected_roles'] = top_roles_list\n",
    "skill_summary['is_role_concentrated'] = is_conc_list\n",
    "skill_summary['max_role_share_pct'] = max_role_pct_list\n",
    "\n",
    "print('=== TOP 10 ORGANIZATION SKILL GAPS WITH ROLE BREAKDOWN ===\\n')\n",
    "for idx in range(10):\n",
    "    r = skill_summary.iloc[idx]\n",
    "    flag = 'ROLE-CONCENTRATED (Targeted)' if r['is_role_concentrated'] else 'CROSS-CUTTING (Company-Wide)'\n",
    "    print(f'{idx+1:2d}. {r[\"skill_name\"]} (Total Missing: {r[\"total_missing_count\"]}) — [{r[\"severity\"]}] — {flag}')\n",
    "    print(f'    Max Role Share : {r[\"max_role_share_pct\"]}%')\n",
    "    print(f'    Role Breakdown : {r[\"top_affected_roles\"]}\\n')\n",
    "\n",
    "# Concentration summary\n",
    "conc_counts = skill_summary['is_role_concentrated'].value_counts()\n",
    "print('=== CONCENTRATION DISTRIBUTION OVER ALL 33 SKILLS ===')\n",
    "print(f'  Role-Concentrated (>=80% from 1 role) : {conc_counts.get(True, 0)} skills')\n",
    "print(f'  Cross-Cutting (<80% concentration)    : {conc_counts.get(False, 0)} skills')"
]),

# ── Step 4: Save Output Artifact ─────────────────────────────────────────────
md("md-step4", [
    "---\n",
    "## Step 4 · Save Output Dataset (`organization_skill_gaps.csv`)\n",
    "\n",
    "Saving the organization-wide gap inventory to `data/processed/organization_skill_gaps.csv`.  \n",
    "The synthetic warning comment is retained in line 1."
]),

code("cell-save", [
    "out_file = os.path.join(PROC, 'organization_skill_gaps.csv')\n",
    "\n",
    "# Export columns requested\n",
    "export_cols = ['skill_name', 'total_missing_count', 'severity', 'top_affected_roles', 'is_role_concentrated']\n",
    "df_export = skill_summary[export_cols]\n",
    "\n",
    "warning_comment = (\n",
    "    '# SYNTHETIC DATA — employee current-skill possession was not present in any source file '\n",
    "    'and has been simulated using a tenure/training-based heuristic for MVP demonstration purposes only. '\n",
    "    'This must NOT be presented to stakeholders as real observed skill data. Real deployment requires '\n",
    "    'an actual skills inventory (HRIS export, LMS completion records, or self-assessment survey).\\n'\n",
    ")\n",
    "\n",
    "with open(out_file, 'w', encoding='utf-8') as f:\n",
    "    f.write(warning_comment)\n",
    "    df_export.to_csv(f, index=False)\n",
    "\n",
    "file_size = os.path.getsize(out_file)\n",
    "print(f'Saved organization skill gaps to: {out_file}')\n",
    "print(f'File size: {file_size:,} bytes')\n",
    "print(f'Total skills recorded: {len(df_export)}')\n",
    "\n",
    "# Round-trip reload verification\n",
    "df_reloaded = pd.read_csv(out_file, comment='#')\n",
    "assert len(df_reloaded) == 33, 'Row count mismatch on reload!'\n",
    "assert list(df_reloaded.columns) == export_cols, 'Column mismatch on reload!'\n",
    "print('CONFIRMED: Round-trip verification passed cleanly.')"
])

]

out_path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\13_organization_skill_gap.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Written: {out_path}')
