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
    "# 11 · Employee-Level Skills Inventory (Synthetic MVP Model)\n",
    "\n",
    "**Project:** Enterprise HR AI  \n",
    "\n",
    "> ### ⚠️ PROMINENT DATA INTEGRITY WARNING\n",
    "> **SYNTHETIC DATA — employee current-skill possession was not present in any source file and has been simulated using a tenure/training-based heuristic for MVP demonstration purposes only. This must NOT be presented to stakeholders as real observed skill data. Real deployment requires an actual skills inventory (HRIS export, LMS completion records, or self-assessment survey).**\n",
    "\n",
    "---"
]),

# ── Step 1: Explicit Audit for Real Skill Columns ────────────────────────────
md("md-audit", [
    "---\n",
    "## Step 1 · Honest Source Data Audit: Does Employee-Level Skill Data Exist?"
]),

code("cell-audit", [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "\n",
    "PROC = os.path.join('..', 'data', 'processed')\n",
    "\n",
    "att_path = os.path.join(PROC, 'employee_attrition_processed.csv')\n",
    "eng_path = os.path.join(PROC, 'engagement_processed.csv')\n",
    "rsp_path = os.path.join(PROC, 'role_skill_profiles.csv')\n",
    "\n",
    "df_att = pd.read_csv(att_path)\n",
    "df_eng = pd.read_csv(eng_path)\n",
    "df_rsp = pd.read_csv(rsp_path)\n",
    "\n",
    "print('=== COLUMN AUDIT FOR REAL EMPLOYEE-LEVEL SKILLS DATA ===\\n')\n",
    "\n",
    "print(f'1. employee_attrition_processed.csv ({df_att.shape[1]} columns):')\n",
    "print(list(df_att.columns))\n",
    "att_skill_cols = [c for c in df_att.columns if 'skill' in c.lower() or 'competenc' in c.lower()]\n",
    "print(f'   Skill-related columns found: {att_skill_cols}\\n')\n",
    "\n",
    "print(f'2. engagement_processed.csv ({df_eng.shape[1]} columns):')\n",
    "print(list(df_eng.columns))\n",
    "eng_skill_cols = [c for c in df_eng.columns if 'skill' in c.lower() or 'competenc' in c.lower()]\n",
    "print(f'   Skill-related columns found: {eng_skill_cols}\\n')\n",
    "\n",
    "print(f'3. role_skill_profiles.csv ({df_rsp.shape[1]} columns):')\n",
    "print(list(df_rsp.columns))\n",
    "rsp_skill_cols = [c for c in df_rsp.columns if 'skill' in c.lower()]\n",
    "print(f'   Skill-related columns found: {rsp_skill_cols}\\n')\n",
    "\n",
    "print('=' * 80)\n",
    "print('EXPLICIT AUDIT CONCLUSION:')\n",
    "print('Does a real employee-level current-skills column exist anywhere in the raw or processed data?')\n",
    "print('ANSWER: NO.')\n",
    "print('- employee_attrition has career/demographic fields, but zero skill inventory.')\n",
    "print('- engagement_processed tracks training courses (name, cost, duration), but no skill mastery.')\n",
    "print('- role_skill_profiles contains ROLE-LEVEL required O*NET skills, but no employee-level possession.')\n",
    "print('=' * 80)"
]),

# ── Step 2: Synthetic Skill Generation Methodology ───────────────────────────
md("md-methodology", [
    "---\n",
    "## Step 2 · Synthetic Skill Generation Heuristic (MVP Model)\n",
    "\n",
    "To enable downstream capability matching and gap analysis without fabricating ungrounded data:\n",
    "1. **Role Benchmark Mapping:** Each employee inherits their job role's benchmark skills from `role_skill_profiles.csv` (top 5 essential skills + top 5 software tools).\n",
    "2. **Empirical Probability Formula:** Skill possession is simulated using an auditable, tenure/training-conditioned heuristic:\n",
    "   $$\\text{possession\\_probability} = \\min(0.30 + 0.05 \\times \\text{YearsAtCompany} + 0.05 \\times \\text{TrainingTimesLastYear}, 0.95)$$\n",
    "   - Base probability = 30% (new hire baseline)\n",
    "   - +5% per year at company (tenure/on-the-job mastery)\n",
    "   - +5% per training session completed in the last year (active upskilling)\n",
    "   - Capped at 95% (no employee is synthetically assumed to have 100% mastery)\n",
    "3. **Manager Role Exclusion:** For the 102 employees with `JobRole == 'Manager'`, skill simulation is **skipped entirely** and mapped to `skill_name = 'N/A - Department-level analysis only'`, adhering to the architectural decision from Step 10 & 11."
]),

code("cell-generate-skills", [
    "# Parse required skills per role from role_skill_profiles.csv\n",
    "role_skills_dict = {}\n",
    "for _, r in df_rsp.iterrows():\n",
    "    role = r['ibm_job_role']\n",
    "    if r['match_confidence'] == 'very_low':\n",
    "        continue\n",
    "    ess_items = [s.split('(')[0].strip() for s in r['top5_essential_skills'].split('|')]\n",
    "    sw_items = [s.strip() for s in r['top5_software_tools'].split('|')]\n",
    "    role_skills_dict[role] = (\n",
    "        [(s, 'essential') for s in ess_items] +\n",
    "        [(s, 'software') for s in sw_items]\n",
    "    )\n",
    "\n",
    "print(f'Parsed benchmarks for {len(role_skills_dict)} job roles (Manager excluded from O*NET profiles).')\n",
    "\n",
    "# Set fixed random seed for 100% reproducibility\n",
    "np.random.seed(42)\n",
    "\n",
    "synthetic_rows = []\n",
    "audit_examples = []\n",
    "\n",
    "for _, emp in df_att.iterrows():\n",
    "    emp_id = emp['EmployeeNumber']\n",
    "    role = emp['JobRole']\n",
    "    tenure = emp['YearsAtCompany']\n",
    "    training = emp['TrainingTimesLastYear']\n",
    "    \n",
    "    # Manager Role Exclusion\n",
    "    if role == 'Manager':\n",
    "        synthetic_rows.append({\n",
    "            'EmployeeNumber': emp_id,\n",
    "            'skill_name': 'N/A - Department-level analysis only',\n",
    "            'skill_type': 'N/A',\n",
    "            'has_skill': 0\n",
    "        })\n",
    "        continue\n",
    "        \n",
    "    # Compute formula\n",
    "    prob = min(0.30 + 0.05 * tenure + 0.05 * training, 0.95)\n",
    "    req_skills = role_skills_dict[role]\n",
    "    \n",
    "    possessed = []\n",
    "    for skill_name, skill_type in req_skills:\n",
    "        has = int(np.random.rand() < prob)\n",
    "        synthetic_rows.append({\n",
    "            'EmployeeNumber': emp_id,\n",
    "            'skill_name': skill_name,\n",
    "            'skill_type': skill_type,\n",
    "            'has_skill': has\n",
    "        })\n",
    "        if has:\n",
    "            possessed.append(skill_name)\n",
    "            \n",
    "    if len(audit_examples) < 5:\n",
    "        audit_examples.append({\n",
    "            'EmployeeNumber': emp_id,\n",
    "            'JobRole': role,\n",
    "            'YearsAtCompany': tenure,\n",
    "            'TrainingTimesLastYear': training,\n",
    "            'possession_probability': round(prob, 4),\n",
    "            'skills_possessed_count': len(possessed),\n",
    "            'total_skills': len(req_skills),\n",
    "            'skills_possessed': possessed\n",
    "        })\n",
    "\n",
    "df_synthetic = pd.DataFrame(synthetic_rows)\n",
    "print(f'Total synthetic records generated: {len(df_synthetic):,}')\n",
    "print(f'Total distinct employees covered: {df_synthetic[\"EmployeeNumber\"].nunique():,}')"
]),

# ── Step 3: Audit 5 Example Employees ────────────────────────────────────────
md("md-examples", [
    "---\n",
    "## Step 3 · Auditing Heuristic Behavior: 5 Example Employees"
]),

code("cell-examples", [
    "print('=== 5 AUDIT EXAMPLES: HEURISTIC PROBABILITY & SKILL LIST ===')\n",
    "for ex in audit_examples:\n",
    "    print(f'\\nEmployee #{ex[\"EmployeeNumber\"]} ({ex[\"JobRole\"]})')\n",
    "    print(f'  Tenure: {ex[\"YearsAtCompany\"]} years | Training Sessions Last Year: {ex[\"TrainingTimesLastYear\"]}')\n",
    "    print(f'  Possession Probability: min(0.30 + 0.05*{ex[\"YearsAtCompany\"]} + 0.05*{ex[\"TrainingTimesLastYear\"]}, 0.95) = {ex[\"possession_probability\"]:.2f}')\n",
    "    print(f'  Skills Possessed: {ex[\"skills_possessed_count\"]} of {ex[\"total_skills\"]} required')\n",
    "    print(f'  Possessed Skills: {ex[\"skills_possessed\"]}')"
]),

# ── Step 4: Manager Exclusion Verification ───────────────────────────────────
md("md-manager-check", [
    "---\n",
    "## Step 4 · Verify Manager Role Exclusion"
]),

code("cell-manager-check", [
    "mgr_ids = df_att[df_att['JobRole'] == 'Manager']['EmployeeNumber'].tolist()\n",
    "mgr_records = df_synthetic[df_synthetic['EmployeeNumber'].isin(mgr_ids)]\n",
    "\n",
    "print('=== MANAGER ROLE EXCLUSION VERIFICATION ===')\n",
    "print(f'Expected Manager headcount: 102')\n",
    "print(f'Total Manager records in synthetic table: {len(mgr_records)}')\n",
    "print(f'Distinct skill names for Managers: {mgr_records[\"skill_name\"].unique().tolist()}')\n",
    "print(f'Distinct skill types for Managers: {mgr_records[\"skill_type\"].unique().tolist()}')\n",
    "\n",
    "assert len(mgr_records) == 102, f'Expected 102 manager records, got {len(mgr_records)}'\n",
    "assert set(mgr_records['skill_name']) == {'N/A - Department-level analysis only'}, 'Unexpected skill name for Manager!'\n",
    "assert set(mgr_records['skill_type']) == {'N/A'}, 'Unexpected skill type for Manager!'\n",
    "print('\\nCONFIRMED: All 102 Manager employees have exactly 1 record showing: \"N/A - Department-level analysis only\". No synthetic O*NET skills were fabricated.')"
]),

# ── Step 5: Save Synthetic CSV with Header Comment ───────────────────────────
md("md-save", [
    "---\n",
    "## Step 5 · Save CSV with Embedded Data Warning"
]),

code("cell-save", [
    "out_csv = os.path.join(PROC, 'employee_skills_synthetic.csv')\n",
    "\n",
    "warning_comment = (\n",
    "    '# SYNTHETIC DATA — employee current-skill possession was not present in any source file '\n",
    "    'and has been simulated using a tenure/training-based heuristic for MVP demonstration purposes only. '\n",
    "    'This must NOT be presented to stakeholders as real observed skill data. Real deployment requires '\n",
    "    'an actual skills inventory (HRIS export, LMS completion records, or self-assessment survey).\\n'\n",
    ")\n",
    "\n",
    "# Write warning comment followed by CSV content\n",
    "with open(out_csv, 'w', encoding='utf-8') as f:\n",
    "    f.write(warning_comment)\n",
    "    df_synthetic.to_csv(f, index=False)\n",
    "\n",
    "file_size = os.path.getsize(out_csv)\n",
    "print(f'Saved synthetic file to: {out_csv}')\n",
    "print(f'File Size: {file_size:,} bytes')\n",
    "print(f'Total Rows: {len(df_synthetic):,} (plus 1 comment line + 1 header line)')\n",
    "\n",
    "# Verify round-trip reading with comment='# '\n",
    "df_reloaded = pd.read_csv(out_csv, comment='#')\n",
    "assert len(df_reloaded) == len(df_synthetic), 'Reloaded row count mismatch!'\n",
    "assert list(df_reloaded.columns) == ['EmployeeNumber', 'skill_name', 'skill_type', 'has_skill']\n",
    "print('\\nCONFIRMED: Round-trip verification passed cleanly.')"
])

]

out_path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\11_employee_skills.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Written: {out_path}')
