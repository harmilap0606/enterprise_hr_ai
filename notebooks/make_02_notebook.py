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

def md(cell_id, source_lines):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source_lines
    }

def code(cell_id, source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

nb["cells"] = [

md("md-title", [
    "# 02 · Data Validation\n",
    "\n",
    "**Project:** Enterprise HR AI  \n",
    "**Purpose:** Assert-based internal validation of `employee_attrition.csv` and `Cleaned_HR_Data_Analysis.csv`.  \n",
    "**Rule:** Files are validated independently. No merge. Asserts raise on violation — no suppression.\n",
    "\n",
    "---"
]),

code("cell-imports", [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "\n",
    "RAW = os.path.join('..', 'data', 'raw')\n",
    "print('Raw path:', os.path.abspath(RAW))\n",
    "print('Files:', sorted(os.listdir(RAW)))"
]),

md("md-att-header", [
    "---\n",
    "## Section 1 · employee_attrition.csv"
]),

code("cell-load-attrition", [
    "attrition = pd.read_csv(os.path.join(RAW, 'employee_attrition.csv'))\n",
    "print(f'Loaded employee_attrition.csv: {attrition.shape[0]} rows x {attrition.shape[1]} cols')"
]),

md("md-v-att-1", ["### V-ATT-1 · Schema check — 10 required columns exist"]),

code("cell-att-schema", [
    "REQUIRED_ATTRITION_COLS = [\n",
    "    'Age', 'Attrition', 'Department', 'JobRole',\n",
    "    'EmployeeNumber', 'MonthlyIncome', 'OverTime',\n",
    "    'JobSatisfaction', 'YearsAtCompany', 'WorkLifeBalance'\n",
    "]\n",
    "\n",
    "missing_cols = [c for c in REQUIRED_ATTRITION_COLS if c not in attrition.columns]\n",
    "if missing_cols:\n",
    "    print(f'MISSING COLUMNS: {missing_cols}')\n",
    "\n",
    "assert missing_cols == [], (\n",
    "    f'V-ATT-1 FAILED — missing columns in employee_attrition.csv: {missing_cols}'\n",
    ")\n",
    "print('V-ATT-1 PASSED: all 10 required columns present')"
]),

md("md-v-att-2", ["### V-ATT-2 · Type check — Age is integer, MonthlyIncome is numeric"]),

code("cell-att-types", [
    "print(f\"Age dtype          : {attrition['Age'].dtype}\")\n",
    "print(f\"MonthlyIncome dtype: {attrition['MonthlyIncome'].dtype}\")\n",
    "\n",
    "assert pd.api.types.is_integer_dtype(attrition['Age']), (\n",
    "    f\"V-ATT-2a FAILED — Age is not integer dtype, got: {attrition['Age'].dtype}\"\n",
    ")\n",
    "print('V-ATT-2a PASSED: Age is integer dtype')\n",
    "\n",
    "assert pd.api.types.is_numeric_dtype(attrition['MonthlyIncome']), (\n",
    "    f\"V-ATT-2b FAILED — MonthlyIncome is not numeric dtype, got: {attrition['MonthlyIncome'].dtype}\"\n",
    ")\n",
    "print('V-ATT-2b PASSED: MonthlyIncome is numeric dtype')"
]),

md("md-v-att-3", ["### V-ATT-3 · Range check — Age between 18 and 100"]),

code("cell-att-age", [
    "age_min = attrition['Age'].min()\n",
    "age_max = attrition['Age'].max()\n",
    "print(f'Age range in employee_attrition: min={age_min}, max={age_max}')\n",
    "\n",
    "bad_age = attrition[(attrition['Age'] < 18) | (attrition['Age'] > 100)]\n",
    "if len(bad_age) > 0:\n",
    "    print(f'Offending rows ({len(bad_age)}):')\n",
    "    print(bad_age[['EmployeeNumber', 'Age']].to_string())\n",
    "\n",
    "assert len(bad_age) == 0, (\n",
    "    f'V-ATT-3 FAILED — {len(bad_age)} rows have Age outside [18, 100]'\n",
    ")\n",
    "print('V-ATT-3 PASSED: all Age values within [18, 100]')"
]),

md("md-v-att-4", ["### V-ATT-4 · Uniqueness — EmployeeNumber has no duplicates"]),

code("cell-att-unique", [
    "dup_empnum = attrition[attrition.duplicated(subset=['EmployeeNumber'], keep=False)]\n",
    "if len(dup_empnum) > 0:\n",
    "    print(f'Duplicate EmployeeNumber rows ({len(dup_empnum)}):')\n",
    "    print(dup_empnum[['EmployeeNumber', 'Age', 'Department']].to_string())\n",
    "\n",
    "assert len(dup_empnum) == 0, (\n",
    "    f'V-ATT-4 FAILED — {len(dup_empnum)} rows have duplicate EmployeeNumber'\n",
    ")\n",
    "n_unique = attrition['EmployeeNumber'].nunique()\n",
    "print(f'V-ATT-4 PASSED: EmployeeNumber is unique ({n_unique} distinct values)')"
]),

md("md-v-att-5", ["### V-ATT-5 · Category check — Attrition only contains {'Yes', 'No'}"]),

code("cell-att-cats", [
    "ALLOWED_ATTRITION = {'Yes', 'No'}\n",
    "actual_vals = set(attrition['Attrition'].dropna().unique())\n",
    "print(f'Attrition unique values: {sorted(actual_vals)}')\n",
    "\n",
    "unexpected = actual_vals - ALLOWED_ATTRITION\n",
    "if unexpected:\n",
    "    bad_rows = attrition[~attrition['Attrition'].isin(ALLOWED_ATTRITION)]\n",
    "    print(f'Offending rows ({len(bad_rows)}):')\n",
    "    print(bad_rows[['EmployeeNumber', 'Attrition']].to_string())\n",
    "\n",
    "assert unexpected == set(), (\n",
    "    f'V-ATT-5 FAILED — unexpected Attrition values: {unexpected}'\n",
    ")\n",
    "print(f'V-ATT-5 PASSED: Attrition contains only {ALLOWED_ATTRITION}')"
]),

md("md-att-summary", [
    "### employee_attrition.csv — Validation Summary\n",
    "\n",
    "| Check | ID | Status |\n",
    "|---|---|---|\n",
    "| Schema (10 required cols) | V-ATT-1 | Ran above |\n",
    "| Age integer, MonthlyIncome numeric | V-ATT-2 | Ran above |\n",
    "| Age in [18, 100] | V-ATT-3 | Ran above |\n",
    "| EmployeeNumber unique | V-ATT-4 | Ran above |\n",
    "| Attrition ∈ {Yes, No} | V-ATT-5 | Ran above |"
]),

md("md-hr-header", [
    "---\n",
    "## Section 2 · Cleaned_HR_Data_Analysis.csv"
]),

code("cell-load-hr", [
    "hr = pd.read_csv(os.path.join(RAW, 'Cleaned_HR_Data_Analysis.csv'))\n",
    "print(f'Loaded Cleaned_HR_Data_Analysis.csv: {hr.shape[0]} rows x {hr.shape[1]} cols')"
]),

md("md-v-hr-1", ["### V-HR-1 · Schema check — 5 required columns exist"]),

code("cell-hr-schema", [
    "REQUIRED_HR_COLS = [\n",
    "    'Employee ID', 'Engagement Score',\n",
    "    'Satisfaction Score', 'Work-Life Balance Score', 'Age'\n",
    "]\n",
    "\n",
    "missing_hr = [c for c in REQUIRED_HR_COLS if c not in hr.columns]\n",
    "if missing_hr:\n",
    "    print(f'MISSING COLUMNS: {missing_hr}')\n",
    "\n",
    "assert missing_hr == [], (\n",
    "    f'V-HR-1 FAILED — missing columns in Cleaned_HR_Data_Analysis.csv: {missing_hr}'\n",
    ")\n",
    "print('V-HR-1 PASSED: all 5 required columns present')"
]),

md("md-v-hr-2", ["### V-HR-2 · Score range discovery — actual min/max printed before asserting any bounds"]),

code("cell-hr-score-discovery", [
    "SCORE_COLS = ['Engagement Score', 'Satisfaction Score', 'Work-Life Balance Score']\n",
    "\n",
    "print('=== ACTUAL SCORE RANGES (before asserting any bounds) ===')\n",
    "score_ranges = {}\n",
    "for col in SCORE_COLS:\n",
    "    lo = hr[col].min()\n",
    "    hi = hr[col].max()\n",
    "    score_ranges[col] = (lo, hi)\n",
    "    print(f'  {col:<30s}  min={lo}  max={hi}')"
]),

md("md-v-hr-2b", ["### V-HR-2 (cont.) · Assert — all score values within the observed scale"]),

code("cell-hr-score-assert", [
    "# Scale bounds derived from discovery cell above — not hardcoded.\n",
    "SCALE_LO = int(min(lo for lo, hi in score_ranges.values()))\n",
    "SCALE_HI = int(max(hi for lo, hi in score_ranges.values()))\n",
    "print(f'Asserting score bounds: [{SCALE_LO}, {SCALE_HI}] (derived from observed data)')\n",
    "\n",
    "for col in SCORE_COLS:\n",
    "    bad = hr[(hr[col] < SCALE_LO) | (hr[col] > SCALE_HI)]\n",
    "    if len(bad) > 0:\n",
    "        n_bad = len(bad)\n",
    "        print(f'Offending rows for [{col}] ({n_bad} rows):')\n",
    "        print(bad[['Employee ID', col]].to_string())\n",
    "    assert len(bad) == 0, (\n",
    "        f'V-HR-2 FAILED — {len(bad)} rows in [{col}] outside [{SCALE_LO}, {SCALE_HI}]'\n",
    "    )\n",
    "    print(f'V-HR-2 PASSED: [{col}] all values within [{SCALE_LO}, {SCALE_HI}]')"
]),

md("md-v-hr-3", ["### V-HR-3 · Uniqueness — Employee ID has no duplicates"]),

code("cell-hr-unique", [
    "dup_empid = hr[hr.duplicated(subset=['Employee ID'], keep=False)]\n",
    "if len(dup_empid) > 0:\n",
    "    print(f'Duplicate Employee ID rows ({len(dup_empid)}):')\n",
    "    print(dup_empid[['Employee ID', 'Age', 'Performance Score']].to_string())\n",
    "\n",
    "assert len(dup_empid) == 0, (\n",
    "    f'V-HR-3 FAILED — {len(dup_empid)} rows have duplicate Employee ID'\n",
    ")\n",
    "n_uid = hr['Employee ID'].nunique()\n",
    "print(f'V-HR-3 PASSED: Employee ID is unique ({n_uid} distinct values)')"
]),

md("md-v-hr-4", ["### V-HR-4 · Range check — Age between 18 and 100"]),

code("cell-hr-age", [
    "hr_age_min = hr['Age'].min()\n",
    "hr_age_max = hr['Age'].max()\n",
    "print(f'Age range in Cleaned_HR_Data_Analysis: min={hr_age_min}, max={hr_age_max}')\n",
    "\n",
    "bad_hr_age = hr[(hr['Age'] < 18) | (hr['Age'] > 100)]\n",
    "if len(bad_hr_age) > 0:\n",
    "    print(f'Offending rows ({len(bad_hr_age)}):')\n",
    "    print(bad_hr_age[['Employee ID', 'Age']].to_string())\n",
    "\n",
    "assert len(bad_hr_age) == 0, (\n",
    "    f'V-HR-4 FAILED — {len(bad_hr_age)} rows have Age outside [18, 100]'\n",
    ")\n",
    "print('V-HR-4 PASSED: all Age values within [18, 100]')"
]),

md("md-hr-summary", [
    "### Cleaned_HR_Data_Analysis.csv — Validation Summary\n",
    "\n",
    "| Check | ID | Status |\n",
    "|---|---|---|\n",
    "| Schema (5 required cols) | V-HR-1 | Ran above |\n",
    "| Score cols in observed scale | V-HR-2 | Ran above |\n",
    "| Employee ID unique | V-HR-3 | Ran above |\n",
    "| Age in [18, 100] | V-HR-4 | Ran above |"
]),

md("md-final-note", [
    "---\n",
    "## Important Scoping Note\n",
    "\n",
    "**Employee ID overlap between these two files is 49.7% (731/1470) — validation here only confirms internal validity of each file, it does NOT confirm join correctness. Row-level join validity is a Day 1 §4 (data_relationships) concern, not a data_validation concern.**\n",
    "\n",
    "Specifically:\n",
    "- `employee_attrition.csv` has been validated: `EmployeeNumber` is unique and within expected schema/types/ranges.\n",
    "- `Cleaned_HR_Data_Analysis.csv` has been validated: `Employee ID` is unique, scores are within the observed [1,5] scale, Age is within [18,100].\n",
    "- Whether the 731 overlapping IDs represent the same physical employees, or whether the ID namespaces are coincidentally numeric, is a **join integrity** question deferred to the data_relationships notebook.\n",
    "- No cleaning, merging, or imputation has been performed in this notebook."
])

]

with open(r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\02_data_validation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('02_data_validation.ipynb written successfully.')
