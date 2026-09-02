import json
import pandas as pd
import os

# ── 1. Inspect existing notebook ──────────────────────────────────────────────
nb_path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\05_feature_engineering.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Existing cell count: {len(nb["cells"])}')

# ── 2. Build the diagnostic cell source ───────────────────────────────────────
diag_source = [
    "# ============================================================\n",
    "# DIAGNOSTIC CELL — added post-execution\n",
    "# Purpose: audit column composition of features_unscaled.csv\n",
    "# Rule: does NOT re-save any CSV; read-only diagnostic.\n",
    "# ============================================================\n",
    "import pandas as pd\n",
    "import os\n",
    "\n",
    "PROC = os.path.join('..', 'data', 'processed')\n",
    "df_check = pd.read_csv(os.path.join(PROC, 'features_unscaled.csv'))\n",
    "all_cols = df_check.columns.tolist()\n",
    "print(f'Total columns in features_unscaled.csv: {len(all_cols)}')\n",
    "print()\n",
    "\n",
    "# ── Known groups from Step 5 ──────────────────────────────────\n",
    "\n",
    "TARGET_COL = 'Attrition'\n",
    "\n",
    "ENGINEERED = [\n",
    "    'income_per_year_at_company',\n",
    "    'years_since_promotion_ratio',\n",
    "    'overall_satisfaction_score',\n",
    "    'experience_ratio',\n",
    "]\n",
    "\n",
    "# Binary-encoded (original categorical, collapsed to 0/1 in Step 5)\n",
    "BINARY_ENCODED = ['Gender', 'OverTime']\n",
    "\n",
    "# OHE parents (drop_first=True was used in pd.get_dummies)\n",
    "OHE_PARENTS = {\n",
    "    'BusinessTravel':  3,   # 3 levels -> 2 dummies with drop_first\n",
    "    'Department':      3,   # 3 levels -> 2 dummies\n",
    "    'EducationField':  6,   # 6 levels -> 5 dummies\n",
    "    'JobRole':         9,   # 9 levels -> 8 dummies\n",
    "    'MaritalStatus':   3,   # 3 levels -> 2 dummies\n",
    "}\n",
    "\n",
    "# Collect all OHE dummy column names actually present\n",
    "ohe_dummies = [c for c in all_cols\n",
    "               if any(c.startswith(p + '_') for p in OHE_PARENTS)]\n",
    "\n",
    "# Numeric/continuous = everything else except target, engineered, binary, OHE dummies\n",
    "accounted = set([TARGET_COL] + ENGINEERED + BINARY_ENCODED + ohe_dummies)\n",
    "numeric_continuous = [c for c in all_cols if c not in accounted]\n",
    "\n",
    "# ── Print groups ─────────────────────────────────────────────\n",
    "print('=' * 60)\n",
    "print('GROUP 1 — Numeric/continuous columns (kept as-is from raw data)')\n",
    "print('=' * 60)\n",
    "for c in numeric_continuous:\n",
    "    print(f'  {c}')\n",
    "print(f'  COUNT: {len(numeric_continuous)}')\n",
    "print()\n",
    "\n",
    "print('=' * 60)\n",
    "print('GROUP 2 — Categorical columns (encoded in Step 5)')\n",
    "print('=' * 60)\n",
    "print('  2a. Binary-encoded (0/1 mapping, original 2-level categoricals):')\n",
    "for c in BINARY_ENCODED:\n",
    "    print(f'    {c} -> 1 column (binary 0/1)')\n",
    "print(f'    COUNT: {len(BINARY_ENCODED)}')\n",
    "print()\n",
    "print('  2b. One-hot encoded (pd.get_dummies, drop_first=True):')\n",
    "ohe_col_count = 0\n",
    "for parent, n_levels in OHE_PARENTS.items():\n",
    "    actual_dummies = [c for c in ohe_dummies if c.startswith(parent + '_')]\n",
    "    expected_dummies = n_levels - 1  # drop_first removes 1\n",
    "    match_str = 'OK' if len(actual_dummies) == expected_dummies else f'MISMATCH — expected {expected_dummies}'\n",
    "    print(f'    {parent} ({n_levels} levels) -> {len(actual_dummies)} dummies [drop_first=True, {match_str}]')\n",
    "    for d in actual_dummies:\n",
    "        print(f'      {d}')\n",
    "    ohe_col_count += len(actual_dummies)\n",
    "print(f'    OHE dummy column COUNT: {ohe_col_count}')\n",
    "total_cat = len(BINARY_ENCODED) + ohe_col_count\n",
    "print(f'    TOTAL categorical-derived columns: {total_cat}')\n",
    "print()\n",
    "\n",
    "print('=' * 60)\n",
    "print('GROUP 3 — Engineered features')\n",
    "print('=' * 60)\n",
    "for c in ENGINEERED:\n",
    "    present = '✓ present' if c in all_cols else '✗ MISSING'\n",
    "    print(f'  {c}  [{present}]')\n",
    "print(f'  COUNT: {len(ENGINEERED)}')\n",
    "print()\n",
    "\n",
    "print('=' * 60)\n",
    "print('GROUP 4 — Target column')\n",
    "print('=' * 60)\n",
    "print(f'  {TARGET_COL}  [{\"✓ present\" if TARGET_COL in all_cols else \"✗ MISSING\"}]')\n",
    "print(f'  COUNT: 1')\n",
    "print()\n",
    "\n",
    "# ── Tally & verify ───────────────────────────────────────────\n",
    "tally = len(numeric_continuous) + total_cat + len(ENGINEERED) + 1\n",
    "print('=' * 60)\n",
    "print('COLUMN TALLY')\n",
    "print('=' * 60)\n",
    "print(f'  Group 1 — Numeric/continuous          : {len(numeric_continuous)}')\n",
    "print(f'  Group 2a — Binary-encoded categoricals: {len(BINARY_ENCODED)}')\n",
    "print(f'  Group 2b — OHE dummy columns          : {ohe_col_count}')\n",
    "print(f'  Group 3  — Engineered features        : {len(ENGINEERED)}')\n",
    "print(f'  Group 4  — Target (Attrition)         : 1')\n",
    "print(f'  TOTAL                                 : {tally}')\n",
    "print(f'  ACTUAL columns in CSV                 : {len(all_cols)}')\n",
    "print()\n",
    "if tally == len(all_cols):\n",
    "    print(f'CONFIRMED: groups sum to {tally} == actual column count {len(all_cols)} ✅')\n",
    "else:\n",
    "    diff = len(all_cols) - tally\n",
    "    print(f'DISCREPANCY: groups sum to {tally}, actual is {len(all_cols)} (delta={diff:+d})')\n",
    "    unaccounted = [c for c in all_cols\n",
    "                   if c not in numeric_continuous\n",
    "                   and c not in BINARY_ENCODED\n",
    "                   and c not in ohe_dummies\n",
    "                   and c not in ENGINEERED\n",
    "                   and c != TARGET_COL]\n",
    "    if unaccounted:\n",
    "        print(f'  Unaccounted columns ({len(unaccounted)}): {unaccounted}')\n",
    "    missing_from_csv = [c for c in ENGINEERED + BINARY_ENCODED\n",
    "                        if c not in all_cols]\n",
    "    if missing_from_csv:\n",
    "        print(f'  Expected but absent from CSV: {missing_from_csv}')\n",
    "\n",
    "print()\n",
    "print('drop_first CONFIRMATION:')\n",
    "print('  pd.get_dummies(..., drop_first=True, dtype=int) was used in Step 5, cell-ohe.')\n",
    "print('  This means the FIRST alphabetical level of each OHE column was dropped:')\n",
    "for parent, n_levels in OHE_PARENTS.items():\n",
    "    actual_dummies = [c for c in ohe_dummies if c.startswith(parent + '_')]\n",
    "    n_dropped = n_levels - len(actual_dummies)\n",
    "    print(f'    {parent}: dropped {n_dropped} level(s) as reference category')\n"
]

# ── 3. Create new cells (one markdown header + one code cell) ─────────────────
new_md_cell = {
    "cell_type": "markdown",
    "id": "diag-md-header",
    "metadata": {},
    "source": [
        "---\n",
        "## Diagnostic · Column Composition Audit\n",
        "\n",
        "Added post-execution. **Read-only — does not re-save any output CSV.**  \n",
        "Breaks down the 49 columns in `features_unscaled.csv` by group and confirms\n",
        "the `drop_first=True` flag used in `pd.get_dummies()`.\n",
        "\n",
        "---"
    ]
}

new_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "diag-code-main",
    "metadata": {},
    "outputs": [],
    "source": diag_source
}

# Append to notebook
nb["cells"].append(new_md_cell)
nb["cells"].append(new_code_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Appended 2 new cells. Total cells now: {len(nb["cells"])}')
print('Notebook saved.')
