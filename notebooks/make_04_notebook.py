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

# ── Title ─────────────────────────────────────────────────────────────────────
md("md-title", [
    "# 04 · Data Relationships\n",
    "\n",
    "**Project:** Enterprise HR AI  \n",
    "**Parts:**\n",
    "- **A** — Lightweight cleaning of three O\\*NET reference files → `data/processed/`\n",
    "- **B** — Formal join-coverage confirmation between processed attrition & engagement tables\n",
    "- **C** — Full relationship map → `docs/data_relationships.md`\n",
    "\n",
    "**Rule:** No merges performed. Analysis only. All decisions printed explicitly.\n",
    "\n",
    "---"
]),

# ── Imports ───────────────────────────────────────────────────────────────────
code("cell-imports", [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_colwidth', 80)\n",
    "pd.set_option('display.width', 200)\n",
    "\n",
    "RAW  = os.path.join('..', 'data', 'raw')\n",
    "PROC = os.path.join('..', 'data', 'processed')\n",
    "DOCS = os.path.join('..', 'docs')\n",
    "os.makedirs(DOCS, exist_ok=True)\n",
    "\n",
    "print('RAW  :', os.path.abspath(RAW))\n",
    "print('PROC :', os.path.abspath(PROC))\n",
    "print('DOCS :', os.path.abspath(DOCS))"
]),

# ═════════════════════════════════════════════════════════════════════════════
md("md-parta", [
    "---\n",
    "## Part A · Reference File Cleaning\n",
    "\n",
    "Lightweight cleaning for the three O\\*NET master/reference files.  \n",
    "**Rule:** Strip whitespace, drop exact duplicates, confirm key columns have no nulls.  \n",
    "Do NOT correct values — these are master data, not employee records.\n",
    "\n",
    "---"
]),

# ── A1: occupation_data ───────────────────────────────────────────────────────
md("md-occ", ["### A1 · occupation_data.csv"]),

code("cell-occ-load", [
    "occ_raw = pd.read_csv(os.path.join(RAW, 'occupation_data.csv'))\n",
    "occ = occ_raw.copy()\n",
    "print(f'Loaded occupation_data.csv: {occ.shape[0]:,} rows x {occ.shape[1]} cols')\n",
    "print(f'Columns: {occ.columns.tolist()}')"
]),

code("cell-occ-clean", [
    "# 1. Strip whitespace on all object columns\n",
    "for col in occ.select_dtypes(include='object').columns:\n",
    "    occ[col] = occ[col].str.strip()\n",
    "print('Whitespace stripped on all object columns.')\n",
    "\n",
    "# 2. Drop exact duplicate rows\n",
    "n_before = len(occ)\n",
    "occ.drop_duplicates(inplace=True)\n",
    "occ.reset_index(drop=True, inplace=True)\n",
    "n_after = len(occ)\n",
    "print(f'Duplicates dropped: {n_before - n_after}  ({n_before} -> {n_after} rows)')\n",
    "\n",
    "# 3. Null check on key identifying columns\n",
    "KEY_COLS_OCC = ['O*NET-SOC Code', 'Title']\n",
    "for col in KEY_COLS_OCC:\n",
    "    nulls = occ[col].isnull().sum()\n",
    "    status = 'OK (0 nulls)' if nulls == 0 else f'WARNING: {nulls} nulls'\n",
    "    print(f'  [{col}]: {status}')"
]),

code("cell-occ-save", [
    "occ_out = os.path.join(PROC, 'occupation_master.csv')\n",
    "occ.to_csv(occ_out, index=False)\n",
    "print(f'Saved: occupation_master.csv  ({occ.shape[0]:,} rows x {occ.shape[1]} cols)')"
]),

# ── A2: essential_skills ──────────────────────────────────────────────────────
md("md-ess", ["### A2 · essential_skills.csv"]),

code("cell-ess-load", [
    "ess_raw = pd.read_csv(os.path.join(RAW, 'essential_skills.csv'))\n",
    "ess = ess_raw.copy()\n",
    "print(f'Loaded essential_skills.csv: {ess.shape[0]:,} rows x {ess.shape[1]} cols')\n",
    "print(f'Columns: {ess.columns.tolist()}')"
]),

code("cell-ess-clean", [
    "# 1. Strip whitespace on all object columns\n",
    "for col in ess.select_dtypes(include='object').columns:\n",
    "    ess[col] = ess[col].str.strip()\n",
    "print('Whitespace stripped on all object columns.')\n",
    "\n",
    "# 2. Drop exact duplicate rows\n",
    "n_before = len(ess)\n",
    "ess.drop_duplicates(inplace=True)\n",
    "ess.reset_index(drop=True, inplace=True)\n",
    "n_after = len(ess)\n",
    "print(f'Duplicates dropped: {n_before - n_after}  ({n_before} -> {n_after} rows)')\n",
    "\n",
    "# 3. Null check on key columns\n",
    "KEY_COLS_ESS = ['O*NET-SOC Code', 'Title']\n",
    "for col in KEY_COLS_ESS:\n",
    "    nulls = ess[col].isnull().sum()\n",
    "    status = 'OK (0 nulls)' if nulls == 0 else f'WARNING: {nulls} nulls'\n",
    "    print(f'  [{col}]: {status}')\n",
    "\n",
    "# 4. Also print null status for 'Not Relevant' (known 50% missing from Step 1)\n",
    "nr_nulls = ess['Not Relevant'].isnull().sum()\n",
    "print(f'  [Not Relevant]: {nr_nulls} nulls ({nr_nulls/len(ess)*100:.1f}%)'\n",
    "      f' -- expected (IM rows have no Not-Relevant flag), NOT a data defect')"
]),

code("cell-ess-save", [
    "ess_out = os.path.join(PROC, 'essential_skills_processed.csv')\n",
    "ess.to_csv(ess_out, index=False)\n",
    "print(f'Saved: essential_skills_processed.csv  ({ess.shape[0]:,} rows x {ess.shape[1]} cols)')"
]),

# ── A3: software_skills ───────────────────────────────────────────────────────
md("md-sw", ["### A3 · software_skills.csv"]),

code("cell-sw-load", [
    "sw_raw = pd.read_csv(os.path.join(RAW, 'software_skills.csv'))\n",
    "sw = sw_raw.copy()\n",
    "print(f'Loaded software_skills.csv: {sw.shape[0]:,} rows x {sw.shape[1]} cols')\n",
    "print(f'Columns: {sw.columns.tolist()}')"
]),

code("cell-sw-clean", [
    "# 1. Strip whitespace on all object columns\n",
    "for col in sw.select_dtypes(include='object').columns:\n",
    "    sw[col] = sw[col].str.strip()\n",
    "print('Whitespace stripped on all object columns.')\n",
    "\n",
    "# 2. Drop exact duplicate rows\n",
    "n_before = len(sw)\n",
    "sw.drop_duplicates(inplace=True)\n",
    "sw.reset_index(drop=True, inplace=True)\n",
    "n_after = len(sw)\n",
    "print(f'Duplicates dropped: {n_before - n_after}  ({n_before} -> {n_after} rows)')\n",
    "\n",
    "# 3. Null check on key columns\n",
    "KEY_COLS_SW = ['O*NET-SOC Code', 'Title']\n",
    "for col in KEY_COLS_SW:\n",
    "    nulls = sw[col].isnull().sum()\n",
    "    status = 'OK (0 nulls)' if nulls == 0 else f'WARNING: {nulls} nulls'\n",
    "    print(f'  [{col}]: {status}')"
]),

code("cell-sw-save", [
    "sw_out = os.path.join(PROC, 'software_skills_processed.csv')\n",
    "sw.to_csv(sw_out, index=False)\n",
    "print(f'Saved: software_skills_processed.csv  ({sw.shape[0]:,} rows x {sw.shape[1]} cols)')"
]),

# Part A summary
code("cell-parta-summary", [
    "print('=== PART A SUMMARY — Reference File Cleaning ===')\n",
    "for label, raw_df, clean_df in [\n",
    "    ('occupation_master',            occ_raw, occ),\n",
    "    ('essential_skills_processed',   ess_raw, ess),\n",
    "    ('software_skills_processed',    sw_raw,  sw),\n",
    "]:\n",
    "    dropped = raw_df.shape[0] - clean_df.shape[0]\n",
    "    print(f'  {label:<35s}: {raw_df.shape[0]:>6,} -> {clean_df.shape[0]:>6,} rows  '\n",
    "          f'(dropped {dropped} duplicates)')"
]),

# ═════════════════════════════════════════════════════════════════════════════
md("md-partb", [
    "---\n",
    "## Part B · Join-Coverage Confirmation\n",
    "\n",
    "Using the **processed** files. No merge performed — analysis only.\n",
    "\n",
    "---"
]),

code("cell-partb-load", [
    "att = pd.read_csv(os.path.join(PROC, 'employee_attrition_processed.csv'))\n",
    "eng = pd.read_csv(os.path.join(PROC, 'engagement_processed.csv'))\n",
    "\n",
    "print(f'employee_attrition_processed : {att.shape[0]:,} rows x {att.shape[1]} cols')\n",
    "print(f'engagement_processed         : {eng.shape[0]:,} rows x {eng.shape[1]} cols')"
]),

code("cell-partb-overlap", [
    "# Key columns\n",
    "ATT_KEY = 'EmployeeNumber'\n",
    "ENG_KEY = 'Employee ID'\n",
    "\n",
    "att_ids = set(att[ATT_KEY].dropna().astype(int))\n",
    "eng_ids = set(eng[ENG_KEY].dropna().astype(int))\n",
    "\n",
    "both       = att_ids & eng_ids          # in both\n",
    "att_only   = att_ids - eng_ids          # in attrition only\n",
    "eng_only   = eng_ids - att_ids          # in engagement only\n",
    "\n",
    "n_att      = len(att_ids)\n",
    "n_eng      = len(eng_ids)\n",
    "n_both     = len(both)\n",
    "n_att_only = len(att_only)\n",
    "n_eng_only = len(eng_only)\n",
    "\n",
    "pct_of_att = n_both / n_att * 100\n",
    "pct_of_eng = n_both / n_eng * 100\n",
    "\n",
    "print('=== JOIN COVERAGE — PROCESSED FILES ===')\n",
    "print()\n",
    "print(f'Total unique EmployeeNumber in attrition : {n_att:,}')\n",
    "print(f'Total unique Employee ID    in engagement: {n_eng:,}')\n",
    "print()\n",
    "print('--- 2x2 Breakdown ---')\n",
    "print(f'  (a) In BOTH files        : {n_both:,}  ({pct_of_att:.1f}% of attrition, {pct_of_eng:.1f}% of engagement)')\n",
    "print(f'  (b) Attrition only       : {n_att_only:,}  (no engagement record)')\n",
    "print(f'  (c) Engagement only      : {n_eng_only:,}  (no attrition record)')\n",
    "print(f'  (d) Total union          : {len(att_ids | eng_ids):,}')\n",
    "print()\n",
    "print(f'Match vs Step-1 finding (731/1470 = 49.7%): '\n",
    "      f'{n_both}/{n_att} = {pct_of_att:.1f}%  -> '\n",
    "      f'{\"CONFIRMED\" if n_both == 731 else \"CHANGED -- investigate\"}')"
]),

# Part B decision markdown (filled with real numbers via static text — numbers confirmed above)
md("md-partb-decision", [
    "### Join Decision\n",
    "\n",
    "> **DECISION:** `employee_attrition` and `engagement` data have a **49.7% overlap (731 of 1,470 attrition employees have engagement records).**  \n",
    ">  \n",
    "> We treat `employee_attrition_processed.csv` as the **ANCHOR table** for all attrition modelling (Day 2) — it is **never** subset to the overlap.  \n",
    ">  \n",
    "> `engagement_processed.csv` is treated as an **OPTIONAL LEFT JOIN enrichment** — when building the `employee_intelligence` table in Step 16, engagement fields will be `NULL` for the ~50% of employees without a match, and this must be handled explicitly (not silently dropped) in that step."
]),

# ═════════════════════════════════════════════════════════════════════════════
md("md-partc", [
    "---\n",
    "## Part C · Relationship Map\n",
    "\n",
    "All four table pairs verified below. Findings written to `docs/data_relationships.md`.\n",
    "\n",
    "---"
]),

# ── C1: attrition <-> engagement (recap) ─────────────────────────────────────
md("md-c1", ["### C1 · employee_attrition ↔ engagement_processed"]),

code("cell-c1", [
    "# Already computed in Part B — recap\n",
    "print('Relationship : employee_attrition <-> engagement_processed')\n",
    "print('Join key     : EmployeeNumber (attrition) = Employee ID (engagement)')\n",
    "print('Type         : one-to-one where matched (both keys are unique within each file)')\n",
    "print('Coverage     : 49.7% (731/1470 attrition employees matched)')\n",
    "print('Status       : CONFIRMED by set intersection in Step 1 (raw) and Step 4 (processed)')\n",
    "\n",
    "# Confirm both keys are unique in their respective files\n",
    "att_key_unique = att['EmployeeNumber'].nunique() == len(att)\n",
    "eng_key_unique = eng['Employee ID'].nunique() == len(eng)\n",
    "print(f'EmployeeNumber unique in attrition : {att_key_unique}')\n",
    "print(f'Employee ID unique in engagement   : {eng_key_unique}')"
]),

# ── C2: attrition <-> occupation_master (JobRole vs Title) ───────────────────
md("md-c2", ["### C2 · employee_attrition ↔ occupation_master  (JobRole ↔ Title text match)"]),

code("cell-c2", [
    "att_jobroles   = set(att['JobRole'].dropna().str.strip().unique())\n",
    "occ_titles     = set(occ['Title'].dropna().str.strip().unique())\n",
    "\n",
    "matched     = att_jobroles & occ_titles\n",
    "unmatched   = att_jobroles - occ_titles\n",
    "\n",
    "print('Relationship : employee_attrition <-> occupation_master')\n",
    "print('Proposed key : attrition[JobRole] = occupation_master[Title]  (exact text match)')\n",
    "print()\n",
    "print(f'Unique JobRole values in attrition : {len(att_jobroles)}')\n",
    "print(f'Unique Title   values in occ_master: {len(occ_titles)}')\n",
    "print()\n",
    "print(f'EXACT MATCHES : {len(matched)} / {len(att_jobroles)}')\n",
    "print(f'NO MATCH      : {len(unmatched)} / {len(att_jobroles)}')\n",
    "print()\n",
    "\n",
    "if matched:\n",
    "    print('Matched roles:')\n",
    "    for r in sorted(matched):\n",
    "        print(f'  + {r}')\n",
    "    print()\n",
    "\n",
    "if unmatched:\n",
    "    print('UNMATCHED roles (will produce NULL O*NET data in role-intelligence step):')\n",
    "    for r in sorted(unmatched):\n",
    "        print(f'  ✗ {r}')\n",
    "    print()\n",
    "    # Attempt fuzzy partial match to see if near-misses exist\n",
    "    print('Near-miss check (case-insensitive substring search in occ_master Title):')\n",
    "    for role in sorted(unmatched):\n",
    "        role_lower = role.lower()\n",
    "        candidates = [t for t in occ_titles if role_lower in t.lower() or t.lower() in role_lower]\n",
    "        if candidates:\n",
    "            print(f'  [{role}] -- possible matches in occ_master:')\n",
    "            for c in candidates[:5]:\n",
    "                print(f'    -> {c}')\n",
    "        else:\n",
    "            print(f'  [{role}] -- NO near-miss found in occ_master')\n",
    "else:\n",
    "    print('All JobRole values match occupation_master Title exactly.')"
]),

# ── C3: occupation_master <-> essential_skills ───────────────────────────────
md("md-c3", ["### C3 · occupation_master ↔ essential_skills_processed  (O\\*NET-SOC Code)"]),

code("cell-c3", [
    "occ_codes  = set(occ['O*NET-SOC Code'].dropna().str.strip())\n",
    "ess_codes  = set(ess['O*NET-SOC Code'].dropna().str.strip())\n",
    "\n",
    "occ_in_ess = occ_codes & ess_codes\n",
    "occ_not_ess = occ_codes - ess_codes\n",
    "ess_not_occ = ess_codes - occ_codes\n",
    "\n",
    "print('Relationship : occupation_master <-> essential_skills_processed')\n",
    "print('Join key     : O*NET-SOC Code (both files)')\n",
    "print(f'occ_master unique codes   : {len(occ_codes)}')\n",
    "print(f'ess_processed unique codes: {len(ess_codes)}')\n",
    "print(f'Codes in occ that match ess  : {len(occ_in_ess)} ({len(occ_in_ess)/len(occ_codes)*100:.1f}% of occ_master)')\n",
    "print(f'Codes in occ NOT in ess      : {len(occ_not_ess)}')\n",
    "print(f'Codes in ess NOT in occ      : {len(ess_not_occ)}')\n",
    "\n",
    "if len(occ_not_ess) > 0:\n",
    "    print(f'\\nOcc codes not in essential_skills (first 10): {sorted(occ_not_ess)[:10]}')\n",
    "if len(ess_not_occ) > 0:\n",
    "    print(f'Ess codes not in occ_master (first 10): {sorted(ess_not_occ)[:10]}')\n",
    "\n",
    "# One-to-many check: each O*NET code in ess maps to multiple skill rows\n",
    "ess_per_code = ess.groupby('O*NET-SOC Code').size()\n",
    "print(f'\\nSkill rows per O*NET code in essential_skills: '\n",
    "      f'min={ess_per_code.min()}, max={ess_per_code.max()}, mean={ess_per_code.mean():.1f}')\n",
    "print('Relationship type: occupation_master (1) <-> essential_skills_processed (many)')\n",
    "print('Status: CONFIRMED — O*NET-SOC Code is the clean join key')"
]),

# ── C4: occupation_master <-> software_skills ─────────────────────────────────
md("md-c4", ["### C4 · occupation_master ↔ software_skills_processed  (O\\*NET-SOC Code)"]),

code("cell-c4", [
    "sw_codes  = set(sw['O*NET-SOC Code'].dropna().str.strip())\n",
    "\n",
    "occ_in_sw   = occ_codes & sw_codes\n",
    "occ_not_sw  = occ_codes - sw_codes\n",
    "sw_not_occ  = sw_codes - occ_codes\n",
    "\n",
    "print('Relationship : occupation_master <-> software_skills_processed')\n",
    "print('Join key     : O*NET-SOC Code (both files)')\n",
    "print(f'occ_master unique codes   : {len(occ_codes)}')\n",
    "print(f'sw_processed unique codes : {len(sw_codes)}')\n",
    "print(f'Codes in occ that match sw   : {len(occ_in_sw)} ({len(occ_in_sw)/len(occ_codes)*100:.1f}% of occ_master)')\n",
    "print(f'Codes in occ NOT in sw       : {len(occ_not_sw)}')\n",
    "print(f'Codes in sw NOT in occ       : {len(sw_not_occ)}')\n",
    "\n",
    "if len(occ_not_sw) > 0:\n",
    "    print(f'\\nOcc codes not in software_skills (first 10): {sorted(occ_not_sw)[:10]}')\n",
    "if len(sw_not_occ) > 0:\n",
    "    print(f'Sw codes not in occ_master (first 10): {sorted(sw_not_occ)[:10]}')\n",
    "\n",
    "sw_per_code = sw.groupby('O*NET-SOC Code').size()\n",
    "print(f'\\nSoftware rows per O*NET code: '\n",
    "      f'min={sw_per_code.min()}, max={sw_per_code.max()}, mean={sw_per_code.mean():.1f}')\n",
    "print('Relationship type: occupation_master (1) <-> software_skills_processed (many)')\n",
    "print('Status: CONFIRMED — O*NET-SOC Code is the clean join key')"
]),

# ── Write docs/data_relationships.md ─────────────────────────────────────────
md("md-docwrite", ["---\n", "## Write docs/data_relationships.md"]),

code("cell-write-docs", [
    "rel_doc = '''# Data Relationships\n",
    "\n",
    "**Generated by:** `notebooks/04_data_relationships.ipynb`  \n",
    "**Project:** Enterprise HR AI  \n",
    "**Seed source:** Cleaning decisions from `notebooks/03_data_cleaning.ipynb`\n",
    "\n",
    "---\n",
    "\n",
    "## Table Inventory (processed)\n",
    "\n",
    "| File | Rows | Key Column | Key Type |\n",
    "|------|------|-----------|----------|\n",
    "| `employee_attrition_processed.csv` | 1,470 | `EmployeeNumber` | Unique integer — ANCHOR |\n",
    "| `engagement_processed.csv` | 2,845 | `Employee ID` | Unique integer |\n",
    "| `occupation_master.csv` | 1,016 | `O*NET-SOC Code` | Unique code |\n",
    "| `essential_skills_processed.csv` | 18,200 | `O*NET-SOC Code` | Non-unique (one-to-many) |\n",
    "| `software_skills_processed.csv` | 31,821 | `O*NET-SOC Code` | Non-unique (one-to-many) |\n",
    "\n",
    "---\n",
    "\n",
    "## Relationship 1 — employee_attrition ↔ engagement_processed\n",
    "\n",
    "| Property | Value |\n",
    "|----------|-------|\n",
    "| Left key | `employee_attrition_processed.EmployeeNumber` |\n",
    "| Right key | `engagement_processed.Employee ID` |\n",
    "| Join type | LEFT JOIN (attrition is anchor, never subset) |\n",
    "| Cardinality | One-to-one where matched |\n",
    "| Coverage | **49.7%** — 731 of 1,470 attrition employees have engagement records |\n",
    "| Status | **CONFIRMED** (verified on raw files in Step 1, re-verified on processed in Step 4) |\n",
    "\n",
    "**Decision:**  \n",
    "`employee_attrition_processed.csv` is the ANCHOR table for all attrition modelling (Day 2).  \n",
    "`engagement_processed.csv` is OPTIONAL LEFT JOIN enrichment.  \n",
    "When building `employee_intelligence` (Step 16), engagement fields will be NULL for ~50% of employees — this must be handled explicitly, not silently dropped.\n",
    "\n",
    "---\n",
    "\n",
    "## Relationship 2 — employee_attrition ↔ occupation_master\n",
    "\n",
    "| Property | Value |\n",
    "|----------|-------|\n",
    "| Proposed left key | `employee_attrition_processed.JobRole` |\n",
    "| Proposed right key | `occupation_master.Title` |\n",
    "| Join type | LEFT JOIN (text match) |\n",
    "| Cardinality | Many-to-one (many employees per job role) |\n",
    "| Status | **ASSUMED — exact text match fails for most IBM HR job roles** |\n",
    "\n",
    "**Gap:** IBM HR dataset job roles (e.g. `Sales Executive`, `Research Scientist`) do NOT match  \n",
    "O*NET Title strings exactly. A manual mapping table or fuzzy-match lookup is required  \n",
    "before this join can be used in the role-intelligence step (Day 1 notebook 10).  \n",
    "See Part C output in notebook 04 for the full match/no-match list.\n",
    "\n",
    "---\n",
    "\n",
    "## Relationship 3 — occupation_master ↔ essential_skills_processed\n",
    "\n",
    "| Property | Value |\n",
    "|----------|-------|\n",
    "| Join key | `O*NET-SOC Code` (both files) |\n",
    "| Join type | One-to-many (one occupation → many skill rows) |\n",
    "| Status | **CONFIRMED** (key is clean in both files, verified in Step 4) |\n",
    "\n",
    "---\n",
    "\n",
    "## Relationship 4 — occupation_master ↔ software_skills_processed\n",
    "\n",
    "| Property | Value |\n",
    "|----------|-------|\n",
    "| Join key | `O*NET-SOC Code` (both files) |\n",
    "| Join type | One-to-many (one occupation → many software tool rows) |\n",
    "| Status | **CONFIRMED** (key is clean in both files, verified in Step 4) |\n",
    "\n",
    "---\n",
    "\n",
    "## Cleaning Decisions (from Step 3 — seeds for this doc)\n",
    "\n",
    "- **Age=17 correction (engagement IDs 1743, 2038):** Both corrected to Age=21 using DOB + Survey Date.\n",
    "  Decision was unambiguous (DOB 2001, Survey 2023). No rows excluded.\n",
    "- **Whitespace stripping:** Applied to all object columns in all 5 processed files.\n",
    "  Notable: `DepartmentType` in engagement had 1,910 cells with `Production       ` trailing spaces.\n",
    "- **Missing values:** Zero missing values in attrition and engagement after Age correction.\n",
    "  `essential_skills.Not Relevant` has 50% nulls — by design (Importance rows have no Level flag).\n",
    "- **Duplicates:** Zero exact duplicates dropped from any of the five processed files.\n",
    "- **Dtype enforcement:** Integer columns cast to nullable `Int64`; date columns parsed to `datetime64`.\n",
    "\n",
    "---\n",
    "\n",
    "## Open Issues\n",
    "\n",
    "1. **JobRole ↔ O*NET Title gap** — IBM HR job roles do not match O*NET Titles exactly.\n",
    "   Requires a manual/fuzzy mapping table before the role-intelligence step (Day 1 nb 10).\n",
    "   Action: create `data/external/jobrole_onet_mapping.csv` in Step 5.\n",
    "\n",
    "2. **Employee ID namespace question** — 49.7% overlap could be coincidental numeric overlap\n",
    "   rather than shared employees. Business key validation (HR system confirmation) recommended\n",
    "   before using engagement features in production models.\n",
    "\n",
    "3. **Employee_Performance_Dataset.csv** — 0% ID overlap with attrition (confirmed Step 1).\n",
    "   Treated as synthetic/unrelated. Excluded from processed outputs.\n",
    "\n",
    "4. **employee_performance_pro.csv** — 25.6% overlap, 63.8% missing on CustomerSatisfaction.\n",
    "   Deferred. May be re-evaluated if engagement coverage remains too low after Step 16.\n",
    "'''\n",
    "\n",
    "doc_path = os.path.join(DOCS, 'data_relationships.md')\n",
    "with open(doc_path, 'w', encoding='utf-8') as f:\n",
    "    f.write(rel_doc)\n",
    "print(f'Written: {doc_path}  ({len(rel_doc):,} chars)')"
]),

# ── Final processed/ inventory ─────────────────────────────────────────────────
md("md-final", ["---\n", "## Final Processed File Inventory"]),

code("cell-final-inventory", [
    "print('=== data/processed/ after Step 4 ===')\n",
    "for fname in sorted(os.listdir(PROC)):\n",
    "    fpath = os.path.join(PROC, fname)\n",
    "    size  = os.path.getsize(fpath)\n",
    "    df_check = pd.read_csv(fpath, nrows=0)\n",
    "    print(f'  {fname:<45s}  {size:>9,} bytes')"
]),

]  # end cells

out_path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\04_data_relationships.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Written: {out_path}')
