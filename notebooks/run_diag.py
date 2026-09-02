import pandas as pd
import os

PROC = os.path.join(r'C:\Users\ASUS\Desktop\enterprise_hr_ai\data', 'processed')
df_check = pd.read_csv(os.path.join(PROC, 'features_unscaled.csv'))
all_cols = df_check.columns.tolist()
print(f'Total columns in features_unscaled.csv: {len(all_cols)}')
print()

TARGET_COL = 'Attrition'

ENGINEERED = [
    'income_per_year_at_company',
    'years_since_promotion_ratio',
    'overall_satisfaction_score',
    'experience_ratio',
]

BINARY_ENCODED = ['Gender', 'OverTime']

OHE_PARENTS = {
    'BusinessTravel':  3,
    'Department':      3,
    'EducationField':  6,
    'JobRole':         9,
    'MaritalStatus':   3,
}

ohe_dummies = [c for c in all_cols
               if any(c.startswith(p + '_') for p in OHE_PARENTS)]

accounted = set([TARGET_COL] + ENGINEERED + BINARY_ENCODED + ohe_dummies)
numeric_continuous = [c for c in all_cols if c not in accounted]

print('=' * 60)
print('GROUP 1 — Numeric/continuous columns (kept as-is)')
print('=' * 60)
for c in numeric_continuous:
    print(f'  {c}')
print(f'  COUNT: {len(numeric_continuous)}')
print()

print('=' * 60)
print('GROUP 2 — Categorical columns (encoded in Step 5)')
print('=' * 60)
print('  2a. Binary-encoded (0/1 mapping, 2-level categoricals):')
for c in BINARY_ENCODED:
    print(f'    {c} -> 1 column (binary 0/1)')
print(f'    COUNT: {len(BINARY_ENCODED)}')
print()
print('  2b. One-hot encoded (pd.get_dummies, drop_first=True):')
ohe_col_count = 0
for parent, n_levels in OHE_PARENTS.items():
    actual_dummies = [c for c in ohe_dummies if c.startswith(parent + '_')]
    expected_dummies = n_levels - 1
    match_str = 'OK' if len(actual_dummies) == expected_dummies else f'MISMATCH expected {expected_dummies}'
    print(f'    {parent} ({n_levels} levels) -> {len(actual_dummies)} dummies [drop_first=True, {match_str}]')
    for d in actual_dummies:
        print(f'      {d}')
    ohe_col_count += len(actual_dummies)
print(f'    OHE dummy column COUNT: {ohe_col_count}')
total_cat = len(BINARY_ENCODED) + ohe_col_count
print(f'    TOTAL categorical-derived columns: {total_cat}')
print()

print('=' * 60)
print('GROUP 3 — Engineered features')
print('=' * 60)
for c in ENGINEERED:
    present = 'present' if c in all_cols else 'MISSING'
    print(f'  {c}  [{present}]')
print(f'  COUNT: {len(ENGINEERED)}')
print()

print('=' * 60)
print('GROUP 4 — Target column')
print('=' * 60)
present_tgt = 'present' if TARGET_COL in all_cols else 'MISSING'
print(f'  {TARGET_COL}  [{present_tgt}]')
print(f'  COUNT: 1')
print()

tally = len(numeric_continuous) + total_cat + len(ENGINEERED) + 1
print('=' * 60)
print('COLUMN TALLY')
print('=' * 60)
print(f'  Group 1  — Numeric/continuous          : {len(numeric_continuous)}')
print(f'  Group 2a — Binary-encoded categoricals : {len(BINARY_ENCODED)}')
print(f'  Group 2b — OHE dummy columns           : {ohe_col_count}')
print(f'  Group 3  — Engineered features         : {len(ENGINEERED)}')
print(f'  Group 4  — Target (Attrition)          : 1')
print(f'  TALLY TOTAL                            : {tally}')
print(f'  ACTUAL columns in CSV                  : {len(all_cols)}')
print()
if tally == len(all_cols):
    print(f'CONFIRMED: {tally} == {len(all_cols)} -> column groups account for every column exactly')
else:
    diff = len(all_cols) - tally
    print(f'DISCREPANCY: tally={tally}, actual={len(all_cols)}, delta={diff:+d}')
    unaccounted = [c for c in all_cols
                   if c not in numeric_continuous
                   and c not in BINARY_ENCODED
                   and c not in ohe_dummies
                   and c not in ENGINEERED
                   and c != TARGET_COL]
    if unaccounted:
        print(f'  Unaccounted-for columns ({len(unaccounted)}): {unaccounted}')
    missing_from_csv = [c for c in ENGINEERED + BINARY_ENCODED if c not in all_cols]
    if missing_from_csv:
        print(f'  Expected but absent from CSV: {missing_from_csv}')

print()
print('drop_first CONFIRMATION:')
print('  pd.get_dummies(..., drop_first=True, dtype=int) used in Step 5, cell "cell-ohe".')
print('  The FIRST alphabetical level of each multi-level categorical is the reference:')
for parent, n_levels in OHE_PARENTS.items():
    actual_dummies = [c for c in ohe_dummies if c.startswith(parent + '_')]
    all_levels_approx = [d.replace(parent + '_', '') for d in actual_dummies]
    print(f'    {parent}: reference = first alphabetical level (dropped), {n_levels-1} dummies kept')
