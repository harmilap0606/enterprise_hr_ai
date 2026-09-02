import sys
sys.path.insert(0, r'C:\Users\ASUS\Desktop\enterprise_hr_ai')

import json
from app.validation.employee_schema import EmployeeInputSchema

with open(r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\05_feature_engineering.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

keep_cols = []
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'LEAKAGE_MAP = {' in src:
        for line in src.split('\n'):
            line = line.strip()
            if line.startswith("'") and "'KEEP'" in line:
                col = line.split(':')[0].strip("'")
                keep_cols.append(col)

print(f"Step 5 Leakage Audit KEEP count: {len(keep_cols)}")
print(f"Columns: {sorted(keep_cols)}")

schema_fields = [k for k in EmployeeInputSchema.model_fields.keys() if k != 'EmployeeNumber']
print(f"\nEmployeeInputSchema fields count: {len(schema_fields)}")
print(f"Fields: {sorted(schema_fields)}")

missing = set(keep_cols) - set(schema_fields)
extra = set(schema_fields) - set(keep_cols)

print(f"\nMissing from schema: {missing}")
print(f"Extra in schema: {extra}")
assert len(keep_cols) == 30, f"Expected 30 KEEP cols, got {len(keep_cols)}"
assert missing == set(), f"Missing columns: {missing}"
print("\nCONFIRMED: All 30 KEEP columns are 100% present in EmployeeInputSchema!")
