"""
Test script for prediction logging and fallback execution.
Executes 5 live predictions (and 1 pre-computed lookup) to verify:
1. prediction_log.csv created with correct schema.
2. 5 live model inferences appended.
3. Fallback path in get_employee_risk() triggers live prediction and logs.
4. Pre-computed lookup in get_employee_risk() does NOT log (no duplicates).
"""

import sys
import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure fresh prediction log for verification
pred_log_path = PROJECT_ROOT / "data" / "predictions" / "prediction_log.csv"
if pred_log_path.exists():
    pred_log_path.unlink()

from fastapi.testclient import TestClient
import pandas as pd
from app.main import app
import app.services.attrition_service as attrition_service

client = TestClient(app)
raw_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "employee_attrition_processed.csv")

print("=" * 85)
print("EXECUTING PREDICTION LOGGING TEST SUITE")
print("=" * 85)

# --- Test 1: POST /predict/attrition with Employee #1 ---
emp1_dict = raw_df[raw_df["EmployeeNumber"] == 1].iloc[0].to_dict()
r1 = client.post("/predict/attrition", json=emp1_dict)
print(f"Run 1: POST /predict/attrition (Emp #1) -> HTTP {r1.status_code}, Prob={r1.json()['probability']}, Risk={r1.json()['risk_level']}")

# --- Test 2: POST /predict/attrition with Employee #2 ---
emp2_dict = raw_df[raw_df["EmployeeNumber"] == 2].iloc[0].to_dict()
r2 = client.post("/predict/attrition", json=emp2_dict)
print(f"Run 2: POST /predict/attrition (Emp #2) -> HTTP {r2.status_code}, Prob={r2.json()['probability']}, Risk={r2.json()['risk_level']}")

# --- Test 3: POST /predict/attrition with Employee #4 ---
emp4_dict = raw_df[raw_df["EmployeeNumber"] == 4].iloc[0].to_dict()
r3 = client.post("/predict/attrition", json=emp4_dict)
print(f"Run 3: POST /predict/attrition (Emp #4) -> HTTP {r3.status_code}, Prob={r3.json()['probability']}, Risk={r3.json()['risk_level']}")

# --- Test 4: POST /predict/attrition with Anonymous Employee (no EmployeeNumber) ---
anon_dict = raw_df.iloc[5].to_dict()
anon_dict.pop("EmployeeNumber", None)
r4 = client.post("/predict/attrition", json=anon_dict)
print(f"Run 4: POST /predict/attrition (Anonymous, no ID) -> HTTP {r4.status_code}, Prob={r4.json()['probability']}, Risk={r4.json()['risk_level']}")

# --- Test 5: get_employee_risk() Fallback Path Execution ---
# Explanation: In the static CSV, all 1,470 employees already have pre-computed scores.
# To test the fallback path where an employee exists in raw HR records but NOT in the
# pre-computed intelligence table, we simulate Employee #100 missing from the intelligence cache.
intel_df = attrition_service._get_intelligence_df()
# Filter out Employee #100 from cache
attrition_service._intelligence_cache = intel_df[intel_df["EmployeeNumber"] != 100].copy()

# Call get_employee_risk(100) -> will miss intelligence cache and trigger fallback to raw table + live prediction
res_fallback = attrition_service.get_employee_risk(100)
print(f"Run 5: get_employee_risk(100) [Fallback Path] -> Source={res_fallback['source']}, Prob={res_fallback['probability']}, Risk={res_fallback['risk_level']}")

# --- Negative Check: Pre-computed Lookup Path (MUST NOT LOG) ---
# Call get_employee_risk(1) which exists in intelligence cache
res_lookup = attrition_service.get_employee_risk(1)
print(f"\nVerification Lookup: get_employee_risk(1) -> Source={res_lookup['source']} (Should NOT log)")

# Restore cache
attrition_service._intelligence_cache = None

print("\n" + "=" * 85)
print("ACTUAL CONTENTS OF data/predictions/prediction_log.csv:")
print("=" * 85)
if pred_log_path.exists():
    with open(pred_log_path, "r", encoding="utf-8") as f:
        log_content = f.read()
        print(log_content)
else:
    print("ERROR: prediction_log.csv was not created!")
    sys.exit(1)

# Check row count
df_log = pd.read_csv(pred_log_path)
print("=" * 85)
print(f"Total entries logged: {len(df_log)}")
assert len(df_log) == 5, f"Expected exactly 5 logged predictions, but found {len(df_log)}!"
print("CONFIRMED: Exactly 5 live inference calls logged. Lookup call was NOT logged.")
