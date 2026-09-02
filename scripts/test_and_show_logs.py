"""
Script to execute the 3 required requests and display logs/app.log.
"""

import sys
import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Clean previous app.log so we capture clean run
log_file = PROJECT_ROOT / "logs" / "app.log"
if log_file.exists():
    log_file.unlink()

from fastapi.testclient import TestClient
import pandas as pd
from app.main import app

raw_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "employee_attrition_processed.csv")
emp1_payload = raw_df.iloc[0].to_dict()

print("Executing requests via TestClient with lifespan context...")
with TestClient(app) as client:
    # 1. POST /predict/attrition
    print("\n--- Request 1: POST /predict/attrition ---")
    r1 = client.post("/predict/attrition", json=emp1_payload)
    print(f"Status: {r1.status_code}")
    print(f"Response: {r1.json()}")
    
    # 2. GET /employees/1
    print("\n--- Request 2: GET /employees/1 ---")
    r2 = client.get("/employees/1")
    print(f"Status: {r2.status_code}")
    print(f"Response (summary): EmployeeNumber={r2.json().get('EmployeeNumber')}, Role={r2.json().get('JobRole')}, Risk={r2.json().get('RiskLevel')}")
    
    # 3. GET /employees/999999
    print("\n--- Request 3: GET /employees/999999 ---")
    r3 = client.get("/employees/999999")
    print(f"Status: {r3.status_code}")
    print(f"Response: {r3.json()}")

print("\n" + "=" * 85)
print("ACTUAL CONTENTS OF logs/app.log:")
print("=" * 85)
if log_file.exists():
    with open(log_file, "r", encoding="utf-8") as f:
        print(f.read())
else:
    print("ERROR: logs/app.log not found!")
print("=" * 85)
