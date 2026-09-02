"""
Automated Test Script for All FastAPI Endpoints.
Uses FastAPI TestClient to test:
1. POST /predict/attrition (valid payload)
2. GET  /dashboard/summary
3. GET  /dashboard/attrition-by-department
4. GET  /dashboard/skill-gaps
5. GET  /dashboard/recommendations?department=Sales&risk_level=HIGH
6. GET  /employees/{employee_id} (both regular employee and Manager)
7. GET  /skills/recommendations/{employee_id}
8. POST /predict/attrition with malformed/missing fields (validation error handling)
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
import pandas as pd
from app.main import app

client = TestClient(app)

def run_tests():
    print("=" * 85)
    print("RUNNING AUTOMATED FASTAPI ENDPOINTS TEST SUITE")
    print("=" * 85)
    
    # 0. Health / Root
    r0 = client.get("/")
    assert r0.status_code == 200, f"Root failed: {r0.status_code}"
    print("[PASS] GET / -> 200 OK")
    
    # 1. POST /predict/attrition (Valid Payload)
    raw_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "employee_attrition_processed.csv")
    emp1_dict = raw_df.iloc[0].to_dict()
    r1 = client.post("/predict/attrition", json=emp1_dict)
    assert r1.status_code == 200, f"POST /predict/attrition failed: {r1.text}"
    d1 = r1.json()
    assert d1["probability"] == 0.8976, f"Expected 0.8976, got {d1['probability']}"
    assert d1["risk_level"] == "HIGH"
    print(f"[PASS] 1. POST /predict/attrition (Valid) -> 200 OK (prob={d1['probability']}, level={d1['risk_level']})")
    
    # 2. GET /dashboard/summary
    r2 = client.get("/dashboard/summary")
    assert r2.status_code == 200, f"GET /dashboard/summary failed: {r2.text}"
    d2 = r2.json()
    assert d2["total_employees"] == 1470
    assert d2["high_risk_count"] == 585
    assert d2["average_engagement"] == 2.95, f"Expected 2.95, got {d2['average_engagement']}"
    assert "engagement_coverage_note" in d2
    print(f"[PASS] 2. GET /dashboard/summary -> 200 OK (total={d2['total_employees']}, high_risk={d2['high_risk_count']}, avg_eng={d2['average_engagement']})")
    print(f"       - Note: {d2['engagement_coverage_note']}")
    
    # 3. GET /dashboard/attrition-by-department
    r3 = client.get("/dashboard/attrition-by-department")
    assert r3.status_code == 200, f"GET /dashboard/attrition-by-department failed: {r3.text}"
    d3 = r3.json()
    assert len(d3) == 3, f"Expected 3 departments, got {len(d3)}"
    total_check = sum(x["total_employees"] for x in d3)
    high_check = sum(x["high_risk_count"] for x in d3)
    assert total_check == 1470
    assert high_check == 585
    print(f"[PASS] 3. GET /dashboard/attrition-by-department -> 200 OK (3 depts, {total_check} total, {high_check} high risk)")
    for dept in d3:
        print(f"       - {dept['department']}: {dept['high_risk_count']} high risk ({dept['high_risk_percentage']}%) of {dept['total_employees']}")
        
    # 4. GET /dashboard/skill-gaps
    r4 = client.get("/dashboard/skill-gaps")
    assert r4.status_code == 200, f"GET /dashboard/skill-gaps failed: {r4.text}"
    d4 = r4.json()
    assert len(d4) == 33, f"Expected 33 skill gaps, got {len(d4)}"
    print(f"[PASS] 4. GET /dashboard/skill-gaps -> 200 OK ({len(d4)} organizational skills ranked)")
    print(f"       - Top gap: {d4[0]['skill_name']} ({d4[0]['total_missing_count']} missing, {d4[0]['severity']})")
    
    # 5. GET /dashboard/recommendations?department=Sales&risk_level=HIGH
    r5 = client.get("/dashboard/recommendations?department=Sales&risk_level=HIGH")
    assert r5.status_code == 200, f"GET /dashboard/recommendations failed: {r5.text}"
    d5 = r5.json()
    assert len(d5) == 237, f"Expected 237 Sales HIGH risk employees, got {len(d5)}"
    assert all(x["Department"] == "Sales" and x["RiskLevel"] == "HIGH" for x in d5)
    print(f"[PASS] 5. GET /dashboard/recommendations?department=Sales&risk_level=HIGH -> 200 OK ({len(d5)} filtered records)")
    print(f"       - Sample: Emp #{d5[0]['EmployeeNumber']} ({d5[0]['JobRole']}) -> {d5[0]['Top3Recommendations'][:50]}...")
    
    # 6. GET /employees/{employee_id}
    r6 = client.get("/employees/1")
    assert r6.status_code == 200, f"GET /employees/1 failed: {r6.text}"
    d6 = r6.json()
    assert d6["EmployeeNumber"] == 1
    assert d6["JobRole"] == "Sales Executive"
    assert d6["RiskScore"] == 0.8976
    assert d6["RiskLevel"] == "HIGH"
    assert "ONET_Title" in d6
    assert "SkillGapSeverity" in d6
    print(f"[PASS] 6. GET /employees/1 -> 200 OK (Full 360 profile loaded for Emp #1)")
    
    # Test Manager on /employees/{id}
    r6_mgr = client.get("/employees/23")
    assert r6_mgr.status_code == 200
    d6_mgr = r6_mgr.json()
    assert d6_mgr["JobRole"] == "Manager"
    assert d6_mgr["SkillGapSeverity"] == "N/A - Manager"
    print(f"       - Emp #23 (Manager): SkillGapSeverity='{d6_mgr['SkillGapSeverity']}', ONET_Title='{d6_mgr['ONET_Title']}'")
    
    # Test 404 for invalid employee
    r6_404 = client.get("/employees/999999")
    assert r6_404.status_code == 404
    print(f"       - Invalid Emp #999999 -> 404 Not Found handled properly")
    
    # 7. GET /skills/recommendations/{employee_id}
    r7 = client.get("/skills/recommendations/1")
    assert r7.status_code == 200, f"GET /skills/recommendations/1 failed: {r7.text}"
    d7 = r7.json()
    assert d7["EmployeeNumber"] == 1
    assert "top_3_recommendations" in d7
    print(f"[PASS] 7. GET /skills/recommendations/1 -> 200 OK")
    print(f"       - Emp #1 Recs: {d7['top_3_recommendations'][:60]}...")
    
    # Test Manager on /skills/recommendations/{id}
    r7_mgr = client.get("/skills/recommendations/23")
    assert r7_mgr.status_code == 200
    d7_mgr = r7_mgr.json()
    assert "Manager" in d7_mgr["top_3_recommendations"]
    print(f"       - Emp #23 (Manager): {d7_mgr['top_3_recommendations']}")
    
    # 8. POST /predict/attrition with Malformed / Missing Fields (Validation Error Handling)
    print("\n" + "-" * 85)
    print("8. TESTING SCHEMA VALIDATION FAILURE ON MALFORMED / MISSING REQUESTS")
    print("-" * 85)
    
    # 8a: Missing required field MonthlyIncome
    bad_dict_missing = emp1_dict.copy()
    del bad_dict_missing["MonthlyIncome"]
    r8a = client.post("/predict/attrition", json=bad_dict_missing)
    print(f"[PASS] 8a. Missing 'MonthlyIncome': HTTP {r8a.status_code} returned (FastAPI default 422 Unprocessable Entity)")
    print(f"       Full Response Body:\n{json.dumps(r8a.json(), indent=2)}\n")
    assert r8a.status_code == 422, f"Expected 422, got {r8a.status_code}"
    err_loc_8a = r8a.json()["detail"][0]["loc"]
    assert "MonthlyIncome" in err_loc_8a
    
    # 8b: Wrong-typed field YearsAtCompany as string "five"
    bad_dict_wrong_type = emp1_dict.copy()
    bad_dict_wrong_type["YearsAtCompany"] = "five"
    r8b = client.post("/predict/attrition", json=bad_dict_wrong_type)
    print(f"[PASS] 8b. Wrong-typed 'YearsAtCompany=\"five\"': HTTP {r8b.status_code} returned (FastAPI default 422 Unprocessable Entity)")
    print(f"       Full Response Body:\n{json.dumps(r8b.json(), indent=2)}")
    assert r8b.status_code == 422, f"Expected 422, got {r8b.status_code}"
    err_loc_8b = r8b.json()["detail"][0]["loc"]
    assert "YearsAtCompany" in err_loc_8b
    
    print("\n" + "=" * 85)
    print("ALL TESTS (INCLUDING SCHEMA VALIDATION ERROR CASES) VERIFIED AND PASSING CLEANLY!")
    print("=" * 85)

if __name__ == "__main__":
    run_tests()
