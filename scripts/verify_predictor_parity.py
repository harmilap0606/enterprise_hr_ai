"""
Standalone Predictor Parity Verification Script.
Validates that app.ml.predictor.predict_attrition_risk produces exact parity with
the notebook-generated RiskScores stored in data/processed/employee_intelligence.csv.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so app modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from app.ml.predictor import predict_attrition_risk, preprocess_features
from app.utils.config import EMPLOYEE_INTELLIGENCE_PATH, EMPLOYEE_ATTRITION_PATH


def run_parity_check():
    print("=" * 85)
    print("PREDICTOR PARITY SANITY CHECK: Notebook vs API Pipeline")
    print("=" * 85)

    if not EMPLOYEE_INTELLIGENCE_PATH.exists():
        raise FileNotFoundError(f"Missing employee intelligence file: {EMPLOYEE_INTELLIGENCE_PATH}")
    if not EMPLOYEE_ATTRITION_PATH.exists():
        raise FileNotFoundError(f"Missing employee attrition file: {EMPLOYEE_ATTRITION_PATH}")

    # 1. Load data sources
    df_intel = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH, comment="#")
    df_raw = pd.read_csv(EMPLOYEE_ATTRITION_PATH)

    print(f"Loaded employee_intelligence.csv : {len(df_intel):,} rows")
    print(f"Loaded employee_attrition_processed.csv : {len(df_raw):,} rows")
    print()

    # 2. Sample 10 random employees spanning a mix of HIGH and LOW risk (at least 3 of each)
    high_pool = df_intel[df_intel["RiskLevel"] == "HIGH"]
    low_pool = df_intel[df_intel["RiskLevel"] == "LOW"]

    # Pick 5 HIGH and 5 LOW with fixed seed for 100% reproducibility
    sample_high = high_pool.sample(n=5, random_state=42)
    sample_low = low_pool.sample(n=5, random_state=42)
    sample_10 = (
        pd.concat([sample_high, sample_low])
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    print(f"Selected 10 sample employees ({len(sample_high)} HIGH risk, {len(sample_low)} LOW risk):")
    print(f"IDs: {sample_10['EmployeeNumber'].tolist()}")
    print()

    # 3. Evaluate each employee
    results = []
    first_failure = None

    for _, row in sample_10.iterrows():
        emp_id = int(row["EmployeeNumber"])
        nb_score = float(row["RiskScore"])
        nb_level = row["RiskLevel"]

        # Fetch raw employee record
        raw_match = df_raw[df_raw["EmployeeNumber"] == emp_id]
        if raw_match.empty:
            raise KeyError(f"Employee #{emp_id} not found in raw attrition dataset!")
        raw_dict = raw_match.iloc[0].to_dict()

        # Run through API predictor
        api_res = predict_attrition_risk(raw_dict)
        api_prob = float(api_res["probability"])
        diff = abs(nb_score - api_prob)
        is_match = diff < 0.001

        results.append({
            "EmployeeNumber": emp_id,
            "JobRole": row["JobRole"],
            "RiskLevel": nb_level,
            "Notebook RiskScore": round(nb_score, 4),
            "API-Code Probability": round(api_prob, 4),
            "Absolute Difference": round(diff, 6),
            "Match": is_match
        })

        if not is_match and first_failure is None:
            first_failure = (emp_id, raw_dict)

    df_results = pd.DataFrame(results)

    # 4. Print comparison table
    print("=" * 85)
    print("PARITY COMPARISON TABLE")
    print("=" * 85)
    
    header = (
        f"{'EmployeeNumber':<16} | {'Notebook Risk':<15} | "
        f"{'API Probability':<15} | {'Abs Difference':<16} | {'Match':<6}"
    )
    print(header)
    print("-" * 85)
    for _, r in df_results.iterrows():
        status = "True" if r["Match"] else "FALSE"
        print(
            f"{r['EmployeeNumber']:<16} | {r['Notebook RiskScore']:<15.4f} | "
            f"{r['API-Code Probability']:<15.4f} | {r['Absolute Difference']:<16.6f} | {status:<6}"
        )
    print("=" * 85)

    all_passed = df_results["Match"].all()
    print(f"\nAll 10 employees passed parity check (diff < 0.001): {all_passed}")

    # 5. Handle failure diagnostics if any
    if not all_passed:
        fail_id, fail_dict = first_failure
        print("\n" + "!" * 85)
        print(f"PARITY FAILURE DETECTED ON EMPLOYEE #{fail_id}")
        print("!" * 85)
        print("\nRaw Input Features:")
        for k, v in fail_dict.items():
            print(f"  {k}: {v}")
        print("\nIntermediate Preprocessed Feature Matrix:")
        X_fail = preprocess_features(fail_dict)
        print(X_fail.to_string())
        sys.exit(1)

    print("\nSUCCESS: Complete statistical parity confirmed between API predictor and notebook pipeline!")
    return df_results


if __name__ == "__main__":
    run_parity_check()
