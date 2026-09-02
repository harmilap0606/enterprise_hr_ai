"""
Attrition Service Module.
Handles workforce flight risk assessment, real-time employee scoring,
pre-computed risk lookups, departmental rollups, full intelligence profile retrieval,
and dedicated prediction inference auditing (data/predictions/prediction_log.csv).
"""

from typing import Dict, Any, List, Optional
import os
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from app.ml.predictor import predict_attrition_risk
from app.utils.config import (
    EMPLOYEE_INTELLIGENCE_PATH,
    EMPLOYEE_ATTRITION_PATH,
    PREDICTIONS_DIR,
    PREDICTION_LOG_PATH
)
from app.utils.logger import logger

_intelligence_cache: Optional[pd.DataFrame] = None
_attrition_cache: Optional[pd.DataFrame] = None

PREDICTION_LOG_COLUMNS = [
    "timestamp",
    "EmployeeNumber",
    "model_version",
    "probability",
    "risk_level",
    "threshold"
]


def _log_prediction_event(prediction_result: Dict[str, Any]) -> None:
    """
    Appends a live model inference record to data/predictions/prediction_log.csv.
    Initializes the file with headers if it does not exist.
    """
    try:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        file_exists = PREDICTION_LOG_PATH.exists()
        
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emp_num = prediction_result.get("EmployeeNumber")
        emp_val = str(emp_num) if emp_num is not None else ""
        
        row_data = {
            "timestamp": timestamp_str,
            "EmployeeNumber": emp_val,
            "model_version": prediction_result.get("model_version", "v3 (logistic_regression_balanced)"),
            "probability": prediction_result.get("probability"),
            "risk_level": prediction_result.get("risk_level"),
            "threshold": prediction_result.get("threshold", 0.40)
        }
        
        with open(PREDICTION_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PREDICTION_LOG_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
            
        logger.info(f"Appended live model inference to {PREDICTION_LOG_PATH.name} for employee: {emp_val or 'Anonymous'}")
    except Exception as e:
        logger.error(f"Failed to append to prediction_log.csv: {e}", exc_info=True)


def _get_intelligence_df() -> pd.DataFrame:
    """Loads and caches the capstone employee intelligence table."""
    global _intelligence_cache
    if _intelligence_cache is None:
        if not EMPLOYEE_INTELLIGENCE_PATH.exists():
            raise FileNotFoundError(f"Employee intelligence file not found at {EMPLOYEE_INTELLIGENCE_PATH}")
        _intelligence_cache = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH, comment="#")
        logger.info(f"Loaded {len(_intelligence_cache):,} records from {EMPLOYEE_INTELLIGENCE_PATH.name}")
    return _intelligence_cache


def _get_raw_attrition_df() -> pd.DataFrame:
    """Loads and caches raw processed attrition records for feature extraction by ID."""
    global _attrition_cache
    if _attrition_cache is None:
        if not EMPLOYEE_ATTRITION_PATH.exists():
            raise FileNotFoundError(f"Employee attrition file not found at {EMPLOYEE_ATTRITION_PATH}")
        _attrition_cache = pd.read_csv(EMPLOYEE_ATTRITION_PATH)
        logger.info(f"Loaded {len(_attrition_cache):,} records from {EMPLOYEE_ATTRITION_PATH.name}")
    return _attrition_cache


def predict_single_employee(employee_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes on-demand live model inference for an employee record using the trained pipeline.
    Appends an audit entry to data/predictions/prediction_log.csv.
    
    Args:
        employee_data: Dictionary containing raw employee feature fields.
        
    Returns:
        Dict with 'probability', 'risk_level', 'threshold', and 'model_version'.
    """
    emp_num = employee_data.get("EmployeeNumber")
    logger.info(f"Executing real-time attrition prediction for employee: {emp_num or 'Anonymous'}")
    
    result = predict_attrition_risk(employee_data)
    result["EmployeeNumber"] = emp_num
    result["model_version"] = "v3 (logistic_regression_balanced)"
    
    # Log every live model execution to data/predictions/prediction_log.csv
    _log_prediction_event(result)
    
    return result


def get_employee_risk(employee_id: int) -> Dict[str, Any]:
    """
    Retrieves stored attrition risk score for an existing employee by EmployeeNumber.
    
    =============================================================================
    CRITICAL ARCHITECTURAL DISTINCTION: LOOKUP VS. LIVE INFERENCE LOGGING
    -----------------------------------------------------------------------------
    1. Pre-computed match in employee_intelligence.csv:
       This is a cached database LOOKUP. No ML pipeline or model inference is run.
       Therefore, we DO NOT log an event to data/predictions/prediction_log.csv.
       This prevents double-counting and duplicate log entries.
       
    2. Fallback execution (employee not found in pre-computed intelligence cache):
       If an employee is missing from employee_intelligence.csv but present in raw HR
       records, this function falls back to predict_single_employee(), executing a
       true LIVE ML INFERENCE. In this case, predict_single_employee() automatically
       logs the live prediction to prediction_log.csv.
    =============================================================================
    """
    df_intel = _get_intelligence_df()
    match = df_intel[df_intel["EmployeeNumber"] == employee_id]
    
    if not match.empty:
        # LOOKUP PATH ONLY: Pre-computed intelligence table hit -> Return directly WITHOUT logging
        row = match.iloc[0]
        return {
            "EmployeeNumber": int(row["EmployeeNumber"]),
            "Department": row["Department"],
            "JobRole": row["JobRole"],
            "probability": float(row["RiskScore"]),
            "risk_level": row["RiskLevel"],
            "model_version": "v3 (logistic_regression_balanced)",
            "source": "precomputed_lookup"
        }
        
    # FALLBACK PATH: Live ML inference on raw HR features -> predict_single_employee logs the event
    df_raw = _get_raw_attrition_df()
    raw_match = df_raw[df_raw["EmployeeNumber"] == employee_id]
    if raw_match.empty:
        raise KeyError(f"Employee with EmployeeNumber {employee_id} not found in records.")
        
    logger.info(f"Triggering real-time model inference fallback for Employee #{employee_id} (not in pre-computed table)")
    res = predict_single_employee(raw_match.iloc[0].to_dict())
    res["source"] = "live_fallback_inference"
    return res


def get_full_employee_intelligence(employee_id: int) -> Dict[str, Any]:
    """
    Retrieves the complete intelligence record for an employee from employee_intelligence.csv,
    combining attrition risk, engagement survey, O*NET role alignment, skill gaps, and recommendations.
    """
    df_intel = _get_intelligence_df()
    match = df_intel[df_intel["EmployeeNumber"] == employee_id]
    
    if match.empty:
        raise KeyError(f"Employee #{employee_id} not found in employee intelligence dataset.")
        
    row = match.iloc[0].to_dict()
    
    # Clean NaNs to None for clean JSON serialization
    clean_record = {}
    for k, v in row.items():
        if pd.isna(v):
            clean_record[k] = None
        elif isinstance(v, (np.int64, np.int32)):
            clean_record[k] = int(v)
        elif isinstance(v, (np.float64, np.float32)):
            clean_record[k] = round(float(v), 4)
        else:
            clean_record[k] = v
            
    # Enrich with tenure (YearsAtCompany) from raw processed HR records if available
    try:
        df_raw = _get_raw_attrition_df()
        raw_match = df_raw[df_raw["EmployeeNumber"] == employee_id]
        if not raw_match.empty and "YearsAtCompany" in raw_match.columns:
            val = raw_match.iloc[0]["YearsAtCompany"]
            clean_record["YearsAtCompany"] = int(val) if pd.notna(val) else None
    except Exception as e:
        logger.warning(f"Could not retrieve YearsAtCompany for #{employee_id}: {e}")

    return clean_record


def get_workforce_risk_summary() -> Dict[str, Any]:
    """Computes organization-wide risk distribution and high-risk employee counts."""
    df_intel = _get_intelligence_df()
    total = len(df_intel)
    counts = df_intel["RiskLevel"].value_counts().to_dict()
    
    high_count = counts.get("HIGH", 0)
    low_count = counts.get("LOW", 0)
    
    return {
        "total_employees": total,
        "high_risk_count": high_count,
        "high_risk_percentage": round((high_count / total) * 100, 2) if total > 0 else 0.0,
        "low_risk_count": low_count,
        "low_risk_percentage": round((low_count / total) * 100, 2) if total > 0 else 0.0,
        "threshold": 0.40
    }


def get_dashboard_summary_kpis() -> Dict[str, Any]:
    """
    Combines overall flight risk counts with workforce average engagement.
    Explicitly caveats that engagement is based on the 731-employee survey sample.
    """
    df_intel = _get_intelligence_df()
    summary = get_workforce_risk_summary()
    
    valid_eng = df_intel[df_intel["EngagementScore"].notnull()]["EngagementScore"]
    survey_count = len(valid_eng)
    avg_eng = round(float(valid_eng.mean()), 2) if survey_count > 0 else None
    
    summary["average_engagement"] = avg_eng
    summary["survey_respondents"] = survey_count
    summary["engagement_coverage_note"] = (
        f"Average engagement score ({avg_eng} / 5.0) is based strictly on the {survey_count} "
        f"employees ({round(survey_count/len(df_intel)*100, 1)}% of workforce) with matched survey data. "
        f"{len(df_intel) - survey_count} employees have null survey data."
    )
    return summary


def get_attrition_by_department() -> List[Dict[str, Any]]:
    """
    Groups employee intelligence table by Department, calculating headcounts,
    high risk counts, and high risk percentages.
    """
    df_intel = _get_intelligence_df()
    dept_stats = []
    
    for dept, group in df_intel.groupby("Department"):
        total = len(group)
        high_count = (group["RiskLevel"] == "HIGH").sum()
        low_count = total - high_count
        high_pct = round((high_count / total) * 100, 2) if total > 0 else 0.0
        low_pct = round((low_count / total) * 100, 2) if total > 0 else 0.0
        mean_prob = round(float(group["RiskScore"].mean()), 4)
        
        dept_stats.append({
            "department": dept,
            "total_employees": total,
            "high_risk_count": int(high_count),
            "high_risk_percentage": high_pct,
            "low_risk_count": int(low_count),
            "low_risk_percentage": low_pct,
            "mean_risk_score": mean_prob
        })
        
    return dept_stats
