"""
Recommendation Service Module.
Serves tailored training pathways and learning recommendations from employee_recommendations.csv
and filtered workforce cohorts from employee_intelligence.csv.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from app.utils.config import EMPLOYEE_RECOMMENDATIONS_PATH, EMPLOYEE_INTELLIGENCE_PATH
from app.utils.logger import logger

_cached_recs_df: Optional[pd.DataFrame] = None
_cached_intel_df: Optional[pd.DataFrame] = None

def _load_recommendations_data() -> pd.DataFrame:
    """Loads and caches the employee recommendations dataset."""
    global _cached_recs_df
    if _cached_recs_df is None:
        if not EMPLOYEE_RECOMMENDATIONS_PATH.exists():
            raise FileNotFoundError(f"Employee recommendations file not found at {EMPLOYEE_RECOMMENDATIONS_PATH}")
        _cached_recs_df = pd.read_csv(EMPLOYEE_RECOMMENDATIONS_PATH, comment="#")
    return _cached_recs_df


def _load_intelligence_data() -> pd.DataFrame:
    """Loads and caches the capstone employee intelligence dataset."""
    global _cached_intel_df
    if _cached_intel_df is None:
        if not EMPLOYEE_INTELLIGENCE_PATH.exists():
            raise FileNotFoundError(f"Employee intelligence file not found at {EMPLOYEE_INTELLIGENCE_PATH}")
        _cached_intel_df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH, comment="#")
    return _cached_intel_df


def get_employee_recommendations(employee_id: int) -> Dict[str, Any]:
    """
    Retrieves personalized top 3 course recommendations for a specific employee.
    Handles Manager exclusion gracefully with architectural note.
    """
    df = _load_recommendations_data()
    match = df[df["EmployeeNumber"] == employee_id]
    
    if match.empty:
        # Check if employee is Manager
        intel_df = _load_intelligence_data()
        emp_row = intel_df[intel_df["EmployeeNumber"] == employee_id]
        if not emp_row.empty and emp_row.iloc[0]["JobRole"] == "Manager":
            return {
                "EmployeeNumber": employee_id,
                "JobRole": "Manager",
                "severity": "N/A - Manager",
                "top_3_missing_skills": "None",
                "top_3_recommendations": "N/A - Manager (use Department-level analysis)",
                "note": "Manager role was excluded from synthetic O*NET skill mapping per Step 10 & 11."
            }
        raise KeyError(f"Employee with EmployeeNumber {employee_id} not found in recommendations table.")
        
    row = match.iloc[0]
    return {
        "EmployeeNumber": int(row["EmployeeNumber"]),
        "JobRole": row["JobRole"],
        "severity": row["severity"],
        "top_3_missing_skills": row["top_3_missing_skills"],
        "top_3_recommendations": row["top_3_recommendations"],
        "note": "Rule-based Version 1 recommendation output (derived from synthetic skills inventory)."
    }


def get_filtered_recommendations(
    department: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Returns filtered list of employee recommendations from employee_intelligence.csv.
    Allows filtering by department (e.g. 'Sales') and/or risk level (e.g. 'HIGH', 'LOW').
    """
    df = _load_intelligence_data()
    subset = df.copy()
    
    if department:
        subset = subset[subset["Department"].str.lower() == department.strip().lower()]
        
    if risk_level:
        subset = subset[subset["RiskLevel"].str.upper() == risk_level.strip().upper()]
        
    cols = [
        "EmployeeNumber", "Department", "JobRole", "RiskScore",
        "RiskLevel", "SkillGapSeverity", "Top3Recommendations"
    ]
    records = subset[cols].copy()
    if limit is not None and limit > 0:
        records = records.head(limit)
        
    # Clean records
    result = []
    for r in records.to_dict(orient="records"):
        clean_r = {}
        for k, v in r.items():
            if pd.isna(v):
                clean_r[k] = None
            elif isinstance(v, (np.int64, np.int32)):
                clean_r[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                clean_r[k] = round(float(v), 4)
            else:
                clean_r[k] = v
        result.append(clean_r)
        
    return result
