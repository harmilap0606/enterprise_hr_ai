"""
Engagement Service Module.
Computes workforce survey metrics and departmental engagement aggregations
from employee_intelligence.csv (adapting Step 13 methodology into reusable business logic).
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from app.utils.config import EMPLOYEE_INTELLIGENCE_PATH
from app.utils.logger import logger

_cached_intelligence_df: Optional[pd.DataFrame] = None

def _load_intelligence_data() -> pd.DataFrame:
    """Loads and caches the capstone employee intelligence table."""
    global _cached_intelligence_df
    if _cached_intelligence_df is None:
        if not EMPLOYEE_INTELLIGENCE_PATH.exists():
            raise FileNotFoundError(f"Employee intelligence file not found at {EMPLOYEE_INTELLIGENCE_PATH}")
        _cached_intelligence_df = pd.read_csv(EMPLOYEE_INTELLIGENCE_PATH, comment="#")
    return _cached_intelligence_df


def get_engagement_summary() -> Dict[str, Any]:
    """
    Computes overall engagement survey metrics across the workforce.
    Preserves and highlights the 49.7% sample limitation.
    """
    df = _load_intelligence_data()
    total = len(df)
    surveyed = df[df["EngagementScore"].notnull()]
    respondent_count = len(surveyed)
    
    mean_eng = float(surveyed["EngagementScore"].mean()) if respondent_count > 0 else 0.0
    mean_sat = float(surveyed["SatisfactionScore"].mean()) if respondent_count > 0 else 0.0
    mean_wlb = float(surveyed["WorkLifeBalanceScore"].mean()) if respondent_count > 0 else 0.0
    
    dept_breakdown = get_engagement_by_department()
    
    return {
        "total_workforce": total,
        "survey_respondents": respondent_count,
        "unmapped_employees": total - respondent_count,
        "workforce_coverage_percentage": round((respondent_count / total) * 100, 2),
        "overall_mean_engagement": round(mean_eng, 2),
        "overall_mean_satisfaction": round(mean_sat, 2),
        "overall_mean_work_life_balance": round(mean_wlb, 2),
        "caveat": (
            "Findings are based on 731 employees (49.7% of workforce) with matched survey data. "
            "Do not generalize to the 739 unmapped employees."
        ),
        "departments": dept_breakdown
    }


def get_engagement_by_department() -> List[Dict[str, Any]]:
    """
    Aggregates survey responses by department (Sales, R&D, HR).
    """
    df = _load_intelligence_data()
    results = []
    
    for dept, group in df.groupby("Department"):
        total_dept = len(group)
        surveyed_dept = group[group["EngagementScore"].notnull()]
        n_surveyed = len(surveyed_dept)
        
        coverage = round((n_surveyed / total_dept) * 100, 2) if total_dept > 0 else 0.0
        mean_eng = round(float(surveyed_dept["EngagementScore"].mean()), 2) if n_surveyed > 0 else None
        mean_sat = round(float(surveyed_dept["SatisfactionScore"].mean()), 2) if n_surveyed > 0 else None
        mean_wlb = round(float(surveyed_dept["WorkLifeBalanceScore"].mean()), 2) if n_surveyed > 0 else None
        
        results.append({
            "Department": dept,
            "total_employees": total_dept,
            "surveyed_count": n_surveyed,
            "coverage_percentage": coverage,
            "mean_engagement": mean_eng,
            "mean_satisfaction": mean_sat,
            "mean_work_life_balance": mean_wlb
        })
        
    return results
