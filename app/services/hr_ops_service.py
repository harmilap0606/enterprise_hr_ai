"""
app/services/hr_ops_service.py
==============================
Read-only HR Operations service module for the Enterprise HR AI Platform.
Serves operational employee profiles, organizational/departmental headcount statistics,
and department staffing roster distributions from the primary anchor table.

Governed by:
- POL-DATA-001 (Data Usage, Anchor Primacy, and Provenance Lineage)
- POL-REVIEW-001 (Human Review Requirements)

MANDATORY PRIVACY DEFENSE:
Under no circumstances are demographic or compensation fields exposed:
Prohibited fields: Age, Gender, MaritalStatus, MonthlyIncome, HourlyRate,
DailyRate, MonthlyRate, PercentSalaryHike, StockOptionLevel.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from app.utils.config import EMPLOYEE_ATTRITION_PATH
from app.utils.logger import logger

_cached_attrition_df: Optional[pd.DataFrame] = None

CANONICAL_DEPARTMENTS = [
    "Sales",
    "Research & Development",
    "Human Resources"
]

DEPARTMENT_ALIASES = {
    "sales": "Sales",
    "r&d": "Research & Development",
    "research & development": "Research & Development",
    "research and development": "Research & Development",
    "rd": "Research & Development",
    "hr": "Human Resources",
    "human resources": "Human Resources",
}

# Strict prohibited PII list for defense-in-depth
PROHIBITED_PII_FIELDS = {
    "Age", "Gender", "MaritalStatus", "MonthlyIncome",
    "HourlyRate", "DailyRate", "MonthlyRate",
    "PercentSalaryHike", "StockOptionLevel",
    "Attrition", "RiskScore", "RiskLevel"
}


def _load_anchor_dataset() -> pd.DataFrame:
    """Loads and caches the primary anchor dataset (1,470 employee records)."""
    global _cached_attrition_df
    if _cached_attrition_df is None:
        if not EMPLOYEE_ATTRITION_PATH.exists():
            raise FileNotFoundError(f"Anchor dataset not found at {EMPLOYEE_ATTRITION_PATH}")
        _cached_attrition_df = pd.read_csv(EMPLOYEE_ATTRITION_PATH)
        logger.info(f"Loaded {len(_cached_attrition_df):,} records from {EMPLOYEE_ATTRITION_PATH.name}")
    return _cached_attrition_df


def normalize_department_name(department: str) -> str:
    """Normalizes and resolves department names across common enterprise aliases."""
    if not department:
        raise ValueError(f"Department must be specified. Valid departments are: {', '.join(CANONICAL_DEPARTMENTS)}")
    dept_clean = department.strip().lower()
    if dept_clean in DEPARTMENT_ALIASES:
        return DEPARTMENT_ALIASES[dept_clean]
    for canon in CANONICAL_DEPARTMENTS:
        if canon.lower() == dept_clean:
            return canon
    raise ValueError(
        f"Invalid department '{department}'. Valid departments are: {', '.join(CANONICAL_DEPARTMENTS)}"
    )


def get_employee_operational_profile(employee_id: int) -> Dict[str, Any]:
    """
    Retrieves operational personnel fields for a specific employee.
    
    STRICT PRIVACY GUARDRAILS:
    Never returns demographic or compensation fields.
    Does NOT calculate or expose attrition probability or risk tiers.
    
    Args:
        employee_id: Unique numeric EmployeeNumber identifier.
        
    Returns:
        Structured dictionary containing operational profile attributes and provenance.
        
    Raises:
        KeyError: If employee is not found in enterprise personnel records.
    """
    df = _load_anchor_dataset()
    match = df[df["EmployeeNumber"] == employee_id]
    
    if match.empty:
        logger.warning(f"Employee #{employee_id} not found in enterprise personnel records.")
        raise KeyError(f"Employee #{employee_id} was not found in enterprise personnel records.")
        
    row = match.iloc[0]
    
    # Operational fields strictly allowed
    profile = {
        "EmployeeNumber": int(row["EmployeeNumber"]),
        "Department": str(row["Department"]),
        "JobRole": str(row["JobRole"]),
        "JobLevel": int(row["JobLevel"]),
        "EducationField": str(row["EducationField"]),
        "Education": int(row["Education"]) if pd.notna(row["Education"]) else 0,
        "BusinessTravel": str(row["BusinessTravel"]),
        "OverTime": str(row["OverTime"]),
        "TotalWorkingYears": int(row["TotalWorkingYears"]) if pd.notna(row["TotalWorkingYears"]) else 0,
        "YearsAtCompany": int(row["YearsAtCompany"]) if pd.notna(row["YearsAtCompany"]) else 0,
        "YearsInCurrentRole": int(row["YearsInCurrentRole"]) if pd.notna(row["YearsInCurrentRole"]) else 0,
        "YearsSinceLastPromotion": int(row["YearsSinceLastPromotion"]) if pd.notna(row["YearsSinceLastPromotion"]) else 0,
        "YearsWithCurrManager": int(row["YearsWithCurrManager"]) if pd.notna(row["YearsWithCurrManager"]) else 0,
        "PerformanceRating": int(row["PerformanceRating"]) if pd.notna(row["PerformanceRating"]) else 0,
        "WorkLifeBalance": int(row["WorkLifeBalance"]) if pd.notna(row["WorkLifeBalance"]) else 0,
        "provenance": [
            {
                "source": "data/processed/employee_attrition_processed.csv",
                "record_type": "operational_personnel_record",
                "employee_id": employee_id,
                "governance": "POL-DATA-001 Rule 1"
            }
        ]
    }
    
    # Defense-in-depth: Ensure prohibited fields are never present
    for prohibited in PROHIBITED_PII_FIELDS:
        if prohibited in profile:
            del profile[prohibited]
            
    return profile


def get_headcount_statistics(department: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes company-wide or departmental headcount and tenure statistics.
    
    Expected totals:
    - Research & Development: 961
    - Sales: 446
    - Human Resources: 63
    - Total: 1470
    
    Args:
        department: Optional department name to filter headcount.
        
    Returns:
        Structured aggregate headcount information.
    """
    df = _load_anchor_dataset()
    total_headcount = len(df)
    
    dept_counts = df["Department"].value_counts().to_dict()
    breakdown = {
        dept: {
            "headcount": int(count),
            "percentage": round((count / total_headcount) * 100, 2)
        }
        for dept, count in dept_counts.items()
    }
    
    provenance = [
        {
            "source": "data/processed/employee_attrition_processed.csv",
            "scope": "workforce_headcount_inventory",
            "governance": "POL-DATA-001 Rule 1 (Row Invariance: 1,470)"
        }
    ]
    
    if not department:
        return {
            "scope": "company_wide",
            "total_headcount": total_headcount,
            "department_breakdown": breakdown,
            "provenance": provenance
        }
        
    # Department-specific aggregate
    norm_dept = normalize_department_name(department)
    dept_df = df[df["Department"] == norm_dept]
    dept_headcount = len(dept_df)
    pct_workforce = round((dept_headcount / total_headcount) * 100, 2)
    
    role_counts = dept_df["JobRole"].value_counts().to_dict()
    mean_tenure = round(float(dept_df["YearsAtCompany"].mean()), 2)
    ot_count = (dept_df["OverTime"] == "Yes").sum()
    ot_rate = round((ot_count / dept_headcount) * 100, 2) if dept_headcount > 0 else 0.0
    
    return {
        "scope": "department",
        "department": norm_dept,
        "headcount": dept_headcount,
        "percentage_of_workforce": pct_workforce,
        "role_headcounts": {role: int(c) for role, c in role_counts.items()},
        "mean_tenure_years": mean_tenure,
        "overtime_rate_pct": ot_rate,
        "provenance": provenance
    }


def get_department_staffing(department: str) -> Dict[str, Any]:
    """
    Computes comprehensive structural staffing distribution for a department.
    
    Includes role counts, JobLevel hierarchy, education field representation,
    average organizational tenure, and overtime distribution.
    
    Args:
        department: Canonical or alias department name.
        
    Returns:
        Structured departmental staffing profile.
    """
    df = _load_anchor_dataset()
    norm_dept = normalize_department_name(department)
    dept_df = df[df["Department"] == norm_dept]
    headcount = len(dept_df)
    
    # Distributions
    role_dist = dept_df["JobRole"].value_counts().to_dict()
    level_dist = {int(k): int(v) for k, v in dept_df["JobLevel"].value_counts().sort_index().to_dict().items()}
    edu_dist = dept_df["EducationField"].value_counts().to_dict()
    
    avg_tenure = round(float(dept_df["YearsAtCompany"].mean()), 2)
    avg_total_working = round(float(dept_df["TotalWorkingYears"].mean()), 2)
    
    ot_yes = (dept_df["OverTime"] == "Yes").sum()
    ot_no = headcount - ot_yes
    ot_rate = round((ot_yes / headcount) * 100, 2) if headcount > 0 else 0.0
    
    return {
        "department": norm_dept,
        "headcount": headcount,
        "role_distribution": {role: int(c) for role, c in role_dist.items()},
        "job_level_distribution": level_dist,
        "education_field_distribution": {f: int(c) for f, c in edu_dist.items()},
        "average_tenure": {
            "years_at_company": avg_tenure,
            "total_working_years": avg_total_working
        },
        "overtime_summary": {
            "overtime_count": int(ot_yes),
            "regular_hours_count": int(ot_no),
            "overtime_rate_pct": ot_rate
        },
        "provenance": [
            {
                "source": "data/processed/employee_attrition_processed.csv",
                "scope": f"department_staffing_{norm_dept}",
                "governance": "POL-DATA-001 Rule 1"
            }
        ]
    }
