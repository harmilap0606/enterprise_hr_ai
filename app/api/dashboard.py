"""
Executive Dashboard API Endpoints Router.
Serves workforce intelligence KPIs, departmental attrition distributions,
organization-wide skill gaps, and filtered employee recommendations.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from app.services.attrition_service import (
    get_dashboard_summary_kpis,
    get_attrition_by_department
)
from app.services.skill_gap_service import get_organization_skill_gaps
from app.services.recommendation_service import get_filtered_recommendations
from app.utils.logger import logger

router = APIRouter(prefix="/dashboard", tags=["Executive HR Dashboard"])


@router.get("/summary", summary="2. Get Workforce Risk & Engagement KPI Summary")
def get_dashboard_summary():
    """
    2. GET /dashboard/summary
    Returns overall workforce counts, high risk totals, and average engagement score.
    Explicitly notes that engagement is calculated from the 731 employees with survey data.
    """
    try:
        return get_dashboard_summary_kpis()
    except Exception as e:
        logger.error(f"Error fetching dashboard summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/attrition-by-department", summary="3. Get Attrition Risk Breakdown by Department")
def get_department_attrition():
    """
    3. GET /dashboard/attrition-by-department
    Groups employees by Department (Sales, R&D, HR) and returns headcounts,
    high risk counts, and high risk percentages.
    """
    try:
        return get_attrition_by_department()
    except Exception as e:
        logger.error(f"Error fetching department attrition: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skill-gaps", summary="4. Get Organization-Wide Skill Gap Rankings")
def get_dashboard_skill_gaps():
    """
    4. GET /dashboard/skill-gaps
    Returns the complete ranked list of 33 organizational skill gaps with
    missing employee counts, severity tiers, and role concentration indicators.
    """
    try:
        return get_organization_skill_gaps()
    except Exception as e:
        logger.error(f"Error fetching organizational skill gaps: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations", summary="5. Get Filtered Workforce Recommendations")
def get_dashboard_recommendations(
    department: Optional[str] = Query(None, description="Filter by department (e.g. Sales, Research & Development, Human Resources)"),
    risk_level: Optional[str] = Query(None, description="Filter by risk tier (HIGH or LOW)"),
    limit: Optional[int] = Query(None, ge=1, le=1470, description="Optional maximum number of records to return")
):
    """
    5. GET /dashboard/recommendations?department=&risk_level=
    Returns list of employees with their risk tier, skill gap severity, and concrete top 3 training recommendations.
    Supports filtering by department and/or risk level.
    """
    try:
        return get_filtered_recommendations(department=department, risk_level=risk_level, limit=limit)
    except Exception as e:
        logger.error(f"Error fetching filtered recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
