"""
Skills & Capability Intelligence API Endpoints Router.
Provides individual employee skill recommendation lookups.
"""

from fastapi import APIRouter, HTTPException, Path
from app.services.recommendation_service import get_employee_recommendations
from app.services.skill_gap_service import get_organization_skill_gaps
from app.utils.logger import logger

router = APIRouter(prefix="/skills", tags=["Skills & Development"])


@router.get("/recommendations/{employee_id}", summary="7. Get Single Employee Skill Recommendations")
def get_recommendations_for_employee(employee_id: int = Path(..., description="Unique Employee Number")):
    """
    7. GET /skills/recommendations/{employee_id}
    Retrieves top 3 skill gaps and corresponding concrete course recommendations for an individual employee.
    Handles Manager exclusion gracefully with architectural guidance.
    """
    try:
        return get_employee_recommendations(employee_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching recommendations for #{employee_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gaps", summary="Organization Skill Gaps (Skills Tag Alias)")
def get_skill_gaps_alias():
    """Alias for /dashboard/skill-gaps under the /skills namespace."""
    try:
        return get_organization_skill_gaps()
    except Exception as e:
        logger.error(f"Error fetching skill gaps: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
