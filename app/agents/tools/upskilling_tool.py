"""
app/agents/tools/upskilling_tool.py
==================================
Upskilling and Learning Recommendation Tools for the Enterprise HR AI Agentic Layer.
Encapsulates access to employee skill gaps, personalized course recommendations,
and organization-wide capability analytics:
- Recommendation Service: app/services/recommendation_service.py
- Skill Gap Service: app/services/skill_gap_service.py

Provides 2 specialized tools:
1. EmployeeUpskillingTool: Individual employee skill gaps, severity tier, and Top-3 course recommendations.
2. OrganizationSkillGapTool: Organization-wide capability gap rankings, severity distribution, and catalog scope.

Security & Execution Guardrails:
- Strict Pydantic parameter schemas.
- Filters out non-workforce PII / demographic fields (Age, Gender, MaritalStatus, MonthlyIncome, HourlyRate).
- Whitelist registry: UPSKILLING_AUTHORIZED_TOOLS.
- Structured provenance attribution.
- Exception shielding (catches KeyError for unknown employee IDs).
- Manager exclusion protocol preservation (POL-LEARN-001, POL-SKILL-001).
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from pydantic import BaseModel, Field

from app.services.recommendation_service import (
    get_employee_recommendations,
    _load_intelligence_data
)
from app.services.skill_gap_service import (
    get_organization_skill_gaps as get_org_gaps_data,
    get_severity_distribution
)
from app.utils.logger import logger


# ==============================================================================
# Pydantic Tool Input Schemas
# ==============================================================================

class EmployeeUpskillingInput(BaseModel):
    """Input schema for individual employee upskilling and course recommendation queries."""
    employee_id: int = Field(
        ...,
        description="Explicit numeric EmployeeNumber identifier (e.g., 1, 100, 1024)."
    )


class OrganizationSkillGapInput(BaseModel):
    """Input schema for organization-wide skill gap analytics."""
    limit: Optional[int] = Field(
        10,
        ge=1,
        le=33,
        description="Maximum number of organizational skill gaps to return (default 10, max 33)."
    )
    severity: Optional[str] = Field(
        None,
        description="Optional severity filter ('HIGH', 'MEDIUM', 'LOW')."
    )


# ==============================================================================
# Tool 1: EmployeeUpskillingTool
# ==============================================================================

class EmployeeUpskillingTool:
    """
    Retrieves personalized course recommendations, missing skill gaps,
    and capability severity for an individual employee.
    
    Adheres strictly to POL-LEARN-001 and POL-SKILL-001:
    - 1,368 non-manager employees receive precomputed Top-3 concrete course recommendations.
    - 102 Manager employees receive structured guidance recommending department-level frameworks.
    - Demographics and sensitive PII are excluded.
    """
    
    TOOL_NAME = "get_employee_upskilling_recommendations"
    
    def __init__(self):
        pass

    def execute(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes individual employee upskilling retrieval with parameter validation
        and exception shielding.
        """
        # 1. Pydantic validation
        validated = EmployeeUpskillingInput(**tool_args)
        emp_id = validated.employee_id
        
        logger.info(f"EmployeeUpskillingTool executing for Employee #{emp_id}")

        # 2. Invoke recommendation service with exception shielding
        try:
            raw_rec = get_employee_recommendations(emp_id)
        except KeyError as e:
            logger.warning(f"EmployeeUpskillingTool: Employee #{emp_id} not found: {e}")
            return {
                "status": "error",
                "error_type": "EMPLOYEE_NOT_FOUND",
                "message": f"Employee #{emp_id} was not found in the employee recommendations records.",
                "EmployeeNumber": emp_id
            }
        except Exception as e:
            logger.error(f"EmployeeUpskillingTool unexpected error for #{emp_id}: {e}", exc_info=True)
            return {
                "status": "error",
                "error_type": "SERVICE_ERROR",
                "message": f"Failed to retrieve recommendations for #{emp_id}: {str(e)}",
                "EmployeeNumber": emp_id
            }

        # 3. Format structured missing skills and recommendations
        raw_missing = raw_rec.get("top_3_missing_skills")
        if raw_missing is not None and not pd.isna(raw_missing):
            str_missing = str(raw_missing).strip()
            if str_missing and str_missing != "None":
                missing_skills = [s.strip() for s in str_missing.split(";") if s.strip()]
            else:
                missing_skills = []
        else:
            missing_skills = []

        raw_courses = raw_rec.get("top_3_recommendations")
        if raw_courses is not None and not pd.isna(raw_courses):
            str_courses = str(raw_courses).strip()
            if str_courses and str_courses != "None":
                recommended_courses = [c.strip() for c in str_courses.split(";") if c.strip()]
            else:
                recommended_courses = []
        else:
            recommended_courses = []

        job_role = raw_rec.get("JobRole", "Unknown")
        is_manager = (job_role == "Manager" or "Manager" in str(raw_rec.get("severity", "")))

        # 4. Assemble clean, privacy-filtered result
        return {
            "status": "success",
            "EmployeeNumber": int(raw_rec.get("EmployeeNumber", emp_id)),
            "JobRole": job_role,
            "severity": raw_rec.get("severity", "UNKNOWN"),
            "missing_skills": missing_skills,
            "recommended_courses": recommended_courses,
            "is_manager": is_manager,
            "notes": raw_rec.get("note", "Rule-based Version 1 recommendation output."),
            "provenance": [
                {
                    "source": "data/processed/employee_recommendations.csv",
                    "description": "Precomputed employee-level 3-course recommendation catalog derived from synthetic O*NET skill inventory.",
                    "record_id": f"EmployeeNumber_{emp_id}"
                },
                {
                    "source": "app/services/recommendation_service.py",
                    "method": "get_employee_recommendations",
                    "policy_basis": "POL-LEARN-001 (Deterministic Top-3 Course Selection Heuristic)"
                }
            ]
        }


# ==============================================================================
# Tool 2: OrganizationSkillGapTool
# ==============================================================================

class OrganizationSkillGapTool:
    """
    Retrieves organization-wide capability gap rankings, severity distribution,
    and enterprise catalog scope across all 33 benchmark skills.
    
    Aggregated operational data only. Zero individual employee records exposed.
    """
    
    TOOL_NAME = "get_organization_skill_gaps"
    
    def __init__(self):
        pass

    def execute(self, tool_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes organization skill gap analysis.
        """
        args = tool_args or {}
        validated = OrganizationSkillGapInput(**args)
        limit = validated.limit or 10
        severity_filter = validated.severity.strip().upper() if validated.severity else None

        logger.info(f"OrganizationSkillGapTool executing with limit={limit}, severity={severity_filter}")

        try:
            all_gaps = get_org_gaps_data()
            dist = get_severity_distribution()
        except Exception as e:
            logger.error(f"OrganizationSkillGapTool error loading data: {e}", exc_info=True)
            return {
                "status": "error",
                "error_type": "SERVICE_ERROR",
                "message": f"Failed to retrieve organization skill gaps: {str(e)}"
            }

        # Apply optional severity filter
        if severity_filter:
            filtered = [g for g in all_gaps if g.get("severity", "").upper() == severity_filter]
        else:
            filtered = all_gaps

        top_gaps = filtered[:limit] if limit > 0 else filtered

        return {
            "status": "success",
            "total_skills_evaluated": 33,
            "severity_breakdown": {
                "HIGH": dist.get("HIGH", 0),
                "MEDIUM": dist.get("MEDIUM", 0),
                "LOW": dist.get("LOW", 0)
            },
            "top_skill_gaps": top_gaps,
            "catalog_scope": "33 Curated Enterprise Courses",
            "provenance": [
                {
                    "source": "data/processed/organization_skill_gaps.csv",
                    "description": "Ranked inventory of all 33 organizational skill gaps mapped to enterprise roles.",
                    "total_skills": 33
                },
                {
                    "source": "app/services/skill_gap_service.py",
                    "method": "get_organization_skill_gaps",
                    "policy_basis": "POL-SKILL-001 (Skill Gap Identification & Severity Classification)"
                }
            ]
        }


# ==============================================================================
# Authorized Tool Registry
# ==============================================================================

UPSKILLING_AUTHORIZED_TOOLS = {
    EmployeeUpskillingTool.TOOL_NAME,
    OrganizationSkillGapTool.TOOL_NAME
}
