"""
app/agents/tools/career_tool.py
===============================
Dedicated Career progression, role mapping, and promotion readiness tools for the Career Agent.

Includes:
- RoleCareerPathwayInput & RoleCareerPathwayTool
- EmployeePromotionReadinessInput & EmployeePromotionReadinessTool
- RoleCompetencyComparisonInput & RoleCompetencyComparisonTool
- CAREER_AUTHORIZED_TOOLS whitelist
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.services.career_service import (
    get_role_career_pathway as svc_get_role_career_pathway,
    get_employee_promotion_readiness as svc_get_employee_promotion_readiness,
    compare_role_competencies as svc_compare_role_competencies
)
from app.utils.logger import logger


# ==============================================================================
# Pydantic Tool Input Schemas
# ==============================================================================

class RoleCareerPathwayInput(BaseModel):
    """Input schema for querying career pathways and O*NET mapping for an internal job role."""
    role_name: str = Field(
        ...,
        description="Internal enterprise job role (e.g., 'Laboratory Technician', 'Sales Representative').",
        examples=["Laboratory Technician", "Research Scientist", "Sales Representative"]
    )


class EmployeePromotionReadinessInput(BaseModel):
    """Input schema for evaluating an individual employee's promotion velocity and stagnation status."""
    employee_id: int = Field(
        ...,
        description="Numeric EmployeeNumber identifier (e.g. 1, 13, 100).",
        examples=[1, 13, 100]
    )


class RoleCompetencyComparisonInput(BaseModel):
    """Input schema for comparing competency transferability between current and target roles."""
    current_role: str = Field(
        ...,
        description="Current/origin job role title (e.g., 'Sales Representative').",
        examples=["Sales Representative", "Laboratory Technician"]
    )
    target_role: str = Field(
        ...,
        description="Target/aspirational job role title (e.g., 'Sales Executive', 'Research Scientist').",
        examples=["Sales Executive", "Research Scientist"]
    )


# ==============================================================================
# Whitelist of Authorized Tools for Career Agent
# ==============================================================================

CAREER_AUTHORIZED_TOOLS = {
    "get_role_career_pathway",
    "get_employee_promotion_readiness",
    "compare_role_competencies"
}


# ==============================================================================
# Tool Implementations
# ==============================================================================

class RoleCareerPathwayTool:
    """
    Tool for querying structured internal career progression ladders,
    lateral mobility options, and canonical O*NET SOC mappings.
    """
    TOOL_NAME = "get_role_career_pathway"

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            validated = RoleCareerPathwayInput(**args)
            logger.info(f"RoleCareerPathwayTool executing for role: '{validated.role_name}'")
            res = svc_get_role_career_pathway(validated.role_name)
            return {
                "status": "success",
                **res
            }
        except ValueError as ve:
            logger.warning(f"RoleCareerPathwayTool validation error: {ve}")
            return {
                "status": "error",
                "error_type": "INVALID_ROLE",
                "message": str(ve)
            }
        except Exception as e:
            logger.error(f"RoleCareerPathwayTool execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error_type": "EXECUTION_FAILURE",
                "message": str(e)
            }


class EmployeePromotionReadinessTool:
    """
    Tool for evaluating employee promotion readiness, tenure velocity,
    and career stagnation under POL-CAREER-001 Rule 3.
    """
    TOOL_NAME = "get_employee_promotion_readiness"

    # Strict list of prohibited demographic PII fields
    PROHIBITED_PII_FIELDS = {
        "Age", "Gender", "MaritalStatus", "MonthlyIncome",
        "HourlyRate", "DailyRate", "MonthlyRate"
    }

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            validated = EmployeePromotionReadinessInput(**args)
            logger.info(f"EmployeePromotionReadinessTool executing for Employee #{validated.employee_id}")
            res = svc_get_employee_promotion_readiness(validated.employee_id)
            
            # Defense-in-depth: Ensure prohibited fields are stripped
            filtered = {k: v for k, v in res.items() if k not in self.PROHIBITED_PII_FIELDS}
            return {
                "status": "success",
                **filtered
            }
        except KeyError as ke:
            logger.warning(f"Employee #{args.get('employee_id')} not found: {ke}")
            return {
                "status": "error",
                "error_type": "EMPLOYEE_NOT_FOUND",
                "message": f"Employee #{args.get('employee_id')} was not found in employee career records."
            }
        except Exception as e:
            logger.error(f"EmployeePromotionReadinessTool execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error_type": "EXECUTION_FAILURE",
                "message": str(e)
            }


class RoleCompetencyComparisonTool:
    """
    Tool for analyzing competency transferability and capability deltas
    between origin and aspirational target roles.
    """
    TOOL_NAME = "compare_role_competencies"

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            validated = RoleCompetencyComparisonInput(**args)
            logger.info(
                f"RoleCompetencyComparisonTool comparing: '{validated.current_role}' -> '{validated.target_role}'"
            )
            res = svc_compare_role_competencies(validated.current_role, validated.target_role)
            return {
                "status": "success",
                **res
            }
        except ValueError as ve:
            logger.warning(f"RoleCompetencyComparisonTool validation error: {ve}")
            return {
                "status": "error",
                "error_type": "INVALID_ROLES",
                "message": str(ve)
            }
        except Exception as e:
            logger.error(f"RoleCompetencyComparisonTool execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error_type": "EXECUTION_FAILURE",
                "message": str(e)
            }
