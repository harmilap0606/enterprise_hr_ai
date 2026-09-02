"""
app/agents/tools/hr_ops_tool.py
===============================
Dedicated HR Operations tools for the HR Operations Agent (Agent 5 of 5).

Includes:
- EmployeeProfileLookupInput & EmployeeProfileLookupTool
- HeadcountAnalyticsInput & HeadcountAnalyticsTool
- DepartmentStaffingInput & DepartmentStaffingTool
- HR_OPS_AUTHORIZED_TOOLS whitelist
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.services.hr_ops_service import (
    get_employee_operational_profile as svc_get_employee_operational_profile,
    get_headcount_statistics as svc_get_headcount_statistics,
    get_department_staffing as svc_get_department_staffing,
    PROHIBITED_PII_FIELDS
)
from app.utils.logger import logger


# ==============================================================================
# Pydantic Input Schemas
# ==============================================================================

class EmployeeProfileLookupInput(BaseModel):
    """Input schema for retrieving operational personnel records for an employee."""
    employee_id: int = Field(
        ...,
        description="Unique numeric EmployeeNumber identifier (e.g. 1, 42, 500).",
        examples=[1, 42, 500]
    )


class HeadcountAnalyticsInput(BaseModel):
    """Input schema for querying company-wide or departmental headcount statistics."""
    department: Optional[str] = Field(
        default=None,
        description="Optional canonical department name (e.g. 'Sales', 'Research & Development', 'Human Resources').",
        examples=["Sales", "Research & Development", "Human Resources"]
    )


class DepartmentStaffingInput(BaseModel):
    """Input schema for querying departmental staffing structure, role distribution, and JobLevel hierarchy."""
    department: str = Field(
        ...,
        description="Canonical department name (e.g. 'Sales', 'Research & Development', 'Human Resources').",
        examples=["Sales", "Research & Development", "Human Resources"]
    )


# ==============================================================================
# Whitelist of Authorized Tools for HR Ops Agent
# ==============================================================================

HR_OPS_AUTHORIZED_TOOLS = {
    "lookup_employee_record",
    "get_headcount_statistics",
    "get_department_staffing",
}


# ==============================================================================
# Tool Implementations
# ==============================================================================

class EmployeeProfileLookupTool:
    """
    Tool for looking up an employee's factual operational personnel profile.
    Enforces strict PII defense by stripping demographic and compensation fields.
    """
    TOOL_NAME = "lookup_employee_record"

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            validated = EmployeeProfileLookupInput(**args)
            logger.info(f"EmployeeProfileLookupTool executing for Employee #{validated.employee_id}")
            profile = svc_get_employee_operational_profile(validated.employee_id)
            
            # Defense-in-depth: Ensure prohibited fields are stripped
            filtered = {k: v for k, v in profile.items() if k not in PROHIBITED_PII_FIELDS}
            return {
                "status": "success",
                **filtered
            }
        except KeyError as ke:
            logger.warning(f"Employee #{args.get('employee_id')} not found: {ke}")
            return {
                "status": "error",
                "error_type": "EMPLOYEE_NOT_FOUND",
                "message": f"Employee #{args.get('employee_id')} was not found in enterprise personnel records."
            }
        except Exception as e:
            logger.error(f"EmployeeProfileLookupTool execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error_type": "EXECUTION_FAILURE",
                "message": str(e)
            }


class HeadcountAnalyticsTool:
    """
    Tool for querying total company headcount and departmental headcounts.
    """
    TOOL_NAME = "get_headcount_statistics"

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            validated = HeadcountAnalyticsInput(**args)
            logger.info(f"HeadcountAnalyticsTool executing (department={validated.department})")
            stats = svc_get_headcount_statistics(validated.department)
            return {
                "status": "success",
                **stats
            }
        except ValueError as ve:
            logger.warning(f"HeadcountAnalyticsTool validation error: {ve}")
            return {
                "status": "error",
                "error_type": "INVALID_DEPARTMENT",
                "message": str(ve)
            }
        except Exception as e:
            logger.error(f"HeadcountAnalyticsTool execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error_type": "EXECUTION_FAILURE",
                "message": str(e)
            }


class DepartmentStaffingTool:
    """
    Tool for analyzing structural staffing, role distributions, and JobLevel hierarchies by department.
    """
    TOOL_NAME = "get_department_staffing"

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            validated = DepartmentStaffingInput(**args)
            logger.info(f"DepartmentStaffingTool executing for department='{validated.department}'")
            staffing = svc_get_department_staffing(validated.department)
            return {
                "status": "success",
                **staffing
            }
        except ValueError as ve:
            logger.warning(f"DepartmentStaffingTool validation error: {ve}")
            return {
                "status": "error",
                "error_type": "INVALID_DEPARTMENT",
                "message": str(ve)
            }
        except Exception as e:
            logger.error(f"DepartmentStaffingTool execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error_type": "EXECUTION_FAILURE",
                "message": str(e)
            }
