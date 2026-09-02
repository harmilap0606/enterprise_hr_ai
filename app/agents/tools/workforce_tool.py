"""
app/agents/tools/workforce_tool.py
==================================
Workforce Intelligence Tools for the Enterprise HR AI Agentic Layer.
Encapsulates access to workforce risk assessment and engagement analytics:
- Attrition Service: app/services/attrition_service.py
- Engagement Service: app/services/engagement_service.py

Provides 3 specialized tools:
1. WorkforceKPITool: Organization-wide flight risk distribution and survey engagement KPIs.
2. DepartmentWorkforceTool: Department-level attrition risk and engagement metrics.
3. EmployeeRiskTool: Individual employee flight risk intelligence and engagement metrics.

Security & Execution Guardrails:
- Strict Pydantic parameter schemas.
- Filters out non-workforce PII / demographic fields (Age, Gender, MaritalStatus).
- Authorized tool registry: WORKFORCE_AUTHORIZED_TOOLS.
- Structured provenance attribution.
- Exception shielding (catches KeyError for unknown employee IDs).
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.services.attrition_service import (
    get_dashboard_summary_kpis,
    get_attrition_by_department,
    get_employee_risk,
    get_full_employee_intelligence
)
from app.services.engagement_service import (
    get_engagement_summary,
    get_engagement_by_department
)
from app.utils.logger import logger


# ==============================================================================
# Pydantic Tool Input Schemas
# ==============================================================================

class WorkforceKPIInput(BaseModel):
    """Empty input schema for organization-wide KPI queries."""
    pass


class DepartmentWorkforceInput(BaseModel):
    """Input schema for department workforce analytics."""
    department: Optional[str] = Field(
        None,
        description="Optional department name to filter by (e.g., 'Sales', 'Research & Development', 'Human Resources')."
    )


class EmployeeRiskInput(BaseModel):
    """Input schema for individual employee flight risk queries."""
    employee_id: int = Field(
        ...,
        description="Explicit numeric EmployeeNumber identifier (e.g., 1, 100, 1024)."
    )


# ==============================================================================
# Tool 1: WorkforceKPITool
# ==============================================================================

class WorkforceKPITool:
    """
    Organization-wide workforce intelligence tool.
    Retrieves high-level attrition risk counts, decision threshold (0.40),
    and workforce engagement metrics with sample coverage caveats.
    """

    name: str = "get_workforce_kpis"
    description: str = (
        "Retrieves organization-wide workforce flight risk distribution and employee engagement KPIs, "
        "including total headcount (1,470), high-risk counts, model threshold (0.40), and survey sample coverage (49.7%)."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": []
    }

    _instance: Optional["WorkforceKPITool"] = None

    @classmethod
    def get_instance(cls) -> "WorkforceKPITool":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def execute(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Validates arguments and retrieves executive workforce KPIs."""
        logger.info("Executing WorkforceKPITool...")
        # Pydantic validation
        WorkforceKPIInput(**tool_args)

        kpis = get_dashboard_summary_kpis()
        eng_summary = get_engagement_summary()

        provenance = [
            {
                "source": "data/processed/employee_intelligence.csv",
                "service": "app/services/attrition_service.py",
                "operation": "get_dashboard_summary_kpis",
                "scope": "organization_aggregate"
            },
            {
                "source": "data/processed/employee_intelligence.csv",
                "service": "app/services/engagement_service.py",
                "operation": "get_engagement_summary",
                "scope": "survey_sample_731_respondents"
            }
        ]

        result_data = {
            "total_employees": kpis.get("total_employees", 1470),
            "high_risk_count": kpis.get("high_risk_count", 585),
            "high_risk_percentage": kpis.get("high_risk_percentage", 39.8),
            "low_risk_count": kpis.get("low_risk_count", 885),
            "low_risk_percentage": kpis.get("low_risk_percentage", 60.2),
            "decision_threshold": kpis.get("threshold", 0.40),
            "average_engagement": kpis.get("average_engagement", 2.95),
            "survey_respondents": kpis.get("survey_respondents", 731),
            "unmapped_employees": eng_summary.get("unmapped_employees", 739),
            "workforce_coverage_percentage": eng_summary.get("workforce_coverage_percentage", 49.73),
            "overall_mean_satisfaction": eng_summary.get("overall_mean_satisfaction", 2.73),
            "overall_mean_work_life_balance": eng_summary.get("overall_mean_work_life_balance", 2.76),
            "engagement_coverage_note": kpis.get("engagement_coverage_note", eng_summary.get("caveat", ""))
        }

        return {
            "status": "success",
            "data": result_data,
            "provenance": provenance
        }


# ==============================================================================
# Tool 2: DepartmentWorkforceTool
# ==============================================================================

class DepartmentWorkforceTool:
    """
    Department-level workforce analytics tool.
    Retrieves attrition breakdown and engagement survey statistics for
    Sales, Research & Development, and Human Resources.
    """

    name: str = "get_department_workforce_breakdown"
    description: str = (
        "Retrieves department-level workforce metrics including headcount, high flight risk counts, "
        "attrition percentages, and engagement/satisfaction survey averages for Sales, R&D, and HR."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "department": {
                "type": "string",
                "description": "Optional department filter: 'Sales', 'Research & Development', or 'Human Resources'."
            }
        },
        "required": []
    }

    _instance: Optional["DepartmentWorkforceTool"] = None

    @classmethod
    def get_instance(cls) -> "DepartmentWorkforceTool":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def execute(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Validates arguments and retrieves department-level breakdowns."""
        logger.info(f"Executing DepartmentWorkforceTool with args: {tool_args}")
        parsed_input = DepartmentWorkforceInput(**tool_args)
        target_dept = parsed_input.department

        dept_attrition = get_attrition_by_department()
        dept_engagement = get_engagement_by_department()

        # Build merged department lookup map
        eng_by_dept = {d["Department"]: d for d in dept_engagement}
        merged_results = []

        for att in dept_attrition:
            dept_name = att["department"]
            eng = eng_by_dept.get(dept_name, {})
            
            # Apply department filter if requested
            if target_dept:
                clean_target = target_dept.strip().lower()
                # Match normalized strings e.g. "sales", "r&d", "research & development"
                if clean_target not in dept_name.lower() and dept_name.lower() not in clean_target:
                    # Special abbreviation aliases
                    if clean_target in ("r&d", "rd") and "research" in dept_name.lower():
                        pass
                    elif clean_target in ("hr",) and "human resources" in dept_name.lower():
                        pass
                    else:
                        continue

            merged_results.append({
                "department": dept_name,
                "total_employees": att.get("total_employees"),
                "high_risk_count": att.get("high_risk_count"),
                "high_risk_percentage": att.get("high_risk_percentage"),
                "low_risk_count": att.get("low_risk_count"),
                "mean_risk_score": att.get("mean_risk_score"),
                "surveyed_count": eng.get("surveyed_count"),
                "survey_coverage_percentage": eng.get("coverage_percentage"),
                "mean_engagement": eng.get("mean_engagement"),
                "mean_satisfaction": eng.get("mean_satisfaction"),
                "mean_work_life_balance": eng.get("mean_work_life_balance")
            })

        provenance = [
            {
                "source": "data/processed/employee_intelligence.csv",
                "service": "app/services/attrition_service.py",
                "operation": "get_attrition_by_department",
                "scope": "department_attrition_aggregates"
            },
            {
                "source": "data/processed/employee_intelligence.csv",
                "service": "app/services/engagement_service.py",
                "operation": "get_engagement_by_department",
                "scope": "department_engagement_aggregates"
            }
        ]

        return {
            "status": "success",
            "data": {
                "departments": merged_results,
                "filtered_department": target_dept,
                "total_matched_departments": len(merged_results)
            },
            "provenance": provenance
        }


# ==============================================================================
# Tool 3: EmployeeRiskTool
# ==============================================================================

class EmployeeRiskTool:
    """
    Individual employee workforce risk intelligence tool.
    Retrieves calibrated attrition probability, risk tier, decision threshold (0.40),
    job role, and engagement survey score.
    Strictly filters out non-workforce demographic PII (Age, Gender, MaritalStatus).
    """

    name: str = "get_employee_attrition_risk"
    description: str = (
        "Retrieves workforce flight risk intelligence for an explicit EmployeeNumber, including predicted "
        "attrition probability, risk level (HIGH/LOW), model threshold (0.40), and survey scores when available."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "employee_id": {
                "type": "integer",
                "description": "Unique integer EmployeeNumber (e.g., 1, 100)."
            }
        },
        "required": ["employee_id"]
    }

    _instance: Optional["EmployeeRiskTool"] = None

    @classmethod
    def get_instance(cls) -> "EmployeeRiskTool":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def execute(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Validates employee_id, retrieves risk record, and filters out PII."""
        logger.info(f"Executing EmployeeRiskTool with args: {tool_args}")
        parsed_input = EmployeeRiskInput(**tool_args)
        emp_id = parsed_input.employee_id

        try:
            # 1. Retrieve risk score record
            risk_record = get_employee_risk(emp_id)
        except KeyError as e:
            logger.warning(f"Employee #{emp_id} not found: {e}")
            return {
                "status": "error",
                "error_type": "EMPLOYEE_NOT_FOUND",
                "message": f"Employee with EmployeeNumber {emp_id} was not found in the employee intelligence dataset.",
                "data": None,
                "provenance": []
            }
        except Exception as e:
            logger.error(f"Unexpected error looking up Employee #{emp_id}: {e}", exc_info=True)
            return {
                "status": "error",
                "error_type": "SERVICE_ERROR",
                "message": f"An error occurred while retrieving data for Employee #{emp_id}: {str(e)}",
                "data": None,
                "provenance": []
            }

        # 2. Enrich with engagement if available in 360 intelligence, without leaking PII
        eng_score = None
        sat_score = None
        wlb_score = None
        try:
            full_intel = get_full_employee_intelligence(emp_id)
            eng_score = full_intel.get("EngagementScore")
            sat_score = full_intel.get("SatisfactionScore")
            wlb_score = full_intel.get("WorkLifeBalanceScore")
        except Exception:
            pass

        # 3. PII Filtering: Only return workforce-relevant information
        workforce_profile = {
            "EmployeeNumber": risk_record.get("EmployeeNumber", emp_id),
            "Department": risk_record.get("Department"),
            "JobRole": risk_record.get("JobRole"),
            "probability": round(float(risk_record.get("probability", 0.0)), 4),
            "risk_level": risk_record.get("risk_level", "UNKNOWN"),
            "decision_threshold": 0.40,
            "model_version": risk_record.get("model_version", "v3 (logistic_regression_balanced)"),
            "data_source_type": risk_record.get("source", "precomputed_lookup"),
            "EngagementScore": eng_score,
            "SatisfactionScore": sat_score,
            "WorkLifeBalanceScore": wlb_score,
            "survey_status": "survey_matched" if eng_score is not None else "no_survey_record"
        }

        # Determine provenance source
        source_file = "data/processed/employee_intelligence.csv"
        if risk_record.get("source") == "live_fallback_inference":
            source_file = "data/processed/employee_attrition_processed.csv (Live Model Inference logged to data/predictions/prediction_log.csv)"

        provenance = [
            {
                "source": source_file,
                "service": "app/services/attrition_service.py",
                "operation": "get_employee_risk",
                "employee_id": emp_id,
                "model_version": workforce_profile["model_version"],
                "threshold": 0.40
            }
        ]

        return {
            "status": "success",
            "data": workforce_profile,
            "provenance": provenance
        }


# ==============================================================================
# Authorized Tool Registry
# ==============================================================================

WORKFORCE_AUTHORIZED_TOOLS = {
    WorkforceKPITool.name: WorkforceKPITool.get_instance,
    DepartmentWorkforceTool.name: DepartmentWorkforceTool.get_instance,
    EmployeeRiskTool.name: EmployeeRiskTool.get_instance
}
