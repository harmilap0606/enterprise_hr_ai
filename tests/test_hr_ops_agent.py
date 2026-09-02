"""
tests/test_hr_ops_agent.py
==========================
Comprehensive test suite for the HR Operations Specialist Agent (Agent 5 of 5).

Verifies:
1. Employee profile lookup for valid employee (Employee #1: clean profile, zero PII).
2. Unknown employee returns structured error (Employee #99999: EMPLOYEE_NOT_FOUND).
3. Missing employee ID does not guess (requests numeric ID, zero tool calls).
4. Headcount analytics company-wide (total=1470, R&D=961, Sales=446, HR=63).
5. Headcount analytics department-specific (Sales headcount=446).
6. Department staffing profile (R&D headcount, roles, JobLevels, tenure, overtime).
7. Invalid department handling (Finance: structured validation error).
8. Execution boundary separation (decision node emits status='pending' only).
9. Unauthorized tool execution blocked (arbitrary tool names rejected).
10. Strict privacy / PII filtering (Age, Gender, MaritalStatus, MonthlyIncome suppressed).
11. Synthetic data disclosure (POL-DATA-001 citations and benchmark disclosures).
12. Orchestrator dynamic routing to HR Ops Agent (Employee #500).
13. HR Ops vs Workforce Intelligence boundary test.
14. HR Ops vs Career Agent boundary test.
"""

import re
import pytest
from unittest.mock import patch

from app.agents.state import AgentState
from app.agents.tools.hr_ops_tool import (
    EmployeeProfileLookupTool,
    HeadcountAnalyticsTool,
    DepartmentStaffingTool,
    HR_OPS_AUTHORIZED_TOOLS
)
from app.agents.hr_ops_agent import (
    HROpsAgent,
    get_hr_ops_agent,
    hr_ops_agent_node,
    hr_ops_tool_node,
    hr_ops_response_node,
    HROpsAgentResult,
    AGENT_HR_OPS_ID
)
from app.agents.orchestrator import (
    GlobalOrchestrator,
    AGENT_POLICY,
    AGENT_WORKFORCE,
    AGENT_UPSKILLING,
    AGENT_CAREER,
    AGENT_HR_OPS
)


# ==============================================================================
# 1. Employee Profile Lookup Valid Employee
# ==============================================================================

def test_employee_profile_lookup_valid_employee():
    """Verify clean operational personnel profile extraction for Employee #1 without PII."""
    tool = EmployeeProfileLookupTool()
    res = tool.execute({"employee_id": 1})

    assert res["status"] == "success"
    assert res["EmployeeNumber"] == 1
    assert res["Department"] == "Sales"
    assert res["JobRole"] == "Sales Executive"
    assert res["JobLevel"] == 2
    assert res["YearsAtCompany"] == 6
    assert res["YearsInCurrentRole"] == 4
    assert res["OverTime"] == "Yes"
    assert len(res["provenance"]) >= 1

    # Verify absence of all sensitive fields
    prohibited = [
        "Age", "Gender", "MaritalStatus", "MonthlyIncome",
        "HourlyRate", "DailyRate", "MonthlyRate", "PercentSalaryHike", "StockOptionLevel"
    ]
    for field in prohibited:
        assert field not in res, f"Prohibited field '{field}' leaked in tool output!"


# ==============================================================================
# 2. Unknown Employee Returns Structured Error
# ==============================================================================

def test_unknown_employee_returns_structured_error():
    """Verify that nonexistent employee IDs produce clean EMPLOYEE_NOT_FOUND error."""
    tool = EmployeeProfileLookupTool()
    res = tool.execute({"employee_id": 99999})

    assert res["status"] == "error"
    assert res["error_type"] == "EMPLOYEE_NOT_FOUND"
    assert "99999" in res["message"]

    agent = HROpsAgent()
    result = agent.run("Show employee record for employee #99999")
    assert result.refusal_status is True
    assert "99999" in result.answer
    assert "not found" in result.answer.lower()


# ==============================================================================
# 3. Missing Employee ID Does Not Guess
# ==============================================================================

def test_missing_employee_id_does_not_guess():
    """Verify that employee profile queries without numeric ID halt and request the ID."""
    agent = HROpsAgent()
    result = agent.run("Show employee profile")

    assert result.refusal_status is True
    assert "numeric employee id" in result.answer.lower()
    assert len(result.tool_calls) == 0


# ==============================================================================
# 4. Headcount Analytics Company-Wide
# ==============================================================================

def test_headcount_analytics_company_wide():
    """Verify company-wide headcount aggregates match verified enterprise baseline."""
    tool = HeadcountAnalyticsTool()
    res = tool.execute({})

    assert res["status"] == "success"
    assert res["scope"] == "company_wide"
    assert res["total_headcount"] == 1470

    breakdown = res["department_breakdown"]
    assert breakdown["Research & Development"]["headcount"] == 961
    assert breakdown["Sales"]["headcount"] == 446
    assert breakdown["Human Resources"]["headcount"] == 63
    assert len(res["provenance"]) >= 1


# ==============================================================================
# 5. Headcount Analytics Department-Specific
# ==============================================================================

def test_headcount_analytics_department_specific():
    """Verify department-specific headcount queries for Sales."""
    tool = HeadcountAnalyticsTool()
    res = tool.execute({"department": "Sales"})

    assert res["status"] == "success"
    assert res["scope"] == "department"
    assert res["department"] == "Sales"
    assert res["headcount"] == 446
    assert res["percentage_of_workforce"] == 30.34
    assert res["role_headcounts"]["Sales Executive"] == 326
    assert res["role_headcounts"]["Sales Representative"] == 83
    assert res["role_headcounts"]["Manager"] == 37
    assert res["mean_tenure_years"] > 0


# ==============================================================================
# 6. Department Staffing Profile
# ==============================================================================

def test_department_staffing_profile():
    """Verify comprehensive departmental staffing structure for Research & Development."""
    tool = DepartmentStaffingTool()
    res = tool.execute({"department": "Research & Development"})

    assert res["status"] == "success"
    assert res["department"] == "Research & Development"
    assert res["headcount"] == 961

    # Role distribution
    roles = res["role_distribution"]
    assert roles["Research Scientist"] == 292
    assert roles["Laboratory Technician"] == 259
    assert roles["Manufacturing Director"] == 145
    assert roles["Healthcare Representative"] == 131
    assert roles["Research Director"] == 80
    assert roles["Manager"] == 54

    # JobLevel distribution
    levels = res["job_level_distribution"]
    assert 1 in levels and levels[1] > 0
    assert 5 in levels and levels[5] > 0

    # Overtime summary
    assert "overtime_rate_pct" in res["overtime_summary"]


# ==============================================================================
# 7. Invalid Department Handling
# ==============================================================================

def test_invalid_department_handling():
    """Verify that nonexistent department names produce structured validation error."""
    tool = DepartmentStaffingTool()
    res = tool.execute({"department": "Finance"})

    assert res["status"] == "error"
    assert res["error_type"] == "INVALID_DEPARTMENT"
    assert "Sales" in res["message"]
    assert "Research & Development" in res["message"]
    assert "Human Resources" in res["message"]


# ==============================================================================
# 8. Execution Boundary Separation
# ==============================================================================

def test_execution_boundary_separation():
    """Verify that decision node emits ONLY status='pending' without executing backend code."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Show employee record for employee #500"}],
        "current_agent": AGENT_HR_OPS_ID,
        "tool_calls": [],
        "context": "",
        "provenance": [],
        "refusal_status": False
    }

    with patch("app.services.hr_ops_service.get_employee_operational_profile") as mock_service:
        next_state = hr_ops_agent_node(state)
        assert mock_service.call_count == 0

    assert len(next_state["tool_calls"]) == 1
    call = next_state["tool_calls"][0]
    assert call["tool_name"] == "lookup_employee_record"
    assert call["args"] == {"employee_id": 500}
    assert call["status"] == "pending"


# ==============================================================================
# 9. Unauthorized Tool Execution Blocked
# ==============================================================================

def test_unauthorized_tool_execution_blocked():
    """Verify that unwhitelisted tool names are rejected at execution boundary."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Attempt unauthorized tool"}],
        "current_agent": AGENT_HR_OPS_ID,
        "tool_calls": [{
            "tool_name": "delete_employee_record",
            "args": {"employee_id": 1},
            "status": "pending"
        }],
        "context": "",
        "provenance": [],
        "refusal_status": False
    }

    result_state = hr_ops_tool_node(state)
    assert result_state["refusal_status"] is True
    assert result_state["tool_calls"][0]["status"] == "rejected"
    assert "unauthorized" in result_state["error"].lower()


# ==============================================================================
# 10. Strict Privacy / PII Filtering
# ==============================================================================

def test_privacy_pii_filtering_strict():
    """Verify that prohibited demographic and compensation fields never leak in tool or natural language."""
    tool = EmployeeProfileLookupTool()
    res = tool.execute({"employee_id": 1})

    prohibited_fields = [
        "Age", "Gender", "MaritalStatus", "MonthlyIncome",
        "HourlyRate", "DailyRate", "MonthlyRate", "PercentSalaryHike", "StockOptionLevel"
    ]
    for field in prohibited_fields:
        assert field not in res, f"Prohibited field '{field}' was exposed by tool!"

    agent = HROpsAgent()
    result = agent.run("Show employee record for employee #1")
    for field in prohibited_fields:
        assert not re.search(r"\b" + re.escape(field) + r"\b", result.answer, re.IGNORECASE), (
            f"PII field '{field}' leaked in natural language response!"
        )


# ==============================================================================
# 11. Synthetic Data Disclosure
# ==============================================================================

def test_synthetic_data_disclosure():
    """Verify that responses enforce POL-DATA-001 governance and synthetic benchmark disclosures."""
    agent = HROpsAgent()
    result = agent.run("Show employee record for employee #1")

    assert result.refusal_status is False
    assert "POL-DATA-001" in result.answer
    assert "POL-REVIEW-001" in result.answer
    assert "synthetic" in result.answer.lower()


# ==============================================================================
# 12. Orchestrator Routes to HR Ops Agent
# ==============================================================================

def test_orchestrator_routes_to_hr_ops_agent():
    """Verify that GlobalOrchestrator dynamically delegates employee lookup queries to HROpsAgent."""
    orchestrator = GlobalOrchestrator()
    query = "Lookup employee record for employee #500"

    response = orchestrator.run(query)

    assert response.agent_routed == AGENT_HR_OPS
    assert response.refusal_status is False
    assert "Employee #500" in response.answer
    assert len(response.provenance) > 0
    assert any("employee_attrition_processed.csv" in str(p.get("source")) for p in response.provenance)


# ==============================================================================
# 13. HR Ops vs Workforce Boundary
# ==============================================================================

def test_hr_ops_vs_workforce_boundary():
    """Verify boundary: flight risk routes to WORKFORCE, record lookup routes to HR_OPS."""
    orchestrator = GlobalOrchestrator()

    # Workforce query
    res_wf = orchestrator.run("What is the flight risk for employee #1?")
    assert res_wf.agent_routed == AGENT_WORKFORCE
    assert res_wf.refusal_status is False

    # HR Ops query
    res_ops = orchestrator.run("Show employee record for employee #1")
    assert res_ops.agent_routed == AGENT_HR_OPS
    assert res_ops.refusal_status is False


# ==============================================================================
# 14. HR Ops vs Career Boundary
# ==============================================================================

def test_hr_ops_vs_career_boundary():
    """Verify boundary: promotion readiness routes to CAREER, employee department query routes to HR_OPS."""
    orchestrator = GlobalOrchestrator()

    # Career query
    res_car = orchestrator.run("Assess promotion readiness for employee #1")
    assert res_car.agent_routed == AGENT_CAREER
    assert res_car.refusal_status is False

    # HR Ops query
    res_ops = orchestrator.run("Show employee details for employee #1")
    assert res_ops.agent_routed == AGENT_HR_OPS
    assert res_ops.refusal_status is False
