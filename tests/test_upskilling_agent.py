"""
tests/test_upskilling_agent.py
==============================
Comprehensive test suite for the Upskilling Specialist Agent (Agent 3 of 5).

Verifies:
1. Employee recommendation tool execution (Employee #1: Sales Executive, 3 recs, MEDIUM).
2. Manager exclusion guidance (Employee #23: Manager, department-level guidance, no crash).
3. Unknown employee handling (Employee #99999: structured refusal, EMPLOYEE_NOT_FOUND).
4. Missing employee ID does not guess (structured request for ID).
5. Organization skill gaps tool (33 skills, severity breakdown, top affected roles).
6. Execution boundary separation (planning node emits only status="pending").
7. Unauthorized tool execution blocked (arbitrary tool names rejected).
8. Privacy / PII filtering (no Age, Gender, MaritalStatus, MonthlyIncome, HourlyRate).
9. Developmental policy caveat (POL-LEARN-001, POL-REVIEW-001: developmental, human review).
10. Synthetic skill data transparency (explicit disclosure of synthetic MVP skill data).
11. Orchestrator dynamic routing to Upskilling Agent.
12. Policy and Workforce agent regression protection.
"""

import re
import pytest
from unittest.mock import patch

from app.agents.state import AgentState
from app.agents.tools.upskilling_tool import (
    EmployeeUpskillingTool,
    OrganizationSkillGapTool,
    UPSKILLING_AUTHORIZED_TOOLS
)
from app.agents.upskilling_agent import (
    UpskillingAgent,
    get_upskilling_agent,
    upskilling_agent_node,
    upskilling_tool_node,
    upskilling_response_node,
    UpskillingAgentResult,
    AGENT_UPSKILLING_ID
)
from app.agents.orchestrator import (
    GlobalOrchestrator,
    AGENT_POLICY,
    AGENT_WORKFORCE,
    AGENT_UPSKILLING
)


# ==============================================================================
# 1. Employee Recommendation Tool Execution
# ==============================================================================

def test_employee_recommendation_tool_execution():
    """Verify that EmployeeUpskillingTool retrieves real records for Employee #1."""
    tool = EmployeeUpskillingTool()
    res = tool.execute({"employee_id": 1})

    assert res["status"] == "success"
    assert res["EmployeeNumber"] == 1
    assert res["JobRole"] == "Sales Executive"
    assert res["severity"] == "MEDIUM"
    assert len(res["recommended_courses"]) == 3
    assert len(res["missing_skills"]) > 0
    assert res["is_manager"] is False
    assert len(res["provenance"]) >= 2
    assert "data/processed/employee_recommendations.csv" in str(res["provenance"])


# ==============================================================================
# 2. Manager Exclusion Guidance
# ==============================================================================

def test_manager_exclusion_guidance():
    """Verify that Manager employees receive proper guidance without raising errors."""
    tool = EmployeeUpskillingTool()
    # Employee #23 is a known Manager in employee_intelligence.csv
    res = tool.execute({"employee_id": 23})

    assert res["status"] == "success"
    assert res["EmployeeNumber"] == 23
    assert res["JobRole"] == "Manager"
    assert res["is_manager"] is True
    assert "N/A - Manager" in res["severity"]
    assert any("department-level" in str(c).lower() for c in res["recommended_courses"]) or len(res["missing_skills"]) == 0

    # Also test end-to-end response for Manager
    agent = UpskillingAgent()
    result = agent.run("What courses are recommended for employee #23?")
    assert result.refusal_status is False
    assert "Managerial Protocol Notice" in result.answer
    assert "POL-LEARN-001" in result.answer
    assert "department-level" in result.answer.lower()


# ==============================================================================
# 3. Unknown Employee Handling
# ==============================================================================

def test_unknown_employee_returns_structured_refusal():
    """Verify that nonexistent employee IDs produce clean structured refusals."""
    tool = EmployeeUpskillingTool()
    res = tool.execute({"employee_id": 99999})

    assert res["status"] == "error"
    assert res["error_type"] == "EMPLOYEE_NOT_FOUND"
    assert "99999" in res["message"]

    # Test through agent facade
    agent = UpskillingAgent()
    result = agent.run("What courses should employee #99999 take?")
    assert result.refusal_status is True
    assert "99999" in result.answer
    assert "not found" in result.answer.lower()


# ==============================================================================
# 4. Missing Employee ID Does Not Guess
# ==============================================================================

def test_missing_employee_id_does_not_guess():
    """Verify that queries with unstated employee ID prompt for an ID rather than guessing."""
    agent = UpskillingAgent()
    result = agent.run("Recommend courses for the employee")

    assert result.refusal_status is True
    assert "numeric employee id" in result.answer.lower()
    assert len(result.tool_calls) == 0


# ==============================================================================
# 5. Organization Skill Gaps Tool
# ==============================================================================

def test_organization_skill_gaps_tool():
    """Verify that OrganizationSkillGapTool aggregates enterprise capability gaps."""
    tool = OrganizationSkillGapTool()
    res = tool.execute({"limit": 10})

    assert res["status"] == "success"
    assert res["total_skills_evaluated"] == 33
    assert "HIGH" in res["severity_breakdown"]
    assert "MEDIUM" in res["severity_breakdown"]
    assert "LOW" in res["severity_breakdown"]
    assert len(res["top_skill_gaps"]) == 10
    assert "top_affected_roles" in res["top_skill_gaps"][0]
    assert "Speaking" in [g["skill_name"] for g in res["top_skill_gaps"]]


# ==============================================================================
# 6. Execution Boundary Separation
# ==============================================================================

def test_execution_boundary_separation():
    """Verify that the decision node emits ONLY status='pending' and never invokes tools."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Recommend training courses for employee #1"}],
        "current_agent": AGENT_UPSKILLING_ID,
        "tool_calls": [],
        "context": "",
        "provenance": [],
        "refusal_status": False
    }

    with patch("app.services.recommendation_service.get_employee_recommendations") as mock_service:
        next_state = upskilling_agent_node(state)
        # Decision node must NOT call the service
        assert mock_service.call_count == 0

    assert len(next_state["tool_calls"]) == 1
    call = next_state["tool_calls"][0]
    assert call["tool_name"] == "get_employee_upskilling_recommendations"
    assert call["args"] == {"employee_id": 1}
    assert call["status"] == "pending"


# ==============================================================================
# 7. Unauthorized Tool Execution Blocked
# ==============================================================================

def test_unauthorized_tool_blocked():
    """Verify that arbitrary or unwhitelisted tools are rejected at execution boundary."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Inject malicious tool"}],
        "current_agent": AGENT_UPSKILLING_ID,
        "tool_calls": [{
            "tool_name": "delete_all_employee_records",
            "args": {"target": "all"},
            "status": "pending"
        }],
        "context": "",
        "provenance": [],
        "refusal_status": False
    }

    result_state = upskilling_tool_node(state)
    assert result_state["refusal_status"] is True
    assert result_state["tool_calls"][0]["status"] == "rejected"
    assert "not authorized" in result_state["error"].lower()


# ==============================================================================
# 8. Privacy / PII Filtering
# ==============================================================================

def test_privacy_pii_filtering():
    """Verify that individual upskilling responses contain zero non-workforce PII."""
    tool = EmployeeUpskillingTool()
    res = tool.execute({"employee_id": 1})

    prohibited_fields = ["Age", "Gender", "MaritalStatus", "MonthlyIncome", "HourlyRate", "DailyRate"]
    for field in prohibited_fields:
        assert field not in res, f"Prohibited PII field '{field}' was exposed by tool output!"

    agent = UpskillingAgent()
    result = agent.run("What courses should employee #1 take?")
    for field in prohibited_fields:
        assert not re.search(r"\b" + re.escape(field) + r"\b", result.answer, re.IGNORECASE), (
            f"PII '{field}' leaked in natural language response!"
        )


# ==============================================================================
# 9. Developmental Policy Caveat
# ==============================================================================

def test_developmental_policy_caveat():
    """Verify that responses enforce POL-LEARN-001 and POL-REVIEW-001 developmental guardrails."""
    agent = UpskillingAgent()
    result = agent.run("What courses are recommended for employee #100?")

    assert result.refusal_status is False
    assert "POL-LEARN-001" in result.answer
    assert "developmental" in result.answer.lower()
    assert "human review" in result.answer.lower() or "manager-employee" in result.answer.lower()
    assert "adverse" in result.answer.lower() or "disciplinary" in result.answer.lower()


# ==============================================================================
# 10. Synthetic Skill Data Transparency
# ==============================================================================

def test_synthetic_skill_data_transparency():
    """Verify that employee skill gap responses disclose synthetic MVP skill inventory data."""
    agent = UpskillingAgent()
    result = agent.run("What skills should employee #1 improve?")

    assert result.refusal_status is False
    assert "synthetic" in result.answer.lower()
    assert "mvp" in result.answer.lower() or "demonstration" in result.answer.lower()


# ==============================================================================
# 11. Orchestrator Routes to Upskilling Agent
# ==============================================================================

def test_orchestrator_routes_to_upskilling_agent():
    """Verify that GlobalOrchestrator dynamically delegates upskilling queries to UpskillingAgent."""
    orchestrator = GlobalOrchestrator()
    query = "What courses are recommended for employee #100?"

    response = orchestrator.run(query)

    assert response.agent_routed == AGENT_UPSKILLING
    assert response.refusal_status is False
    assert len(response.answer) > 0
    assert "POL-LEARN-001" in response.answer
    assert len(response.provenance) > 0
    assert any("employee_recommendations.csv" in str(p.get("source")) for p in response.provenance)


# ==============================================================================
# 12. Policy and Workforce Agent Regression Protection
# ==============================================================================

def test_policy_and_workforce_agent_regression():
    """Verify that Policy Agent and Workforce Agent continue functioning with zero degradation."""
    orchestrator = GlobalOrchestrator()

    # 1. Policy Query
    policy_query = "What does POL-MODEL-001 say about the decision threshold?"
    pol_res = orchestrator.run(policy_query)
    assert pol_res.agent_routed == AGENT_POLICY
    assert pol_res.refusal_status is False
    assert "0.40" in pol_res.answer or "threshold" in pol_res.answer.lower()

    # 2. Workforce Query
    workforce_query = "What is the flight risk for employee #1?"
    wf_res = orchestrator.run(workforce_query)
    assert wf_res.agent_routed == AGENT_WORKFORCE
    assert wf_res.refusal_status is False
    assert "high" in wf_res.answer.lower()
    assert "0.40" in wf_res.answer
