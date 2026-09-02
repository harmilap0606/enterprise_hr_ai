"""
tests/test_workforce_agent.py
=============================
Comprehensive test suite for Workforce Intelligence Agent & Tools:
1. WorkforceKPITool execution (1470 total, 585 high risk, 2.95 engagement).
2. Engagement caveat preservation (731 respondents, 49.7% coverage).
3. DepartmentWorkforceTool execution (Sales, R&D, HR breakdown).
4. EmployeeRiskTool execution (Employee #1: prob 0.8976, HIGH).
5. Unknown employee ID handling (structured error/refusal, zero crash).
6. Missing employee ID guardrail (agent does not guess employee ID).
7. Execution boundary separation (planning node only emits 'pending').
8. Tool authorization check (unauthorized tools blocked).
9. End-to-end WorkforceAgent.run() returns WorkforceAgentResult.
10. Orchestrator routing integration (routes workforce questions to workforce_intelligence_agent).
11. Policy Agent regression check (PolicyAgent remains 100% operational).
"""

import pytest
from app.agents.tools.workforce_tool import (
    WorkforceKPITool,
    DepartmentWorkforceTool,
    EmployeeRiskTool,
    WORKFORCE_AUTHORIZED_TOOLS
)
from app.agents.workforce_agent import (
    WorkforceAgent,
    WorkforceAgentResult,
    workforce_agent_node,
    workforce_tool_node,
    get_workforce_agent,
    AGENT_WORKFORCE_ID
)
from app.agents.orchestrator import GlobalOrchestrator, AGENT_WORKFORCE, AGENT_POLICY
from app.agents.state import AgentState


# ==============================================================================
# 1-3. Tool Unit Tests
# ==============================================================================

def test_workforce_kpi_tool_execution():
    """Verify organization KPI counts: 1470 employees, 585 high risk, 2.95 engagement."""
    tool = WorkforceKPITool.get_instance()
    res = tool.execute({})

    assert res["status"] == "success"
    data = res["data"]
    assert data["total_employees"] == 1470
    assert data["high_risk_count"] == 585
    assert data["average_engagement"] == 2.95
    assert data["survey_respondents"] == 731
    assert data["decision_threshold"] == 0.40
    assert len(res["provenance"]) >= 1


def test_engagement_caveat_preservation():
    """Verify that engagement caveat explicitly notes 731 respondents and 49.7% sample."""
    agent = WorkforceAgent()
    res = agent.run("What is our average employee engagement score?")

    assert isinstance(res, WorkforceAgentResult)
    assert res.refusal_status is False
    assert "2.95" in res.answer
    # Must preserve the 731 respondent caveat
    assert "731" in res.answer
    assert "49.7" in res.answer or "50%" in res.answer
    assert "739" in res.answer or "unmapped" in res.answer.lower() or "not be generalized" in res.answer.lower()


def test_department_workforce_tool_execution():
    """Verify that department tool returns metrics for Sales, R&D, and HR."""
    tool = DepartmentWorkforceTool.get_instance()
    res = tool.execute({})

    assert res["status"] == "success"
    depts = res["data"]["departments"]
    dept_names = {d["department"] for d in depts}
    assert "Sales" in dept_names
    assert "Research & Development" in dept_names
    assert "Human Resources" in dept_names

    # Test department filtering
    filtered_res = tool.execute({"department": "Sales"})
    filtered_depts = filtered_res["data"]["departments"]
    assert len(filtered_depts) == 1
    assert filtered_depts[0]["department"] == "Sales"


# ==============================================================================
# 4-6. Employee Risk & Guardrail Tests
# ==============================================================================

def test_employee_risk_lookup_employee_1():
    """Verify Employee #1 has probability 0.8976 and HIGH risk level."""
    tool = EmployeeRiskTool.get_instance()
    res = tool.execute({"employee_id": 1})

    assert res["status"] == "success"
    data = res["data"]
    assert data["EmployeeNumber"] == 1
    assert data["probability"] == 0.8976
    assert data["risk_level"] == "HIGH"
    assert data["decision_threshold"] == 0.40
    # Guardrail: PII fields like Age, Gender, MaritalStatus must NOT be present
    assert "Age" not in data
    assert "Gender" not in data
    assert "MaritalStatus" not in data


def test_unknown_employee_returns_structured_refusal():
    """Verify looking up an unknown employee ID returns structured refusal without crashing."""
    tool = EmployeeRiskTool.get_instance()
    res = tool.execute({"employee_id": 999999})

    assert res["status"] == "error"
    assert res["error_type"] == "EMPLOYEE_NOT_FOUND"
    assert "999999" in res["message"]

    # End-to-end via agent
    agent = WorkforceAgent()
    agent_res = agent.run("What is employee #999999's flight risk?")
    assert agent_res.refusal_status is True
    assert "not found" in agent_res.answer.lower()


def test_missing_employee_id_does_not_guess():
    """Verify that an individual employee query without an ID requests clarification without guessing."""
    agent = WorkforceAgent()
    res = agent.run("What is the employee flight risk?")

    assert res.refusal_status is True
    assert "please specify an explicit" in res.answer.lower()
    assert len(res.tool_calls) == 0


# ==============================================================================
# 7-8. Boundary & Authorization Tests
# ==============================================================================

def test_execution_boundary_separation():
    """Verify that the decision node ONLY emits status='pending' and does not execute the tool."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "What is our overall workforce flight risk?"}],
        "current_agent": AGENT_WORKFORCE_ID,
        "tool_calls": [],
        "context": "",
        "provenance": [],
        "answer": None,
        "refusal_status": False,
        "error": None
    }

    updated_state = workforce_agent_node(state)

    assert len(updated_state["tool_calls"]) == 1
    call = updated_state["tool_calls"][0]
    assert call["tool_name"] == WorkforceKPITool.name
    assert call["status"] == "pending"
    assert "result" not in call  # Proves execution has not occurred


def test_unauthorized_tool_execution_blocked():
    """Verify that an unauthorized tool name is rejected at the execution boundary."""
    state: AgentState = {
        "messages": [],
        "current_agent": AGENT_WORKFORCE_ID,
        "tool_calls": [
            {
                "tool_name": "unauthorized_sql_drop_table",
                "args": {},
                "status": "pending"
            }
        ],
        "context": "",
        "provenance": [],
        "answer": None,
        "refusal_status": False,
        "error": None
    }

    result_state = workforce_tool_node(state)

    assert result_state["refusal_status"] is True
    call = result_state["tool_calls"][0]
    assert call["status"] == "rejected"
    assert "not in WORKFORCE_AUTHORIZED_TOOLS" in call["error"]


# ==============================================================================
# 9-10. End-to-End & Orchestrator Integration Tests
# ==============================================================================

def test_workforce_agent_e2e_run():
    """Verify end-to-end execution of WorkforceAgent produces valid WorkforceAgentResult."""
    agent = WorkforceAgent()
    res = agent.run("Show workforce intelligence for Employee #1.")

    assert isinstance(res, WorkforceAgentResult)
    assert res.current_agent == AGENT_WORKFORCE_ID
    assert res.refusal_status is False
    assert "Employee #1" in res.answer
    assert "HIGH" in res.answer
    assert "0.40" in res.answer
    assert "human review is mandatory" in res.answer.lower()
    assert res.data is not None
    assert res.data["EmployeeNumber"] == 1
    assert len(res.provenance) >= 1
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["status"] == "completed"


def test_orchestrator_routes_to_workforce_agent():
    """Verify GlobalOrchestrator dynamically routes workforce queries to WorkforceAgent."""
    orchestrator = GlobalOrchestrator()
    res = orchestrator.run("What is our department flight risk in Sales?")

    assert res.agent_routed == AGENT_WORKFORCE
    assert res.refusal_status is False
    assert "Sales" in res.answer
    assert len(res.provenance) >= 1


def test_policy_agent_regression():
    """Verify Policy Agent remains fully operational and unaffected by Workforce additions."""
    orchestrator = GlobalOrchestrator()
    res = orchestrator.run("What does POL-MODEL-001 say about the decision threshold?")

    assert res.agent_routed == AGENT_POLICY
    assert res.refusal_status is False
    assert "0.40" in res.answer or "0.4" in res.answer
    assert len(res.provenance) >= 1
    top_source = res.provenance[0]
    assert "POL-MODEL-001" in top_source.get("chunk_id", "") or "POL-MODEL-001" in top_source.get("source", "")
