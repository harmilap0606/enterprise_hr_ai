"""
tests/test_career_agent.py
==========================
Comprehensive test suite for the Career Specialist Agent (Agent 4 of 5).

Verifies:
1. Role career pathway tool execution (Laboratory Technician -> Research Scientist, O*NET 29-2012.00).
2. Employee promotion readiness on-track assessment (Employee #4: clean metrics, ON_TRACK).
3. Promotion stagnation detection (Employee #13: YearsSinceLastPromotion >= 4, STAGNANT, review required).
4. Manager role governance handling (Rule 4, placeholder 11-9199.00, very_low confidence).
5. Role competency comparison tool (Sales Representative -> Sales Executive delta & score).
6. Unknown employee handling (Employee #99999: structured refusal, EMPLOYEE_NOT_FOUND).
7. Missing employee ID does not guess (structured request for ID).
8. Execution boundary separation (decision node emits only status="pending").
9. Unauthorized tool execution blocked (arbitrary tool names rejected).
10. Privacy / PII filtering (no Age, Gender, MaritalStatus, MonthlyIncome, HourlyRate).
11. Developmental policy caveats (POL-CAREER-001, POL-REVIEW-001: advisory, human review).
12. Orchestrator dynamic routing to Career Agent.
13. Boundary tests across Career, Upskilling, Workforce, and Policy agents.
"""

import re
import pytest
from unittest.mock import patch

from app.agents.state import AgentState
from app.agents.tools.career_tool import (
    RoleCareerPathwayTool,
    EmployeePromotionReadinessTool,
    RoleCompetencyComparisonTool,
    CAREER_AUTHORIZED_TOOLS
)
from app.agents.career_agent import (
    CareerAgent,
    get_career_agent,
    career_agent_node,
    career_tool_node,
    career_response_node,
    CareerAgentResult,
    AGENT_CAREER_ID
)
from app.agents.orchestrator import (
    GlobalOrchestrator,
    AGENT_POLICY,
    AGENT_WORKFORCE,
    AGENT_UPSKILLING,
    AGENT_CAREER
)


# ==============================================================================
# 1. Role Career Pathway Tool Execution
# ==============================================================================

def test_role_career_pathway_tool():
    """Verify that RoleCareerPathwayTool retrieves canonical ladder and O*NET mappings."""
    tool = RoleCareerPathwayTool()
    res = tool.execute({"role_name": "Laboratory Technician"})

    assert res["status"] == "success"
    assert res["role_name"] == "Laboratory Technician"
    assert res["onet_soc_code"] == "29-2012.00"
    assert "Medical and Clinical Laboratory Technicians" in res["onet_title"]
    assert res["match_confidence"] == "medium"
    assert "Research Scientist" in res["vertical_pathways"]
    assert len(res["benchmark_competencies"]["essential_skills"]) > 0
    assert len(res["provenance"]) >= 2


# ==============================================================================
# 2. Employee Promotion Readiness On-Track Assessment
# ==============================================================================

def test_employee_promotion_readiness_on_track():
    """Verify clean promotion readiness evaluation for an on-track employee (Employee #4)."""
    tool = EmployeePromotionReadinessTool()
    res = tool.execute({"employee_id": 4})

    assert res["status"] == "success"
    assert res["EmployeeNumber"] == 4
    assert res["stagnation_status"] == "ON_TRACK"
    assert res["career_pathing_review_required"] is False
    assert res["YearsSinceLastPromotion"] <= 1
    assert res["YearsInCurrentRole"] <= 2
    assert "next_ladder_role" in res
    assert len(res["provenance"]) >= 1


# ==============================================================================
# 3. Promotion Stagnation Detection
# ==============================================================================

def test_promotion_stagnation_detection():
    """Verify that employees with >= 4 years without promotion trigger STAGNANT status."""
    tool = EmployeePromotionReadinessTool()
    # Employee #13 has YearsSinceLastPromotion = 7 (>= 4)
    res = tool.execute({"employee_id": 13})

    assert res["status"] == "success"
    assert res["EmployeeNumber"] == 13
    assert res["YearsSinceLastPromotion"] >= 4
    assert res["stagnation_status"] == "STAGNANT"
    assert res["career_pathing_review_required"] is True

    # End-to-end response check
    agent = CareerAgent()
    result = agent.run("Assess promotion readiness for employee #13")

    assert result.refusal_status is False
    assert "STAGNANT" in result.answer
    assert "POL-CAREER-001 Rule 3" in result.answer
    assert "Career Pathing Review" in result.answer


# ==============================================================================
# 4. Manager Role Governance Handling
# ==============================================================================

def test_manager_role_handling():
    """Verify that Manager role triggers Rule 4 placeholder protocol without error."""
    tool = RoleCareerPathwayTool()
    res = tool.execute({"role_name": "Manager"})

    assert res["status"] == "success"
    assert res["is_manager"] is True
    assert res["onet_soc_code"] == "11-9199.00"
    assert res["match_confidence"] == "very_low"
    assert any("Rule 4" in note for note in res["governance_notes"])

    agent = CareerAgent()
    result = agent.run("What is the career path for Manager?")
    assert result.refusal_status is False
    assert "11-9199.00" in result.answer
    assert "very_low" in result.answer.lower()
    assert "leadership assessment" in result.answer.lower()


# ==============================================================================
# 5. Role Competency Comparison Tool Execution
# ==============================================================================

def test_role_competency_comparison_tool():
    """Verify competency transferability comparison between Sales Rep and Sales Executive."""
    tool = RoleCompetencyComparisonTool()
    res = tool.execute({
        "current_role": "Sales Representative",
        "target_role": "Sales Executive"
    })

    assert res["status"] == "success"
    assert res["current_role"] == "Sales Representative"
    assert res["target_role"] == "Sales Executive"
    assert res["target_soc_code"] == "11-2022.00"
    assert len(res["target_description"]) > 10
    assert "Active Listening" in res["shared_essential_skills"]
    assert len(res["missing_essential_skills"]) > 0
    assert res["transferability_score"] > 0.0

    agent = CareerAgent()
    result = agent.run("Compare Sales Representative to Sales Executive")
    assert result.refusal_status is False
    assert "Transferability Overlap Score" in result.answer
    assert "Shared Essential Skills" in result.answer


# ==============================================================================
# 6. Unknown Employee Returns Structured Refusal
# ==============================================================================

def test_unknown_employee_returns_structured_refusal():
    """Verify that nonexistent employee IDs produce clean structured refusals."""
    tool = EmployeePromotionReadinessTool()
    res = tool.execute({"employee_id": 99999})

    assert res["status"] == "error"
    assert res["error_type"] == "EMPLOYEE_NOT_FOUND"
    assert "99999" in res["message"]

    agent = CareerAgent()
    result = agent.run("Assess promotion readiness for employee #99999")
    assert result.refusal_status is True
    assert "99999" in result.answer
    assert "not found" in result.answer.lower()


# ==============================================================================
# 7. Missing Employee ID Does Not Guess
# ==============================================================================

def test_missing_employee_id_does_not_guess():
    """Verify that promotion readiness requests without numeric ID halt with request."""
    agent = CareerAgent()
    result = agent.run("Assess promotion readiness for this employee")

    assert result.refusal_status is True
    assert "numeric employee id" in result.answer.lower()
    assert len(result.tool_calls) == 0


# ==============================================================================
# 8. Execution Boundary Separation
# ==============================================================================

def test_execution_boundary_separation():
    """Verify that decision node emits ONLY status='pending' and never invokes tools."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Assess promotion readiness for employee #1"}],
        "current_agent": AGENT_CAREER_ID,
        "tool_calls": [],
        "context": "",
        "provenance": [],
        "refusal_status": False
    }

    with patch("app.services.career_service.get_employee_promotion_readiness") as mock_service:
        next_state = career_agent_node(state)
        # Decision node must NOT call the service
        assert mock_service.call_count == 0

    assert len(next_state["tool_calls"]) == 1
    call = next_state["tool_calls"][0]
    assert call["tool_name"] == "get_employee_promotion_readiness"
    assert call["args"] == {"employee_id": 1}
    assert call["status"] == "pending"


# ==============================================================================
# 9. Unauthorized Tool Execution Blocked
# ==============================================================================

def test_unauthorized_tool_blocked():
    """Verify that unwhitelisted tool names are rejected at execution boundary."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "Attempt privilege escalation"}],
        "current_agent": AGENT_CAREER_ID,
        "tool_calls": [{
            "tool_name": "delete_career_records",
            "args": {"target": "all"},
            "status": "pending"
        }],
        "context": "",
        "provenance": [],
        "refusal_status": False
    }

    result_state = career_tool_node(state)
    assert result_state["refusal_status"] is True
    assert result_state["tool_calls"][0]["status"] == "rejected"
    assert "unauthorized" in result_state["error"].lower()


# ==============================================================================
# 10. Privacy / PII Filtering
# ==============================================================================

def test_privacy_pii_filtering():
    """Verify that career readiness responses contain zero non-workforce demographic PII."""
    tool = EmployeePromotionReadinessTool()
    res = tool.execute({"employee_id": 1})

    prohibited_fields = ["Age", "Gender", "MaritalStatus", "MonthlyIncome", "HourlyRate", "DailyRate"]
    for field in prohibited_fields:
        assert field not in res, f"Prohibited PII field '{field}' was exposed by tool output!"

    agent = CareerAgent()
    result = agent.run("Assess promotion readiness for employee #1")
    for field in prohibited_fields:
        assert not re.search(r"\b" + re.escape(field) + r"\b", result.answer, re.IGNORECASE), (
            f"PII '{field}' leaked in natural language response!"
        )


# ==============================================================================
# 11. Developmental Policy Caveat
# ==============================================================================

def test_developmental_policy_caveat():
    """Verify that responses enforce POL-CAREER-001 and POL-REVIEW-001 governance guardrails."""
    agent = CareerAgent()
    result = agent.run("What is the career progression path for Laboratory Technician?")

    assert result.refusal_status is False
    assert "POL-CAREER-001" in result.answer
    assert "POL-REVIEW-001" in result.answer
    assert "developmental" in result.answer.lower()
    assert "guaranteed" in result.answer.lower() or "promise" in result.answer.lower()


# ==============================================================================
# 12. Orchestrator Routes to Career Agent
# ==============================================================================

def test_orchestrator_routes_to_career_agent():
    """Verify that GlobalOrchestrator dynamically delegates career queries to CareerAgent."""
    orchestrator = GlobalOrchestrator()
    query = "What is the career progression path to Senior Scientist?"

    response = orchestrator.run(query)

    assert response.agent_routed == AGENT_CAREER
    assert response.refusal_status is False
    assert len(response.answer) > 0
    assert "POL-CAREER-001" in response.answer
    assert len(response.provenance) > 0
    assert any("jobrole_onet_mapping.csv" in str(p.get("source")) for p in response.provenance)


# ==============================================================================
# 13. Boundary & Precedence Tests
# ==============================================================================

def test_career_vs_upskilling_routing_boundary():
    """Verify that course/training queries route to UPSKILLING rather than CAREER."""
    orchestrator = GlobalOrchestrator()
    query = "Recommend training courses for employee #1"

    response = orchestrator.run(query)
    assert response.agent_routed == AGENT_UPSKILLING
    assert response.refusal_status is False


def test_career_vs_workforce_routing_boundary():
    """Verify that flight risk queries route to WORKFORCE rather than CAREER."""
    orchestrator = GlobalOrchestrator()
    query = "What is the flight risk for employee #1?"

    response = orchestrator.run(query)
    assert response.agent_routed == AGENT_WORKFORCE
    assert response.refusal_status is False


def test_policy_precedence_over_career():
    """Verify that explicit policy identifier queries route to POLICY rather than CAREER."""
    orchestrator = GlobalOrchestrator()
    query = "What does POL-CAREER-001 say about promotion stagnation?"

    response = orchestrator.run(query)
    assert response.agent_routed == AGENT_POLICY
    assert response.refusal_status is False
    assert len(response.provenance) > 0
    assert any("POL-CAREER-001" in str(p.get("title", "")) or "POL-CAREER-001" in str(p.get("source", "")) for p in response.provenance)
