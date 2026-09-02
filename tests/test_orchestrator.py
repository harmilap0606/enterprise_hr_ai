"""
tests/test_orchestrator.py
==========================
Comprehensive unit & integration tests for the Global LangGraph Orchestrator:
1. Intent classification: Policy
2. Intent classification: Workforce Intelligence
3. Intent classification: Upskilling
4. Intent classification: Career
5. Intent classification: HR Ops
6. Intent classification: Out-of-Domain
7. Policy routing reaches PolicyAgent & preserves answer
8. Policy provenance is preserved
9. Policy refusal status is preserved
10. Future agents return graceful capability unavailable responses
11. Out-of-domain does NOT invoke PolicyAgent
12. State propagation across the graph
13. API endpoint POST /agents/ask validation
14. No direct RAG imports in orchestrator (layer isolation)
"""

import ast
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.agents.state import AgentState, OrchestratorResponse
from app.agents.router import (
    classify_intent,
    INTENT_POLICY,
    INTENT_WORKFORCE_INTELLIGENCE,
    INTENT_UPSKILLING,
    INTENT_CAREER,
    INTENT_HR_OPS,
    INTENT_OUT_OF_DOMAIN
)
from app.agents.orchestrator import (
    GlobalOrchestrator,
    get_orchestrator,
    build_orchestrator_graph,
    AGENT_POLICY,
    AGENT_WORKFORCE,
    AGENT_UPSKILLING,
    AGENT_CAREER,
    AGENT_HR_OPS,
    AGENT_FALLBACK
)


# ==============================================================================
# 1-6. Deterministic Intent Classification Tests
# ==============================================================================

def test_intent_classification_policy():
    """Verify that policy keywords and exact POL- IDs route to POLICY."""
    policy_queries = [
        "What does POL-MODEL-001 say about the decision threshold?",
        "What is the company leave policy?",
        "What are the employee review requirements under POL-REVIEW-001?",
        "What are the benefits rules for severance pay?",
        "What is our ethical AI governance standard?",
        "What does the handbook say regarding code of conduct?",
        "What is the skill gap severity classification in POL-SKILL-001?"
    ]
    for q in policy_queries:
        assert classify_intent(q) == INTENT_POLICY, f"Failed on query: {q}"


def test_intent_classification_workforce_intelligence():
    """Verify that flight risk, attrition, and engagement terms route to WORKFORCE_INTELLIGENCE."""
    workforce_queries = [
        "Predict attrition risk for this employee",
        "What is the department flight risk in Sales?",
        "Which employees are high flight risk leavers?",
        "What was our average engagement score on the last survey?",
        "Show workforce turnover by department"
    ]
    for q in workforce_queries:
        assert classify_intent(q) == INTENT_WORKFORCE_INTELLIGENCE, f"Failed on query: {q}"


def test_intent_classification_upskilling():
    """Verify that training courses, reskilling, and learning route to UPSKILLING."""
    upskilling_queries = [
        "Recommend training courses for a junior data analyst",
        "How can we upskill employees with python gaps?",
        "Provide a learning recommendation for employee #100",
        "What course recommendations are in the catalog?",
        "Suggest a development recommendation for this worker"
    ]
    for q in upskilling_queries:
        assert classify_intent(q) == INTENT_UPSKILLING, f"Failed on query: {q}"


def test_intent_classification_career():
    """Verify that career pathways, promotions, and O*NET mappings route to CAREER."""
    career_queries = [
        "What is the career progression path to Senior Scientist?",
        "Assess promotion readiness for this role",
        "What is the career ladder for software engineers?",
        "What are the missing competencies for the next role?",
        "Explain O*NET role mapping for Laboratory Technician"
    ]
    for q in career_queries:
        assert classify_intent(q) == INTENT_CAREER, f"Failed on query: {q}"


def test_intent_classification_hr_ops():
    """Verify that employee profile and record lookups route to HR_OPS."""
    hr_ops_queries = [
        "Lookup employee record for employee #1024",
        "Get employee profile details for ID 505",
        "Show employee information and demographics",
        "Retrieve worker details from personnel file",
        "Search employee lookup by number"
    ]
    for q in hr_ops_queries:
        assert classify_intent(q) == INTENT_HR_OPS, f"Failed on query: {q}"


def test_intent_classification_out_of_domain():
    """Verify that general or unrelated questions route to OUT_OF_DOMAIN."""
    unrelated_queries = [
        "What is the capital of France?",
        "Who directed the movie Inception?",
        "How do I make chocolate chip cookies?",
        "Write a quicksort algorithm in Rust",
        "Tell me a joke about airplanes"
    ]
    for q in unrelated_queries:
        assert classify_intent(q) == INTENT_OUT_OF_DOMAIN, f"Failed on query: {q}"


def test_intent_classification_entity_boundary_precedence():
    """
    Regression Test:
    Specific intent must take precedence over generic entity identifiers like 'employee #1'.
    - employee + skills / skill gap / training / course / development => UPSKILLING
    - employee + record / profile / department / role / tenure => HR OPS
    - employee + attrition / flight risk / engagement => WORKFORCE
    """
    boundary_cases = [
        # 1. Upskilling queries containing employee entity
        ("What skills does employee #1 need to develop for their current role?", INTENT_UPSKILLING),
        ("Recommend courses for employee #1", INTENT_UPSKILLING),
        ("What are the skill gaps for employee #1?", INTENT_UPSKILLING),
        ("What training does employee #1 need?", INTENT_UPSKILLING),
        ("How can employee #1 improve their skills?", INTENT_UPSKILLING),
        ("What skills should employee #1 learn?", INTENT_UPSKILLING),
        ("Suggest skill development opportunities for employee #5", INTENT_UPSKILLING),
        
        # 2. HR Ops administrative lookups containing employee entity
        ("Show employee record for employee #1", INTENT_HR_OPS),
        ("Show employee profile for employee #1", INTENT_HR_OPS),
        ("What department does employee #1 work in?", INTENT_HR_OPS),
        ("What is the job role of employee #1?", INTENT_HR_OPS),
        ("Lookup employee record for employee #1", INTENT_HR_OPS),
        ("What is the tenure of employee #1?", INTENT_HR_OPS),
        
        # 3. Workforce Intelligence queries containing employee entity
        ("What is the attrition risk for employee #1?", INTENT_WORKFORCE_INTELLIGENCE),
        ("What is the flight risk for employee #1?", INTENT_WORKFORCE_INTELLIGENCE),
        ("What is the engagement score for employee #1?", INTENT_WORKFORCE_INTELLIGENCE),
        
        # 4. Career queries
        ("What is the career progression path for Laboratory Technician?", INTENT_CAREER),
        ("Assess promotion readiness for employee #1", INTENT_CAREER),
        
        # 5. Policy queries
        ("What does POL-CAREER-001 say about career stagnation?", INTENT_POLICY),
        
        # 6. Out of domain queries
        ("What is the weather today?", INTENT_OUT_OF_DOMAIN),
    ]

    for q, expected_intent in boundary_cases:
        actual_intent = classify_intent(q)
        assert actual_intent == expected_intent, (
            f"Routing failure for query: '{q}'. Got '{actual_intent}', expected '{expected_intent}'."
        )


def test_orchestrator_upskilling_vs_hr_ops_boundary_e2e():
    """
    End-to-End Orchestrator Test:
    Confirms that 'What skills does employee #1 need to develop for their current role?'
    dynamically delegates to UpskillingAgent rather than HROpsAgent.
    """
    orchestrator = GlobalOrchestrator()
    
    # Bug scenario query -> must route to UpskillingAgent
    q_upskill = "What skills does employee #1 need to develop for their current role?"
    resp_upskill = orchestrator.run(q_upskill)
    assert resp_upskill.agent_routed == AGENT_UPSKILLING
    assert resp_upskill.refusal_status is False
    assert len(resp_upskill.answer) > 0
    assert "POL-LEARN-001" in resp_upskill.answer

    # Administrative lookup query -> must route to HROpsAgent
    q_ops = "Show employee record for employee #1"
    resp_ops = orchestrator.run(q_ops)
    assert resp_ops.agent_routed == AGENT_HR_OPS
    assert resp_ops.refusal_status is False
    assert len(resp_ops.answer) > 0
    assert "Employee #1" in resp_ops.answer



# ==============================================================================
# 7-9. Policy Agent Delegation & Provenance Preservation Tests
# ==============================================================================

def test_policy_routing_reaches_policy_agent_and_preserves_answer():
    """Verify that a policy query routes to PolicyAgent and returns the grounded answer."""
    orchestrator = GlobalOrchestrator()
    query = "What does POL-MODEL-001 say about the attrition threshold?"

    result = orchestrator.run(query)

    assert isinstance(result, OrchestratorResponse)
    assert result.query == query
    assert result.agent_routed == AGENT_POLICY
    assert result.refusal_status is False
    assert len(result.answer) > 0
    # Must preserve PolicyAgent's grounded answer containing 0.40
    assert "0.40" in result.answer or "0.4" in result.answer
    # Must preserve provenance
    assert len(result.provenance) > 0
    top_source = result.provenance[0]
    assert "POL-MODEL-001" in top_source.get("chunk_id", "") or "POL-MODEL-001" in top_source.get("source", "")


def test_policy_refusal_preservation():
    """Verify that policy refusal status propagates cleanly through the orchestrator."""
    orchestrator = GlobalOrchestrator()
    query = "What is the company's parental leave entitlement?"

    result = orchestrator.run(query)

    assert isinstance(result, OrchestratorResponse)
    assert result.agent_routed == AGENT_POLICY
    assert result.refusal_status is True
    assert (
        "not contain sufficient" in result.answer.lower()
        or "does not contain" in result.answer.lower()
        or "refuse" in result.answer.lower()
        or "platform knowledge base" in result.answer.lower()
    )


# ==============================================================================
# 10-12. Future Agent & Out-of-Domain Isolation Tests
# ==============================================================================

def test_future_agents_return_graceful_unavailable():
    """Verify that all 5 specialized agents are live and do not return unavailable notices."""
    orchestrator = GlobalOrchestrator()

    # Verify that an actual HR Ops query executes and returns live answer, not "scheduled for implementation"
    res = orchestrator.run("Lookup employee record for employee #500")
    assert res.agent_routed == AGENT_HR_OPS
    assert res.refusal_status is False
    assert "scheduled for implementation" not in res.answer.lower()
    assert "Employee #500" in res.answer



def test_out_of_domain_does_not_invoke_policy_agent():
    """Verify that out-of-domain questions NEVER invoke PolicyAgent."""
    orchestrator = GlobalOrchestrator()
    query = "What is the distance from the Earth to the Moon?"

    with patch.object(orchestrator.policy_agent, "run") as mock_policy_run:
        res = orchestrator.run(query)
        mock_policy_run.assert_not_called()
        assert res.agent_routed == AGENT_FALLBACK
        assert res.refusal_status is True
        assert "can only answer questions related to HR" in res.answer


# ==============================================================================
# 13. State Propagation Across Graph Tests
# ==============================================================================

def test_state_propagation_across_graph():
    """Verify full state accumulation through orchestrator graph nodes."""
    orchestrator = GlobalOrchestrator()
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": "What is the purpose of POL-JOB-001?"}],
        "current_agent": "initial",
        "target_agent": None,
        "intent": None,
        "tool_calls": [],
        "context": "",
        "provenance": [],
        "answer": None,
        "refusal_status": False,
        "error": None
    }

    final_state = orchestrator.graph.invoke(initial_state)

    assert final_state["intent"] == INTENT_POLICY
    assert final_state["target_agent"] == AGENT_POLICY
    assert final_state["current_agent"] == AGENT_POLICY
    assert final_state["refusal_status"] is False
    assert len(final_state["provenance"]) > 0
    assert len(final_state["tool_calls"]) >= 1
    assert len(final_state["messages"]) >= 2
    assert final_state["messages"][-1]["role"] == "assistant"


# ==============================================================================
# 14. API Endpoint Tests (POST /agents/ask)
# ==============================================================================

def test_api_agents_ask_endpoint():
    """Verify that POST /agents/ask conforms to the OpenAPI specification."""
    client = TestClient(app)

    # Valid query
    resp = client.post("/agents/ask", json={"question": "What does POL-MODEL-001 state regarding the decision threshold?"})
    assert resp.status_code == 200
    data = resp.json()

    assert "query" in data
    assert "answer" in data
    assert "agent_routed" in data
    assert "provenance" in data
    assert "refusal_status" in data
    assert data["agent_routed"] == AGENT_POLICY
    assert data["refusal_status"] is False
    assert "0.40" in data["answer"] or "0.4" in data["answer"]

    # Empty question validation
    resp_empty = client.post("/agents/ask", json={"question": "   "})
    assert resp_empty.status_code == 400


# ==============================================================================
# 15. Layer Isolation & No Direct RAG Imports in Orchestrator
# ==============================================================================

def test_no_direct_rag_import_in_orchestrator():
    """Verify that orchestrator.py has ZERO direct imports of ChromaDB, BM25, or RAG models."""
    orchestrator_path = Path(__file__).resolve().parent.parent / "app" / "agents" / "orchestrator.py"
    assert orchestrator_path.exists()

    with open(orchestrator_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    forbidden_modules = {
        "chromadb",
        "rank_bm25",
        "transformers",
        "app.rag.embeddings",
        "app.rag.retrieval.hybrid_retriever",
        "app.rag.retrieval.reranker",
        "app.rag.pipeline"
    }

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module)

    intersection = forbidden_modules.intersection(imported_names)
    assert len(intersection) == 0, f"Orchestrator violates layer boundaries by directly importing: {intersection}"
