"""
tests/test_policy_agent.py
==========================
Unit & integration tests for the Policy Agent foundation:
1. Policy Agent state initialization & schema compliance
2. Policy retrieval tool execution
3. Policy collection isolation & boundary enforcement
4. Grounded policy answer generation
5. Unsupported policy query & refusal verification
6. Provenance propagation into AgentState
7. Tool execution boundary separation
"""

import pytest
from typing import Dict, Any

from app.agents.state import AgentState, PolicyAgentResult
from app.agents.tools.policy_tool import (
    PolicyRetrievalTool,
    POLICY_COLLECTION_NAME,
    FORBIDDEN_COLLECTION_NAME
)
from app.agents.policy_agent import (
    PolicyAgent,
    policy_agent_node,
    policy_tool_node,
    policy_response_node,
    build_policy_agent_graph
)
from app.rag.pipeline import REFUSAL_MESSAGE


# ==============================================================================
# 1. State Schema & Initialization Tests
# ==============================================================================

def test_agent_state_schema_fields():
    """Verify that AgentState supports all 7 required core fields."""
    state: AgentState = {
        "messages": [{"role": "user", "content": "What is POL-MODEL-001?"}],
        "current_agent": "policy_agent",
        "tool_calls": [{"tool_name": "search_hr_policies", "args": {"query": "test"}, "status": "pending"}],
        "context": "Retrieved context string",
        "provenance": [{"chunk_id": "c1", "source": "POL-MODEL-001.md"}],
        "answer": "Test answer",
        "refusal_status": False
    }

    assert state["current_agent"] == "policy_agent"
    assert len(state["messages"]) == 1
    assert len(state["tool_calls"]) == 1
    assert state["refusal_status"] is False
    assert state["context"] == "Retrieved context string"
    assert len(state["provenance"]) == 1
    assert state["answer"] == "Test answer"


def test_policy_agent_result_pydantic():
    """Verify serialization and validation of PolicyAgentResult."""
    res = PolicyAgentResult(
        query="What is the decision threshold?",
        answer="The decision threshold is 0.40.",
        provenance=[{"source": "POL-MODEL-001.md", "score": 4.5}],
        refusal_status=False,
        current_agent="policy_agent",
        tool_calls=[{"tool_name": "search_hr_policies", "status": "completed"}]
    )

    data = res.model_dump()
    assert data["query"] == "What is the decision threshold?"
    assert data["refusal_status"] is False
    assert data["current_agent"] == "policy_agent"
    assert len(data["provenance"]) == 1
    assert data["tool_calls"][0]["status"] == "completed"


# ==============================================================================
# 2. Collection Isolation Tests
# ==============================================================================

def test_policy_collection_isolation():
    """Verify that PolicyRetrievalTool is bound to enterprise_hr_policies_bge and rejects other collections."""
    tool = PolicyRetrievalTool.get_instance()
    assert tool.collection_name == POLICY_COLLECTION_NAME
    assert tool.collection_name == "enterprise_hr_policies_bge"
    assert tool.collection_name != FORBIDDEN_COLLECTION_NAME

    # Attempting to construct with forbidden collection raises AssertionError
    with pytest.raises(AssertionError) as exc_info:
        PolicyRetrievalTool(collection_name=FORBIDDEN_COLLECTION_NAME)
    assert "must use" in str(exc_info.value) or "must NOT use" in str(exc_info.value)


# ==============================================================================
# 3. Tool Execution Boundary & Separation Tests
# ==============================================================================

def test_tool_execution_boundary_separation():
    """Verify that agent decision node ONLY formulates tool requests without executing them."""
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": "What is the decision threshold in POL-MODEL-001?"}],
        "current_agent": "",
        "tool_calls": [],
        "context": "",
        "provenance": [],
        "answer": None,
        "refusal_status": False
    }

    # 1. Run agent decision node
    state_after_decision = policy_agent_node(initial_state)

    # Agent node must have updated current_agent and added a PENDING tool call
    assert state_after_decision["current_agent"] == "policy_agent"
    assert len(state_after_decision["tool_calls"]) == 1
    tool_call = state_after_decision["tool_calls"][0]
    assert tool_call["status"] == "pending"
    assert tool_call["tool_name"] == "search_hr_policies"
    assert tool_call["args"]["query"] == "What is the decision threshold in POL-MODEL-001?"

    # Context, provenance, and answer must NOT be populated by the decision node
    assert state_after_decision["context"] == ""
    assert state_after_decision["provenance"] == []
    assert state_after_decision["answer"] is None

    # 2. Run tool execution node (the boundary)
    state_after_tool = policy_tool_node(state_after_decision)

    # Tool execution node must complete the call and populate context and provenance
    assert len(state_after_tool["tool_calls"]) == 1
    completed_call = state_after_tool["tool_calls"][0]
    assert completed_call["status"] == "completed"
    assert "result" in completed_call
    assert len(state_after_tool["provenance"]) > 0
    assert len(state_after_tool["context"]) > 0
    assert state_after_tool["answer"] is not None

    # 3. Run response synthesis node
    final_state = policy_response_node(state_after_tool)
    assert len(final_state["messages"]) == 2
    assert final_state["messages"][-1]["role"] == "assistant"
    assert final_state["messages"][-1]["name"] == "policy_agent"


# ==============================================================================
# 4. End-to-End Grounded Policy Answer Tests
# ==============================================================================

def test_grounded_policy_answer_e2e():
    """Test full LangGraph execution for a grounded policy question from fixtures."""
    agent = PolicyAgent()
    query = "What does POL-MODEL-001 say about the attrition threshold?"

    result = agent.run(query)

    assert isinstance(result, PolicyAgentResult)
    assert result.query == query
    assert result.current_agent == "policy_agent"
    assert result.refusal_status is False
    assert len(result.answer) > 0
    # Must contain ground truth threshold 0.40
    assert "0.40" in result.answer or "0.4" in result.answer
    # Provenance must be populated with top source chunks from policy corpus
    assert len(result.provenance) > 0
    first_source = result.provenance[0]
    assert "POL-MODEL-001" in first_source.get("chunk_id", "") or "POL-MODEL-001" in first_source.get("source", "")
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["status"] == "completed"


def test_provenance_propagation():
    """Verify that full audit provenance is correctly propagated into the result."""
    agent = PolicyAgent()
    query = "What is the purpose of POL-JOB-001?"

    result = agent.run(query)

    assert result.refusal_status is False
    assert len(result.provenance) >= 1
    for p in result.provenance:
        assert "chunk_id" in p
        assert "source" in p
        assert "score" in p
        assert "excerpt" in p
        assert p["score"] is not None


# ==============================================================================
# 5. Unsupported Query Refusal Tests
# ==============================================================================

def test_unsupported_query_refusal():
    """Test refusal behavior when query asks for out-of-domain information."""
    agent = PolicyAgent()
    query = "What is the company's parental leave entitlement?"

    result = agent.run(query)

    assert isinstance(result, PolicyAgentResult)
    assert result.refusal_status is True
    assert (
        REFUSAL_MESSAGE.lower() in result.answer.lower()
        or "not contain sufficient" in result.answer.lower()
        or "does not contain" in result.answer.lower()
    )
