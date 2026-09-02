"""
app/agents/orchestrator.py
==========================
Global LangGraph Orchestrator for the Enterprise HR AI Platform.

Graph Architecture:
                    START
                      │
                      ▼
             global_router_node
                      │
             Conditional Edge: route_decision()
              ├── "policy_agent" ───────────────► policy_agent_delegation_node
              ├── "workforce_intelligence_agent" ► workforce_agent_delegation_node
              └── "fallback_handler" ───────────► fallback_node
                      │                                    │
                      └─────────────────┬──────────────────┘
                                        ▼
                               global_response_node
                                        │
                                        ▼
                                       END

Enforces Strict Layer Boundaries:
- The orchestrator does NOT import ChromaDB, BM25, CrossEncoder, or RAG models.
- Policy queries are delegated exclusively to the PolicyAgent facade.
- Workforce queries are delegated exclusively to the WorkforceAgent facade.
- Future specialized agents (Upskilling, Career, HR Ops) return clear, structured notices.
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

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
from app.agents.policy_agent import PolicyAgent, get_policy_agent
from app.agents.workforce_agent import WorkforceAgent, get_workforce_agent
from app.agents.upskilling_agent import UpskillingAgent, get_upskilling_agent
from app.agents.career_agent import CareerAgent, get_career_agent
from app.agents.hr_ops_agent import HROpsAgent, get_hr_ops_agent
from app.utils.logger import logger

# Agent Identifier Constants
AGENT_POLICY = "policy_agent"
AGENT_WORKFORCE = "workforce_intelligence_agent"
AGENT_UPSKILLING = "upskilling_agent"
AGENT_CAREER = "career_agent"
AGENT_HR_OPS = "hr_ops_agent"
AGENT_ORCHESTRATOR = "global_orchestrator"
AGENT_FALLBACK = "fallback_handler"


def global_router_node(state: AgentState) -> AgentState:
    """
    Deterministic Router Node:
    Extracts the user question and classifies the intent into one of 6 domains.
    Assigns intent and target_agent without executing tools or business logic.
    """
    messages = state.get("messages", [])
    query = ""
    if messages:
        last_msg = messages[-1]
        query = last_msg.get("content", "")
    elif state.get("context"):
        query = state["context"]

    intent = classify_intent(query)
    logger.info(f"Global Router classified query: '{query[:60]}...' -> Intent: {intent}")

    target_map = {
        INTENT_POLICY: AGENT_POLICY,
        INTENT_WORKFORCE_INTELLIGENCE: AGENT_WORKFORCE,
        INTENT_UPSKILLING: AGENT_UPSKILLING,
        INTENT_CAREER: AGENT_CAREER,
        INTENT_HR_OPS: AGENT_HR_OPS,
        INTENT_OUT_OF_DOMAIN: AGENT_FALLBACK
    }
    target_agent = target_map.get(intent, AGENT_FALLBACK)

    return {
        **state,
        "intent": intent,
        "target_agent": target_agent,
        "current_agent": AGENT_ORCHESTRATOR
    }


def route_decision(state: AgentState) -> str:
    """
    Conditional routing edge function.
    Routes to policy_agent if intent is POLICY,
    routes to workforce_intelligence_agent if intent is WORKFORCE_INTELLIGENCE,
    otherwise routes to fallback_handler.
    """
    intent = state.get("intent", INTENT_OUT_OF_DOMAIN)
    if intent == INTENT_POLICY:
        return "policy_agent"
    elif intent == INTENT_WORKFORCE_INTELLIGENCE:
        return "workforce_intelligence_agent"
    elif intent == INTENT_UPSKILLING:
        return "upskilling_agent"
    elif intent == INTENT_CAREER:
        return "career_agent"
    elif intent == INTENT_HR_OPS:
        return "hr_ops_agent"
    return "fallback_handler"


def policy_agent_delegation_node(state: AgentState, policy_agent: Optional[PolicyAgent] = None) -> AgentState:
    """
    Policy Agent Delegation Node:
    Delegates execution directly to the existing PolicyAgent facade.
    Preserves answer, provenance, refusal_status, and tool_calls without rewriting.
    """
    messages = state.get("messages", [])
    query = messages[-1].get("content", "") if messages else state.get("context", "")
    
    agent = policy_agent or get_policy_agent()
    logger.info(f"Delegating query to PolicyAgent facade: '{query[:60]}...'")
    
    result = agent.run(query)

    # Accumulate tool calls from Policy Agent
    existing_tool_calls = list(state.get("tool_calls", []))
    merged_tool_calls = existing_tool_calls + list(result.tool_calls)

    return {
        **state,
        "current_agent": AGENT_POLICY,
        "target_agent": AGENT_POLICY,
        "answer": result.answer,
        "provenance": result.provenance,
        "refusal_status": result.refusal_status,
        "tool_calls": merged_tool_calls,
        "error": None
    }


def workforce_agent_delegation_node(state: AgentState, workforce_agent: Optional[WorkforceAgent] = None) -> AgentState:
    """
    Workforce Agent Delegation Node:
    Delegates execution directly to the WorkforceAgent facade.
    Preserves answer, provenance, refusal_status, and tool_calls without rewriting.
    """
    messages = state.get("messages", [])
    query = messages[-1].get("content", "") if messages else state.get("context", "")
    
    agent = workforce_agent or get_workforce_agent()
    logger.info(f"Delegating query to WorkforceAgent facade: '{query[:60]}...'")
    
    result = agent.run(query)

    existing_tool_calls = list(state.get("tool_calls", []))
    merged_tool_calls = existing_tool_calls + list(result.tool_calls)

    return {
        **state,
        "current_agent": AGENT_WORKFORCE,
        "target_agent": AGENT_WORKFORCE,
        "answer": result.answer,
        "provenance": result.provenance,
        "refusal_status": result.refusal_status,
        "tool_calls": merged_tool_calls,
        "error": None
    }


def upskilling_agent_delegation_node(state: AgentState, upskilling_agent: Optional[UpskillingAgent] = None) -> AgentState:
    """
    Upskilling Agent Delegation Node:
    Delegates execution directly to the UpskillingAgent facade.
    Preserves answer, provenance, refusal_status, and tool_calls without rewriting.
    """
    messages = state.get("messages", [])
    query = messages[-1].get("content", "") if messages else state.get("context", "")
    
    agent = upskilling_agent or get_upskilling_agent()
    logger.info(f"Delegating query to UpskillingAgent facade: '{query[:60]}...'")
    
    result = agent.run(query)

    existing_tool_calls = list(state.get("tool_calls", []))
    merged_tool_calls = existing_tool_calls + list(result.tool_calls)

    return {
        **state,
        "current_agent": AGENT_UPSKILLING,
        "target_agent": AGENT_UPSKILLING,
        "answer": result.answer,
        "provenance": result.provenance,
        "refusal_status": result.refusal_status,
        "tool_calls": merged_tool_calls,
        "error": None
    }


def career_agent_delegation_node(state: AgentState, career_agent: Optional[CareerAgent] = None) -> AgentState:
    """
    Career Agent Delegation Node:
    Delegates execution directly to the CareerAgent facade.
    Preserves answer, provenance, refusal_status, and tool_calls without rewriting.
    """
    messages = state.get("messages", [])
    query = messages[-1].get("content", "") if messages else state.get("context", "")
    
    agent = career_agent or get_career_agent()
    logger.info(f"Delegating query to CareerAgent facade: '{query[:60]}...'")
    
    result = agent.run(query)

    existing_tool_calls = list(state.get("tool_calls", []))
    merged_tool_calls = existing_tool_calls + list(result.tool_calls)

    return {
        **state,
        "current_agent": AGENT_CAREER,
        "target_agent": AGENT_CAREER,
        "answer": result.answer,
        "provenance": result.provenance,
        "refusal_status": result.refusal_status,
        "tool_calls": merged_tool_calls,
        "error": None
    }


def hr_ops_agent_delegation_node(state: AgentState, hr_ops_agent: Optional[HROpsAgent] = None) -> AgentState:
    """
    HR Ops Agent Delegation Node:
    Delegates execution directly to the HROpsAgent facade.
    Preserves answer, provenance, refusal_status, and tool_calls without rewriting.
    """
    messages = state.get("messages", [])
    query = messages[-1].get("content", "") if messages else state.get("context", "")
    
    agent = hr_ops_agent or get_hr_ops_agent()
    logger.info(f"Delegating query to HROpsAgent facade: '{query[:60]}...'")
    
    result = agent.run(query)

    existing_tool_calls = list(state.get("tool_calls", []))
    merged_tool_calls = existing_tool_calls + list(result.tool_calls)

    return {
        **state,
        "current_agent": AGENT_HR_OPS,
        "target_agent": AGENT_HR_OPS,
        "answer": result.answer,
        "provenance": result.provenance,
        "refusal_status": result.refusal_status,
        "tool_calls": merged_tool_calls,
        "error": None
    }





def fallback_node(state: AgentState) -> AgentState:
    """
    Graceful Fallback Node for out-of-domain queries.
    Provides structured explanations without executing arbitrary tools.
    All five specialized agents are now live; fallback only triggers on out-of-domain.
    """
    target = state.get("target_agent", AGENT_FALLBACK)
    logger.info(f"Fallback node executing for target: {target}")

    answer = (
        "The Enterprise HR AI platform can only answer questions related to HR policies, "
        "workforce intelligence, employee upskilling, career progression, and HR operations."
    )
    refusal = True
    error = "Query classified as out-of-domain."

    return {
        **state,
        "current_agent": target,
        "answer": answer,
        "provenance": [],
        "refusal_status": refusal,
        "error": error
    }


def global_response_node(state: AgentState) -> AgentState:
    """
    Global Response Node:
    Finalizes state, records assistant message in conversation history,
    and guarantees a non-empty answer.
    """
    answer = state.get("answer") or "No response could be generated."
    refusal = state.get("refusal_status", False)
    current = state.get("current_agent", AGENT_ORCHESTRATOR)

    messages = list(state.get("messages", []))
    if not messages or messages[-1].get("role") != "assistant":
        messages.append({
            "role": "assistant",
            "name": current,
            "content": answer,
            "refusal": refusal
        })

    return {
        **state,
        "messages": messages,
        "answer": answer
    }


def build_orchestrator_graph(
    policy_agent: Optional[PolicyAgent] = None,
    workforce_agent: Optional[WorkforceAgent] = None,
    upskilling_agent: Optional[UpskillingAgent] = None,
    career_agent: Optional[CareerAgent] = None,
    hr_ops_agent: Optional[HROpsAgent] = None
) -> StateGraph:
    """
    Constructs and compiles the Global LangGraph Orchestrator StateGraph.
    """
    workflow = StateGraph(AgentState)

    def policy_node_wrapper(state: AgentState) -> AgentState:
        return policy_agent_delegation_node(state, policy_agent=policy_agent)

    def workforce_node_wrapper(state: AgentState) -> AgentState:
        return workforce_agent_delegation_node(state, workforce_agent=workforce_agent)

    def upskilling_node_wrapper(state: AgentState) -> AgentState:
        return upskilling_agent_delegation_node(state, upskilling_agent=upskilling_agent)

    def career_node_wrapper(state: AgentState) -> AgentState:
        return career_agent_delegation_node(state, career_agent=career_agent)

    def hr_ops_node_wrapper(state: AgentState) -> AgentState:
        return hr_ops_agent_delegation_node(state, hr_ops_agent=hr_ops_agent)

    # 1. Add nodes
    workflow.add_node("global_router", global_router_node)
    workflow.add_node("policy_agent_delegation", policy_node_wrapper)
    workflow.add_node("workforce_agent_delegation", workforce_node_wrapper)
    workflow.add_node("upskilling_agent_delegation", upskilling_node_wrapper)
    workflow.add_node("career_agent_delegation", career_node_wrapper)
    workflow.add_node("hr_ops_agent_delegation", hr_ops_node_wrapper)
    workflow.add_node("fallback_handler", fallback_node)
    workflow.add_node("global_response", global_response_node)

    # 2. Add entry edge
    workflow.add_edge(START, "global_router")

    # 3. Add conditional edge from router
    workflow.add_conditional_edges(
        "global_router",
        route_decision,
        {
            "policy_agent": "policy_agent_delegation",
            "workforce_intelligence_agent": "workforce_agent_delegation",
            "upskilling_agent": "upskilling_agent_delegation",
            "career_agent": "career_agent_delegation",
            "hr_ops_agent": "hr_ops_agent_delegation",
            "fallback_handler": "fallback_handler"
        }
    )

    # 4. Join branches to global response node
    workflow.add_edge("policy_agent_delegation", "global_response")
    workflow.add_edge("workforce_agent_delegation", "global_response")
    workflow.add_edge("upskilling_agent_delegation", "global_response")
    workflow.add_edge("career_agent_delegation", "global_response")
    workflow.add_edge("hr_ops_agent_delegation", "global_response")
    workflow.add_edge("fallback_handler", "global_response")

    # 5. Add exit edge
    workflow.add_edge("global_response", END)

    return workflow.compile()


class GlobalOrchestrator:
    """
    High-level facade for executing the Global LangGraph Orchestrator.
    """

    def __init__(
        self,
        policy_agent: Optional[PolicyAgent] = None,
        workforce_agent: Optional[WorkforceAgent] = None,
        upskilling_agent: Optional[UpskillingAgent] = None,
        career_agent: Optional[CareerAgent] = None,
        hr_ops_agent: Optional[HROpsAgent] = None
    ):
        self.policy_agent = policy_agent or get_policy_agent()
        self.workforce_agent = workforce_agent or get_workforce_agent()
        self.upskilling_agent = upskilling_agent or get_upskilling_agent()
        self.career_agent = career_agent or get_career_agent()
        self.hr_ops_agent = hr_ops_agent or get_hr_ops_agent()
        self.graph = build_orchestrator_graph(
            policy_agent=self.policy_agent,
            workforce_agent=self.workforce_agent,
            upskilling_agent=self.upskilling_agent,
            career_agent=self.career_agent,
            hr_ops_agent=self.hr_ops_agent
        )

    def run(self, query: str, initial_state: Optional[AgentState] = None) -> OrchestratorResponse:
        """
        Executes the Orchestrator graph for an incoming user query.
        
        Args:
            query: User natural-language question.
            initial_state: Optional previous conversation state.
            
        Returns:
            OrchestratorResponse with answer, provenance, agent_routed, and refusal_status.
        """
        clean_query = query.strip()
        base_state: AgentState = initial_state or {
            "messages": [],
            "current_agent": AGENT_ORCHESTRATOR,
            "target_agent": None,
            "intent": None,
            "tool_calls": [],
            "context": "",
            "provenance": [],
            "answer": None,
            "refusal_status": False,
            "error": None
        }

        # Append incoming query
        messages = list(base_state.get("messages", []))
        messages.append({"role": "user", "content": clean_query})
        base_state["messages"] = messages
        base_state["context"] = clean_query

        logger.info(f"Global Orchestrator invoking graph for query: '{clean_query}'")
        final_state = self.graph.invoke(base_state)

        agent_routed = final_state.get("target_agent") or final_state.get("current_agent") or AGENT_ORCHESTRATOR

        return OrchestratorResponse(
            query=clean_query,
            answer=final_state.get("answer", ""),
            agent_routed=agent_routed,
            provenance=final_state.get("provenance", []),
            refusal_status=final_state.get("refusal_status", False)
        )


_orchestrator_instance: Optional[GlobalOrchestrator] = None


def get_orchestrator() -> GlobalOrchestrator:
    """Singleton accessor for the Global Orchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = GlobalOrchestrator()
    return _orchestrator_instance
