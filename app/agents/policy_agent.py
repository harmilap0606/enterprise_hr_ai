"""
app/agents/policy_agent.py
==========================
Policy Specialist Agent implemented as an isolated LangGraph StateGraph.

Architecture:
1. policy_agent_node (Decision Logic):
   - Inspects conversation state and query.
   - Formulates structured tool call requests without direct access to data indexes.
2. policy_tool_node (Execution Boundary):
   - Enforces authorization and invokes PolicyRetrievalTool.
   - Updates state with retrieved context, provenance, and execution results.
3. policy_response_node (Synthesis & Guardrails):
   - Sets final grounded answer, refusal status, and audit records.
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END

from app.agents.state import AgentState, PolicyAgentResult
from app.agents.tools.policy_tool import PolicyRetrievalTool
from app.utils.logger import logger


def policy_agent_node(state: AgentState) -> AgentState:
    """
    Agent Decision Node:
    Extracts query and formulates tool call request.
    Strictly isolated: does NOT touch ChromaDB, BM25, or RAG models.
    """
    messages = state.get("messages", [])
    query = ""
    if messages:
        last_msg = messages[-1]
        query = last_msg.get("content", "")
    elif state.get("context"):
        query = state["context"]

    logger.info(f"PolicyAgent decision node triggered for query: '{query[:80]}...'")

    tool_call = {
        "tool_name": PolicyRetrievalTool.name,
        "args": {"query": query},
        "status": "pending"
    }

    updated_tool_calls = list(state.get("tool_calls", [])) + [tool_call]

    return {
        **state,
        "current_agent": "policy_agent",
        "tool_calls": updated_tool_calls
    }


def policy_tool_node(state: AgentState, tool: Optional[PolicyRetrievalTool] = None) -> AgentState:
    """
    Tool Execution Boundary:
    Validates the generated tool request, invokes PolicyRetrievalTool,
    and returns audited provenance and context.
    """
    retrieval_tool = tool or PolicyRetrievalTool.get_instance()
    tool_calls = list(state.get("tool_calls", []))
    
    if not tool_calls:
        logger.warning("No tool calls found in state during tool execution node.")
        return state

    # Execute the most recent pending tool call
    latest_call = tool_calls[-1]
    if latest_call.get("status") == "pending" and latest_call.get("tool_name") == retrieval_tool.name:
        logger.info(f"Executing authorized tool '{retrieval_tool.name}' at execution boundary...")
        tool_args = latest_call.get("args", {})
        result = retrieval_tool.execute(tool_args)

        latest_call["status"] = "completed"
        latest_call["result"] = result

        # Construct concise context summary for graph state
        context_chunks = [
            f"[{p.get('source', 'Unknown')}] {p.get('excerpt', '')}"
            for p in result.get("provenance", [])
        ]
        context_str = "\n\n".join(context_chunks)

        return {
            **state,
            "tool_calls": tool_calls,
            "context": context_str,
            "provenance": result.get("provenance", []),
            "answer": result.get("answer", ""),
            "refusal_status": result.get("refusal_status", False)
        }

    return state


def policy_response_node(state: AgentState) -> AgentState:
    """
    Response Synthesis Node:
    Finalizes output answer, preserves refusal status, and appends agent message.
    """
    answer = state.get("answer", "")
    refusal = state.get("refusal_status", False)
    messages = list(state.get("messages", []))

    assistant_msg = {
        "role": "assistant",
        "name": "policy_agent",
        "content": answer,
        "refusal": refusal
    }
    messages.append(assistant_msg)

    return {
        **state,
        "messages": messages,
        "current_agent": "policy_agent"
    }


def build_policy_agent_graph(tool: Optional[PolicyRetrievalTool] = None) -> StateGraph:
    """
    Constructs and compiles the isolated Policy Agent LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    def tool_node_wrapper(state: AgentState) -> AgentState:
        return policy_tool_node(state, tool=tool)

    workflow.add_node("policy_agent", policy_agent_node)
    workflow.add_node("policy_tool_executor", tool_node_wrapper)
    workflow.add_node("policy_response", policy_response_node)

    workflow.set_entry_point("policy_agent")
    workflow.add_edge("policy_agent", "policy_tool_executor")
    workflow.add_edge("policy_tool_executor", "policy_response")
    workflow.add_edge("policy_response", END)

    return workflow.compile()


class PolicyAgent:
    """
    High-level functional facade for running the Policy Agent StateGraph.
    """

    def __init__(self, tool: Optional[PolicyRetrievalTool] = None):
        self.tool = tool or PolicyRetrievalTool.get_instance()
        self.graph = build_policy_agent_graph(tool=self.tool)

    def run(self, query: str, initial_state: Optional[AgentState] = None) -> PolicyAgentResult:
        """
        Executes the Policy Agent graph for a user query.
        
        Args:
            query: User question regarding HR policies or AI governance.
            initial_state: Optional existing AgentState dictionary to continue conversation.
            
        Returns:
            PolicyAgentResult with answer, audited provenance, and refusal status.
        """
        clean_query = query.strip()
        base_state: AgentState = initial_state or {
            "messages": [],
            "current_agent": "policy_agent",
            "tool_calls": [],
            "context": "",
            "provenance": [],
            "answer": None,
            "refusal_status": False
        }

        # Append user query to messages
        messages = list(base_state.get("messages", []))
        messages.append({"role": "user", "content": clean_query})
        base_state["messages"] = messages
        base_state["context"] = clean_query

        logger.info(f"Invoking Policy Agent graph for query: '{clean_query}'")
        final_state = self.graph.invoke(base_state)

        return PolicyAgentResult(
            query=clean_query,
            answer=final_state.get("answer", ""),
            provenance=final_state.get("provenance", []),
            refusal_status=final_state.get("refusal_status", False),
            current_agent=final_state.get("current_agent", "policy_agent"),
            tool_calls=final_state.get("tool_calls", [])
        )


_policy_agent_instance: Optional[PolicyAgent] = None


def get_policy_agent() -> PolicyAgent:
    """Singleton provider for PolicyAgent."""
    global _policy_agent_instance
    if _policy_agent_instance is None:
        _policy_agent_instance = PolicyAgent()
    return _policy_agent_instance
