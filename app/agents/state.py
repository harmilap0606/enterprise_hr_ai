"""
app/agents/state.py
===================
Standardized typed state schema for the Enterprise HR AI Agentic Layer.
Built on LangGraph StateGraph specifications.
"""

from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    """
    Minimal, typed agent state container for LangGraph orchestration.
    
    Fields:
        messages: Chronological conversation history of messages (role, content).
        current_agent: Identifier of the currently executing specialized agent node.
        target_agent: Target specialized agent identified by the Global Router.
        intent: Domain classification label (e.g. POLICY, WORKFORCE_INTELLIGENCE).
        tool_calls: Structured records of generated tool requests and execution results.
        context: Aggregated textual background or retrieved evidence for generation.
        provenance: Verified source attribution records propagating from retrieval tools.
        answer: Final or intermediate synthesized natural-language response.
        refusal_status: Boolean indicating whether the request was refused due to lack of evidence.
        error: Optional error or availability notice.
    """
    messages: List[Dict[str, Any]]
    current_agent: str
    target_agent: Optional[str]
    intent: Optional[str]
    tool_calls: List[Dict[str, Any]]
    context: str
    provenance: List[Dict[str, Any]]
    answer: Optional[str]
    refusal_status: bool
    error: Optional[str]


class PolicyAgentResult(BaseModel):
    """
    Structured Pydantic response model returned by the Policy Agent.
    Guarantees clean serialization for API, tests, and orchestrator nodes.
    """
    query: str = Field(..., description="Original user question.")
    answer: str = Field(..., description="Grounded natural-language answer or verified refusal string.")
    provenance: List[Dict[str, Any]] = Field(default_factory=list, description="Audit provenance source items.")
    refusal_status: bool = Field(default=False, description="True if query could not be verified in the policy corpus.")
    current_agent: str = Field(default="policy_agent", description="Active agent responsible for output.")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Audited list of tool calls executed.")


class OrchestratorResponse(BaseModel):
    """
    Structured Pydantic response model returned by the Global LangGraph Orchestrator.
    Unified response contract for API and client applications.
    """
    query: str = Field(..., description="Original user question submitted to the orchestrator.")
    answer: str = Field(..., description="Synthesized answer, grounded policy response, or refusal message.")
    agent_routed: str = Field(..., description="Specialized agent identifier responsible for handling the query.")
    provenance: List[Dict[str, Any]] = Field(default_factory=list, description="Audit provenance records from tool execution.")
    refusal_status: bool = Field(default=False, description="True if the query was refused or out-of-domain.")
