"""
app/agents/upskilling_agent.py
==============================
Upskilling Specialist Agent implemented as an isolated LangGraph StateGraph.

Graph Architecture:
1. upskilling_agent_node (Decision Logic):
   - Deterministically parses query intent and extracts parameters (e.g. employee_id, limit).
   - Formulates structured pending tool requests without direct access to databases, services, or models.
2. upskilling_tool_node (Execution Boundary):
   - Validates requests against UPSKILLING_AUTHORIZED_TOOLS.
   - Executes authorized tools safely, catching expected domain exceptions (e.g. KeyError).
   - Populates state with structured data, context, and provenance.
3. upskilling_response_node (Synthesis & Policy Guardrails):
   - Synthesizes grounded natural-language responses.
   - Enforces synthetic MVP skill data disclosure.
   - Enforces voluntary developmental nature of recommendations (no punitive/adverse use, POL-LEARN-001).
   - Enforces mandatory human-in-the-loop review (POL-REVIEW-001).
   - Enforces Manager exclusion protocol (department-level analysis guidance, POL-LEARN-001 Rule 3).
"""

import re
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.tools.upskilling_tool import (
    EmployeeUpskillingTool,
    OrganizationSkillGapTool,
    UPSKILLING_AUTHORIZED_TOOLS
)
from app.utils.logger import logger

AGENT_UPSKILLING_ID = "upskilling_agent"


# ==============================================================================
# Structured Output Model
# ==============================================================================

class UpskillingAgentResult(BaseModel):
    """
    Structured Pydantic response model returned by the Upskilling Agent.
    Guarantees consistent serialization across API, orchestrator, and test harnesses.
    """
    query: str = Field(..., description="Original user question.")
    answer: str = Field(..., description="Synthesized upskilling report or guidance.")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Structured tool output payload.")
    provenance: List[Dict[str, Any]] = Field(default_factory=list, description="Audit provenance source items.")
    refusal_status: bool = Field(default=False, description="True if query could not be fulfilled or employee was missing.")
    current_agent: str = Field(default=AGENT_UPSKILLING_ID, description="Active specialized agent identifier.")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Audited list of tool calls executed.")


# ==============================================================================
# Node 1: Decision Logic
# ==============================================================================

def extract_employee_id(query: str) -> Optional[int]:
    """Extracts explicit numeric employee ID from natural language queries."""
    clean_q = query.lower()
    patterns = [
        r"\b(?:employee|emp|worker|id)\s*#?\s*(\d+)\b",
        r"\b#(\d+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, clean_q)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


def is_individual_upskilling_query(query: str) -> bool:
    """Detects whether query targets a specific employee's courses or skills."""
    clean_q = query.lower()
    individual_patterns = [
        r"\b(?:the employee|this employee|an employee|single employee|individual employee|particular employee)\b",
        r"\b(?:the worker|this worker|an worker|single worker|individual worker)\b",
        r"\b(?:for employee|for the employee|for an employee|for worker|for the worker)\b",
        r"\b(?:employee's|worker's)\b",
        r"\bwhat course(?:s)? should (?:the|an|this)?\s*employee\b",
        r"\bwhat skill(?:s)? should (?:the|an|this)?\s*employee\b",
        r"\brecommend course(?:s)? for (?:the|an|this)?\s*employee\b",
        r"\blearning recommendation for (?:the|an|this)?\s*employee\b",
        r"\btraining for employee\b",
        r"\bcourses for employee\b",
        r"\bemployee #\d+\b",
        r"\bemployee \d+\b",
    ]
    return any(re.search(pat, clean_q) for pat in individual_patterns)


def is_organization_skill_query(query: str) -> bool:
    """Detects whether query asks about organization-wide skill gaps or course catalog."""
    clean_q = query.lower()
    org_patterns = [
        r"\borganization(?:al)?\b",
        r"\bcompany\b",
        r"\bworkforce\b",
        r"\bacross the company\b",
        r"\bbiggest skill gap\b",
        r"\btop skill gap\b",
        r"\boverall skill gap\b",
        r"\ball skill gap\b",
        r"\bcatalog\b",
        r"\bcurriculum\b",
        r"\bcourse catalog\b",
    ]
    return any(re.search(pat, clean_q) for pat in org_patterns)


def upskilling_agent_node(state: AgentState) -> AgentState:
    """
    Pure Decision / Planning Node for Upskilling:
    - Parses user query to determine whether individual or organizational data is needed.
    - Extracts parameters (e.g. employee_id).
    - Formulates a pending structured tool request.
    - NEVER directly executes backend services or touches data files.
    """
    messages = state.get("messages", [])
    query = messages[-1].get("content", "") if messages else state.get("context", "")
    logger.info(f"UpskillingAgent decision node evaluating query: '{query}'")

    clean_q = query.strip()
    emp_id = extract_employee_id(clean_q)
    is_individual = is_individual_upskilling_query(clean_q) or emp_id is not None
    is_org = is_organization_skill_query(clean_q)

    # Disambiguation: explicit aggregate preference
    if is_org and not emp_id:
        is_individual = False

    tool_calls = list(state.get("tool_calls", []))

    # Scenario A: Individual Employee Upskilling Request
    if is_individual:
        if emp_id is None:
            logger.info("Individual upskilling requested without explicit numeric employee ID. Halting with refusal.")
            return {
                **state,
                "current_agent": AGENT_UPSKILLING_ID,
                "answer": (
                    "Please specify the numeric Employee ID (for example, 'Recommend courses for employee #100') "
                    "to retrieve personalized upskilling pathways and skill-gap diagnostics."
                ),
                "refusal_status": True,
                "tool_calls": tool_calls,
                "error": None
            }

        pending_call = {
            "tool_name": "get_employee_upskilling_recommendations",
            "args": {"employee_id": emp_id},
            "status": "pending"
        }
        tool_calls.append(pending_call)
        return {
            **state,
            "current_agent": AGENT_UPSKILLING_ID,
            "tool_calls": tool_calls,
            "refusal_status": False,
            "error": None
        }

    # Scenario B: Organization-Wide Skill Gap Analysis
    pending_call = {
        "tool_name": "get_organization_skill_gaps",
        "args": {"limit": 10},
        "status": "pending"
    }
    tool_calls.append(pending_call)
    return {
        **state,
        "current_agent": AGENT_UPSKILLING_ID,
        "tool_calls": tool_calls,
        "refusal_status": False,
        "error": None
    }


# ==============================================================================
# Node 2: Tool Execution Boundary
# ==============================================================================

def upskilling_tool_node(
    state: AgentState,
    tools: Optional[Dict[str, Any]] = None
) -> AgentState:
    """
    Authorized Tool Execution Boundary:
    - Reads pending tool call from state.
    - Enforces whitelist authorization against UPSKILLING_AUTHORIZED_TOOLS.
    - Executes tool safely and updates tool call status to 'completed', 'failed', or 'rejected'.
    - Populates state context, data payload, and provenance.
    """
    tool_calls = list(state.get("tool_calls", []))
    if not tool_calls:
        logger.warning("Upskilling tool node invoked but no tool calls found in state.")
        return state

    latest_call = dict(tool_calls[-1])
    tool_name = latest_call.get("tool_name")
    args = latest_call.get("args", {})

    # Tool instances lookup
    available_tools = tools or {
        EmployeeUpskillingTool.TOOL_NAME: EmployeeUpskillingTool(),
        OrganizationSkillGapTool.TOOL_NAME: OrganizationSkillGapTool(),
    }

    logger.info(f"Upskilling execution boundary checking tool: '{tool_name}'")

    # Authorization Check
    if tool_name not in UPSKILLING_AUTHORIZED_TOOLS or tool_name not in available_tools:
        logger.error(f"Unauthorized or unknown upskilling tool execution attempted: '{tool_name}'")
        latest_call["status"] = "rejected"
        latest_call["error"] = f"Tool '{tool_name}' is not authorized. Allowed tools: {sorted(list(UPSKILLING_AUTHORIZED_TOOLS))}"
        tool_calls[-1] = latest_call
        return {
            **state,
            "tool_calls": tool_calls,
            "error": latest_call["error"],
            "refusal_status": True
        }

    # Authorized Execution
    target_tool = available_tools[tool_name]
    try:
        raw_result = target_tool.execute(args)
        
        # Check domain error from tool (e.g., EMPLOYEE_NOT_FOUND)
        if raw_result.get("status") == "error":
            latest_call["status"] = "failed"
            latest_call["error"] = raw_result.get("message")
            tool_calls[-1] = latest_call
            return {
                **state,
                "tool_calls": tool_calls,
                "context": str(raw_result),
                "error": raw_result.get("message"),
                "refusal_status": True
            }

        latest_call["status"] = "completed"
        latest_call["result"] = raw_result
        tool_calls[-1] = latest_call

        existing_provenance = list(state.get("provenance", []))
        tool_provenance = raw_result.get("provenance", [])
        combined_provenance = existing_provenance + tool_provenance

        return {
            **state,
            "tool_calls": tool_calls,
            "context": str(raw_result),
            "provenance": combined_provenance,
            "error": None,
            "refusal_status": False
        }

    except Exception as e:
        logger.error(f"Upskilling tool execution threw unhandled exception for '{tool_name}': {e}", exc_info=True)
        latest_call["status"] = "failed"
        latest_call["error"] = str(e)
        tool_calls[-1] = latest_call
        return {
            **state,
            "tool_calls": tool_calls,
            "error": str(e),
            "refusal_status": True
        }


# ==============================================================================
# Node 3: Synthesis & Policy Guardrails
# ==============================================================================

def upskilling_response_node(state: AgentState) -> AgentState:
    """
    Response Synthesis Node:
    - Generates concise, grounded natural-language responses.
    - Enforces synthetic MVP skill data disclosure.
    - Enforces voluntary developmental nature of recommendations (POL-LEARN-001).
    - Enforces mandatory human review (POL-REVIEW-001).
    - Enforces Manager exclusion protocol (POL-LEARN-001 Rule 3, POL-SKILL-001 Rule 2).
    """
    # If already refused or in error
    if state.get("refusal_status") and state.get("answer"):
        return state

    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return {
            **state,
            "answer": "No upskilling tools were executed to answer the query.",
            "refusal_status": True
        }

    latest_call = tool_calls[-1]
    tool_status = latest_call.get("status")
    tool_name = latest_call.get("tool_name")
    result = latest_call.get("result", {})

    # Handle Tool Failure
    if tool_status == "failed":
        err_msg = latest_call.get("error", "The requested upskilling record could not be found.")
        return {
            **state,
            "answer": f"Unable to retrieve upskilling details: {err_msg}",
            "refusal_status": True
        }

    # Handle Tool Rejection
    if tool_status == "rejected":
        return {
            **state,
            "answer": f"Execution halted: {latest_call.get('error')}",
            "refusal_status": True
        }

    # Synthesis 1: Individual Employee Upskilling
    if tool_name == EmployeeUpskillingTool.TOOL_NAME:
        emp_id = result.get("EmployeeNumber")
        role = result.get("JobRole")
        severity = result.get("severity")
        missing_skills = result.get("missing_skills", [])
        recs = result.get("recommended_courses", [])
        is_mgr = result.get("is_manager", False)

        # Case 1A: Manager Exclusion Protocol (POL-LEARN-001 Rule 3)
        if is_mgr:
            answer = (
                f"### Employee #{emp_id} Upskilling Profile\n"
                f"- **Job Role:** {role}\n"
                f"- **Skill Gap Severity:** {severity}\n"
                f"- **Automated Course Recommendations:** N/A — Manager Cohort\n\n"
                f"**Managerial Protocol Notice (POL-LEARN-001 Rule 3 & POL-SKILL-001 Rule 2):**\n"
                f"Employees holding the generic Job Role 'Manager' (102 employees) are excluded from automated "
                f"O*NET benchmark skill comparisons due to low mapping confidence. Professional development "
                f"for managers must be coordinated directly through departmental HR and department-level leadership frameworks "
                f"rather than automated course triples.\n\n"
                f"**Governance & Developmental Policy (POL-REVIEW-001):**\n"
                f"All learning recommendations are developmental suggestions for professional growth. "
                f"Manager-employee dialogue is required before formal curriculum planning."
            )
            return {
                **state,
                "answer": answer,
                "refusal_status": False
            }

        # Case 1B: Standard Employee Recommendations
        missing_str = ", ".join(missing_skills) if missing_skills else "None identified"
        courses_formatted = "\n".join([f"  {idx+1}. {c}" for idx, c in enumerate(recs)])

        answer = (
            f"### Personalized Upskilling Recommendations for Employee #{emp_id}\n"
            f"- **Job Role:** {role}\n"
            f"- **Skill Gap Severity:** {severity}\n"
            f"- **Identified Missing Benchmark Skills:** {missing_str}\n\n"
            f"**Recommended Course Pathways (Top-3):**\n"
            f"{courses_formatted}\n\n"
            f"**Developmental Policy & Human Review (POL-LEARN-001 & POL-REVIEW-001):**\n"
            f"- *Developmental Nature:* In accordance with POL-LEARN-001, these course recommendations represent voluntary "
            f"growth opportunities and must never be used for disciplinary remediation or adverse employment decisions.\n"
            f"- *Human Review Required:* A manager-employee developmental discussion is required prior to formal course assignment.\n"
            f"- *Synthetic Skill Data:* Identified skill gaps are derived from synthetic MVP skill inventory data "
            f"simulated for demonstration purposes."
        )
        return {
            **state,
            "answer": answer,
            "refusal_status": False
        }

    # Synthesis 2: Organization-Wide Skill Gaps
    if tool_name == OrganizationSkillGapTool.TOOL_NAME:
        total_eval = result.get("total_skills_evaluated", 33)
        dist = result.get("severity_breakdown", {})
        top_gaps = result.get("top_skill_gaps", [])
        scope = result.get("catalog_scope", "33 Curated Enterprise Courses")

        gaps_summary = []
        for idx, g in enumerate(top_gaps[:5]):
            gaps_summary.append(
                f"  {idx+1}. **{g.get('skill_name')}** — {g.get('total_missing_count')} employees missing "
                f"[{g.get('severity')} Severity]"
            )
        gaps_text = "\n".join(gaps_summary)

        answer = (
            f"### Organization-Wide Skill Gap Overview\n"
            f"- **Total Benchmark Skills Evaluated:** {total_eval}\n"
            f"- **Severity Distribution:** {dist.get('HIGH', 0)} HIGH, {dist.get('MEDIUM', 0)} MEDIUM, {dist.get('LOW', 0)} LOW\n"
            f"- **Enterprise Course Catalog Scope:** {scope}\n\n"
            f"**Top Critical Capability Gaps Across Enterprise:**\n"
            f"{gaps_text}\n\n"
            f"**Governance Notice (POL-SKILL-001 & POL-LEARN-001):**\n"
            f"Organizational skill gaps aggregate individual capability deltas against O*NET occupational standards. "
            f"All prioritized competencies map to accredited enterprise courses. Results reflect synthetic MVP benchmark data."
        )
        return {
            **state,
            "answer": answer,
            "refusal_status": False
        }

    return state


# ==============================================================================
# LangGraph Graph Construction
# ==============================================================================

def build_upskilling_agent_graph():
    """
    Constructs and compiles the isolated StateGraph for the Upskilling Agent.
    Topology:
      START -> upskilling_agent_node -> upskilling_tool_node -> upskilling_response_node -> END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("upskilling_agent_node", upskilling_agent_node)
    workflow.add_node("upskilling_tool_node", upskilling_tool_node)
    workflow.add_node("upskilling_response_node", upskilling_response_node)

    workflow.set_entry_point("upskilling_agent_node")
    workflow.add_edge("upskilling_agent_node", "upskilling_tool_node")
    workflow.add_edge("upskilling_tool_node", "upskilling_response_node")
    workflow.add_edge("upskilling_response_node", END)

    return workflow.compile()


# ==============================================================================
# Upskilling Agent Facade
# ==============================================================================

class UpskillingAgent:
    """
    Facade providing synchronous query execution against the Upskilling Agent StateGraph.
    Adheres to the same interface as PolicyAgent and WorkforceAgent.
    """

    def __init__(self, compiled_graph=None):
        self.graph = compiled_graph or build_upskilling_agent_graph()

    def run(self, query: str) -> UpskillingAgentResult:
        """
        Executes a user query through the Upskilling LangGraph workflow.
        
        Args:
            query: Natural language question regarding courses, skills, or upskilling.
            
        Returns:
            UpskillingAgentResult with answer, data, provenance, refusal_status, and tool_calls.
        """
        logger.info(f"Invoking Upskilling Agent graph for query: '{query}'")

        initial_state: AgentState = {
            "messages": [{"role": "user", "content": query}],
            "current_agent": AGENT_UPSKILLING_ID,
            "target_agent": AGENT_UPSKILLING_ID,
            "intent": "UPSKILLING",
            "tool_calls": [],
            "context": "",
            "provenance": [],
            "answer": None,
            "refusal_status": False,
            "error": None
        }

        final_state = self.graph.invoke(initial_state)

        # Extract structured tool data
        tool_calls = final_state.get("tool_calls", [])
        data_payload = None
        for tc in reversed(tool_calls):
            if tc.get("status") == "completed" and "result" in tc:
                data_payload = tc["result"]
                break

        return UpskillingAgentResult(
            query=query,
            answer=final_state.get("answer") or "No response generated.",
            data=data_payload,
            provenance=final_state.get("provenance", []),
            refusal_status=final_state.get("refusal_status", False),
            current_agent=final_state.get("current_agent", AGENT_UPSKILLING_ID),
            tool_calls=tool_calls
        )


_UPSKILLING_AGENT_INSTANCE: Optional[UpskillingAgent] = None

def get_upskilling_agent() -> UpskillingAgent:
    """Singleton accessor for UpskillingAgent."""
    global _UPSKILLING_AGENT_INSTANCE
    if _UPSKILLING_AGENT_INSTANCE is None:
        _UPSKILLING_AGENT_INSTANCE = UpskillingAgent()
    return _UPSKILLING_AGENT_INSTANCE
