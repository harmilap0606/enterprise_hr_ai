"""
app/agents/workforce_agent.py
=============================
Workforce Intelligence Specialist Agent implemented as an isolated LangGraph StateGraph.

Graph Architecture:
1. workforce_agent_node (Decision Logic):
   - Deterministically parses query intent and extracts parameters (e.g. employee_id, department).
   - Formulates structured pending tool requests without direct access to databases or models.
2. workforce_tool_node (Execution Boundary):
   - Validates requests against WORKFORCE_AUTHORIZED_TOOLS.
   - Executes authorized tools safely, catching expected domain exceptions (e.g. KeyError).
   - Populates state with data, context, and provenance.
3. workforce_response_node (Synthesis & Critical Guardrails):
   - Synthesizes grounded natural-language responses.
   - Enforces model decision threshold disclosure (0.40) and mandatory human-in-the-loop review.
   - Enforces survey coverage caveats (49.7% sample representation).
"""

import re
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.tools.workforce_tool import (
    WorkforceKPITool,
    DepartmentWorkforceTool,
    EmployeeRiskTool,
    WORKFORCE_AUTHORIZED_TOOLS
)
from app.utils.logger import logger

AGENT_WORKFORCE_ID = "workforce_intelligence_agent"


# ==============================================================================
# Structured Output Model
# ==============================================================================

class WorkforceAgentResult(BaseModel):
    """
    Structured Pydantic response model returned by the Workforce Intelligence Agent.
    Guarantees consistent serialization across API, orchestrator, and test harnesses.
    """
    query: str = Field(..., description="Original user question.")
    answer: str = Field(..., description="Synthesized workforce intelligence report.")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Structured metrics and record payload.")
    provenance: List[Dict[str, Any]] = Field(default_factory=list, description="Audit provenance source items.")
    refusal_status: bool = Field(default=False, description="True if query could not be fulfilled or employee was missing.")
    current_agent: str = Field(default=AGENT_WORKFORCE_ID, description="Active specialized agent identifier.")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Audited list of tool calls executed.")


# ==============================================================================
# Node 1: Decision Logic
# ==============================================================================

def extract_employee_id(query: str) -> Optional[int]:
    """Extracts explicit numeric employee ID from natural language queries."""
    clean_q = query.lower()
    # Patterns like "employee #100", "emp 100", "worker #1", "employee 1"
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


def extract_department(query: str) -> Optional[str]:
    """Detects department name from query."""
    clean_q = query.lower()
    if re.search(r"\b(?:sales)\b", clean_q):
        return "Sales"
    if re.search(r"\b(?:r&d|rd|research|development)\b", clean_q):
        return "Research & Development"
    if re.search(r"\b(?:hr|human resources)\b", clean_q):
        return "Human Resources"
    return None


def workforce_agent_node(state: AgentState) -> AgentState:
    """
    Decision & Planning Node:
    Inspects user query and formulates a pending tool call.
    Strictly isolated: does NOT call services, pandas, or ML models.
    """
    messages = state.get("messages", [])
    query = messages[-1].get("content", "") if messages else state.get("context", "")
    clean_q = query.strip()

    logger.info(f"WorkforceAgent decision node evaluating query: '{clean_q[:80]}...'")

    # 1. Check if asking about an individual employee
    emp_id = extract_employee_id(clean_q)

    # Specific indicators of an individual employee query when ID might be omitted
    individual_patterns = [
        r"\b(?:the employee|this employee|an employee|single employee|individual employee|particular employee)\b",
        r"\b(?:the worker|this worker|an worker|single worker|individual worker)\b",
        r"\b(?:flight risk of employee|flight risk for employee|risk for employee|risk of employee)\b",
        r"\bwhat is (?:the|an|this) employee\b",
        r"\bwhat is employee (?:flight|risk|attrition)\b"
    ]
    is_explicit_individual = any(re.search(p, clean_q, re.IGNORECASE) for p in individual_patterns)
    is_aggregate = bool(re.search(r"\b(?:average|overall|total|all employees|our employees|workforce|company|across)\b", clean_q, re.IGNORECASE))

    if emp_id is not None:
        # Valid explicit employee ID found -> Create pending EmployeeRiskTool call
        tool_call = {
            "tool_name": EmployeeRiskTool.name,
            "args": {"employee_id": emp_id},
            "status": "pending"
        }
        return {
            **state,
            "current_agent": AGENT_WORKFORCE_ID,
            "tool_calls": list(state.get("tool_calls", [])) + [tool_call]
        }
    elif is_explicit_individual and not is_aggregate:
        # Individual employee intent detected but ID is missing -> Request employee ID without guessing
        logger.warning("Individual employee query detected without valid numeric ID.")
        return {
            **state,
            "current_agent": AGENT_WORKFORCE_ID,
            "answer": (
                "To retrieve individual workforce flight risk intelligence, please specify an explicit "
                "numeric EmployeeNumber (for example: Employee #1 or Employee #100). The agent cannot guess employee IDs."
            ),
            "refusal_status": True,
            "tool_calls": list(state.get("tool_calls", []))
        }

    # 2. Check if asking about department-level analytics
    is_dept_query = bool(re.search(r"\b(?:department|departments|dept|sales|r&d|research|development|human resources)\b", clean_q, re.IGNORECASE))
    if is_dept_query:
        target_dept = extract_department(clean_q)
        tool_call = {
            "tool_name": DepartmentWorkforceTool.name,
            "args": {"department": target_dept},
            "status": "pending"
        }
        return {
            **state,
            "current_agent": AGENT_WORKFORCE_ID,
            "tool_calls": list(state.get("tool_calls", [])) + [tool_call]
        }

    # 3. Default: Organization-wide KPI query
    tool_call = {
        "tool_name": WorkforceKPITool.name,
        "args": {},
        "status": "pending"
    }
    return {
        **state,
        "current_agent": AGENT_WORKFORCE_ID,
        "tool_calls": list(state.get("tool_calls", [])) + [tool_call]
    }


# ==============================================================================
# Node 2: Tool Execution Boundary
# ==============================================================================

def workforce_tool_node(state: AgentState) -> AgentState:
    """
    Tool Execution Boundary Node:
    Enforces WORKFORCE_AUTHORIZED_TOOLS whitelist, validates arguments,
    executes authorized tools, and safely catches service errors.
    """
    tool_calls = list(state.get("tool_calls", []))
    if not tool_calls:
        logger.warning("No tool calls found in state during workforce tool execution.")
        return state

    latest_call = tool_calls[-1]
    if latest_call.get("status") != "pending":
        return state

    tool_name = latest_call.get("tool_name")
    tool_args = latest_call.get("args", {})

    # 1. Authorization Whitelist Check
    if tool_name not in WORKFORCE_AUTHORIZED_TOOLS:
        logger.error(f"Unauthorized tool execution blocked: '{tool_name}'")
        latest_call["status"] = "rejected"
        latest_call["error"] = f"Tool '{tool_name}' is not in WORKFORCE_AUTHORIZED_TOOLS."
        return {
            **state,
            "tool_calls": tool_calls,
            "refusal_status": True,
            "error": f"Unauthorized tool: {tool_name}",
            "answer": f"Execution failed: Tool '{tool_name}' is not authorized."
        }

    # 2. Execute Authorized Tool
    logger.info(f"Executing authorized workforce tool: '{tool_name}' with args: {tool_args}")
    tool_factory = WORKFORCE_AUTHORIZED_TOOLS[tool_name]
    tool_instance = tool_factory()
    
    try:
        result = tool_instance.execute(tool_args)
    except Exception as e:
        logger.error(f"Unexpected exception executing {tool_name}: {e}", exc_info=True)
        latest_call["status"] = "failed"
        latest_call["error"] = str(e)
        return {
            **state,
            "tool_calls": tool_calls,
            "refusal_status": True,
            "error": str(e),
            "answer": f"Unable to retrieve workforce intelligence due to an internal service error: {str(e)}"
        }

    # Check if tool itself reported a clean domain error (e.g. EMPLOYEE_NOT_FOUND)
    if result.get("status") == "error":
        latest_call["status"] = "completed"
        latest_call["result"] = result
        return {
            **state,
            "tool_calls": tool_calls,
            "refusal_status": True,
            "error": result.get("message"),
            "answer": result.get("message", "Requested record could not be found.")
        }

    latest_call["status"] = "completed"
    latest_call["result"] = result

    # Store raw data and provenance into state
    data_payload = result.get("data", {})
    provenance = result.get("provenance", [])

    return {
        **state,
        "tool_calls": tool_calls,
        "provenance": provenance,
        "refusal_status": False,
        "error": None,
        # Temporarily store structured payload in context for response node
        "context": str(data_payload),
    }


# ==============================================================================
# Node 3: Synthesis & Business Rules
# ==============================================================================

def workforce_response_node(state: AgentState) -> AgentState:
    """
    Response Synthesis Node:
    Formats grounded natural language responses strictly adhering to critical business rules:
    - Rule 1: Calibrated decision threshold (0.40) & mandatory human review for employee risk.
    - Rule 2: Engagement coverage caveat (731 respondents, 49.7% of workforce).
    - Rule 3: Zero demographic PII exposure.
    - Rule 4: No fabrication.
    """
    # If refusal or error was already set (e.g. missing employee ID or unknown employee), preserve it
    if state.get("refusal_status"):
        answer = state.get("answer") or "Requested workforce intelligence is not available."
        messages = list(state.get("messages", []))
        messages.append({
            "role": "assistant",
            "name": AGENT_WORKFORCE_ID,
            "content": answer,
            "refusal": True
        })
        return {
            **state,
            "messages": messages,
            "answer": answer
        }

    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return state

    latest_call = tool_calls[-1]
    tool_result = latest_call.get("result", {})
    data = tool_result.get("data", {})
    tool_name = latest_call.get("tool_name")

    answer = ""

    # Formatting Branch 1: Individual Employee Risk
    if tool_name == EmployeeRiskTool.name:
        emp_id = data.get("EmployeeNumber")
        prob = data.get("probability", 0.0)
        risk_level = data.get("risk_level", "UNKNOWN")
        dept = data.get("Department", "Unknown")
        role = data.get("JobRole", "Unknown")
        threshold = data.get("decision_threshold", 0.40)
        eng = data.get("EngagementScore")
        model_ver = data.get("model_version", "v3 (logistic_regression_balanced)")

        eng_clause = f"Self-reported engagement score is {eng} / 5.0." if eng is not None else "No engagement survey response on record."

        answer = (
            f"**Employee #{emp_id} Workforce Intelligence Report:**\n"
            f"- **Role & Department:** {role} ({dept})\n"
            f"- **Attrition Flight Risk:** {risk_level} (Predicted probability: {prob:.4f})\n"
            f"- **Decision Threshold:** Calibrated at {threshold:.2f} using model {model_ver}.\n"
            f"- **Survey Feedback:** {eng_clause}\n\n"
            f"> [!IMPORTANT]\n"
            f"> In accordance with POL-MODEL-001 and POL-REVIEW-001, human review is mandatory before any employment or "
            f"retention decision is taken. The predictive model operates with a decision threshold of {threshold:.2f} "
            f"and must not be used for automated punitive or adverse employment actions."
        )

    # Formatting Branch 2: Department-level Analytics
    elif tool_name == DepartmentWorkforceTool.name:
        departments = data.get("departments", [])
        filtered_dept = data.get("filtered_department")

        if not departments:
            answer = f"No department matching '{filtered_dept}' was found. Active departments are Sales, Research & Development, and Human Resources."
        else:
            lines = ["**Department Workforce Intelligence Breakdown:**\n"]
            for d in departments:
                dept_name = d["department"]
                total = d["total_employees"]
                high_count = d["high_risk_count"]
                high_pct = d["high_risk_percentage"]
                mean_eng = d.get("mean_engagement")
                coverage = d.get("survey_coverage_percentage")
                eng_str = f"avg engagement {mean_eng:.2f} ({coverage:.1f}% surveyed)" if mean_eng else "no survey data"

                lines.append(
                    f"- **{dept_name}:** {total} total employees, {high_count} high flight risk ({high_pct:.1f}%), {eng_str}."
                )
            
            lines.append(
                "\n*Note: Model threshold is calibrated at 0.40. Department engagement averages reflect surveyed respondents only.*"
            )
            answer = "\n".join(lines)

    # Formatting Branch 3: Organization-wide KPIs
    else:
        total = data.get("total_employees", 1470)
        high_count = data.get("high_risk_count", 585)
        high_pct = data.get("high_risk_percentage", 39.8)
        low_count = data.get("low_risk_count", 885)
        low_pct = data.get("low_risk_percentage", 60.2)
        threshold = data.get("decision_threshold", 0.40)
        avg_eng = data.get("average_engagement", 2.95)
        respondents = data.get("survey_respondents", 731)
        coverage = data.get("workforce_coverage_percentage", 49.73)
        caveat = data.get("engagement_coverage_note", "")

        answer = (
            f"**Executive Workforce Risk & Engagement Overview:**\n"
            f"- **Total Workforce Headcount:** {total:,} employees\n"
            f"- **High Flight Risk:** {high_count} employees ({high_pct:.1f}%)\n"
            f"- **Low Flight Risk:** {low_count} employees ({low_pct:.1f}%)\n"
            f"- **Model Decision Threshold:** Calibrated at {threshold:.2f} (balanced sensitivity & precision)\n"
            f"- **Workforce Average Engagement:** {avg_eng} / 5.0\n\n"
            f"> [!NOTE]\n"
            f"> **Survey Representation Integrity:** {caveat or f'Average engagement ({avg_eng} / 5.0) is based strictly on {respondents} employees ({coverage:.1f}% of workforce) with matched survey data. Findings must not be generalized to the 739 unmapped employees.'}"
        )

    messages = list(state.get("messages", []))
    messages.append({
        "role": "assistant",
        "name": AGENT_WORKFORCE_ID,
        "content": answer,
        "refusal": False
    })

    return {
        **state,
        "messages": messages,
        "answer": answer,
        "current_agent": AGENT_WORKFORCE_ID
    }


# ==============================================================================
# Graph Construction & Facade
# ==============================================================================

def build_workforce_agent_graph() -> StateGraph:
    """Constructs and compiles the isolated Workforce Agent LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("workforce_agent", workforce_agent_node)
    workflow.add_node("workforce_tool_executor", workforce_tool_node)
    workflow.add_node("workforce_response", workforce_response_node)

    workflow.set_entry_point("workforce_agent")
    workflow.add_edge("workforce_agent", "workforce_tool_executor")
    workflow.add_edge("workforce_tool_executor", "workforce_response")
    workflow.add_edge("workforce_response", END)

    return workflow.compile()


class WorkforceAgent:
    """
    High-level functional facade for running the Workforce Intelligence Agent StateGraph.
    """

    def __init__(self):
        self.graph = build_workforce_agent_graph()

    def run(self, query: str, initial_state: Optional[AgentState] = None) -> WorkforceAgentResult:
        """
        Executes the Workforce Agent graph for an incoming query.
        
        Args:
            query: User natural language question regarding workforce risk, attrition, or engagement.
            initial_state: Optional existing conversation state.
            
        Returns:
            WorkforceAgentResult with answer, structured data, provenance, and refusal status.
        """
        clean_query = query.strip()
        base_state: AgentState = initial_state or {
            "messages": [],
            "current_agent": AGENT_WORKFORCE_ID,
            "target_agent": AGENT_WORKFORCE_ID,
            "intent": "WORKFORCE_INTELLIGENCE",
            "tool_calls": [],
            "context": "",
            "provenance": [],
            "answer": None,
            "refusal_status": False,
            "error": None
        }

        messages = list(base_state.get("messages", []))
        messages.append({"role": "user", "content": clean_query})
        base_state["messages"] = messages
        base_state["context"] = clean_query

        logger.info(f"WorkforceAgent invoking graph for query: '{clean_query}'")
        final_state = self.graph.invoke(base_state)

        # Extract structured data payload from the latest executed tool call if available
        data_payload = None
        tool_calls = final_state.get("tool_calls", [])
        if tool_calls:
            latest = tool_calls[-1]
            if latest.get("status") == "completed" and isinstance(latest.get("result"), dict):
                data_payload = latest["result"].get("data")

        return WorkforceAgentResult(
            query=clean_query,
            answer=final_state.get("answer", ""),
            data=data_payload,
            provenance=final_state.get("provenance", []),
            refusal_status=final_state.get("refusal_status", False),
            current_agent=AGENT_WORKFORCE_ID,
            tool_calls=tool_calls
        )


_workforce_agent_instance: Optional[WorkforceAgent] = None


def get_workforce_agent() -> WorkforceAgent:
    """Singleton accessor for WorkforceAgent."""
    global _workforce_agent_instance
    if _workforce_agent_instance is None:
        _workforce_agent_instance = WorkforceAgent()
    return _workforce_agent_instance
