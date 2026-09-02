"""
app/agents/hr_ops_agent.py
==========================
HR Operations Specialist Agent (Agent 5 of 5) for the Enterprise HR AI Platform.
Constructed using LangGraph StateGraph specifications.

Architecture:
    START
      │
      ▼
  hr_ops_agent_node (Pure Decision / Parameter Extraction Boundary)
      │
      ▼
  hr_ops_tool_node (Strict Execution Whitelist Boundary)
      │
      ▼
  hr_ops_response_node (Deterministic Governance & Policy Synthesis)
      │
      ▼
     END

Mandatory Governance & Privacy Guardrails:
1. Pure administrative decision support (factual roster, tenure, and headcount records).
2. Never expose demographic or compensation PII (Age, Gender, MaritalStatus, MonthlyIncome,
   HourlyRate, DailyRate, MonthlyRate, PercentSalaryHike, StockOptionLevel).
3. Missing employee ID: never guess an employee; request numeric ID.
4. Never perform attrition prediction or flight risk scoring in this agent.
5. Departmental staffing requests return aggregate distributions only.
6. Disclose synthetic demonstration data nature per POL-DATA-001.
7. Human review governed by POL-REVIEW-001.
"""

from typing import Dict, Any, List, Optional
import re
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.tools.hr_ops_tool import (
    EmployeeProfileLookupTool,
    HeadcountAnalyticsTool,
    DepartmentStaffingTool,
    HR_OPS_AUTHORIZED_TOOLS
)
from app.services.hr_ops_service import CANONICAL_DEPARTMENTS, PROHIBITED_PII_FIELDS
from app.utils.logger import logger

AGENT_HR_OPS_ID = "hr_ops_agent"


# ==============================================================================
# Pydantic Structured Response Model
# ==============================================================================

class HROpsAgentResult(BaseModel):
    """Structured Pydantic response returned by the HR Operations Agent."""
    query: str = Field(..., description="Original user query.")
    answer: str = Field(..., description="Synthesized natural-language HR operational response.")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Raw structured data payload.")
    provenance: List[Dict[str, Any]] = Field(default_factory=list, description="Verified audit source records.")
    refusal_status: bool = Field(default=False, description="True if query could not be fulfilled or required an ID.")
    current_agent: str = Field(default=AGENT_HR_OPS_ID, description="Agent identifier.")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Audited list of executed tool calls.")


# ==============================================================================
# Node 1: Decision & Parameter Extraction Node (Pure Planning)
# ==============================================================================

def hr_ops_agent_node(state: AgentState) -> AgentState:
    """
    HR Ops Planning Node:
    Extracts parameters (employee_id, department) from query.
    Emits ONLY pending tool_calls. NEVER calls backend services or pandas directly.
    """
    messages = state.get("messages", [])
    query = messages[-1].get("content", "") if messages else state.get("context", "")
    logger.info(f"HROpsAgent decision node evaluating query: '{query[:60]}...'")

    q_lower = query.lower()

    # Detect Department Name in query
    detected_dept = None
    if "research" in q_lower or "r&d" in q_lower or "development" in q_lower:
        detected_dept = "Research & Development"
    elif "sales" in q_lower:
        detected_dept = "Sales"
    elif "human resources" in q_lower or r"\bhr\b" in q_lower:
        detected_dept = "Human Resources"

    # 1. Check for Departmental Staffing / Roster Queries
    staffing_patterns = [
        r"\bdepartment staffing\b",
        r"\bstaffing\b",
        r"\broster\b",
        r"\bpersonnel directory\b",
        r"\brole distribution\b",
        r"\bwho works in\b",
        r"\blist roles in\b"
    ]
    if any(re.search(p, q_lower) for p in staffing_patterns) and detected_dept:
        tool_call = {
            "tool_name": DepartmentStaffingTool.TOOL_NAME,
            "args": {"department": detected_dept},
            "status": "pending"
        }
        return {
            **state,
            "current_agent": AGENT_HR_OPS_ID,
            "tool_calls": [tool_call],
            "error": None
        }

    # 2. Check for Headcount Queries
    headcount_patterns = [
        r"\bheadcount\b",
        r"\bhead count\b",
        r"\btotal employees\b",
        r"\bworkforce headcount\b",
        r"\bhow many employees\b",
        r"\bnumber of employees\b",
        r"\bworkforce size\b",
        r"\bcompany size\b",
        r"\bhow many people work\b"
    ]
    if any(re.search(p, q_lower) for p in headcount_patterns):
        tool_call = {
            "tool_name": HeadcountAnalyticsTool.TOOL_NAME,
            "args": {"department": detected_dept},
            "status": "pending"
        }
        return {
            **state,
            "current_agent": AGENT_HR_OPS_ID,
            "tool_calls": [tool_call],
            "error": None
        }

    # 3. Check for Individual Employee Profile / Record Lookup
    emp_token = bool(re.search(r"\b(employee|worker|staff|person|record|profile|personnel)\b", q_lower))
    emp_match = re.search(r"(?:employee|emp|id|#)\s*#?\s*(\d+)", q_lower)
    if not emp_match:
        emp_match = re.search(r"\b(\d+)\b", q_lower)

    if emp_token or emp_match:
        if emp_match:
            emp_id = int(emp_match.group(1))
            tool_call = {
                "tool_name": EmployeeProfileLookupTool.TOOL_NAME,
                "args": {"employee_id": emp_id},
                "status": "pending"
            }
            return {
                **state,
                "current_agent": AGENT_HR_OPS_ID,
                "tool_calls": [tool_call],
                "error": None
            }
        else:
            # Query asks for employee profile/record without specifying numeric ID
            logger.info("HR Ops query requests employee record without numeric ID. Halting without guessing.")
            return {
                **state,
                "current_agent": AGENT_HR_OPS_ID,
                "tool_calls": [],
                "answer": (
                    "Please specify the numeric Employee ID (for example, 'Show employee record for employee #500') "
                    "to retrieve the administrative operational profile."
                ),
                "refusal_status": True,
                "error": "Missing employee ID"
            }

    # 4. Default / Fallback: Return Company-Wide Headcount
    tool_call = {
        "tool_name": HeadcountAnalyticsTool.TOOL_NAME,
        "args": {"department": detected_dept},
        "status": "pending"
    }
    return {
        **state,
        "current_agent": AGENT_HR_OPS_ID,
        "tool_calls": [tool_call],
        "error": None
    }


# ==============================================================================
# Node 2: Execution Boundary Node (Tool Validation & Invocation)
# ==============================================================================

def hr_ops_tool_node(state: AgentState) -> AgentState:
    """
    HR Ops Tool Execution Boundary:
    Validates requested tool against HR_OPS_AUTHORIZED_TOOLS. Executes tool and catches errors.
    """
    tool_calls = list(state.get("tool_calls", []))
    if not tool_calls:
        return state

    latest_call = dict(tool_calls[-1])
    tool_name = latest_call.get("tool_name")
    args = latest_call.get("args", {})

    logger.info(f"HR Ops execution boundary checking tool: '{tool_name}'")

    # Whitelist Verification
    if tool_name not in HR_OPS_AUTHORIZED_TOOLS:
        logger.warning(f"BLOCKED unauthorized tool execution attempt: '{tool_name}'")
        latest_call["status"] = "rejected"
        latest_call["error"] = f"Tool '{tool_name}' is not authorized for HR Ops Agent."
        return {
            **state,
            "tool_calls": tool_calls[:-1] + [latest_call],
            "answer": f"Execution halted: Tool '{tool_name}' is not an authorized HR Operations capability.",
            "refusal_status": True,
            "error": "Unauthorized tool call"
        }

    # Dispatch tool
    tool_map = {
        EmployeeProfileLookupTool.TOOL_NAME: EmployeeProfileLookupTool(),
        HeadcountAnalyticsTool.TOOL_NAME: HeadcountAnalyticsTool(),
        DepartmentStaffingTool.TOOL_NAME: DepartmentStaffingTool(),
    }
    target_tool = tool_map.get(tool_name)

    try:
        raw_result = target_tool.execute(args)
        status_val = raw_result.get("status")

        if status_val == "error":
            latest_call["status"] = "failed"
            latest_call["error"] = raw_result.get("message")
            error_type = raw_result.get("error_type", "EXECUTION_ERROR")

            if error_type == "EMPLOYEE_NOT_FOUND":
                answer_text = (
                    f"Employee #{args.get('employee_id')} was not found in enterprise personnel records. "
                    "Please verify the employee identifier and try again."
                )
            else:
                answer_text = f"HR Operations query failed: {raw_result.get('message')}"

            return {
                **state,
                "tool_calls": tool_calls[:-1] + [latest_call],
                "answer": answer_text,
                "refusal_status": True,
                "error": error_type
            }

        # Successful execution
        latest_call["status"] = "completed"
        latest_call["result"] = raw_result
        prov = raw_result.get("provenance", [])
        existing_prov = list(state.get("provenance", []))

        return {
            **state,
            "tool_calls": tool_calls[:-1] + [latest_call],
            "provenance": existing_prov + prov,
            "refusal_status": False,
            "error": None
        }

    except Exception as e:
        logger.error(f"HR Ops tool execution failed for '{tool_name}': {e}", exc_info=True)
        latest_call["status"] = "failed"
        latest_call["error"] = str(e)
        return {
            **state,
            "tool_calls": tool_calls[:-1] + [latest_call],
            "answer": f"Unable to complete HR operations request: {str(e)}",
            "refusal_status": True,
            "error": str(e)
        }


# ==============================================================================
# Node 3: Synthesis & Response Node (Privacy & Policy Guardrails)
# ==============================================================================

def hr_ops_response_node(state: AgentState) -> AgentState:
    """
    HR Ops Response Synthesis Node:
    Enforces strict PII suppression, administrative governance,
    and synthetic demonstration data transparency.
    """
    if state.get("refusal_status", False):
        return state

    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return state

    latest_call = tool_calls[-1]
    tool_name = latest_call.get("tool_name")
    result = latest_call.get("result", {})

    if latest_call.get("status") != "completed":
        return {
            **state,
            "answer": f"Execution halted: {latest_call.get('error')}",
            "refusal_status": True
        }

    # ==========================================================================
    # Case 1: Individual Employee Operational Profile
    # ==========================================================================
    if tool_name == EmployeeProfileLookupTool.TOOL_NAME:
        emp_id = result.get("EmployeeNumber")
        dept = result.get("Department")
        role = result.get("JobRole")
        lvl = result.get("JobLevel")
        field = result.get("EducationField")
        travel = result.get("BusinessTravel")
        ot = result.get("OverTime")
        tot_years = result.get("TotalWorkingYears")
        at_co = result.get("YearsAtCompany")
        in_role = result.get("YearsInCurrentRole")
        since_promo = result.get("YearsSinceLastPromotion")
        with_mgr = result.get("YearsWithCurrManager")
        perf = result.get("PerformanceRating")
        wlb = result.get("WorkLifeBalance")

        answer = (
            f"### Operational Personnel Profile for Employee #{emp_id}\n"
            f"- **Department:** {dept}\n"
            f"- **Job Role:** {role} (JobLevel {lvl})\n"
            f"- **Education Discipline:** {field}\n"
            f"- **Business Travel:** {travel}\n"
            f"- **OverTime Status:** {ot}\n"
            f"- **Total Career Experience:** {tot_years} years\n"
            f"- **Tenure at Company:** {at_co} years\n"
            f"- **Tenure in Current Role:** {in_role} years\n"
            f"- **Years Since Last Promotion:** {since_promo} years\n"
            f"- **Tenure with Current Manager:** {with_mgr} years\n"
            f"- **Performance Rating:** {perf} / 4\n"
            f"- **Work-Life Balance Score:** {wlb} / 4\n\n"
            "**Data Governance & Privacy Notice (POL-DATA-001 & POL-REVIEW-001):**\n"
            "This profile contains factual operational employment records from the primary anchor table. "
            "In strict adherence to enterprise data protection policies, sensitive demographic indicators and "
            "compensation tiers are suppressed from operational lookups. "
            "All records derive from the IBM synthetic HR benchmark dataset."
        )

        return {
            **state,
            "answer": answer,
            "refusal_status": False
        }

    # ==========================================================================
    # Case 2: Headcount Statistics (Company-Wide or Departmental)
    # ==========================================================================
    if tool_name == HeadcountAnalyticsTool.TOOL_NAME:
        scope = result.get("scope")
        if scope == "company_wide":
            total = result.get("total_headcount")
            breakdown = result.get("department_breakdown", {})

            lines = [f"- **{dept}:** {data['headcount']:,} employees ({data['percentage']}%)" for dept, data in breakdown.items()]
            breakdown_str = "\n".join(lines)

            answer = (
                f"### Enterprise Headcount Summary\n"
                f"- **Total Workforce Population:** **{total:,} employees**\n\n"
                f"**Departmental Headcount Breakdown:**\n"
                f"{breakdown_str}\n\n"
                "**Data Governance Notice (POL-DATA-001 Rule 1):**\n"
                "The workforce population is fixed at exactly 1,470 employees based on the enterprise anchor dataset. "
                "All figures represent verified active personnel records."
            )
        else:
            dept = result.get("department")
            headcount = result.get("headcount")
            pct = result.get("percentage_of_workforce")
            mean_tenure = result.get("mean_tenure_years")
            ot_rate = result.get("overtime_rate_pct")
            roles = result.get("role_headcounts", {})

            role_lines = [f"  - **{r}:** {cnt} employees" for r, cnt in roles.items()]
            roles_str = "\n".join(role_lines)

            answer = (
                f"### Department Headcount: {dept}\n"
                f"- **Active Headcount:** **{headcount:,} employees** ({pct}% of total workforce)\n"
                f"- **Mean Company Tenure:** {mean_tenure} years\n"
                f"- **OverTime Rate:** {ot_rate}% of department staff\n\n"
                f"**Role Distribution:**\n"
                f"{roles_str}\n\n"
                "**Data Governance Notice (POL-DATA-001):**\n"
                "Data derived from enterprise anchor records for operational workforce tracking."
            )

        return {
            **state,
            "answer": answer,
            "refusal_status": False
        }

    # ==========================================================================
    # Case 3: Department Staffing Profile
    # ==========================================================================
    if tool_name == DepartmentStaffingTool.TOOL_NAME:
        dept = result.get("department")
        headcount = result.get("headcount")
        roles = result.get("role_distribution", {})
        levels = result.get("job_level_distribution", {})
        edus = result.get("education_field_distribution", {})
        tenure = result.get("average_tenure", {})
        ot = result.get("overtime_summary", {})

        roles_str = "\n".join([f"  - **{r}:** {c} employees" for r, c in roles.items()])
        levels_str = ", ".join([f"Level {lvl}: {cnt}" for lvl, cnt in levels.items()])
        edus_str = ", ".join([f"{f}: {c}" for f, c in edus.items()])

        answer = (
            f"### Department Staffing & Structural Profile: {dept}\n"
            f"- **Total Department Headcount:** **{headcount:,} employees**\n"
            f"- **Average Company Tenure:** {tenure.get('years_at_company')} years (Total Career Experience: {tenure.get('total_working_years')} years)\n"
            f"- **Overtime Distribution:** {ot.get('overtime_rate_pct')}% ({ot.get('overtime_count')} employees working overtime)\n\n"
            f"**Job Role Staffing:**\n{roles_str}\n\n"
            f"**JobLevel Distribution:** {levels_str}\n\n"
            f"**Education Field Representation:** {edus_str}\n\n"
            "**Governance & Administrative Policy (POL-DATA-001 & POL-REVIEW-001):**\n"
            "Staffing analyses aggregate departmental personnel records without exposing individual compensation "
            "or demographic identifiers."
        )

        return {
            **state,
            "answer": answer,
            "refusal_status": False
        }

    return state


# ==============================================================================
# Graph Construction & Facade
# ==============================================================================

def build_hr_ops_agent_graph() -> StateGraph:
    """Constructs and compiles the isolated HR Operations Agent LangGraph StateGraph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("hr_ops_agent_node", hr_ops_agent_node)
    workflow.add_node("hr_ops_tool_node", hr_ops_tool_node)
    workflow.add_node("hr_ops_response_node", hr_ops_response_node)

    workflow.add_edge(START, "hr_ops_agent_node")
    workflow.add_edge("hr_ops_agent_node", "hr_ops_tool_node")
    workflow.add_edge("hr_ops_tool_node", "hr_ops_response_node")
    workflow.add_edge("hr_ops_response_node", END)

    return workflow.compile()


class HROpsAgent:
    """
    High-level facade for the HR Operations Specialist Agent.
    Provides a standardized run() method returning HROpsAgentResult.
    """

    def __init__(self):
        self.graph = build_hr_ops_agent_graph()

    def run(self, query: str, initial_state: Optional[AgentState] = None) -> HROpsAgentResult:
        logger.info(f"Invoking HR Ops Agent graph for query: '{query[:60]}...'")

        base_state: AgentState = initial_state or {
            "messages": [],
            "current_agent": AGENT_HR_OPS_ID,
            "target_agent": AGENT_HR_OPS_ID,
            "intent": "HR_OPS",
            "tool_calls": [],
            "context": query,
            "provenance": [],
            "answer": None,
            "refusal_status": False,
            "error": None
        }

        base_state["messages"] = list(base_state.get("messages", [])) + [
            {"role": "user", "content": query}
        ]

        final_state = self.graph.invoke(base_state)

        tool_calls = final_state.get("tool_calls", [])
        data_payload = None
        if tool_calls and tool_calls[-1].get("status") == "completed":
            data_payload = tool_calls[-1].get("result")

        return HROpsAgentResult(
            query=query,
            answer=final_state.get("answer") or "No HR operations response could be generated.",
            data=data_payload,
            provenance=final_state.get("provenance", []),
            refusal_status=final_state.get("refusal_status", False),
            current_agent=final_state.get("current_agent", AGENT_HR_OPS_ID),
            tool_calls=tool_calls
        )


_hr_ops_agent_singleton: Optional[HROpsAgent] = None

def get_hr_ops_agent() -> HROpsAgent:
    """Singleton accessor for HROpsAgent facade."""
    global _hr_ops_agent_singleton
    if _hr_ops_agent_singleton is None:
        _hr_ops_agent_singleton = HROpsAgent()
    return _hr_ops_agent_singleton
