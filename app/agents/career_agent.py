"""
app/agents/career_agent.py
==========================
Career Specialist Agent (Agent 4 of 5) for the Enterprise HR AI Platform.
Constructed using LangGraph StateGraph specifications.

Architecture:
    START
      │
      ▼
  career_agent_node (Pure Decision / Parameter Extraction Boundary)
      │
      ▼
  career_tool_node (Strict Execution Whitelist Boundary)
      │
      ▼
  career_response_node (Deterministic Governance & Policy Synthesis)
      │
      ▼
     END

Mandatory Governance:
1. Career progression is developmental/advisory.
2. Never promise promotion or guaranteed placement.
3. Human review is required per POL-REVIEW-001.
4. If YearsSinceLastPromotion >= 4, explicitly cite POL-CAREER-001 Rule 3 and recommend
   initiating an active Career Pathing Review between employee, manager, and HR partner.
5. Manager role: disclose O*NET placeholder 11-9199.00 (very_low confidence) and require
   individualized departmental leadership assessment.
6. Healthcare Representative & Sales Representative: disclose dual mapping to 41-3091.00.
7. Missing employee ID: never guess an employee; request numeric ID.
8. Disclose synthetic MVP data nature where benchmark skills or baseline tenures are discussed.
"""

from typing import Dict, Any, List, Optional
import re
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.tools.career_tool import (
    RoleCareerPathwayTool,
    EmployeePromotionReadinessTool,
    RoleCompetencyComparisonTool,
    CAREER_AUTHORIZED_TOOLS
)
from app.utils.logger import logger

AGENT_CAREER_ID = "career_agent"


# ==============================================================================
# Pydantic Structured Response Model
# ==============================================================================

class CareerAgentResult(BaseModel):
    """Structured Pydantic response returned by the Career Agent."""
    query: str = Field(..., description="Original user query.")
    answer: str = Field(..., description="Synthesized natural-language career guidance.")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Raw structured data payload.")
    provenance: List[Dict[str, Any]] = Field(default_factory=list, description="Verified audit source records.")
    refusal_status: bool = Field(default=False, description="True if query could not be fulfilled or required an ID.")
    current_agent: str = Field(default=AGENT_CAREER_ID, description="Agent identifier.")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Audited list of executed tool calls.")


# ==============================================================================
# Node 1: Decision & Parameter Extraction Node (Pure Planning)
# ==============================================================================

def career_agent_node(state: AgentState) -> AgentState:
    """
    Career Planning Node:
    Extracts parameters (employee_id, role_name, current_role, target_role) from query.
    Emits ONLY pending tool_calls. NEVER calls backend services directly.
    """
    messages = state.get("messages", [])
    query = messages[-1].get("content", "") if messages else state.get("context", "")
    logger.info(f"CareerAgent decision node evaluating query: '{query[:60]}...'")

    q_lower = query.lower()

    # 1. Check for Employee Promotion Readiness queries
    has_employee_token = bool(re.search(r"\b(employee|worker|staff|person|candidate)\b", q_lower))
    has_readiness_token = bool(
        re.search(r"\b(promotion|promotion readiness|readiness|stagnant|stagnation|velocity)\b", q_lower)
    )
    
    emp_match = re.search(r"(?:employee|emp|id|#)\s*#?\s*(\d+)", q_lower)
    if not emp_match:
        emp_match = re.search(r"\b(\d+)\b", q_lower)

    if (has_employee_token and has_readiness_token) or (emp_match and has_readiness_token):
        if emp_match:
            emp_id = int(emp_match.group(1))
            tool_call = {
                "tool_name": EmployeePromotionReadinessTool.TOOL_NAME,
                "args": {"employee_id": emp_id},
                "status": "pending"
            }
            return {
                **state,
                "current_agent": AGENT_CAREER_ID,
                "tool_calls": [tool_call],
                "error": None
            }
        else:
            # Query asks for an employee's promotion readiness but omits the ID
            logger.info("Career query requests employee promotion readiness without numeric ID. Halting without guessing.")
            return {
                **state,
                "current_agent": AGENT_CAREER_ID,
                "tool_calls": [],
                "answer": (
                    "Please specify the numeric Employee ID (for example, 'Assess promotion readiness for employee #13') "
                    "to retrieve empirical career velocity, promotion interval, and tenure stagnation diagnostics."
                ),
                "refusal_status": True,
                "error": "Missing employee ID"
            }

    # 2. Check for Role Competency Comparison (current -> target)
    comparison_patterns = [
        r"compare\s+([a-zA-Z\s]+?)\s+(?:to|with|and|vs|versus)\s+([a-zA-Z\s]+)",
        r"transition\s+(?:from\s+)?([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)",
        r"competenc(?:y|ies)\s+(?:for|between|from)\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)",
        r"missing\s+competencies\s+for\s+(?:the\s+)?next\s+role\s+from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)"
    ]
    
    for pat in comparison_patterns:
        m = re.search(pat, q_lower)
        if m:
            r1 = m.group(1).strip()
            r2 = m.group(2).strip()
            # Clean trailing question terms
            r2 = re.sub(r"\?.*", "", r2).strip()
            tool_call = {
                "tool_name": RoleCompetencyComparisonTool.TOOL_NAME,
                "args": {"current_role": r1, "target_role": r2},
                "status": "pending"
            }
            return {
                **state,
                "current_agent": AGENT_CAREER_ID,
                "tool_calls": [tool_call],
                "error": None
            }

    # Check for general "missing competencies for the next role" without explicit pair
    if "missing competencies" in q_lower or "competency gap" in q_lower:
        # Check if role mentioned
        known_roles = [
            "sales representative", "sales executive", "laboratory technician",
            "research scientist", "research director", "healthcare representative",
            "manufacturing director", "human resources", "manager"
        ]
        matched_role = None
        for kr in known_roles:
            if kr in q_lower:
                matched_role = kr
                break
        if matched_role:
            # Map to its default progressive pair
            default_targets = {
                "sales representative": "Sales Executive",
                "sales executive": "Manager",
                "laboratory technician": "Research Scientist",
                "research scientist": "Research Director",
                "research director": "Manager",
                "healthcare representative": "Sales Executive",
                "manufacturing director": "Manager",
                "human resources": "Manager"
            }
            targ = default_targets.get(matched_role, "Manager")
            tool_call = {
                "tool_name": RoleCompetencyComparisonTool.TOOL_NAME,
                "args": {"current_role": matched_role, "target_role": targ},
                "status": "pending"
            }
            return {
                **state,
                "current_agent": AGENT_CAREER_ID,
                "tool_calls": [tool_call],
                "error": None
            }

    # 3. Default / Fallback: Role Career Pathway & O*NET Mapping Query
    known_roles = [
        "sales representative", "sales executive", "laboratory technician",
        "research scientist", "research director", "healthcare representative",
        "manufacturing director", "human resources", "manager",
        "senior scientist", "lead engineer", "software engineer"
    ]
    detected_role = "Laboratory Technician"  # default standard reference role
    for kr in known_roles:
        if kr in q_lower:
            detected_role = kr
            break

    tool_call = {
        "tool_name": RoleCareerPathwayTool.TOOL_NAME,
        "args": {"role_name": detected_role},
        "status": "pending"
    }
    return {
        **state,
        "current_agent": AGENT_CAREER_ID,
        "tool_calls": [tool_call],
        "error": None
    }


# ==============================================================================
# Node 2: Execution Boundary Node (Tool Validation & Invocation)
# ==============================================================================

def career_tool_node(state: AgentState) -> AgentState:
    """
    Career Tool Execution Boundary:
    Validates tool against CAREER_AUTHORIZED_TOOLS. Executes tool and catches errors.
    """
    tool_calls = list(state.get("tool_calls", []))
    if not tool_calls:
        return state

    latest_call = dict(tool_calls[-1])
    tool_name = latest_call.get("tool_name")
    args = latest_call.get("args", {})

    logger.info(f"Career execution boundary checking tool: '{tool_name}'")

    # Whitelist Verification
    if tool_name not in CAREER_AUTHORIZED_TOOLS:
        logger.warning(f"BLOCKED unauthorized tool execution attempt: '{tool_name}'")
        latest_call["status"] = "rejected"
        latest_call["error"] = f"Tool '{tool_name}' is not authorized for Career Agent."
        return {
            **state,
            "tool_calls": tool_calls[:-1] + [latest_call],
            "answer": f"Execution halted: Tool '{tool_name}' is not an authorized Career capability.",
            "refusal_status": True,
            "error": "Unauthorized tool call"
        }

    # Dispatch tool
    tool_map = {
        RoleCareerPathwayTool.TOOL_NAME: RoleCareerPathwayTool(),
        EmployeePromotionReadinessTool.TOOL_NAME: EmployeePromotionReadinessTool(),
        RoleCompetencyComparisonTool.TOOL_NAME: RoleCompetencyComparisonTool(),
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
                    f"Employee #{args.get('employee_id')} was not found in employee career records. "
                    "Please verify the employee identifier and try again."
                )
            else:
                answer_text = f"Career analysis failed: {raw_result.get('message')}"

            return {
                **state,
                "tool_calls": tool_calls[:-1] + [latest_call],
                "answer": answer_text,
                "refusal_status": True,
                "error": error_type
            }

        # Successful tool execution
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
        logger.error(f"Career tool execution threw unhandled exception for '{tool_name}': {e}", exc_info=True)
        latest_call["status"] = "failed"
        latest_call["error"] = str(e)
        return {
            **state,
            "tool_calls": tool_calls[:-1] + [latest_call],
            "answer": f"Unable to complete career analysis: {str(e)}",
            "refusal_status": True,
            "error": str(e)
        }


# ==============================================================================
# Node 3: Synthesis & Response Node (Policy Guardrails)
# ==============================================================================

def career_response_node(state: AgentState) -> AgentState:
    """
    Career Response Synthesis Node:
    Enforces mandatory corporate governance, developmental guardrails,
    manager cohort notices, and synthetic data transparency.
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
    # Case 1: Role Career Pathway & O*NET Mapping
    # ==========================================================================
    if tool_name == RoleCareerPathwayTool.TOOL_NAME:
        role = result.get("role_name")
        lvl_range = result.get("current_level_range")
        onet_code = result.get("onet_soc_code")
        onet_title = result.get("onet_title")
        confidence = result.get("match_confidence")
        vertical = result.get("vertical_pathways", [])
        lateral = result.get("lateral_pathways", [])
        comps = result.get("benchmark_competencies", {})
        essential = comps.get("essential_skills", [])
        software = comps.get("software_tools", [])
        is_mgr = result.get("is_manager", False)
        is_dual = result.get("is_dual_mapped", False)

        vert_str = " -> ".join([role] + vertical) if vertical else f"{role} (Top Functional Tier)"
        lat_str = ", ".join(lateral) if lateral else "No direct lateral crosswalk documented"
        ess_str = ", ".join(essential[:5]) if essential else "Department-level assessment required"
        soft_str = ", ".join(software[:5]) if software else "Department-level assessment required"

        answer = (
            f"### Career Progression Pathway: {role}\n"
            f"- **Observed Level Tier:** {lvl_range}\n"
            f"- **Canonical O*NET-SOC Code:** `{onet_code}` — *{onet_title}*\n"
            f"- **Taxonomic Match Confidence:** `{confidence}`\n"
            f"- **Vertical Progression Ladder:** {vert_str}\n"
            f"- **Lateral / Cross-Functional Pathways:** {lat_str}\n"
            f"- **Benchmark Essential Competencies:** {ess_str}\n"
            f"- **Benchmark Software Tools:** {soft_str}\n\n"
        )

        if is_mgr:
            answer += (
                "**Managerial Protocol Notice (POL-CAREER-001 Rule 4):**\n"
                "The generic Job Role 'Manager' is mapped to placeholder `11-9199.00` with `very_low` confidence. "
                "Career advancement to executive functional leadership requires individualized leadership assessment "
                "coordinated directly through departmental HR leadership frameworks.\n\n"
            )

        if is_dual:
            answer += (
                "**Dual-Mapping Notice (POL-CAREER-001 Rule 4):**\n"
                f"`{role}` is dual-mapped with its peer sales cohort to O*NET `41-3091.00`. "
                "Lateral cross-functional mobility between these roles is highly transferable.\n\n"
            )

        answer += (
            "**Governance & Developmental Policy (POL-CAREER-001 & POL-REVIEW-001):**\n"
            "All career pathways represent developmental milestones along JobLevel tiers (1 to 5). "
            "Pathway modeling indicates capability readiness and does not constitute a promise of promotion "
            "or guaranteed staffing placement. Written manager and department leader review is required."
        )

        return {
            **state,
            "answer": answer,
            "refusal_status": False
        }

    # ==========================================================================
    # Case 2: Employee Promotion Readiness & Stagnation
    # ==========================================================================
    if tool_name == EmployeePromotionReadinessTool.TOOL_NAME:
        emp_id = result.get("EmployeeNumber")
        role = result.get("JobRole")
        lvl = result.get("JobLevel")
        dept = result.get("Department")
        in_role = result.get("YearsInCurrentRole")
        since_promo = result.get("YearsSinceLastPromotion")
        at_co = result.get("YearsAtCompany")
        perf = result.get("PerformanceRating")
        status_val = result.get("stagnation_status")
        review_req = result.get("career_pathing_review_required")
        rationale = result.get("stagnation_rationale")
        next_role = result.get("next_ladder_role")
        is_mgr = result.get("is_manager", False)

        answer = (
            f"### Promotion Readiness Diagnostic for Employee #{emp_id}\n"
            f"- **Current Role:** {role} (JobLevel {lvl}) — {dept}\n"
            f"- **Tenure in Current Role:** {in_role} years\n"
            f"- **Years Since Last Promotion:** {since_promo} years\n"
            f"- **Total Company Tenure:** {at_co} years\n"
            f"- **Performance Rating:** {perf} / 4\n"
            f"- **Promotion Velocity Status:** **{status_val}**\n"
            f"- **Active Career Review Required:** {'YES' if review_req else 'NO'}\n"
            f"- **Next Vertical Ladder Role:** {next_role}\n\n"
            f"**Diagnostic Rationale:**\n{rationale}\n\n"
        )

        if since_promo >= 4 or status_val == "STAGNANT":
            answer += (
                "**Stagnation Mitigation Protocol (POL-CAREER-001 Rule 3):**\n"
                "Because tenure without promotion meets or exceeds 4 years, an active Career Pathing Review must be "
                "initiated between the employee, direct manager, and HR business partner to evaluate advancement "
                "or lateral cross-functional mobility to mitigate turnover risk.\n\n"
            )
        elif status_val == "REVIEW_RECOMMENDED":
            answer += (
                "**Career Pathing Recommendation (POL-CAREER-001 Rule 3):**\n"
                "Employee is approaching the promotional stagnation boundary. A developmental career review "
                "is recommended to establish clear milestone targets for advancing to the next JobLevel tier.\n\n"
            )

        if is_mgr:
            answer += (
                "**Managerial Advancement Notice (POL-CAREER-001 Rule 4):**\n"
                "Advancement from Manager to Executive Leadership requires departmental succession planning "
                "rather than automated promotion velocity scoring.\n\n"
            )

        answer += (
            "**Governance & Developmental Policy (POL-CAREER-001 & POL-REVIEW-001):**\n"
            "Promotion readiness assessments are developmental decision-support tools. Career advancement is subject "
            "to organizational vacancies, business need, and formalized human review. Career intervals derive "
            "from synthetic baseline data for workforce planning demonstrations."
        )

        return {
            **state,
            "answer": answer,
            "refusal_status": False
        }

    # ==========================================================================
    # Case 3: Role Competency Comparison & Transferability Delta
    # ==========================================================================
    if tool_name == RoleCompetencyComparisonTool.TOOL_NAME:
        c_role = result.get("current_role")
        t_role = result.get("target_role")
        t_soc = result.get("target_soc_code")
        t_title = result.get("target_onet_title")
        t_desc = result.get("target_description")
        shared_ess = result.get("shared_essential_skills", [])
        miss_ess = result.get("missing_essential_skills", [])
        shared_soft = result.get("shared_software_tools", [])
        miss_soft = result.get("missing_software_tools", [])
        score = result.get("transferability_score", 0.0)

        shared_ess_str = ", ".join(shared_ess) if shared_ess else "None identified"
        miss_ess_str = ", ".join(miss_ess) if miss_ess else "None (Full Essential Overlap)"
        shared_soft_str = ", ".join(shared_soft) if shared_soft else "None identified"
        miss_soft_str = ", ".join(miss_soft) if miss_soft else "None (Full Software Overlap)"

        answer = (
            f"### Competency Transferability Analysis: {c_role} -> {t_role}\n"
            f"- **Target O*NET-SOC Code:** `{t_soc}` — *{t_title}*\n"
            f"- **Canonical Target Role Summary:** {t_desc}\n"
            f"- **Transferability Overlap Score:** {int(score * 100)}%\n\n"
            f"**Competency Breakdown:**\n"
            f"- **Shared Essential Skills:** {shared_ess_str}\n"
            f"- **Net-New Essential Skills Required:** {miss_ess_str}\n"
            f"- **Shared Software Tools:** {shared_soft_str}\n"
            f"- **Net-New Software Tools Required:** {miss_soft_str}\n\n"
            "**Governance & Developmental Policy (POL-CAREER-001 Rule 1 & POL-SKILL-001):**\n"
            "Competency transferability analysis measures capability overlap against canonical O*NET benchmark profiles. "
            "High overlap indicates developmental readiness for role transition, not guaranteed placement or immediate staffing. "
            "Formal cross-functional transfers require written authorization per POL-REVIEW-001."
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

def build_career_agent_graph() -> StateGraph:
    """Constructs and compiles the isolated Career Agent LangGraph StateGraph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("career_agent_node", career_agent_node)
    workflow.add_node("career_tool_node", career_tool_node)
    workflow.add_node("career_response_node", career_response_node)

    workflow.add_edge(START, "career_agent_node")
    workflow.add_edge("career_agent_node", "career_tool_node")
    workflow.add_edge("career_tool_node", "career_response_node")
    workflow.add_edge("career_response_node", END)

    return workflow.compile()


class CareerAgent:
    """
    High-level facade for the Career Specialist Agent.
    Provides a standardized run() method returning CareerAgentResult.
    """

    def __init__(self):
        self.graph = build_career_agent_graph()

    def run(self, query: str, initial_state: Optional[AgentState] = None) -> CareerAgentResult:
        logger.info(f"Invoking Career Agent graph for query: '{query[:60]}...'")

        base_state: AgentState = initial_state or {
            "messages": [],
            "current_agent": AGENT_CAREER_ID,
            "target_agent": AGENT_CAREER_ID,
            "intent": "CAREER",
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

        return CareerAgentResult(
            query=query,
            answer=final_state.get("answer") or "No career guidance could be synthesized.",
            data=data_payload,
            provenance=final_state.get("provenance", []),
            refusal_status=final_state.get("refusal_status", False),
            current_agent=final_state.get("current_agent", AGENT_CAREER_ID),
            tool_calls=tool_calls
        )


_career_agent_singleton: Optional[CareerAgent] = None

def get_career_agent() -> CareerAgent:
    """Singleton accessor for CareerAgent facade."""
    global _career_agent_singleton
    if _career_agent_singleton is None:
        _career_agent_singleton = CareerAgent()
    return _career_agent_singleton
