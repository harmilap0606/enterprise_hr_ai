"""
app/agents package.
Enterprise HR AI Agentic Layer.
"""

from app.agents.state import AgentState, PolicyAgentResult, OrchestratorResponse
from app.agents.policy_agent import PolicyAgent, get_policy_agent, build_policy_agent_graph
from app.agents.workforce_agent import (
    WorkforceAgent,
    get_workforce_agent,
    build_workforce_agent_graph,
    WorkforceAgentResult
)
from app.agents.upskilling_agent import (
    UpskillingAgent,
    get_upskilling_agent,
    build_upskilling_agent_graph,
    UpskillingAgentResult
)
from app.agents.career_agent import (
    CareerAgent,
    get_career_agent,
    build_career_agent_graph,
    CareerAgentResult
)
from app.agents.hr_ops_agent import (
    HROpsAgent,
    get_hr_ops_agent,
    build_hr_ops_agent_graph,
    HROpsAgentResult
)
from app.agents.tools.policy_tool import PolicyRetrievalTool
from app.agents.tools.workforce_tool import (
    WorkforceKPITool,
    DepartmentWorkforceTool,
    EmployeeRiskTool,
    WORKFORCE_AUTHORIZED_TOOLS
)
from app.agents.tools.upskilling_tool import (
    EmployeeUpskillingTool,
    OrganizationSkillGapTool,
    UPSKILLING_AUTHORIZED_TOOLS
)
from app.agents.tools.career_tool import (
    RoleCareerPathwayTool,
    EmployeePromotionReadinessTool,
    RoleCompetencyComparisonTool,
    CAREER_AUTHORIZED_TOOLS
)
from app.agents.tools.hr_ops_tool import (
    EmployeeProfileLookupTool,
    HeadcountAnalyticsTool,
    DepartmentStaffingTool,
    HR_OPS_AUTHORIZED_TOOLS
)
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
    AGENT_ORCHESTRATOR,
    AGENT_FALLBACK
)

__all__ = [
    "AgentState",
    "PolicyAgentResult",
    "WorkforceAgentResult",
    "UpskillingAgentResult",
    "CareerAgentResult",
    "HROpsAgentResult",
    "OrchestratorResponse",
    "PolicyAgent",
    "get_policy_agent",
    "build_policy_agent_graph",
    "WorkforceAgent",
    "get_workforce_agent",
    "build_workforce_agent_graph",
    "UpskillingAgent",
    "get_upskilling_agent",
    "build_upskilling_agent_graph",
    "CareerAgent",
    "get_career_agent",
    "build_career_agent_graph",
    "HROpsAgent",
    "get_hr_ops_agent",
    "build_hr_ops_agent_graph",
    "PolicyRetrievalTool",
    "WorkforceKPITool",
    "DepartmentWorkforceTool",
    "EmployeeRiskTool",
    "WORKFORCE_AUTHORIZED_TOOLS",
    "EmployeeUpskillingTool",
    "OrganizationSkillGapTool",
    "UPSKILLING_AUTHORIZED_TOOLS",
    "RoleCareerPathwayTool",
    "EmployeePromotionReadinessTool",
    "RoleCompetencyComparisonTool",
    "CAREER_AUTHORIZED_TOOLS",
    "EmployeeProfileLookupTool",
    "HeadcountAnalyticsTool",
    "DepartmentStaffingTool",
    "HR_OPS_AUTHORIZED_TOOLS",
    "classify_intent",
    "GlobalOrchestrator",
    "get_orchestrator",
    "build_orchestrator_graph",
    "INTENT_POLICY",
    "INTENT_WORKFORCE_INTELLIGENCE",
    "INTENT_UPSKILLING",
    "INTENT_CAREER",
    "INTENT_HR_OPS",
    "INTENT_OUT_OF_DOMAIN",
    "AGENT_POLICY",
    "AGENT_WORKFORCE",
    "AGENT_UPSKILLING",
    "AGENT_CAREER",
    "AGENT_HR_OPS",
    "AGENT_ORCHESTRATOR",
    "AGENT_FALLBACK"
]
