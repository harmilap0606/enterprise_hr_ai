"""
app/api/agents.py
=================
API Router for the Global Enterprise HR AI Agentic Orchestrator.
Endpoint: POST /agents/ask

Dispatches natural-language questions through the LangGraph Orchestrator:
- Intent Classification & Dynamic Routing
- Specialized Agent Delegation (Policy Agent)
- Audited Provenance & Structured Response
"""

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.agents.orchestrator import get_orchestrator
from app.agents.state import OrchestratorResponse
from app.utils.logger import logger

router = APIRouter(prefix="/agents", tags=["Agentic Orchestrator"])


class AgentQuestionRequest(BaseModel):
    question: str = Field(..., description="User question to be analyzed and answered by the agentic platform.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What does POL-MODEL-001 state regarding the decision threshold?"
            }
        }
    }


@router.post("/ask", response_model=OrchestratorResponse, summary="Query Global Agentic Orchestrator")
def ask_agentic_platform(payload: AgentQuestionRequest):
    """
    POST /agents/ask
    Dispatches a user query to the Global LangGraph Orchestrator.
    Routes queries to specialized agents (e.g. Policy Agent) or provides
    explainable capability notices.
    """
    try:
        q = payload.question.strip()
        if not q:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        logger.info(f"Received question on /agents/ask: '{q}'")
        orchestrator = get_orchestrator()
        result = orchestrator.run(q)
        logger.info(f"Orchestrator completed query -> Agent: {result.agent_routed}, Refusal: {result.refusal_status}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing agent query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
