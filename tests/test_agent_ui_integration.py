"""
tests/test_agent_ui_integration.py
==================================
Tests for Streamlit Frontend integration with the Global Agentic Orchestrator:
1. Frontend isolation: confirms ZERO direct agent imports in frontend/dashboard.py.
2. Portal view registration: confirms 'Enterprise AI Agents' exists in portal mode radio.
3. UI execution flow: confirms queries typed in UI dispatch to POST /agents/ask.
4. Example button triggers: confirms clicking example buttons executes through the API.
"""

import ast
from pathlib import Path
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest


def test_no_direct_agent_imports_in_frontend():
    """
    Architectural isolation test:
    Verifies that frontend/dashboard.py NEVER imports agent or service modules directly.
    Frontend must communicate exclusively via HTTP API.
    """
    dashboard_path = Path("frontend/dashboard.py")
    assert dashboard_path.exists(), "frontend/dashboard.py not found!"

    tree = ast.parse(dashboard_path.read_text(encoding="utf-8"))

    prohibited_prefixes = (
        "app.agents",
        "app.services",
        "app.models",
        "app.rag",
        "app.orchestrator"
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in prohibited_prefixes:
                    assert not alias.name.startswith(prefix), (
                        f"Direct backend import '{alias.name}' detected in frontend/dashboard.py!"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for prefix in prohibited_prefixes:
                assert not module.startswith(prefix), (
                    f"Direct backend import from '{module}' detected in frontend/dashboard.py!"
                )


def test_agent_ui_portal_view_registered():
    """Verify that 'Enterprise AI Agents' is registered in portal radio options."""
    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "total_employees": 1470,
            "attrition_rate": 0.161,
            "average_engagement": 3.2,
            "high_risk_count": 120,
            "high_risk_percentage": 8.16,
            "monthly_attrition_risk": 0.05,
            "models": [],
            "roles": [],
            "departments": [{"department": "Sales", "headcount": 446}]
        }
        return resp

    with patch("requests.get", side_effect=mock_get):
        at = AppTest.from_file("../frontend/dashboard.py", default_timeout=30).run()
        assert "Enterprise AI Agents" in at.radio[0].options
        assert "Platform Assistant (RAG)" in at.radio[0].options


def test_agent_ui_dispatch_to_api_endpoint():
    """Verify that submitting a query in the UI sends it to POST /agents/ask."""
    api_calls = []

    def mock_post(url, json=None, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        if "/agents/ask" in url:
            q = json.get("question", "") if json else ""
            api_calls.append(q)
            resp.json.return_value = {
                "query": q,
                "answer": f"Simulated response for: {q}",
                "agent_routed": "policy_agent",
                "provenance": [{"source": "POL-CAREER-001.md", "record_type": "policy"}],
                "refusal_status": False
            }
        else:
            resp.json.return_value = {}
        return resp

    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "total_employees": 1470,
            "attrition_rate": 0.161,
            "average_engagement": 3.2,
            "high_risk_count": 120,
            "high_risk_percentage": 8.16,
            "monthly_attrition_risk": 0.05,
            "models": [],
            "roles": [],
            "departments": [{"department": "Sales", "headcount": 446}]
        }
        return resp

    with patch("requests.post", side_effect=mock_post), \
         patch("requests.get", side_effect=mock_get):
        at = AppTest.from_file("../frontend/dashboard.py", default_timeout=30).run()
        at.radio[0].set_value("Enterprise AI Agents").run()

        # Submit question
        q_text = "What does POL-CAREER-001 say about career stagnation?"
        at.text_input("agent_user_input").input(q_text)
        submit_btn = [b for b in at.button if "Ask Agent Platform" in b.label][0]
        submit_btn.click().run()

        assert len(api_calls) == 1
        assert api_calls[-1] == q_text
        assert at.session_state["agent_active_query"] == q_text
        assert at.session_state["agent_last_result"]["agent_routed"] == "policy_agent"
        assert at.session_state["agent_last_result"]["refusal_status"] is False


def test_agent_ui_example_button_dispatch():
    """Verify that clicking an example button executes the query via POST /agents/ask."""
    api_calls = []

    def mock_post(url, json=None, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        if "/agents/ask" in url:
            q = json.get("question", "") if json else ""
            api_calls.append(q)
            resp.json.return_value = {
                "query": q,
                "answer": "Answer for example query",
                "agent_routed": "career_agent",
                "provenance": [],
                "refusal_status": False
            }
        else:
            resp.json.return_value = {}
        return resp

    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "total_employees": 1470,
            "attrition_rate": 0.161,
            "average_engagement": 3.2,
            "high_risk_count": 120,
            "high_risk_percentage": 8.16,
            "monthly_attrition_risk": 0.05,
            "models": [],
            "roles": [],
            "departments": [{"department": "Sales", "headcount": 446}]
        }
        return resp

    with patch("requests.post", side_effect=mock_post), \
         patch("requests.get", side_effect=mock_get):
        at = AppTest.from_file("../frontend/dashboard.py", default_timeout=30).run()
        at.radio[0].set_value("Enterprise AI Agents").run()

        # Click Career example button (agent_ex4)
        at.button("agent_ex4").click().run()

        assert len(api_calls) == 1
        assert api_calls[-1] == "What is the career progression path for Laboratory Technician?"
        assert at.session_state["agent_active_query"] == "What is the career progression path for Laboratory Technician?"
        assert at.session_state["agent_last_result"]["agent_routed"] == "career_agent"


def test_in_process_embedded_adapter_fallback():
    """
    Verify that when no external server is running (ConnectionError),
    _get and _post transparently fallback to in-process FastAPI TestClient adapter.
    """
    import requests
    from frontend.dashboard import _get, _post

    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Offline")):
        res = _get("/dashboard/summary")
        assert res is not None, "In-process fallback failed for /dashboard/summary!"
        assert "total_employees" in res
        assert res["total_employees"] == 1470

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Offline")):
        res = _post("/agents/ask", json_data={"question": "What is the weather today?"})
        assert res is not None, "In-process fallback failed for /agents/ask!"
        assert res["agent_routed"] == "fallback_handler"
        assert res["refusal_status"] is True
