"""
Regression test for Streamlit RAG UI state management and provenance expander behavior.
Verifies that:
1. Example queries trigger correctly and update input state.
2. Submitting a new typed question sends that exact question.
3. Rerunning (e.g. toggling the provenance expander) does NOT re-execute the query or revert to old state.
4. Sequential queries execute independently without stale state contamination.
"""
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest


def test_rag_ui_state_management():
    mock_responses = {
        "What occupation is associated with the Scientist role?": {
            "answer": "Computer and Information Research Scientists.",
            "sources": [{"chunk_id": "occ_19-1042.00_c01", "score": 2.27, "source": "occupation_master.csv"}]
        },
        "What is the purpose of jobrole_onet_mapping.csv and occupation_master.csv, and how are they related?": {
            "answer": "To map the roles.",
            "sources": [{"chunk_id": "data_relationships_open_issues_08_c01", "score": 2.45, "source": "docs/data_relationships.md"}]
        },
        "What does O*NET code 19-1042.00 represent?": {
            "answer": "Medical Scientists, Except Epidemiologists.",
            "sources": [{"chunk_id": "occ_19-1042.00_c01", "score": 2.30, "source": "occupation_master.csv"}]
        },
        "What does a Research Scientist do?": {
            "answer": "Conduct research.",
            "sources": [{"chunk_id": "map_research_scientist_c01", "score": 2.06, "source": "jobrole_onet_mapping.csv"}]
        }
    }

    call_history = []

    def mock_requests_post(url, json=None, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        if "/rag/ask" in url:
            q = json.get("question", "")
            call_history.append(q)
            data = mock_responses.get(q, {
                "answer": f"Mock answer for: {q}",
                "sources": [{"chunk_id": "mock_c01", "score": 1.0, "source": "mock.csv"}]
            })
            resp.json.return_value = data
        else:
            resp.json.return_value = {}
        return resp

    def mock_requests_get(url, params=None, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "total_employees": 1470,
            "attrition_rate": 0.161,
            "average_engagement": 3.2,
            "high_risk_count": 120,
            "monthly_attrition_risk": 0.05,
            "models": [],
            "roles": [],
            "departments": []
        }
        return resp

    with patch("requests.post", side_effect=mock_requests_post), \
         patch("requests.get", side_effect=mock_requests_get):
        at = AppTest.from_file("../frontend/dashboard.py", default_timeout=30).run()
        # Switch to Platform Assistant (RAG)
        at.radio[0].set_value("Platform Assistant (RAG)").run()

        # Initial state: no RAG queries run yet
        assert len(call_history) == 0

        # Step 1: Click Example Question 1
        at.button("rag_ex1").click().run()
        assert len(call_history) == 1
        assert call_history[-1] == "What does a Research Scientist do?"
        assert at.session_state["rag_active_query"] == "What does a Research Scientist do?"
        assert at.session_state["rag_last_result"]["answer"] == "Conduct research."

        # Step 2: Simulate rerun (e.g. toggling expander) without submitting
        at.run()
        # Ensure no additional call was made!
        assert len(call_history) == 1
        assert at.session_state["rag_active_query"] == "What does a Research Scientist do?"

        # Step 3: Type Query A: "What occupation is associated with the Scientist role?"
        q_a = "What occupation is associated with the Scientist role?"
        at.text_input("rag_user_input").input(q_a)
        at.button[5].click().run()  # Submit button inside form
        assert len(call_history) == 2
        assert call_history[-1] == q_a
        assert at.session_state["rag_active_query"] == q_a
        assert at.session_state["rag_last_result"]["answer"] == "Computer and Information Research Scientists."

        # Step 4: Simulate expander toggle on Query A
        at.run()
        assert len(call_history) == 2  # No re-call!
        assert at.session_state["rag_active_query"] == q_a

        # Step 5: Type Query B: "What is the purpose of jobrole_onet_mapping.csv and occupation_master.csv, and how are they related?"
        q_b = "What is the purpose of jobrole_onet_mapping.csv and occupation_master.csv, and how are they related?"
        at.text_input("rag_user_input").input(q_b)
        at.button[5].click().run()
        assert len(call_history) == 3
        assert call_history[-1] == q_b
        assert at.session_state["rag_active_query"] == q_b
        assert at.session_state["rag_last_result"]["answer"] == "To map the roles."

        # Step 6: Simulate expander toggle on Query B
        at.run()
        assert len(call_history) == 3  # No re-call!
        assert at.session_state["rag_active_query"] == q_b

        # Step 7: Type Query C: "What does O*NET code 19-1042.00 represent?"
        q_c = "What does O*NET code 19-1042.00 represent?"
        at.text_input("rag_user_input").input(q_c)
        at.button[5].click().run()
        assert len(call_history) == 4
        assert call_history[-1] == q_c
        assert at.session_state["rag_active_query"] == q_c
        assert at.session_state["rag_last_result"]["answer"] == "Medical Scientists, Except Epidemiologists."

        # Step 8: Simulate expander toggle on Query C
        at.run()
        assert len(call_history) == 4  # No re-call!
        assert at.session_state["rag_active_query"] == q_c
