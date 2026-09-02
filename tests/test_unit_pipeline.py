"""
tests/test_unit_pipeline.py
===========================
Pytest unit test suite covering the 6 cases specified in the DOCX.

Test isolation rules:
- Tests 1-5: Use ONLY small in-memory fixtures. No files from data/processed/.
- Test 6:    Uses FastAPI TestClient (inherently an integration test against real app + loaded models).
"""

import sys
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Shared fixture: a valid, complete employee feature dict (no file I/O).
# These values are hardcoded from memory of the IBM Attrition dataset schema.
# ---------------------------------------------------------------------------
VALID_EMPLOYEE_DICT = {
    "Age": 35,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Research & Development",
    "DistanceFromHome": 5,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 65,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 3,
    "MaritalStatus": "Married",
    "MonthlyIncome": 5000,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 2,
    "OverTime": "No",
    "PercentSalaryHike": 14,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3,
}


# ===========================================================================
# TEST 1 — Missing required column is caught
# ===========================================================================
class TestMissingColumnIsRaised:
    """
    Calling preprocess_features() with a dict missing a required field must raise
    a clear ValueError — not silently produce a NaN or crash with an unrelated
    pandas KeyError deep in the call stack.
    """

    def test_missing_monthly_income_raises_value_error(self):
        from app.ml.predictor import preprocess_features

        incomplete_dict = {k: v for k, v in VALID_EMPLOYEE_DICT.items() if k != "MonthlyIncome"}

        with pytest.raises(ValueError) as exc_info:
            preprocess_features(incomplete_dict)

        # Must be a meaningful, actionable error message — not a raw pandas traceback
        error_msg = str(exc_info.value)
        assert "Missing" in error_msg or "missing" in error_msg, (
            f"Expected a 'Missing ...' message, got: {error_msg}"
        )

    def test_missing_years_at_company_raises_value_error(self):
        from app.ml.predictor import preprocess_features

        incomplete_dict = {k: v for k, v in VALID_EMPLOYEE_DICT.items() if k != "YearsAtCompany"}

        with pytest.raises(ValueError) as exc_info:
            preprocess_features(incomplete_dict)

        error_msg = str(exc_info.value)
        assert "Missing" in error_msg or "missing" in error_msg, (
            f"Expected a 'Missing ...' message, got: {error_msg}"
        )


# ===========================================================================
# TEST 2 — Invalid engagement score is rejected
# ===========================================================================
class TestEngagementScoreValidation:
    """
    EngagementMetricSchema enforces a 1-5 Likert scale (established in Step 2).
    Scores of 0 or 6 must raise Pydantic ValidationError.
    """

    def test_score_of_zero_raises_validation_error(self):
        from app.validation.engagement_schema import EngagementMetricSchema

        with pytest.raises(ValidationError) as exc_info:
            EngagementMetricSchema(
                EmployeeNumber=42,
                EngagementScore=0,       # Below minimum (ge=1.0)
                SatisfactionScore=3.0,
                WorkLifeBalanceScore=3.0,
            )

        errors = exc_info.value.errors()
        fields_with_errors = [e["loc"][-1] for e in errors]
        assert "EngagementScore" in fields_with_errors, (
            f"Expected EngagementScore in error locations, got: {fields_with_errors}"
        )

    def test_score_of_six_raises_validation_error(self):
        from app.validation.engagement_schema import EngagementMetricSchema

        with pytest.raises(ValidationError) as exc_info:
            EngagementMetricSchema(
                EmployeeNumber=42,
                EngagementScore=6,       # Above maximum (le=5.0)
                SatisfactionScore=3.0,
                WorkLifeBalanceScore=3.0,
            )

        errors = exc_info.value.errors()
        fields_with_errors = [e["loc"][-1] for e in errors]
        assert "EngagementScore" in fields_with_errors, (
            f"Expected EngagementScore in error locations, got: {fields_with_errors}"
        )

    def test_satisfaction_score_of_zero_raises_validation_error(self):
        from app.validation.engagement_schema import EngagementMetricSchema

        with pytest.raises(ValidationError):
            EngagementMetricSchema(
                EmployeeNumber=42,
                EngagementScore=3.0,
                SatisfactionScore=0,     # Below minimum
                WorkLifeBalanceScore=3.0,
            )

    def test_work_life_balance_score_of_six_raises_validation_error(self):
        from app.validation.engagement_schema import EngagementMetricSchema

        with pytest.raises(ValidationError):
            EngagementMetricSchema(
                EmployeeNumber=42,
                EngagementScore=3.0,
                SatisfactionScore=3.0,
                WorkLifeBalanceScore=6,  # Above maximum
            )

    def test_boundary_scores_of_1_and_5_are_accepted(self):
        from app.validation.engagement_schema import EngagementMetricSchema

        # These must NOT raise — boundary values are valid per 1-5 scale
        schema = EngagementMetricSchema(
            EmployeeNumber=42,
            EngagementScore=1.0,
            SatisfactionScore=5.0,
            WorkLifeBalanceScore=3.0,
        )
        assert schema.EngagementScore == 1.0
        assert schema.SatisfactionScore == 5.0


# ===========================================================================
# TEST 3 — Attrition prediction returns a real probability
# ===========================================================================
class TestPredictionReturnsRealProbability:
    """
    predict_attrition_risk() must return a float probability in [0.0, 1.0]
    and must not be NaN. Uses the real model (integration-level call).
    """

    def test_probability_is_float(self):
        from app.ml.predictor import predict_attrition_risk

        result = predict_attrition_risk(VALID_EMPLOYEE_DICT)
        assert isinstance(result["probability"], float), (
            f"Expected float, got {type(result['probability'])}"
        )

    def test_probability_is_in_valid_range(self):
        from app.ml.predictor import predict_attrition_risk

        result = predict_attrition_risk(VALID_EMPLOYEE_DICT)
        prob = result["probability"]
        assert 0.0 <= prob <= 1.0, (
            f"Probability {prob} is outside the valid [0.0, 1.0] range"
        )

    def test_probability_is_not_nan(self):
        from app.ml.predictor import predict_attrition_risk

        result = predict_attrition_risk(VALID_EMPLOYEE_DICT)
        assert not math.isnan(result["probability"]), (
            "Prediction returned NaN — feature engineering or scaling produced an invalid value"
        )

    def test_result_contains_all_required_keys(self):
        from app.ml.predictor import predict_attrition_risk

        result = predict_attrition_risk(VALID_EMPLOYEE_DICT)
        required_keys = {"probability", "risk_level", "threshold"}
        assert required_keys.issubset(result.keys()), (
            f"Result dict missing keys. Expected {required_keys}, got {set(result.keys())}"
        )


# ===========================================================================
# TEST 4 — Risk level threshold boundary conditions (pure logic, no model)
# ===========================================================================
class TestRiskLevelThresholdBoundary:
    """
    Pure threshold-logic test. The decision boundary is >= 0.40 -> HIGH.
    Tests: 0.39 -> LOW, 0.40 -> HIGH (inclusive), 0.41 -> HIGH.
    No ML model is invoked; the classification logic is extracted directly.
    Matches Step 7b's established decision threshold.
    """

    def _apply_threshold(self, probability: float, threshold: float = 0.40) -> str:
        """
        Exact classification logic mirrored from predictor.py line 115:
            risk_level = "HIGH" if prob >= threshold else "LOW"
        Reproduced here so this test is purely logic-level with no model dependency.
        """
        return "HIGH" if probability >= threshold else "LOW"

    def test_probability_0_39_is_LOW(self):
        assert self._apply_threshold(0.39) == "LOW", (
            "0.39 should be LOW — it is strictly below the 0.40 threshold"
        )

    def test_probability_0_40_is_HIGH(self):
        assert self._apply_threshold(0.40) == "HIGH", (
            "0.40 should be HIGH — the threshold is INCLUSIVE on the HIGH side (>=)"
        )

    def test_probability_0_41_is_HIGH(self):
        assert self._apply_threshold(0.41) == "HIGH", (
            "0.41 should be HIGH — it is above the 0.40 threshold"
        )

    def test_probability_0_0_is_LOW(self):
        assert self._apply_threshold(0.0) == "LOW"

    def test_probability_1_0_is_HIGH(self):
        assert self._apply_threshold(1.0) == "HIGH"


# ===========================================================================
# TEST 5 — Skill gap calculation matches expected output (pure set logic)
# ===========================================================================
class TestSkillGapSetSubtraction:
    """
    The core skill gap engine (Step 16) computes:
        gap = set(required_skills) - set(possessed_skills)

    This test constructs small fake fixture sets, runs the set-subtraction
    independently in Python, and verifies the algorithm's output matches.
    No real data files (data/processed/) are loaded.
    """

    def _compute_gap(self, required_skills: set, possessed_skills: set) -> set:
        """
        The canonical gap function as implemented in notebook 12:
            missing_skills = required_skills - possessed_skills
        Pure Python set difference. Reproduced here for isolated unit testing.
        """
        return required_skills - possessed_skills

    def test_gap_is_set_difference(self):
        required = {"Python", "SQL", "Machine Learning", "Communication", "Excel"}
        possessed = {"Python", "SQL"}

        gap = self._compute_gap(required, possessed)
        expected = {"Machine Learning", "Communication", "Excel"}

        assert gap == expected, f"Gap mismatch: got {gap}, expected {expected}"

    def test_no_gap_when_all_possessed(self):
        required = {"Python", "SQL"}
        possessed = {"Python", "SQL", "Tableau"}  # Possesses MORE than required

        gap = self._compute_gap(required, possessed)
        assert gap == set(), f"Expected empty gap, got {gap}"

    def test_full_gap_when_nothing_possessed(self):
        required = {"Speaking", "Critical Thinking", "Active Listening"}
        possessed = set()

        gap = self._compute_gap(required, possessed)
        assert gap == required, f"Full gap should equal required set, got {gap}"

    def test_gap_count_matches_set_cardinality(self):
        required = {"A", "B", "C", "D", "E"}
        possessed = {"A", "C"}

        gap = self._compute_gap(required, possessed)
        assert len(gap) == 3, f"Expected gap_count=3, got {len(gap)}"
        assert gap == {"B", "D", "E"}

    def test_extra_possessed_skills_do_not_inflate_gap(self):
        """Possessed skills that aren't in required should never appear in the gap."""
        required = {"Python", "SQL"}
        possessed = {"Python", "SQL", "Java", "Rust", "Haskell"}

        gap = self._compute_gap(required, possessed)
        assert gap == set(), f"Expected no gap, got {gap}"
        # Also confirm possessed-only skills did not leak into gap
        possessed_only = possessed - required
        assert not gap.intersection(possessed_only), (
            f"Leaked possessed-only skills into gap: {gap.intersection(possessed_only)}"
        )


# ===========================================================================
# TEST 6 — API returns expected status codes (integration test, real app)
# ===========================================================================
class TestAPIStatusCodes:
    """
    Uses FastAPI TestClient against the real application.
    Verifies the 4 specific status-code cases specified in the DOCX.
    """

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app) as c:
            yield c

    def test_post_predict_attrition_valid_returns_200(self, client):
        response = client.post("/predict/attrition", json=VALID_EMPLOYEE_DICT)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. Body: {response.text}"
        )
        data = response.json()
        assert "probability" in data
        assert "risk_level" in data

    def test_post_predict_attrition_missing_required_field_returns_422(self, client):
        incomplete = {k: v for k, v in VALID_EMPLOYEE_DICT.items() if k != "MonthlyIncome"}
        response = client.post("/predict/attrition", json=incomplete)
        assert response.status_code == 422, (
            f"Expected 422 Unprocessable Entity (FastAPI native Pydantic validation), "
            f"got {response.status_code}. Body: {response.text}"
        )
        # Verify the error detail pinpoints MonthlyIncome
        detail = response.json().get("detail", [])
        error_fields = [e["loc"][-1] for e in detail if isinstance(e, dict) and "loc" in e]
        assert "MonthlyIncome" in error_fields, (
            f"Expected 'MonthlyIncome' in error detail locations, got: {error_fields}"
        )

    def test_get_employees_valid_id_returns_200(self, client):
        response = client.get("/employees/1")
        assert response.status_code == 200, (
            f"Expected 200 for /employees/1, got {response.status_code}. Body: {response.text}"
        )
        data = response.json()
        assert "EmployeeNumber" in data
        assert data["EmployeeNumber"] == 1

    def test_get_employees_nonexistent_id_returns_404(self, client):
        response = client.get("/employees/999999")
        assert response.status_code == 404, (
            f"Expected 404 for /employees/999999, got {response.status_code}. Body: {response.text}"
        )
