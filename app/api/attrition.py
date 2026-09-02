"""
Attrition & Employee Intelligence API Endpoints Router.
Provides on-demand employee attrition scoring and full workforce intelligence lookups.
"""

from fastapi import APIRouter, HTTPException, Path
from app.validation.employee_schema import EmployeeInputSchema, PredictionResponse
from app.services.attrition_service import (
    predict_single_employee,
    get_full_employee_intelligence
)
from app.utils.logger import logger

router = APIRouter(tags=["Attrition & Employee Intelligence"])


@router.post("/predict/attrition", response_model=PredictionResponse, summary="1. Predict Attrition Risk for Single Employee")
def predict_employee_attrition(payload: EmployeeInputSchema):
    """
    1. POST /predict/attrition
    Computes real-time attrition risk probability using the balanced logistic regression model (threshold = 0.40).
    Requires the 30 KEEP pre-exit features identified in Step 5's leakage audit.
    """
    try:
        data = payload.model_dump()
        result = predict_single_employee(data)
        logger.info(
            f"Prediction completed using model version '{result['model_version']}' "
            f"for employee {result.get('EmployeeNumber') or 'Anonymous'}: "
            f"probability={result['probability']}, risk_level={result['risk_level']}"
        )
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction pipeline failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction pipeline error: {str(e)}")


@router.get("/employees/{employee_id}", summary="6. Get Full Employee Intelligence Record")
def get_employee_record(employee_id: int = Path(..., description="Unique Employee Number")):
    """
    6. GET /employees/{employee_id}
    Returns complete 360-degree intelligence record for an employee,
    combining attrition flight risk, engagement survey metrics, O*NET role alignment,
    synthetic skill gaps, and course recommendations.
    """
    try:
        record = get_full_employee_intelligence(employee_id)
        logger.info(f"Successfully retrieved 360 intelligence record for employee #{employee_id}")
        return record
    except KeyError as e:
        logger.warning(f"Employee #{employee_id} not found in intelligence records (HTTP 404): {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching employee intelligence for #{employee_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
