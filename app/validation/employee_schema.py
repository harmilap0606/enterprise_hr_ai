"""
Pydantic Schemas for Employee Features and Prediction Responses.
Defines input contract based on the 30 KEEP columns identified in Step 5's leakage audit.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field

class EmployeeInputSchema(BaseModel):
    """Input contract for predicting attrition risk on a single employee."""
    # Identifier (Optional)
    EmployeeNumber: Optional[int] = Field(None, description="Unique employee identifier")
    
    # ── Numeric Career & Demographic Features (Pre-exit facts) ──
    Age: int = Field(..., ge=18, le=75, description="Employee age in years", example=35)
    DailyRate: int = Field(..., ge=100, le=2000, description="Daily compensation rate", example=1102)
    DistanceFromHome: int = Field(..., ge=1, le=100, description="Commute distance in miles", example=5)
    Education: int = Field(..., ge=1, le=5, description="Education level (1-5)", example=3)
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4, description="Environment satisfaction (1-4)", example=3)
    HourlyRate: int = Field(..., ge=20, le=150, description="Hourly wage rate", example=65)
    JobInvolvement: int = Field(..., ge=1, le=4, description="Job involvement rating (1-4)", example=3)
    JobLevel: int = Field(..., ge=1, le=5, description="Job level (1-5)", example=2)
    JobSatisfaction: int = Field(..., ge=1, le=4, description="Job satisfaction rating (1-4)", example=3)
    MonthlyIncome: int = Field(..., ge=1000, le=25000, description="Monthly base salary", example=5993)
    MonthlyRate: int = Field(..., ge=1000, le=30000, description="Monthly billing rate", example=19479)
    NumCompaniesWorked: int = Field(..., ge=0, le=15, description="Number of previous employers", example=2)
    PercentSalaryHike: int = Field(..., ge=10, le=30, description="Last salary percentage hike", example=14)
    PerformanceRating: int = Field(..., ge=3, le=4, description="Performance appraisal rating (3-4)", example=3)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4, description="Workplace relationship satisfaction (1-4)", example=3)
    StockOptionLevel: int = Field(..., ge=0, le=3, description="Stock option grant tier (0-3)", example=1)
    TotalWorkingYears: int = Field(..., ge=0, le=45, description="Total professional career experience", example=10)
    TrainingTimesLastYear: int = Field(..., ge=0, le=10, description="Training sessions completed in prior year", example=2)
    WorkLifeBalance: int = Field(..., ge=1, le=4, description="Work-life balance rating (1-4)", example=3)
    YearsAtCompany: int = Field(..., ge=0, le=40, description="Tenure at current company", example=5)
    YearsInCurrentRole: int = Field(..., ge=0, le=25, description="Tenure in current job role", example=3)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=20, description="Years since last career promotion", example=1)
    YearsWithCurrManager: int = Field(..., ge=0, le=25, description="Tenure reporting to current manager", example=3)
    
    # ── Categorical Features ──
    BusinessTravel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"] = Field(
        ..., description="Business travel cadence", example="Travel_Rarely"
    )
    Department: Literal["Sales", "Research & Development", "Human Resources"] = Field(
        ..., description="Organizational department", example="Research & Development"
    )
    EducationField: Literal["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"] = Field(
        ..., description="Primary academic field", example="Life Sciences"
    )
    Gender: Literal["Male", "Female"] = Field(
        ..., description="Gender", example="Male"
    )
    JobRole: Literal[
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ] = Field(..., description="Job role title", example="Research Scientist")
    MaritalStatus: Literal["Single", "Married", "Divorced"] = Field(
        ..., description="Marital status", example="Married"
    )
    OverTime: Literal["Yes", "No"] = Field(
        ..., description="Whether employee works overtime regularly", example="No"
    )


class PredictionResponse(BaseModel):
    """Response payload for attrition prediction."""
    EmployeeNumber: Optional[int] = Field(None, description="Employee identifier if provided")
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of attrition")
    risk_level: Literal["HIGH", "LOW"] = Field(..., description="Risk tier based on threshold")
    threshold: float = Field(..., description="Decision threshold applied")
    model_version: str = Field("v3 (logistic_regression_balanced)", description="Model version identifier")
