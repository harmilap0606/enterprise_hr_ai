"""
Pydantic Schemas for Engagement Metrics & Survey Summaries.
Enforces the discrete 1-5 scale established in Step 2 validation.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class EngagementMetricSchema(BaseModel):
    """Schema for individual survey responses on a discrete 1-5 Likert scale."""
    EmployeeNumber: int = Field(..., description="Employee ID")
    EngagementScore: Optional[float] = Field(None, ge=1.0, le=5.0, description="Self-reported engagement (1-5 scale)")
    SatisfactionScore: Optional[float] = Field(None, ge=1.0, le=5.0, description="Self-reported job satisfaction (1-5 scale)")
    WorkLifeBalanceScore: Optional[float] = Field(None, ge=1.0, le=5.0, description="Self-reported work-life balance (1-5 scale)")


class DepartmentEngagementMetrics(BaseModel):
    """Department-level aggregation of engagement survey metrics."""
    Department: str = Field(..., description="Department name")
    surveyed_count: int = Field(..., description="Employees in department with survey records")
    total_employees: int = Field(..., description="Total employees in department")
    coverage_percentage: float = Field(..., description="Survey coverage percentage")
    mean_engagement: Optional[float] = Field(None, description="Average engagement score (1-5)")
    mean_satisfaction: Optional[float] = Field(None, description="Average satisfaction score (1-5)")
    mean_work_life_balance: Optional[float] = Field(None, description="Average work-life balance score (1-5)")


class EngagementSummaryResponse(BaseModel):
    """Workforce-wide executive engagement summary."""
    total_workforce: int = Field(1470, description="Total workforce count")
    survey_respondents: int = Field(731, description="Employees with matched survey data")
    unmapped_employees: int = Field(739, description="Employees without survey data (preserved as null)")
    workforce_coverage_percentage: float = Field(49.73, description="Survey coverage percentage")
    overall_mean_engagement: float = Field(..., description="Mean engagement score among respondents (1-5)")
    overall_mean_satisfaction: float = Field(..., description="Mean satisfaction score among respondents (1-5)")
    overall_mean_work_life_balance: float = Field(..., description="Mean work-life balance score among respondents (1-5)")
    caveat: str = Field(
        "Findings are based on 731 employees (49.7% of workforce) with matched survey data. "
        "Do not generalize to the 739 unmapped employees.",
        description="Methodological integrity caveat"
    )
    departments: List[DepartmentEngagementMetrics] = Field(..., description="Per-department breakdowns")
