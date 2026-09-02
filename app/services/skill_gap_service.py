"""
Skill Gap Service Module.
Serves organization-wide and role-level capability gap inventories from organization_skill_gaps.csv.
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from app.utils.config import ORGANIZATION_SKILL_GAPS_PATH
from app.utils.logger import logger

_cached_org_gaps_df: Optional[pd.DataFrame] = None

def _load_org_gaps_data() -> pd.DataFrame:
    """Loads and caches the organization-wide skill gap inventory."""
    global _cached_org_gaps_df
    if _cached_org_gaps_df is None:
        if not ORGANIZATION_SKILL_GAPS_PATH.exists():
            raise FileNotFoundError(f"Organization skill gaps file not found at {ORGANIZATION_SKILL_GAPS_PATH}")
        _cached_org_gaps_df = pd.read_csv(ORGANIZATION_SKILL_GAPS_PATH, comment="#")
    return _cached_org_gaps_df


def get_organization_skill_gaps() -> List[Dict[str, Any]]:
    """Returns the full ranked list of all 33 organizational skill gaps."""
    df = _load_org_gaps_data()
    return df.to_dict(orient="records")


def get_severity_distribution() -> Dict[str, int]:
    """Returns headcount of skills by severity band (HIGH, MEDIUM, LOW)."""
    df = _load_org_gaps_data()
    counts = df["severity"].value_counts().to_dict()
    return {
        "HIGH": counts.get("HIGH", 0),
        "MEDIUM": counts.get("MEDIUM", 0),
        "LOW": counts.get("LOW", 0),
        "TOTAL": len(df)
    }


def get_top_gaps(limit: int = 10) -> List[Dict[str, Any]]:
    """Returns top-N critical organizational skill gaps."""
    df = _load_org_gaps_data()
    return df.head(limit).to_dict(orient="records")


def get_role_concentrated_skills() -> List[Dict[str, Any]]:
    """Returns all skills flagged as heavily concentrated in a single role (>=80%)."""
    df = _load_org_gaps_data()
    concentrated = df[df["is_role_concentrated"] == True]
    return concentrated.to_dict(orient="records")
