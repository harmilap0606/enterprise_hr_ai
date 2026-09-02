"""
app/services/career_service.py
==============================
Career Service Module.
Provides deterministic, read-only career pathway analysis, promotion readiness evaluation,
and role-to-role competency transferability comparisons.

Authoritative source datasets:
- data/external/jobrole_onet_mapping.csv
- data/processed/occupation_master.csv
- data/processed/role_skill_profiles.csv
- data/processed/employee_attrition_processed.csv
- data/knowledge_base/hr_policies/POL-CAREER-001.md
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import pandas as pd
from pathlib import Path

from app.utils.config import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    EMPLOYEE_ATTRITION_PATH,
    JOBROLE_ONET_MAPPING_PATH,
    ROLE_SKILL_PROFILES_PATH
)
from app.utils.logger import logger

OCCUPATION_MASTER_PATH = PROCESSED_DATA_DIR / "occupation_master.csv"

# Global Caches for Performance
_jobrole_mapping_cache: Optional[pd.DataFrame] = None
_occupation_master_cache: Optional[pd.DataFrame] = None
_role_skill_profiles_cache: Optional[pd.DataFrame] = None
_employee_attrition_cache: Optional[pd.DataFrame] = None

# Canonical IBM Job Roles in the Enterprise
CANONICAL_ROLES = [
    "Healthcare Representative",
    "Human Resources",
    "Laboratory Technician",
    "Manager",
    "Manufacturing Director",
    "Research Director",
    "Research Scientist",
    "Sales Executive",
    "Sales Representative"
]

# Established Career Ladders from Repo Data & POL-CAREER-001
VERTICAL_LADDERS: Dict[str, List[str]] = {
    "Sales Representative": ["Sales Executive", "Manager"],
    "Sales Executive": ["Manager"],
    "Laboratory Technician": ["Research Scientist", "Research Director"],
    "Research Scientist": ["Research Director", "Manager"],
    "Research Director": ["Executive Leadership / Functional Head"],
    "Healthcare Representative": ["Sales Executive", "Manager"],
    "Manufacturing Director": ["Manager"],
    "Human Resources": ["Manager"],
    "Manager": ["Senior Executive / Functional Leadership"]
}

# Feasible Lateral / Cross-Functional Pathways
LATERAL_PATHWAYS: Dict[str, List[str]] = {
    "Sales Representative": ["Healthcare Representative", "Human Resources"],
    "Sales Executive": ["Healthcare Representative"],
    "Laboratory Technician": ["Manufacturing Director"],
    "Research Scientist": ["Laboratory Technician"],
    "Healthcare Representative": ["Sales Representative", "Human Resources"],
    "Manufacturing Director": ["Research Director"],
    "Human Resources": ["Sales Representative"],
    "Research Director": ["Manufacturing Director"],
    "Manager": ["Cross-Departmental General Management"]
}

# Next Step Role along Hierarchy
NEXT_LADDER_ROLES: Dict[str, str] = {
    "Sales Representative": "Sales Executive",
    "Sales Executive": "Manager",
    "Laboratory Technician": "Research Scientist",
    "Research Scientist": "Research Director",
    "Research Director": "Manager (Executive Leadership)",
    "Healthcare Representative": "Sales Executive",
    "Manufacturing Director": "Manager",
    "Human Resources": "Manager",
    "Manager": "Senior Executive / Functional Leadership"
}


# ==============================================================================
# Internal Cached Loaders
# ==============================================================================

def _load_jobrole_mapping_df() -> pd.DataFrame:
    global _jobrole_mapping_cache
    if _jobrole_mapping_cache is None:
        if not JOBROLE_ONET_MAPPING_PATH.exists():
            raise FileNotFoundError(f"JobRole O*NET mapping file not found at {JOBROLE_ONET_MAPPING_PATH}")
        _jobrole_mapping_cache = pd.read_csv(JOBROLE_ONET_MAPPING_PATH)
    return _jobrole_mapping_cache


def _load_occupation_master_df() -> pd.DataFrame:
    global _occupation_master_cache
    if _occupation_master_cache is None:
        if not OCCUPATION_MASTER_PATH.exists():
            raise FileNotFoundError(f"Occupation master file not found at {OCCUPATION_MASTER_PATH}")
        _occupation_master_cache = pd.read_csv(OCCUPATION_MASTER_PATH)
    return _occupation_master_cache


def _load_role_skill_profiles_df() -> pd.DataFrame:
    global _role_skill_profiles_cache
    if _role_skill_profiles_cache is None:
        if not ROLE_SKILL_PROFILES_PATH.exists():
            raise FileNotFoundError(f"Role skill profiles file not found at {ROLE_SKILL_PROFILES_PATH}")
        _role_skill_profiles_cache = pd.read_csv(ROLE_SKILL_PROFILES_PATH)
    return _role_skill_profiles_cache


def _load_employee_attrition_df() -> pd.DataFrame:
    global _employee_attrition_cache
    if _employee_attrition_cache is None:
        if not EMPLOYEE_ATTRITION_PATH.exists():
            raise FileNotFoundError(f"Employee attrition file not found at {EMPLOYEE_ATTRITION_PATH}")
        _employee_attrition_cache = pd.read_csv(EMPLOYEE_ATTRITION_PATH)
    return _employee_attrition_cache


def _normalize_role_name(role_str: str) -> Optional[str]:
    """Matches a user-supplied role name case-insensitively to the canonical list."""
    if not role_str:
        return None
    cleaned = role_str.strip().lower()
    
    # Direct match
    for r in CANONICAL_ROLES:
        if r.lower() == cleaned:
            return r
            
    # Common variations / abbreviations
    alias_map = {
        "lab tech": "Laboratory Technician",
        "laboratory tech": "Laboratory Technician",
        "lab technician": "Laboratory Technician",
        "sales rep": "Sales Representative",
        "sales exec": "Sales Executive",
        "hr": "Human Resources",
        "hr specialist": "Human Resources",
        "scientist": "Research Scientist",
        "researcher": "Research Scientist",
        "mfg director": "Manufacturing Director",
        "manufacturing": "Manufacturing Director",
        "senior scientist": "Research Scientist",
        "lead engineer": "Research Scientist",
        "software engineer": "Research Scientist",
    }
    if cleaned in alias_map:
        return alias_map[cleaned]
        
    for r in CANONICAL_ROLES:
        if cleaned in r.lower() or r.lower() in cleaned:
            return r
            
    return None


def _parse_pipe_separated_skills(raw_str: str) -> List[str]:
    """Extracts clean skill names from 'Skill Name (3.88) | Next Skill (4.0)'."""
    if not raw_str or pd.isna(raw_str):
        return []
    if "Insufficient mapping confidence" in str(raw_str):
        return []
    items = str(raw_str).split("|")
    clean = []
    for it in items:
        # Strip score suffix e.g. (3.88)
        s = re.sub(r"\s*\([\d\.]+\)", "", it).strip()
        if s and s not in clean:
            clean.append(s)
    return clean


# ==============================================================================
# Public Service Functions
# ==============================================================================

def get_role_career_pathway(role_name: str) -> Dict[str, Any]:
    """
    Returns structured career ladder, O*NET SOC mapping, match confidence,
    and benchmark competencies for a given job role.
    
    Args:
        role_name: Name of the internal job role.
        
    Returns:
        Dict containing career pathway hierarchy and O*NET metadata.
    """
    canonical_role = _normalize_role_name(role_name)
    if not canonical_role:
        raise ValueError(
            f"Unknown job role '{role_name}'. Valid enterprise roles: {', '.join(CANONICAL_ROLES)}"
        )

    # 1. Retrieve O*NET mapping crosswalk
    mapping_df = _load_jobrole_mapping_df()
    map_row = mapping_df[mapping_df["ibm_job_role"] == canonical_role]
    if map_row.empty:
        raise ValueError(f"No O*NET mapping crosswalk found for role '{canonical_role}'.")
    m = map_row.iloc[0]
    onet_code = str(m["onet_soc_code"])
    onet_title = str(m["onet_title"])
    match_conf = str(m["match_confidence"])
    mapping_note = str(m["mapping_note"])

    # 2. Retrieve Level Range from empirical HR records
    attr_df = _load_employee_attrition_df()
    role_emp = attr_df[attr_df["JobRole"] == canonical_role]
    if not role_emp.empty and "JobLevel" in role_emp.columns:
        min_lvl = int(role_emp["JobLevel"].min())
        max_lvl = int(role_emp["JobLevel"].max())
        level_range = f"Level {min_lvl} – {max_lvl}"
    else:
        level_range = "Level 1 – 5"

    # 3. Retrieve benchmark competencies
    skill_df = _load_role_skill_profiles_df()
    skill_row = skill_df[skill_df["ibm_job_role"] == canonical_role]
    essential_skills: List[str] = []
    software_tools: List[str] = []
    if not skill_row.empty:
        s = skill_row.iloc[0]
        essential_skills = _parse_pipe_separated_skills(s.get("top5_essential_skills", ""))
        software_tools = _parse_pipe_separated_skills(s.get("top5_software_tools", ""))

    # 4. Compile Vertical & Lateral Pathways
    vertical = VERTICAL_LADDERS.get(canonical_role, [])
    lateral = LATERAL_PATHWAYS.get(canonical_role, [])

    # 5. Role-specific governance notes
    is_manager = (canonical_role == "Manager")
    is_dual_mapped = (canonical_role in ["Healthcare Representative", "Sales Representative"])
    
    governance_notes = []
    if is_manager:
        governance_notes.append(
            "POL-CAREER-001 Rule 4: The generic role 'Manager' is mapped to placeholder 11-9199.00 "
            "with very_low confidence. Development and career transitions require individualized "
            "departmental leadership assessment."
        )
    if is_dual_mapped:
        governance_notes.append(
            f"POL-CAREER-001 Rule 4: {canonical_role} is dual-mapped with its peer role to O*NET "
            f"41-3091.00 (Sales Representatives of Services). Lateral mobility between these cohorts is highly viable."
        )

    return {
        "role_name": canonical_role,
        "current_level_range": level_range,
        "onet_soc_code": onet_code,
        "onet_title": onet_title,
        "match_confidence": match_conf,
        "mapping_note": mapping_note,
        "vertical_pathways": vertical,
        "lateral_pathways": lateral,
        "benchmark_competencies": {
            "essential_skills": essential_skills,
            "software_tools": software_tools
        },
        "is_manager": is_manager,
        "is_dual_mapped": is_dual_mapped,
        "governance_notes": governance_notes,
        "provenance": [
            {
                "source": "data/external/jobrole_onet_mapping.csv",
                "record_id": f"role_{canonical_role}",
                "onet_soc_code": onet_code
            },
            {
                "source": "data/processed/role_skill_profiles.csv",
                "description": "Precomputed benchmark essential skills and software tools"
            },
            {
                "source": "data/knowledge_base/hr_policies/POL-CAREER-001.md",
                "rules_applied": ["Rule 1 (Canonical Descriptions)", "Rule 2 (JobLevel Hierarchy)", "Rule 4 (Dual/Placeholder Roles)"]
            }
        ]
    }


def get_employee_promotion_readiness(employee_id: int) -> Dict[str, Any]:
    """
    Assesses promotion readiness, tenure velocity, and career stagnation status
    for a specific employee under POL-CAREER-001.
    
    Strictly filters out demographic PII (Age, Gender, MaritalStatus, MonthlyIncome, HourlyRate).
    
    Args:
        employee_id: Numeric EmployeeNumber.
        
    Returns:
        Dict with career metrics, stagnation status, and review flags.
    """
    attr_df = _load_employee_attrition_df()
    match = attr_df[attr_df["EmployeeNumber"] == employee_id]
    
    if match.empty:
        raise KeyError(f"Employee #{employee_id} not found in employee attrition dataset.")
        
    row = match.iloc[0]
    
    job_role = str(row.get("JobRole", "Unknown"))
    job_level = int(row.get("JobLevel", 1))
    years_in_role = int(row.get("YearsInCurrentRole", 0))
    years_since_promo = int(row.get("YearsSinceLastPromotion", 0))
    years_at_company = int(row.get("YearsAtCompany", 0))
    perf_rating = int(row.get("PerformanceRating", 3))
    dept = str(row.get("Department", "General"))

    # Stagnation Logic per POL-CAREER-001 Rule 3:
    # STAGNANT: YearsSinceLastPromotion >= 4 OR YearsInCurrentRole >= 5
    # REVIEW_RECOMMENDED: Level 2/3 with >3 years without promotion (e.g. >=3) or YearsInCurrentRole >= 4
    # Otherwise: ON_TRACK
    is_stagnant = (years_since_promo >= 4) or (years_in_role >= 5)
    is_review_rec = (not is_stagnant) and (job_level in [2, 3]) and (years_since_promo >= 3 or years_in_role >= 4)

    if is_stagnant:
        stagnation_status = "STAGNANT"
        review_required = True
        stagnation_rationale = (
            f"Employee has {years_since_promo} years since last promotion and {years_in_role} years in current role. "
            "Exceeds enterprise promotion velocity threshold (POL-CAREER-001 Rule 3: >=4 years without promotion or >=5 years in role)."
        )
    elif is_review_rec:
        stagnation_status = "REVIEW_RECOMMENDED"
        review_required = True
        stagnation_rationale = (
            f"Employee is at JobLevel {job_level} with {years_since_promo} years since last promotion. "
            "Approaching stagnation threshold; active Career Pathing Review is recommended."
        )
    else:
        stagnation_status = "ON_TRACK"
        review_required = False
        stagnation_rationale = (
            f"Tenure velocity ({years_since_promo} years since promotion, {years_in_role} years in role) "
            "is within normal expected intervals."
        )

    next_role = NEXT_LADDER_ROLES.get(job_role, "Senior Executive Leadership")

    return {
        "EmployeeNumber": int(employee_id),
        "JobRole": job_role,
        "JobLevel": job_level,
        "Department": dept,
        "YearsInCurrentRole": years_in_role,
        "YearsSinceLastPromotion": years_since_promo,
        "YearsAtCompany": years_at_company,
        "PerformanceRating": perf_rating,
        "stagnation_status": stagnation_status,
        "career_pathing_review_required": review_required,
        "stagnation_rationale": stagnation_rationale,
        "next_ladder_role": next_role,
        "is_manager": (job_role == "Manager"),
        "provenance": [
            {
                "source": "data/processed/employee_attrition_processed.csv",
                "record_id": f"EmployeeNumber_{employee_id}",
                "fields": ["JobRole", "JobLevel", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsAtCompany", "PerformanceRating"]
            },
            {
                "source": "data/knowledge_base/hr_policies/POL-CAREER-001.md",
                "rules_applied": ["Rule 2 (JobLevel Tiers 1-5)", "Rule 3 (Mitigating Turnover Stagnation)"]
            }
        ]
    }


def compare_role_competencies(current_role: str, target_role: str) -> Dict[str, Any]:
    """
    Compares essential skills and software tools between origin and aspirational target roles
    to compute competency transferability and identify net-new capability requirements.
    
    Args:
        current_role: Origin job role title.
        target_role: Target/aspirational job role title.
        
    Returns:
        Dict with shared competencies, missing competencies, and transferability score.
    """
    canonical_current = _normalize_role_name(current_role)
    canonical_target = _normalize_role_name(target_role)

    if not canonical_current:
        raise ValueError(f"Unknown current role '{current_role}'. Valid roles: {', '.join(CANONICAL_ROLES)}")
    if not canonical_target:
        raise ValueError(f"Unknown target role '{target_role}'. Valid roles: {', '.join(CANONICAL_ROLES)}")

    # 1. Target description from occupation_master.csv
    mapping_df = _load_jobrole_mapping_df()
    target_map = mapping_df[mapping_df["ibm_job_role"] == canonical_target]
    
    target_soc = target_map.iloc[0]["onet_soc_code"] if not target_map.empty else "Unknown"
    target_title = target_map.iloc[0]["onet_title"] if not target_map.empty else canonical_target

    occ_df = _load_occupation_master_df()
    occ_match = occ_df[occ_df["O*NET-SOC Code"] == target_soc]
    target_desc = occ_match.iloc[0]["Description"] if not occ_match.empty else "Canonical description not found in master catalog."

    # 2. Competency profiles from role_skill_profiles.csv
    skill_df = _load_role_skill_profiles_df()
    curr_s_row = skill_df[skill_df["ibm_job_role"] == canonical_current]
    targ_s_row = skill_df[skill_df["ibm_job_role"] == canonical_target]

    curr_essential = _parse_pipe_separated_skills(curr_s_row.iloc[0]["top5_essential_skills"]) if not curr_s_row.empty else []
    curr_software = _parse_pipe_separated_skills(curr_s_row.iloc[0]["top5_software_tools"]) if not curr_s_row.empty else []

    targ_essential = _parse_pipe_separated_skills(targ_s_row.iloc[0]["top5_essential_skills"]) if not targ_s_row.empty else []
    targ_software = _parse_pipe_separated_skills(targ_s_row.iloc[0]["top5_software_tools"]) if not targ_s_row.empty else []

    # 3. Manager cohort handling (Rule 4)
    if canonical_target == "Manager" or canonical_current == "Manager":
        notes = (
            "Manager role is excluded from automated O*NET competency profiling per POL-CAREER-001 Rule 4 "
            "due to placeholder classification (11-9199.00, very_low confidence). Individualized leadership review required."
        )
        return {
            "current_role": canonical_current,
            "target_role": canonical_target,
            "target_soc_code": target_soc,
            "target_onet_title": target_title,
            "target_description": target_desc,
            "shared_essential_skills": [],
            "missing_essential_skills": ["Departmental Leadership Assessment Required"],
            "shared_software_tools": [],
            "missing_software_tools": ["Departmental Software Tools"],
            "transferability_score": 0.0,
            "notes": notes,
            "provenance": [
                {"source": "data/processed/occupation_master.csv", "onet_soc_code": target_soc},
                {"source": "data/knowledge_base/hr_policies/POL-CAREER-001.md", "rule": "Rule 4 (Catch-All Roles)"}
            ]
        }

    # 4. Overlap & Delta Calculation
    shared_essential = [s for s in targ_essential if s in curr_essential]
    missing_essential = [s for s in targ_essential if s not in curr_essential]

    shared_software = [s for s in targ_software if s in curr_software]
    missing_software = [s for s in targ_software if s not in curr_software]

    total_target_competencies = len(targ_essential) + len(targ_software)
    total_shared = len(shared_essential) + len(shared_software)
    
    if total_target_competencies > 0:
        transferability_score = round(total_shared / total_target_competencies, 2)
    else:
        transferability_score = 0.0

    return {
        "current_role": canonical_current,
        "target_role": canonical_target,
        "target_soc_code": target_soc,
        "target_onet_title": target_title,
        "target_description": target_desc,
        "shared_essential_skills": shared_essential,
        "missing_essential_skills": missing_essential,
        "shared_software_tools": shared_software,
        "missing_software_tools": missing_software,
        "transferability_score": transferability_score,
        "advisory_note": (
            "Competency transferability analysis measures benchmark skill overlap and delta. "
            "In accordance with POL-CAREER-001, high overlap indicates capability readiness, not guaranteed job placement."
        ),
        "provenance": [
            {
                "source": "data/processed/occupation_master.csv",
                "onet_soc_code": target_soc,
                "title": target_title
            },
            {
                "source": "data/processed/role_skill_profiles.csv",
                "comparison": f"{canonical_current} -> {canonical_target}"
            },
            {
                "source": "data/knowledge_base/hr_policies/POL-CAREER-001.md",
                "rules_applied": ["Rule 1 (Canonical Descriptions)", "Rule 2 (JobLevel Hierarchy)"]
            }
        ]
    }
