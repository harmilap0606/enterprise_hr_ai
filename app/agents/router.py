"""
app/agents/router.py
====================
Deterministic, explainable Intent Router for the Enterprise HR AI Orchestrator.
Operates 100% offline with zero external cloud API dependencies.

Supported Intent Classes:
- POLICY: HR policies, AI ethics & governance, model thresholds, compliance rules.
- WORKFORCE_INTELLIGENCE: Attrition risk prediction, flight risk, engagement analytics.
- UPSKILLING: Learning recommendations, training courses, skill development programs.
- CAREER: Career progression, role readiness, promotion paths, O*NET mapping.
- HR_OPS: Employee record lookups, demographic details, operational profiles.
- OUT_OF_DOMAIN: Questions completely outside the HR, AI, workforce, or career domains.
"""

import re
from typing import List, Tuple

# Supported Intent Constants
INTENT_POLICY = "POLICY"
INTENT_WORKFORCE_INTELLIGENCE = "WORKFORCE_INTELLIGENCE"
INTENT_UPSKILLING = "UPSKILLING"
INTENT_CAREER = "CAREER"
INTENT_HR_OPS = "HR_OPS"
INTENT_OUT_OF_DOMAIN = "OUT_OF_DOMAIN"


# Precedence Rule 1: Exact Policy Patterns & Compliance Rules
POLICY_PATTERNS = [
    r"\bpol[-_][a-z]+[-_]\d+\b",                       # POL-MODEL-001, POL-AI-001, etc.
    r"\bpolicy\b",                                     # explicit word policy
    r"\bpolicies\b",
    r"\bhandbook\b",
    r"\bgovernance\b",
    r"\bcompliance\b",
    r"\bcode of conduct\b",
    r"\bleave policy\b",
    r"\bparental leave\b",
    r"\bmaternity leave\b",
    r"\bpaternity leave\b",
    r"\bvacation allowance\b",
    r"\bannual leave\b",
    r"\bbenefits\b",
    r"\bhealth insurance\b",
    r"\bcopay\b",
    r"\bseverance\b",
    r"\bhuman review requirement\b",
    r"\bhuman override\b",
    r"\bdecision threshold\b",
    r"\bthreshold standard\b",
    r"\balgorithmic decision\b",
    r"\bautonomous decision\b",
    r"\bethical ai\b",
    r"\bmodel governance\b",
    r"\bdata provenance governance\b",
    r"\bmonitoring policy\b",
    r"\brisk review policy\b",
    r"\bskill gap severity\b",
    r"\bseverity classification\b",
    r"\bmanager exclusion\b",
    r"\bemployment law\b",
    r"\blabor law\b",
]

# Precedence Rule 2: Workforce Intelligence (Attrition & Engagement)
WORKFORCE_PATTERNS = [
    r"\battrition\b",
    r"\bflight risk\b",
    r"\bleaver\b",
    r"\bleavers\b",
    r"\bstayer\b",
    r"\bstayers\b",
    r"\bretention risk\b",
    r"\bturnover\b",
    r"\bengagement score\b",
    r"\bengagement survey\b",
    r"\bworkforce risk\b",
    r"\bdepartment attrition\b",
    r"\bdepartment risk\b",
    r"\bpredict attrition\b",
    r"\bpredict flight\b",
    r"\bhigh flight risk\b",
    r"\brisk probability\b",
    r"\brisk level\b",
    r"\bsatisfaction score\b",
    r"\bworkplace engagement\b",
]

# Precedence Rule 3: Upskilling & Course Recommendations
UPSKILLING_PATTERNS = [
    r"\bupskill\b",
    r"\bupskilling\b",
    r"\breskill\b",
    r"\breskilling\b",
    r"\bcourse\b",
    r"\bcourses\b",
    r"\btraining\b",
    r"\btrainings\b",
    r"\blearning recommendation\b",
    r"\blearning recommendations\b",
    r"\bdevelopment recommendation\b",
    r"\bdevelopment recommendations\b",
    r"\bcourse recommendation\b",
    r"\bcourse recommendations\b",
    r"\blearning plan\b",
    r"\btraining session\b",
    r"\beducation program\b",
    r"\bcurriculum\b",
    r"\bskill gap\b",
    r"\bskill gaps\b",
    r"\bskill deficiency\b",
    r"\bskill deficiencies\b",
    r"\bskill development\b",
    r"\bdevelop.*skills?\b",
    r"\bskills?.*develop\b",
    r"\bimprove.*skills?\b",
    r"\bskills?.*improve\b",
    r"\benhance.*skills?\b",
    r"\bskills?.*enhance\b",
    r"\bwhat skills\b",
    r"\bwhich skills\b",
    r"\bskills?\s+(does|do|should|can|to|for|needed|need|require|required)\b",
    r"\b(need|needs|needed|require|requires|required)\s+to\s+develop\b",
]

# Precedence Rule 4: Career Progression & Role Readiness
CAREER_PATTERNS = [
    r"\bcareer path\b",
    r"\bcareer progression\b",
    r"\bcareer pathway\b",
    r"\bcareer pathways\b",
    r"\bcareer development\b",
    r"\bpromotion readiness\b",
    r"\bcareer ladder\b",
    r"\btarget role\b",
    r"\bnext role\b",
    r"\brole transition\b",
    r"\brole readiness\b",
    r"\bpromotion\b",
    r"\bo\*net\b",
    r"\bonet\b",
    r"\bsoc code\b",
    r"\brole mapping\b",
    r"\boccupation mapping\b",
    r"\boccupation mappings\b",
    r"\bmissing competencies\b",
    r"\bcompetency gap\b",
]

# Precedence Rule 5: HR Operations & Employee Profile Lookups
HR_OPS_PATTERNS = [
    r"\bemployee record\b",
    r"\bemployee profile\b",
    r"\bemployee information\b",
    r"\bemployee lookup\b",
    r"\bemployee details\b",
    r"\blookup employee\b",
    r"\blook up employee\b",
    r"\bemployee #\b",
    r"\bemployee number\b",
    r"\bemployee id\b",
    r"\bworker details\b",
    r"\bworker profile\b",
    r"\bpersonnel file\b",
    r"\bpersonnel record\b",
    r"\bpersonnel profile\b",
    r"\bheadcount\b",
    r"\bhead count\b",
    r"\btotal employees\b",
    r"\bworkforce headcount\b",
    r"\bdepartment staffing\b",
    r"\bdepartment roster\b",
    r"\b(what|which)\s+department\b",
    r"\bdepartment\s+(does|of|is)\b",
    r"\bjob\s+role\b",
    r"\btenure\b",
    r"\byears\s+at\s+company\b",
    r"\bhr operation\b",
    r"\bhr ops\b",
]


def normalize_query(query: str) -> str:
    """Normalizes query string for robust regex token matching."""
    if not query:
        return ""
    q = query.lower().strip()
    # Normalize multiple whitespaces
    q = re.sub(r"\s+", " ", q)
    return q


def classify_intent(query: str) -> str:
    """
    Classifies a user query into one of the 6 supported intent domains.
    
    Precedence Rules (Ordered to prevent false positives):
    1. POLICY: Checked first because policy questions often mention topics from other
       domains (e.g. "What is the policy regarding attrition thresholds?" or POL-MODEL-001).
       Any explicit policy identifier (POL-...) or governance term routes to POLICY.
    2. WORKFORCE_INTELLIGENCE: Attrition prediction, flight risk, and engagement analytics.
    3. UPSKILLING: Learning, course, and training recommendations.
    4. CAREER: Career progression, role pathways, O*NET mappings, and skill gap evaluations.
    5. HR_OPS: Employee profile and personnel record lookups.
    6. OUT_OF_DOMAIN: Fallback for queries with no matching enterprise HR concepts.
    
    Args:
        query: Raw natural-language question.
        
    Returns:
        Intent constant: POLICY, WORKFORCE_INTELLIGENCE, UPSKILLING, CAREER, HR_OPS, or OUT_OF_DOMAIN.
    """
    clean_q = normalize_query(query)
    if not clean_q:
        return INTENT_OUT_OF_DOMAIN

    # Precedence 1: Policy
    for pattern in POLICY_PATTERNS:
        if re.search(pattern, clean_q, re.IGNORECASE):
            return INTENT_POLICY

    # Precedence 2: Workforce Intelligence
    for pattern in WORKFORCE_PATTERNS:
        if re.search(pattern, clean_q, re.IGNORECASE):
            return INTENT_WORKFORCE_INTELLIGENCE

    # Precedence 3: Upskilling
    for pattern in UPSKILLING_PATTERNS:
        if re.search(pattern, clean_q, re.IGNORECASE):
            return INTENT_UPSKILLING

    # Precedence 4: Career
    for pattern in CAREER_PATTERNS:
        if re.search(pattern, clean_q, re.IGNORECASE):
            return INTENT_CAREER

    # Precedence 5: HR Operations
    for pattern in HR_OPS_PATTERNS:
        if re.search(pattern, clean_q, re.IGNORECASE):
            return INTENT_HR_OPS

    # Precedence 6: Out of domain fallback
    return INTENT_OUT_OF_DOMAIN
