"""
scripts/validate_and_build_policy_manifest.py
Validates the synthetic HR policy corpus in data/knowledge_base/hr_policies/
and builds data/knowledge_base/hr_policies/manifest.json.
"""

import os
import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
POLICY_DIR = BASE_DIR / "data" / "knowledge_base" / "hr_policies"
MANIFEST_PATH = POLICY_DIR / "manifest.json"

EXPECTED_POLICIES = [
    {
        "id": "POL-JOB-001",
        "filename": "POL-JOB-001.md",
        "title": "Job Role Classification and O*NET Mapping Policy",
        "domain": "Occupational Architecture & Role Classification"
    },
    {
        "id": "POL-AI-001",
        "filename": "POL-AI-001.md",
        "title": "HR AI Decision-Support Governance Policy",
        "domain": "AI Governance & Ethical Operations"
    },
    {
        "id": "POL-MODEL-001",
        "filename": "POL-MODEL-001.md",
        "title": "Attrition Model Usage Policy",
        "domain": "Predictive Analytics & Model Operations"
    },
    {
        "id": "POL-RISK-001",
        "filename": "POL-RISK-001.md",
        "title": "Workforce Risk Review Policy",
        "domain": "Talent Retention & Risk Mitigation"
    },
    {
        "id": "POL-SKILL-001",
        "filename": "POL-SKILL-001.md",
        "title": "Skill Gap Identification Policy",
        "domain": "Skills Architecture & Gap Analysis"
    },
    {
        "id": "POL-LEARN-001",
        "filename": "POL-LEARN-001.md",
        "title": "Employee Upskilling Recommendation Policy",
        "domain": "Learning & Professional Development"
    },
    {
        "id": "POL-CAREER-001",
        "filename": "POL-CAREER-001.md",
        "title": "Career and Occupation Mapping Policy",
        "domain": "Internal Mobility & Career Pathways"
    },
    {
        "id": "POL-DATA-001",
        "filename": "POL-DATA-001.md",
        "title": "HR Data Usage and Source Provenance Policy",
        "domain": "Data Governance & Source Verification"
    },
    {
        "id": "POL-REVIEW-001",
        "filename": "POL-REVIEW-001.md",
        "title": "Human Review of AI-Assisted HR Decisions Policy",
        "domain": "Human Oversight & Operational Safeguards"
    },
    {
        "id": "POL-MONITOR-001",
        "filename": "POL-MONITOR-001.md",
        "title": "HR AI Monitoring and Model Limitations Policy",
        "domain": "Model Lifecycle & Limitation Management"
    }
]

REQUIRED_SECTIONS = [
    "## Policy Metadata",
    "## Purpose",
    "## Scope",
    "## Definitions",
    "## Policy Rules",
    "## Procedure",
    "## Exceptions / Limitations",  # or "## Exceptions and Limitations"
    "## Human Review Requirements",
    "## Data / Source References",  # or "## Data and Source References"
    "## Related Policies"
]

def validate_and_build():
    manifest_entries = []
    errors = []
    
    print("=" * 80)
    print("VALIDATING HR POLICY CORPUS")
    print("=" * 80)
    
    for exp in EXPECTED_POLICIES:
        fpath = POLICY_DIR / exp["filename"]
        if not fpath.exists():
            errors.append(f"Missing file: {fpath}")
            continue
            
        content = fpath.read_text(encoding="utf-8")
        word_count = len(content.split())
        
        # Check title
        if not content.startswith(f"# {exp['title']}"):
            # Check if slight title variation
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            doc_title = title_match.group(1) if title_match else "UNKNOWN"
        else:
            doc_title = exp["title"]
            
        # Check Policy ID in metadata
        id_match = re.search(r"\*\*Policy ID:\*\*\s*(POL-[A-Z]+-\d+)", content)
        doc_id = id_match.group(1) if id_match else None
        if doc_id != exp["id"]:
            errors.append(f"{exp['filename']}: Expected ID {exp['id']}, found {doc_id}")
            
        # Check Status: Synthetic Demo Policy
        status_match = re.search(r"\*\*Status:\*\*\s*(Synthetic Demo Policy)", content)
        if not status_match:
            errors.append(f"{exp['filename']}: Missing '**Status:** Synthetic Demo Policy'")
            
        # Check Version
        ver_match = re.search(r"\*\*Version:\*\*\s*([\d\.]+)", content)
        version = ver_match.group(1) if ver_match else "1.0"
        
        # Check sections
        for sec in REQUIRED_SECTIONS:
            sec_clean = sec.replace("## ", "").replace("/", "").split()[0]
            if not re.search(rf"##\s+.*{sec_clean}", content, re.IGNORECASE):
                errors.append(f"{exp['filename']}: Missing section matching '{sec}'")
                
        # Find related policies
        related_matches = re.findall(r"(POL-[A-Z]+-\d+)", content)
        # Filter out self
        related_ids = sorted(list(set([m for m in related_matches if m != exp["id"] and m in [p["id"] for p in EXPECTED_POLICIES]])))
        
        print(f"Policy: {exp['id']} | Words: {word_count:4d} | Status: Synthetic Demo Policy | Related: {related_ids}")
        
        manifest_entries.append({
            "policy_id": exp["id"],
            "filename": exp["filename"],
            "title": doc_title,
            "domain": exp["domain"],
            "version": version,
            "status": "Synthetic Demo Policy",
            "word_count": word_count,
            "related_policies": related_ids
        })
        
    print("-" * 80)
    if errors:
        print(f"FAILED with {len(errors)} errors:")
        for err in errors:
            print(f"  - {err}")
        raise ValueError("Corpus validation failed")
    else:
        print("ALL 10 POLICIES VALIDATED SUCCESSFULLY!")
        
    # Write manifest
    manifest_data = {
        "corpus_name": "Enterprise HR AI Synthetic Demo Policy Corpus",
        "description": "Controlled synthetic demonstration policy corpus grounded in enterprise project data and governance architecture.",
        "status": "Synthetic Demo Policy",
        "policy_count": len(manifest_entries),
        "total_words": sum(e["word_count"] for e in manifest_entries),
        "policies": manifest_entries
    }
    
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
        
    print(f"Manifest written to: {MANIFEST_PATH}")
    print(f"Total Policies: {len(manifest_entries)} | Total Word Count: {manifest_data['total_words']}")

if __name__ == "__main__":
    validate_and_build()
