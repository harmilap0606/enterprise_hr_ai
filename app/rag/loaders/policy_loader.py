"""
app/rag/loaders/policy_loader.py
================================
Deterministic markdown document loader for synthetic HR policy corpus:
Location: data/knowledge_base/hr_policies/
Loads 10 verified policy documents:
- POL-JOB-001, POL-AI-001, POL-MODEL-001, POL-RISK-001, POL-SKILL-001,
  POL-LEARN-001, POL-CAREER-001, POL-DATA-001, POL-REVIEW-001, POL-MONITOR-001

Extracts structured header metadata (Policy ID, Domain, Version, Status, Scope, Source Basis)
and parses document sections deterministically while preserving section titles and headings.
"""

import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.rag.schemas import Document
from app.rag.normalization import normalize_text
from app.rag.loaders.markdown_loader import parse_markdown_sections, _clean_slug, _extract_doc_title
from app.utils.config import BASE_DIR
from app.utils.logger import logger

DEFAULT_POLICY_DIR = BASE_DIR / "data" / "knowledge_base" / "hr_policies"


def _extract_policy_metadata(raw_text: str) -> Dict[str, str]:
    """
    Extracts key metadata attributes from the ## Policy Metadata section:
    Policy ID, Title, Domain, Version, Status, Effective Date, Owner, Scope, Classification, Source Basis.
    """
    meta: Dict[str, str] = {}
    
    patterns = {
        "policy_id": r"\*\*Policy ID:\*\*\s*(POL-[A-Z]+-\d+)",
        "policy_title": r"\*\*Policy Title:\*\*\s*(.+)",
        "policy_domain": r"\*\*Policy Domain:\*\*\s*(.+)",
        "policy_version": r"\*\*Version:\*\*\s*([\d\.]+)",
        "policy_status": r"\*\*Status:\*\*\s*(.+)",
        "effective_date": r"\*\*Effective Date:\*\*\s*([\d-]+)",
        "owner": r"\*\*Owner:\*\*\s*(.+)",
        "scope": r"\*\*Scope:\*\*\s*(.+)",
        "classification": r"\*\*Classification:\*\*\s*(.+)",
        "source_basis": r"\*\*Source Basis:\*\*\s*(.+)",
    }
    
    for key, pat in patterns.items():
        match = re.search(pat, raw_text, re.IGNORECASE)
        if match:
            meta[key] = match.group(1).strip()
            
    return meta


def load_single_policy_file(file_path: Path) -> List[Document]:
    """
    Loads a single synthetic HR policy markdown file into structured Document sections.
    """
    if not file_path.exists():
        logger.warning(f"Policy file not found: {file_path}")
        return []

    rel_source = str(file_path.relative_to(BASE_DIR)).replace("\\", "/") if file_path.is_relative_to(BASE_DIR) else file_path.name
    raw_text = file_path.read_text(encoding="utf-8")
    
    # 1. Extract metadata
    extracted_meta = _extract_policy_metadata(raw_text)
    policy_id = extracted_meta.get("policy_id", file_path.stem.upper())
    policy_title = extracted_meta.get("policy_title", _extract_doc_title(raw_text, file_path.stem))
    policy_domain = extracted_meta.get("policy_domain", "Human Resources Governance")
    policy_version = extracted_meta.get("policy_version", "1.0")
    policy_status = extracted_meta.get("policy_status", "Synthetic Demo Policy")
    
    # 2. Parse hierarchical markdown sections
    parsed_sections = parse_markdown_sections(raw_text, policy_title)
    
    documents: List[Document] = []
    file_stem = file_path.stem.lower().replace("-", "_")
    
    for idx, (section_title, level, section_raw) in enumerate(parsed_sections, 1):
        norm_text = normalize_text(section_raw)
        if not norm_text:
            continue
            
        slug = _clean_slug(section_title)
        doc_id = f"{file_stem}_{slug}_{idx:02d}"
        
        hierarchy = [policy_title]
        if section_title != policy_title and section_title != "Overview":
            hierarchy.append(section_title)
            
        metadata: Dict[str, Any] = {
            "source": rel_source,
            "source_file": rel_source,
            "source_type": "synthetic_hr_policy",
            "doc_id": doc_id,
            "policy_id": policy_id,
            "policy_title": policy_title,
            "policy_domain": policy_domain,
            "policy_version": policy_version,
            "policy_status": policy_status,
            "effective_date": extracted_meta.get("effective_date", "2026-09-01"),
            "owner": extracted_meta.get("owner", "Enterprise HR AI Governance"),
            "scope": extracted_meta.get("scope", "Enterprise demonstration"),
            "classification": extracted_meta.get("classification", "Internal Demonstration Standard"),
            "source_basis": extracted_meta.get("source_basis", ""),
            "section": section_title,
            "section_level": level,
            "section_hierarchy": hierarchy,
            "document_type": "synthetic_hr_policy"
        }
        
        documents.append(Document(
            doc_id=doc_id,
            source=rel_source,
            file_type="markdown",
            title=f"{policy_id}: {policy_title} — {section_title}",
            text=norm_text,
            metadata=metadata
        ))
        
    return documents


def load_all_hr_policies(policy_dir: Optional[Path] = None) -> List[Document]:
    """
    Loads all synthetic HR policy markdown files from the policy directory.
    
    Returns:
        List[Document]: Collection of normalized Document instances across all 10 policies.
    """
    target_dir = policy_dir or DEFAULT_POLICY_DIR
    if not target_dir.exists():
        logger.error(f"Policy directory does not exist: {target_dir}")
        return []
        
    policy_files = sorted(list(target_dir.glob("POL-*.md")))
    if not policy_files:
        logger.warning(f"No POL-*.md files found in {target_dir}")
        return []
        
    all_documents: List[Document] = []
    for pf in policy_files:
        docs = load_single_policy_file(pf)
        all_documents.extend(docs)
        
    logger.info(f"Loaded {len(all_documents)} structured section documents from {len(policy_files)} policy files in {target_dir}")
    return all_documents
