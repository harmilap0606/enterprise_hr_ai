"""
app/rag/loaders/markdown_loader.py
==================================
Deterministic markdown document loader for platform documentation:
1. docs/model_card.md (governance, metrics, threshold, limitations)
2. docs/data_relationships.md (relational architecture, join keys, open issues)

Preserves document titles, section headings, heading hierarchy, and source paths.
Does not flatten entire files into a single huge document.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from app.rag.schemas import Document
from app.rag.normalization import normalize_text
from app.utils.config import BASE_DIR
from app.utils.logger import logger

MODEL_CARD_PATH = BASE_DIR / "docs" / "model_card.md"
DATA_RELATIONSHIPS_PATH = BASE_DIR / "docs" / "data_relationships.md"


def _clean_slug(text: str) -> str:
    """Produces a clean alphanumeric slug from section titles."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return cleaned or "section"


def _extract_doc_title(text: str, default: str) -> str:
    """Extracts top-level # Title from markdown string if present."""
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return default


def parse_markdown_sections(text: str, doc_title: str) -> List[Tuple[str, int, str]]:
    """
    Parses a markdown document into hierarchical section blocks:
    Returns list of (section_title, heading_level, section_content).
    """
    lines = text.splitlines()
    sections: List[Tuple[str, int, List[str]]] = []
    
    current_title = "Overview"
    current_level = 1
    current_lines: List[str] = []

    for line in lines:
        header_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if header_match:
            hashes, title = header_match.groups()
            level = len(hashes)
            
            # If line is level 1, treat as main document header unless already seen
            if level == 1:
                doc_title = title.strip()
                if current_lines:
                    sections.append((current_title, current_level, current_lines))
                    current_lines = []
                current_title = title.strip()
                current_level = 1
                current_lines.append(line)
            else:
                # Level 2+ represents section break
                if current_lines:
                    sections.append((current_title, current_level, current_lines))
                    current_lines = []
                current_title = title.strip()
                current_level = level
                current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_level, current_lines))

    result = []
    for title, level, lines_list in sections:
        content = "\n".join(lines_list).strip()
        if content:
            result.append((title, level, content))

    return result


def load_markdown_file(file_path: Path, document_type: str, default_title: str) -> List[Document]:
    """
    Loads a markdown documentation file into section-level Document objects.
    Preserves document title, section headings, hierarchy, and provenance.
    """
    if not file_path.exists():
        logger.warning(f"Markdown file not found: {file_path}")
        return []

    rel_source = str(file_path.relative_to(BASE_DIR)).replace("\\", "/") if file_path.is_relative_to(BASE_DIR) else file_path.name
    raw_text = file_path.read_text(encoding="utf-8")
    doc_title = _extract_doc_title(raw_text, default_title)
    parsed_sections = parse_markdown_sections(raw_text, doc_title)

    documents = []
    file_stem = file_path.stem.lower()

    for idx, (section_title, level, section_raw) in enumerate(parsed_sections, 1):
        norm_text = normalize_text(section_raw)
        if not norm_text:
            continue

        slug = _clean_slug(section_title)
        doc_id = f"{file_stem}_{slug}_{idx:02d}"

        hierarchy = [doc_title]
        if section_title != doc_title and section_title != "Overview":
            hierarchy.append(section_title)

        metadata: Dict[str, Any] = {
            "source": rel_source,
            "doc_id": doc_id,
            "document_title": doc_title,
            "section": section_title,
            "section_level": level,
            "section_hierarchy": hierarchy,
            "document_type": document_type
        }

        documents.append(Document(
            doc_id=doc_id,
            source=rel_source,
            file_type="markdown",
            title=f"{doc_title} — {section_title}",
            text=norm_text,
            metadata=metadata
        ))

    logger.info(f"Loaded {len(documents)} structured section documents from {rel_source}")
    return documents


def load_model_card(file_path: Optional[Path] = None) -> List[Document]:
    """Loads docs/model_card.md sections."""
    path = file_path or MODEL_CARD_PATH
    return load_markdown_file(
        path,
        document_type="governance",
        default_title="Enterprise HR Attrition Risk Model Card"
    )


def load_data_relationships(file_path: Optional[Path] = None) -> List[Document]:
    """Loads docs/data_relationships.md sections."""
    path = file_path or DATA_RELATIONSHIPS_PATH
    return load_markdown_file(
        path,
        document_type="architecture",
        default_title="Data Relationships Specification"
    )
