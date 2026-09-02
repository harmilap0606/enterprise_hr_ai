"""
app/rag/loaders/csv_loader.py
=============================
Deterministic CSV document loaders for approved tabular knowledge sources:
1. data/processed/occupation_master.csv (O*NET occupational catalog)
2. data/external/jobrole_onet_mapping.csv (IBM HR JobRole to O*NET crosswalk)

Strictly loads only reference/knowledge CSVs; never loads employee-level data.
"""

import csv
import re
from pathlib import Path
from typing import List, Optional

from app.rag.schemas import Document
from app.rag.normalization import normalize_text
from app.utils.config import BASE_DIR
from app.utils.logger import logger

OCCUPATION_MASTER_PATH = BASE_DIR / "data" / "processed" / "occupation_master.csv"
JOBROLE_MAPPING_PATH = BASE_DIR / "data" / "external" / "jobrole_onet_mapping.csv"


def _clean_id(val: str) -> str:
    """Creates a deterministic URL/file-safe identifier from arbitrary string."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", val.strip()).strip("_").lower()


def load_occupation_master(file_path: Optional[Path] = None) -> List[Document]:
    """
    Loads data/processed/occupation_master.csv into a list of Document objects.
    Produces exactly one logical Document per occupation row.

    Preserves:
    - O*NET-SOC Code
    - Title
    - Description

    Metadata:
    - source
    - doc_id
    - occupation_code
    - occupation_title
    - document_type="occupation"
    """
    path = file_path or OCCUPATION_MASTER_PATH
    if not path.exists():
        logger.error(f"Occupation master CSV not found at: {path}")
        return []

    rel_source = str(path.relative_to(BASE_DIR)).replace("\\", "/") if path.is_relative_to(BASE_DIR) else path.name
    documents = []

    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            soc_code = str(row.get("O*NET-SOC Code", "")).strip()
            title = str(row.get("Title", "")).strip()
            description = str(row.get("Description", "")).strip()

            if not soc_code and not title:
                continue

            doc_id = f"occ_{soc_code}"
            raw_text = (
                f"O*NET Occupation: {title}\n"
                f"O*NET-SOC Code: {soc_code}\n"
                f"Description: {description}"
            )
            norm_text = normalize_text(raw_text)

            metadata = {
                "source": rel_source,
                "doc_id": doc_id,
                "occupation_code": soc_code,
                "occupation_title": title,
                "document_type": "occupation"
            }

            documents.append(Document(
                doc_id=doc_id,
                source=rel_source,
                file_type="csv",
                title=f"O*NET: {title}",
                text=norm_text,
                metadata=metadata
            ))

    logger.info(f"Loaded {len(documents)} occupation documents from {rel_source}")
    return documents


def load_jobrole_mapping(file_path: Optional[Path] = None) -> List[Document]:
    """
    Loads data/external/jobrole_onet_mapping.csv into Document objects.
    Produces one logical Document per mapping record.

    Preserves:
    - IBM job role
    - O*NET title
    - mapping note

    Metadata:
    - source
    - doc_id
    - ibm_job_role
    - onet_title
    - onet_soc_code
    - match_confidence
    - document_type="role_mapping"
    """
    path = file_path or JOBROLE_MAPPING_PATH
    if not path.exists():
        logger.warning(f"Job role mapping CSV not found at: {path}")
        return []

    rel_source = str(path.relative_to(BASE_DIR)).replace("\\", "/") if path.is_relative_to(BASE_DIR) else path.name
    documents = []

    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            ibm_role = str(row.get("ibm_job_role", "")).strip()
            soc_code = str(row.get("onet_soc_code", "")).strip()
            onet_title = str(row.get("onet_title", "")).strip()
            confidence = str(row.get("match_confidence", "")).strip()
            note = str(row.get("mapping_note", "")).strip()

            if not ibm_role:
                continue

            doc_id = f"map_{_clean_id(ibm_role)}"
            raw_text = (
                f"IBM Job Role: {ibm_role}\n"
                f"Mapped O*NET Title: {onet_title}\n"
                f"O*NET-SOC Code: {soc_code}\n"
                f"Match Confidence: {confidence}\n"
                f"Mapping Note: {note}"
            )
            norm_text = normalize_text(raw_text)

            metadata = {
                "source": rel_source,
                "doc_id": doc_id,
                "ibm_job_role": ibm_role,
                "onet_title": onet_title,
                "onet_soc_code": soc_code,
                "match_confidence": confidence,
                "document_type": "role_mapping"
            }

            documents.append(Document(
                doc_id=doc_id,
                source=rel_source,
                file_type="csv",
                title=f"Role Mapping: {ibm_role} -> {onet_title}",
                text=norm_text,
                metadata=metadata
            ))

    logger.info(f"Loaded {len(documents)} role mapping documents from {rel_source}")
    return documents
