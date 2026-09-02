"""
app/rag/loaders/__init__.py
===========================
Unified loader interface for the Enterprise HR AI RAG knowledge corpus.
Only loads verified reference CSVs and technical markdown documentation.
Strictly excludes all employee-level data, surveys, and prediction tables.
"""

from typing import List
from app.rag.schemas import Document
from app.rag.loaders.csv_loader import load_occupation_master, load_jobrole_mapping
from app.rag.loaders.markdown_loader import load_model_card, load_data_relationships
from app.rag.loaders.policy_loader import load_all_hr_policies


def load_all_knowledge_documents() -> List[Document]:
    """
    Loads all approved real knowledge sources into a unified list of Document objects:
    1. data/processed/occupation_master.csv (O*NET descriptions)
    2. data/external/jobrole_onet_mapping.csv (JobRole crosswalk notes)
    3. docs/model_card.md (Governance & model limitations)
    4. docs/data_relationships.md (Relational data architecture & open issues)
    
    Returns:
        List[Document]: Collection of normalized Document instances.
    """
    documents: List[Document] = []
    
    # 1. Occupational catalog (1,016 documents)
    documents.extend(load_occupation_master())
    
    # 2. Job role to O*NET mapping (9 documents)
    documents.extend(load_jobrole_mapping())
    
    # 3. Model governance card (structured sections)
    documents.extend(load_model_card())
    
    # 4. Data relationships documentation (structured sections)
    documents.extend(load_data_relationships())
    
    return documents


__all__ = [
    "load_all_knowledge_documents",
    "load_occupation_master",
    "load_jobrole_mapping",
    "load_model_card",
    "load_data_relationships",
    "load_all_hr_policies",
]

