"""
tests/test_rag_ingestion.py
===========================
Unit tests for the RAG foundation ingestion, normalization, and chunking pipeline:
1. CSV loading (occupation_master.csv and jobrole_onet_mapping.csv)
2. Markdown loading (model_card.md and data_relationships.md)
3. Deterministic normalization (unicode, whitespace, table preservation)
4. Metadata preservation
5. Chunk IDs are deterministic and reproducible
6. Chunking respects document boundaries and heading structures
7. Codes (O*NET-SOC codes like 19-1042.00, decision threshold 0.40, SHAP) remain intact
8. No empty chunks produced
9. Contextual text header generation
"""

import pytest
from app.rag.schemas import Document, Chunk
from app.rag.normalization import normalize_text
from app.rag.loaders.csv_loader import load_occupation_master, load_jobrole_mapping
from app.rag.loaders.markdown_loader import load_model_card, load_data_relationships
from app.rag.loaders import load_all_knowledge_documents
from app.rag.chunking.chunker import chunk_document, chunk_all_documents, ChunkConfig
from app.rag.chunking.enricher import create_contextual_text


def test_normalization_preserves_codes_and_headings():
    """Verify normalization does not strip codes, markdown headings, or table pipes."""
    raw = (
        "## Performance Metrics (Threshold = 0.40)\r\n\r\n\r\n"
        "O*NET-SOC Code:   19-1042.00   \n\n\n"
        "Top SHAP Driver:   OverTime\r\n"
        "| Metric |   Value |\n| Recall |   0.7872 |"
    )
    norm = normalize_text(raw)
    
    assert "0.40" in norm
    assert "19-1042.00" in norm
    assert "O*NET-SOC Code:" in norm
    assert "SHAP" in norm
    assert "## Performance Metrics (Threshold = 0.40)" in norm
    assert "| Metric | Value |" in norm
    assert "\r" not in norm
    assert "\n\n\n" not in norm


def test_csv_loading_occupation_master():
    """Verify occupation_master.csv loads exactly 1016 documents with required metadata."""
    docs = load_occupation_master()
    assert len(docs) == 1016
    
    first = docs[0]
    assert first.file_type == "csv"
    assert first.doc_id.startswith("occ_")
    assert "occupation_code" in first.metadata
    assert "occupation_title" in first.metadata
    assert first.metadata["document_type"] == "occupation"
    assert "Description:" in first.text
    assert len(first.text) > 20


def test_csv_loading_jobrole_mapping():
    """Verify jobrole_onet_mapping.csv loads exactly 9 mapping crosswalk documents."""
    docs = load_jobrole_mapping()
    assert len(docs) == 9
    
    for d in docs:
        assert d.file_type == "csv"
        assert d.doc_id.startswith("map_")
        assert d.metadata["document_type"] == "role_mapping"
        assert "ibm_job_role" in d.metadata
        assert "onet_title" in d.metadata
        assert "onet_soc_code" in d.metadata
        assert "Mapping Note:" in d.text


def test_markdown_loading_sections_preserved():
    """Verify model_card.md and data_relationships.md are split into section documents."""
    mc_docs = load_model_card()
    assert len(mc_docs) >= 8
    
    dr_docs = load_data_relationships()
    assert len(dr_docs) >= 7
    
    # Check model card sections
    sec_names = [d.metadata["section"] for d in mc_docs]
    assert any("Performance Metrics" in s for s in sec_names)
    assert any("Known Limitations" in s for s in sec_names)
    assert any("Top 5 SHAP Feature Drivers" in s for s in sec_names)

    # Check data relationships sections
    dr_names = [d.metadata["section"] for d in dr_docs]
    assert any("Open Issues" in s for s in dr_names)
    assert any("Relationship 1" in s for s in dr_names)


def test_metadata_preservation():
    """Verify all documents preserve document_title, source, doc_id, and document_type."""
    docs = load_all_knowledge_documents()
    assert len(docs) == 1042  # 1016 + 9 + 9 + 8

    for d in docs:
        assert d.doc_id
        assert d.source
        assert d.file_type in ("csv", "markdown")
        assert d.title
        assert d.text
        assert "document_type" in d.metadata


def test_chunk_ids_are_deterministic():
    """Verify chunk generation produces 100% deterministic chunk_ids across multiple calls."""
    docs = load_all_knowledge_documents()
    
    chunks_run1 = chunk_all_documents(docs)
    chunks_run2 = chunk_all_documents(docs)
    
    assert len(chunks_run1) == len(chunks_run2)
    for c1, c2 in zip(chunks_run1, chunks_run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.doc_id == c2.doc_id
        assert c1.contextual_text == c2.contextual_text
        assert c1.token_count == c2.token_count


def test_chunking_respects_document_structure():
    """Verify chunks do not slice headings away from text or create malformed tables."""
    mc_docs = load_model_card()
    lim_doc = next(d for d in mc_docs if "Known Limitations" in d.metadata["section"])
    
    chunks = chunk_document(lim_doc)
    assert len(chunks) >= 1
    
    for c in chunks:
        assert "Known Limitations" in c.section
        assert c.token_count <= 400
        # Codes in limitations must remain present
        assert "O*NET" in c.text or "SHAP" in c.text or "Synthetic" in c.text


def test_contextual_text_generation():
    """Verify contextual_text formats standardized deterministic headers."""
    ctx = create_contextual_text(
        title="Model Card",
        section="Performance Metrics",
        document_type="governance",
        text="Recall is 0.7872 at threshold 0.40."
    )
    assert ctx.startswith("[Document: Model Card]\n[Section: Performance Metrics]\n[Document Type: governance]\n\n")
    assert "Recall is 0.7872 at threshold 0.40." in ctx


def test_no_empty_chunks_produced():
    """Verify that chunking produces no empty text chunks or zero-token items."""
    docs = load_all_knowledge_documents()
    chunks = chunk_all_documents(docs)
    
    assert len(chunks) == len(docs)
    for c in chunks:
        assert len(c.text.strip()) > 0
        assert len(c.contextual_text.strip()) > 0
        assert c.token_count > 0
        assert c.chunk_id.endswith("_c01") or "_c" in c.chunk_id
