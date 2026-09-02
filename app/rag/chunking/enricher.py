"""
app/rag/chunking/enricher.py
============================
Deterministic contextual metadata enrichment for RAG chunks.
Formats standardized header prefixes for transparent provenance tracking.
Zero LLM calls; 100% deterministic and reproducible.
"""

def create_contextual_text(title: str, section: str, document_type: str, text: str) -> str:
    """
    Constructs deterministic contextual header blocks prepended to chunk text.

    Format:
    [Document: {title}]
    [Section: {section}]
    [Document Type: {document_type}]

    {text}

    Args:
        title: Document or entity title.
        section: Section name or occupational title.
        document_type: Category ('occupation', 'role_mapping', 'governance', 'architecture').
        text: Raw chunk body text.

    Returns:
        String with standardized prefix metadata.
    """
    clean_title = title.strip()
    clean_section = section.strip() if section else "General"
    clean_type = document_type.strip() if document_type else "knowledge"

    headers = [
        f"[Document: {clean_title}]",
        f"[Section: {clean_section}]",
        f"[Document Type: {clean_type}]"
    ]
    header_block = "\n".join(headers)
    return f"{header_block}\n\n{text.strip()}"
