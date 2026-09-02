"""
app/rag/chunking/chunker.py
===========================
Structure-aware, offline chunking engine for the Enterprise HR AI RAG pipeline.
Enforces structural boundaries (Markdown headings, tables, occupation descriptions)
and configurable token limits. Never slices sentences or tables arbitrarily.
"""

import re
from typing import List, Optional
from pydantic import BaseModel, Field

from app.rag.schemas import Document, Chunk
from app.rag.chunking.enricher import create_contextual_text
from app.utils.logger import logger


class ChunkConfig(BaseModel):
    """Configuration parameters for structural chunking."""
    target_tokens: int = Field(default=250, description="Ideal target chunk size in approximate tokens (words).")
    min_tokens: int = Field(default=100, description="Minimum soft token floor before merging adjacent units.")
    max_tokens: int = Field(default=400, description="Strict maximum token ceiling before forcing a boundary split.")


DEFAULT_CHUNK_CONFIG = ChunkConfig()


def estimate_tokens(text: str) -> int:
    """Estimates token count deterministically using whitespace-separated word count."""
    if not text:
        return 0
    return len(text.split())


def _split_into_sentences(text: str) -> List[str]:
    """Splits text on sentence boundaries (.!? followed by whitespace), keeping punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _split_paragraphs(text: str) -> List[str]:
    """Splits text on double newline paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def _chunk_text_blocks(
    blocks: List[str],
    doc_heading: str,
    config: ChunkConfig
) -> List[str]:
    """
    Groups paragraphs/blocks into cohesive chunks respecting min, target, and max tokens.
    If a block begins with a markdown table, keeps the table intact.
    If an individual block exceeds max_tokens, splits it on sentence boundaries.
    """
    refined_blocks: List[str] = []
    for b in blocks:
        # Check if block is a markdown table (multiple lines containing |)
        lines = b.splitlines()
        is_table = len(lines) >= 2 and all("|" in line for line in lines[:2])

        if not is_table and estimate_tokens(b) > config.max_tokens:
            # Subdivide long non-table paragraph on sentence boundaries
            sents = _split_into_sentences(b)
            cur_sent_group: List[str] = []
            cur_sent_tokens = 0
            for s in sents:
                s_tok = estimate_tokens(s)
                if cur_sent_group and (cur_sent_tokens + s_tok > config.target_tokens):
                    refined_blocks.append(" ".join(cur_sent_group))
                    cur_sent_group = [s]
                    cur_sent_tokens = s_tok
                else:
                    cur_sent_group.append(s)
                    cur_sent_tokens += s_tok
            if cur_sent_group:
                refined_blocks.append(" ".join(cur_sent_group))
        else:
            refined_blocks.append(b)

    # Now group refined blocks into target-sized chunks
    result_chunks: List[str] = []
    current_chunk_blocks: List[str] = []
    current_tokens = 0

    for block in refined_blocks:
        block_tok = estimate_tokens(block)

        if current_chunk_blocks and (current_tokens + block_tok > config.max_tokens):
            chunk_content = "\n\n".join(current_chunk_blocks)
            # Ensure heading is included if not present at start
            if doc_heading and not chunk_content.startswith("#"):
                chunk_content = f"{doc_heading}\n\n{chunk_content}"
            result_chunks.append(chunk_content)
            current_chunk_blocks = [block]
            current_tokens = block_tok
        else:
            current_chunk_blocks.append(block)
            current_tokens += block_tok

    if current_chunk_blocks:
        chunk_content = "\n\n".join(current_chunk_blocks)
        if doc_heading and not chunk_content.startswith("#"):
            chunk_content = f"{doc_heading}\n\n{chunk_content}"
        result_chunks.append(chunk_content)

    return result_chunks


def chunk_document(doc: Document, config: Optional[ChunkConfig] = None) -> List[Chunk]:
    """
    Converts a single normalized Document into one or more structure-aware Chunk objects.
    
    Rules:
    - Occupation & Role Mapping documents: keep intact when <= max_tokens (sentence split only if oversized).
    - Markdown documents: preserve heading context, keep tables intact, split oversized sections cleanly.
    - Contextual metadata is deterministically attached to every chunk.
    """
    cfg = config or DEFAULT_CHUNK_CONFIG
    doc_type = doc.metadata.get("document_type", "knowledge")
    section_name = doc.metadata.get("section") or doc.metadata.get("occupation_title") or doc.title

    total_tokens = estimate_tokens(doc.text)

    # If document text already fits comfortably in a single chunk, preserve whole
    if total_tokens <= cfg.max_tokens:
        raw_chunks = [doc.text]
    else:
        # Document exceeds max_tokens -> perform structure-aware splitting
        paragraphs = _split_paragraphs(doc.text)
        
        # Extract top heading if present
        heading = ""
        first_line = doc.text.splitlines()[0].strip() if doc.text else ""
        if first_line.startswith("#"):
            heading = first_line

        raw_chunks = _chunk_text_blocks(paragraphs, heading, cfg)

    output_chunks: List[Chunk] = []
    for idx, raw_chunk in enumerate(raw_chunks, 1):
        clean_chunk = raw_chunk.strip()
        if not clean_chunk:
            continue

        chunk_id = f"{doc.doc_id}_c{idx:02d}"
        tok_count = estimate_tokens(clean_chunk)
        contextual_text = create_contextual_text(
            title=doc.title,
            section=section_name,
            document_type=doc_type,
            text=clean_chunk
        )

        chunk_meta = dict(doc.metadata)
        chunk_meta["chunk_id"] = chunk_id
        chunk_meta["chunk_index"] = idx
        chunk_meta["total_chunks_in_doc"] = len(raw_chunks)
        chunk_meta["token_count"] = tok_count

        output_chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            section=section_name,
            document_type=doc_type,
            text=clean_chunk,
            contextual_text=contextual_text,
            token_count=tok_count,
            metadata=chunk_meta
        ))

    return output_chunks


def chunk_all_documents(documents: List[Document], config: Optional[ChunkConfig] = None) -> List[Chunk]:
    """
    Chunks a collection of Document objects into a list of Chunk objects.
    """
    cfg = config or DEFAULT_CHUNK_CONFIG
    all_chunks: List[Chunk] = []
    for doc in documents:
        doc_chunks = chunk_document(doc, cfg)
        all_chunks.extend(doc_chunks)
    logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} structure-aware chunks.")
    return all_chunks
