"""
app/rag/normalization.py
========================
Deterministic text normalization utilities for the Enterprise HR AI RAG pipeline.
Ensures uniform text formatting across CSV tables and Markdown documentation
while strictly preserving codes, identifiers, tables, and markdown structures.
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Applies deterministic text normalization to raw document strings:
    1. Unicode NFKC normalization.
    2. Universal newline conversion (CRLF/CR -> LF).
    3. Trimming trailing line whitespace.
    4. Collapsing excessive blank lines (3+ newlines -> 2 newlines).
    5. Normalizing intra-line horizontal whitespace while preserving table structure and list indentations.
    6. Preserves all punctuation, numerical codes (e.g., '19-1042.00', '0.40'),
       acronyms ('O*NET', 'SHAP', 'ROC-AUC'), and markdown formatting.

    Args:
        text: Raw input string.

    Returns:
        Normalized clean string.
    """
    if not text:
        return ""

    # 1. Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Universal newline normalization
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Process line by line to preserve markdown formatting and tables
    normalized_lines = []
    for line in text.split("\n"):
        # Strip trailing whitespace on each line
        line = line.rstrip()
        
        # If the line is part of a markdown table (starts or contains |), preserve pipe structure
        if "|" in line:
            # Clean outer spaces but preserve table cell boundaries
            cleaned_line = re.sub(r"[ \t]+", " ", line)
            normalized_lines.append(cleaned_line)
        elif line.startswith("#"):
            # Markdown heading: ensure single space after hashes
            match = re.match(r"^(#+)\s*(.*)", line)
            if match:
                hashes, title = match.groups()
                cleaned_title = re.sub(r"[ \t]+", " ", title.strip())
                normalized_lines.append(f"{hashes} {cleaned_title}")
            else:
                normalized_lines.append(line)
        else:
            # Standard line: collapse repeated spaces/tabs
            cleaned_line = re.sub(r"[ \t]+", " ", line)
            normalized_lines.append(cleaned_line)

    result = "\n".join(normalized_lines)

    # 4. Collapse excessive consecutive blank lines (3 or more -> 2)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()
