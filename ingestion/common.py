"""Shared document representation and text chunking helpers."""
from dataclasses import dataclass, asdict
from typing import Any
import re

@dataclass
class Document:
    text: str
    source: str
    source_type: str
    metadata: dict[str, Any]

    def to_dict(self):
        return asdict(self)

def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\x00", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_document(doc: Document, chunk_size: int = 900, overlap: int = 150) -> list[Document]:
    """Split text into overlapping character chunks while preserving source metadata."""
    text = clean_text(doc.text)
    if not text:
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    chunks, start, idx = [], 0, 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            metadata = dict(doc.metadata)
            metadata.update({"chunk_id": idx, "char_start": start, "char_end": end})
            chunks.append(Document(piece, doc.source, doc.source_type, metadata))
            idx += 1
        if end == len(text):
            break
        start = end - overlap
    return chunks
