"""Sentence-window retrieval: index sentence-sized records, return nearby sentence context."""
import re
from ingestion.common import Document

def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def make_sentence_windows(documents, window=2):
    """Create one record per sentence; metadata retains sibling sentences for context restoration."""
    output = []
    for doc in documents:
        sentences = split_sentences(doc.text)
        for i, sentence in enumerate(sentences):
            lo, hi = max(0, i-window), min(len(sentences), i+window+1)
            meta = dict(doc.metadata)
            meta.update({"sentence_index": i, "window_text": " ".join(sentences[lo:hi]),
                         "window_start": lo, "window_end": hi})
            output.append(Document(sentence, doc.source, doc.source_type, meta))
    return output

def restore_window(result):
    doc = result["document"]
    restored = Document(doc.metadata.get("window_text", doc.text), doc.source, doc.source_type, dict(doc.metadata))
    return {**result, "document": restored, "method": "sentence_window"}
