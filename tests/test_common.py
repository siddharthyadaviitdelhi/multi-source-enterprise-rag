import pytest
from ingestion.common import Document, chunk_document, clean_text

def test_clean_text():
    assert clean_text(" a   b \n\n\n c ") == "a b\n\n c"

def test_chunking_overlap_and_metadata():
    doc = Document("abcdefghij", "sample.txt", "text", {"page": 1})
    chunks = chunk_document(doc, chunk_size=6, overlap=2)
    assert [x.text for x in chunks] == ["abcdef", "efghij"]
    assert chunks[0].metadata["page"] == 1
    assert chunks[1].metadata["chunk_id"] == 1

def test_invalid_overlap():
    doc = Document("abc", "a", "text", {})
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=5, overlap=5)
