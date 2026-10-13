from pathlib import Path
from pypdf import PdfReader
from .common import Document, clean_text

def load_pdf(path) -> list[Document]:
    path = Path(path)
    reader = PdfReader(str(path))
    docs = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            docs.append(Document(text, path.name, "pdf", {"page": page_num, "path": str(path)}))
    return docs
