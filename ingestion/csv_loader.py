from pathlib import Path
import pandas as pd
from .common import Document

def load_csv(path) -> list[Document]:
    path = Path(path)
    df = pd.read_csv(path).fillna("")
    docs = []
    for row_idx, row in df.iterrows():
        text = "\n".join(f"{col}: {value}" for col, value in row.items() if str(value).strip())
        if text.strip():
            docs.append(Document(text, path.name, "csv", {"row": int(row_idx) + 2, "path": str(path)}))
    return docs
