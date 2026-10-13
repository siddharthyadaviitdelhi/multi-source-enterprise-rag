"""Read SQLite tables as searchable row documents. Use only databases you trust."""
import sqlite3
from pathlib import Path
import pandas as pd
from .common import Document

def load_sqlite(db_path, table: str | None = None) -> list[Document]:
    db_path = Path(db_path)
    # Read-only connection avoids accidental changes to the database.
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'", conn
        )["name"].tolist()
        if table:
            if table not in tables:
                raise ValueError(f"Table {table!r} not found. Available tables: {tables}")
            tables = [table]
        docs = []
        for table_name in tables:
            # Table name comes from sqlite_master, quote embedded double quotes safely.
            quoted = '"' + table_name.replace('"', '""') + '"'
            df = pd.read_sql_query(f"SELECT * FROM {quoted}", conn).fillna("")
            for row_idx, row in df.iterrows():
                text = "\n".join(f"{col}: {value}" for col, value in row.items() if str(value).strip())
                if text.strip():
                    docs.append(Document(text, f"{db_path.name}:{table_name}", "sqlite",
                                         {"table": table_name, "row": int(row_idx) + 1, "path": str(db_path)}))
        return docs
    finally:
        conn.close()
