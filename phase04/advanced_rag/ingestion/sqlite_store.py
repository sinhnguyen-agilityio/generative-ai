import json
import sqlite3
from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from collections.abc import Iterator, Sequence


class SQLiteDocStore(BaseStore[str, Document]):
    def __init__(self, db_path: str = "data/docstore.db"):
        self.db_path = db_path

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    key TEXT PRIMARY KEY,
                    page_content TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _connection(self):
        return sqlite3.connect(self.db_path)

    def mget(self, keys: Sequence[str]) -> list[Document | None]:
        if not keys:
            return []

        placeholders = ",".join("?" for _ in keys)

        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT key, page_content, metadata
                FROM documents
                WHERE key IN ({placeholders})
                """,
                keys,
            ).fetchall()

        documents = {
            key: Document(
                page_content=page_content,
                metadata=json.loads(metadata),
            )
            for key, page_content, metadata in rows
        }

        return [documents.get(key) for key in keys]

    def mset(
        self,
        key_value_pairs: Sequence[tuple[str, Document]],
    ) -> None:
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO documents (
                    key,
                    page_content,
                    metadata
                )
                VALUES (?, ?, ?)
                """,
                [
                    (
                        key,
                        document.page_content,
                        json.dumps(document.metadata),
                    )
                    for key, document in key_value_pairs
                ],
            )

            conn.commit()

    def mdelete(self, keys: Sequence[str]) -> None:
        if not keys:
            return

        placeholders = ",".join("?" for _ in keys)

        with self._connection() as conn:
            conn.execute(
                f"""
                DELETE FROM documents
                WHERE key IN ({placeholders})
                """,
                keys,
            )
            conn.commit()

    def yield_keys(
        self,
        *,
        prefix: str | None = None,
    ) -> Iterator[str]:
        with self._connection() as conn:
            if prefix:
                rows = conn.execute(
                    """
                    SELECT key
                    FROM documents
                    WHERE key LIKE ?
                    """,
                    (f"{prefix}%",),
                )
            else:
                rows = conn.execute(
                    "SELECT key FROM documents"
                )

            for row in rows:
                yield row[0]
