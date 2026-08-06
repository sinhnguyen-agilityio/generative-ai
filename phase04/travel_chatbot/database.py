import sqlite3


class StyleMemoryDB:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = "memory.db"
        self._create_table()

    def _create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_style_preferences (
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    style_summary TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, thread_id)
                )
            """)
            conn.commit()

    def get_style(
        self,
        user_id: str,
        thread_id: str,
    ) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT style_summary
                FROM user_style_preferences
                WHERE user_id = ?
                AND thread_id = ?
                """,
                (user_id, thread_id),
            )

            row = cursor.fetchone()
            return row[0] if row else ""

    def save_style(
        self,
        user_id: str,
        thread_id: str,
        style_summary: str,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO user_style_preferences (
                    user_id,
                    thread_id,
                    style_summary
                )
                VALUES (?, ?, ?)

                ON CONFLICT(user_id, thread_id)
                DO UPDATE SET
                    style_summary = excluded.style_summary,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    user_id,
                    thread_id,
                    style_summary,
                ),
            )

            conn.commit()

    def get_all_styles(
        self,
        user_id: str,
    ) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT thread_id, style_summary
                FROM user_style_preferences
                WHERE user_id = ?
                ORDER BY updated_at DESC
            """,
                (user_id,),
            )

            rows = cursor.fetchall()

            return "\n".join(
                row[0]
                for row in rows
                if row[0]
            )

    def close(self):
        self.conn.close()
