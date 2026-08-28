import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .sqlite_base import SqliteBase

logger = logging.getLogger(__name__)


@dataclass
class SearchTemplate:
    name: str
    description: str
    alpha: float
    rerank: bool
    top_k: int
    threshold: float
    rewrite_mode: str
    doc_type_filter: list[str] = field(default_factory=list)
    is_builtin: bool = False
    created_at: float = 0.0


BUILTIN_TEMPLATES: list[SearchTemplate] = [
    SearchTemplate(
        name="general",
        description="General-purpose search with balanced hybrid retrieval",
        alpha=0.7,
        rerank=False,
        top_k=10,
        threshold=0.5,
        rewrite_mode="",
        doc_type_filter=[],
        is_builtin=True,
    ),
    SearchTemplate(
        name="code",
        description="Code-optimized search favoring keyword matching with expansion",
        alpha=0.4,
        rerank=True,
        top_k=15,
        threshold=0.3,
        rewrite_mode="expand",
        doc_type_filter=[
            "CODE_PYTHON",
            "CODE_SWIFT",
            "CODE_CPP",
            "CODE_JS",
            "CODE_SHELL",
            "CODE_OTHER",
        ],
        is_builtin=True,
    ),
    SearchTemplate(
        name="design",
        description="Design document search favoring semantic similarity with HyDE",
        alpha=0.8,
        rerank=True,
        top_k=10,
        threshold=0.4,
        rewrite_mode="hyde",
        doc_type_filter=["MD", "HTML"],
        is_builtin=True,
    ),
]


class SearchTemplateManager(SqliteBase):
    def __init__(self, db_path: str):
        self.db_path = db_path
        super().__init__()
        self._create_table()
        self._seed_builtins()
        logger.info("SearchTemplateManager initialized with db=%s", db_path)

    @contextmanager
    def _cursor(self):
        # P2-7: SqliteBase lock guards the shared connection's commit/rollback
        # against cross-thread interleaving (the exact bug SqliteBase exists to
        # fix — this module predates adoption). foreign_keys pragma moved to
        # per-connection init below since SqliteBase owns the conn.
        conn = self._get_conn()
        with self._db_lock:
            cur = conn.cursor()
            try:
                yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception("DB operation failed, rolled back")
                raise
            finally:
                cur.close()

    def _create_table(self):
        with self._cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS search_templates (
                    name TEXT NOT NULL,
                    kb_id TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    alpha REAL NOT NULL DEFAULT 0.7,
                    rerank INTEGER NOT NULL DEFAULT 0,
                    top_k INTEGER NOT NULL DEFAULT 10,
                    threshold REAL NOT NULL DEFAULT 0.5,
                    rewrite_mode TEXT NOT NULL DEFAULT '',
                    doc_type_filter TEXT NOT NULL DEFAULT '[]',
                    is_builtin INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (kb_id, name)
                )
            """)
        logger.debug("search_templates table ensured")

    def _seed_builtins(self):
        with self._cursor() as cur:
            for tpl in BUILTIN_TEMPLATES:
                try:
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO search_templates
                            (name, kb_id, description, alpha, rerank, top_k,
                             threshold, rewrite_mode, doc_type_filter, is_builtin, created_at)
                        VALUES (?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            tpl.name,
                            tpl.description,
                            tpl.alpha,
                            int(tpl.rerank),
                            tpl.top_k,
                            tpl.threshold,
                            tpl.rewrite_mode,
                            json.dumps(tpl.doc_type_filter),
                            int(tpl.is_builtin),
                            tpl.created_at,
                        ),
                    )
                except Exception:
                    logger.exception("Failed to seed builtin template: %s", tpl.name)
                    raise
        logger.debug("Built-in templates seeded")

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        d["rerank"] = bool(d["rerank"])
        d["is_builtin"] = bool(d["is_builtin"])
        d["doc_type_filter"] = json.loads(d.get("doc_type_filter", "[]"))
        return d

    @staticmethod
    def _validate_template(template: SearchTemplate) -> None:
        # L18: custom templates had no param validation — a client could inject
        # alpha=99 or threshold=-5 and break hybrid fusion / filtering. Reject
        # out-of-range values loudly rather than persisting poison config.
        if not template.name or not template.name.strip():
            raise ValueError("template name is required")
        if not 0.0 <= template.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {template.alpha}")
        if not 0.0 <= template.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {template.threshold}")
        if not isinstance(template.top_k, int) or not 1 <= template.top_k <= 200:
            raise ValueError(f"top_k must be an int in [1, 200], got {template.top_k!r}")
        valid_modes = {"", "hyde", "expand", "condense"}
        if template.rewrite_mode not in valid_modes:
            raise ValueError(f"rewrite_mode must be one of {sorted(valid_modes)}, got {template.rewrite_mode!r}")

    def list_templates(self, kb_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM search_templates
                WHERE kb_id = ? OR kb_id = ''
                ORDER BY is_builtin DESC, name ASC
                """,
                (kb_id,),
            )
            rows = cur.fetchall()
        result = [self._row_to_dict(r) for r in rows]
        logger.debug("Listed %d templates for kb_id=%s", len(result), kb_id)
        return result

    def get_template(self, kb_id: str, name: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM search_templates
                WHERE (kb_id = ? OR kb_id = '') AND name = ?
                ORDER BY kb_id DESC
                LIMIT 1
                """,
                (kb_id, name),
            )
            row = cur.fetchone()
        if row is None:
            logger.debug("Template not found: kb_id=%s name=%s", kb_id, name)
            return None
        return self._row_to_dict(row)

    def create_template(self, kb_id: str, template: SearchTemplate) -> dict:
        self._validate_template(template)
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO search_templates
                    (name, kb_id, description, alpha, rerank, top_k,
                     threshold, rewrite_mode, doc_type_filter, is_builtin, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    template.name,
                    kb_id,
                    template.description,
                    template.alpha,
                    int(template.rerank),
                    template.top_k,
                    template.threshold,
                    template.rewrite_mode,
                    json.dumps(template.doc_type_filter),
                    int(template.is_builtin),
                    now,
                ),
            )
        logger.info(
            "Created template: kb_id=%s name=%s alpha=%.2f top_k=%d",
            kb_id,
            template.name,
            template.alpha,
            template.top_k,
        )
        result = self.get_template(kb_id, template.name)
        return result

    def delete_template(self, kb_id: str, name: str) -> bool:
        existing = self.get_template(kb_id, name)
        if existing is None:
            logger.warning("Cannot delete: template not found kb_id=%s name=%s", kb_id, name)
            return False
        if existing.get("is_builtin", False):
            logger.warning("Cannot delete builtin template: kb_id=%s name=%s", kb_id, name)
            return False
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM search_templates WHERE kb_id = ? AND name = ? AND is_builtin = 0",
                (kb_id, name),
            )
            deleted = cur.rowcount > 0
        if deleted:
            logger.info("Deleted template: kb_id=%s name=%s", kb_id, name)
        else:
            logger.warning("Delete had no effect: kb_id=%s name=%s", kb_id, name)
        return deleted
