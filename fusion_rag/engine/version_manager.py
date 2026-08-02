import logging
import os
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class VersionManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _ensure_table(self):
        with self._cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kb_versions (
                    version_id TEXT PRIMARY KEY,
                    kb_id TEXT NOT NULL,
                    snapshot_path TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    doc_count INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_kb_versions_kb_id
                ON kb_versions(kb_id)
            """)

    def create_snapshot(self, kb_id: str, kb_storage_path: str, description: str = "") -> dict:
        version_id = "v_" + str(int(time.time()))
        snapshot_dir = Path(kb_storage_path) / "snapshots" / version_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Creating snapshot %s for kb %s at %s", version_id, kb_id, snapshot_dir)

        vectors_src = Path(kb_storage_path) / "vectors"
        metadata_src = Path(kb_storage_path) / "metadata.db"
        bm25_src = Path(kb_storage_path) / "bm25_index.db"

        if vectors_src.exists():
            vectors_dst = snapshot_dir / "vectors"
            shutil.copytree(str(vectors_src), str(vectors_dst), copy_function=os.link)
            logger.info("Copied vectors via hard links to %s", vectors_dst)

        if metadata_src.exists():
            shutil.copy2(str(metadata_src), str(snapshot_dir / "metadata.db"))
            logger.info("Copied metadata.db to snapshot")

        if bm25_src.exists():
            shutil.copy2(str(bm25_src), str(snapshot_dir / "bm25_index.db"))
            logger.info("Copied bm25_index.db to snapshot")

        doc_count, chunk_count = self._count_records(kb_storage_path)

        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO kb_versions
                   (version_id, kb_id, snapshot_path, description, doc_count, chunk_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (version_id, kb_id, str(snapshot_dir), description, doc_count, chunk_count, now),
            )

        logger.info(
            "Snapshot %s created: doc_count=%d, chunk_count=%d",
            version_id,
            doc_count,
            chunk_count,
        )
        return {
            "version_id": version_id,
            "kb_id": kb_id,
            "snapshot_path": str(snapshot_dir),
            "description": description,
            "doc_count": doc_count,
            "chunk_count": chunk_count,
            "created_at": now,
        }

    def list_snapshots(self, kb_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM kb_versions WHERE kb_id = ? ORDER BY created_at DESC",
                (kb_id,),
            )
            rows = cur.fetchall()
        result = []
        for row in rows:
            result.append(dict(row))
        return result

    def get_snapshot(self, kb_id: str, version_id: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM kb_versions WHERE kb_id = ? AND version_id = ?",
                (kb_id, version_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return dict(row)

    def rollback(self, kb_id: str, kb_storage_path: str, version_id: str) -> dict:
        snapshot = self.get_snapshot(kb_id, version_id)
        if snapshot is None:
            logger.error("Snapshot %s not found for kb %s", version_id, kb_id)
            return {"success": False, "error": f"Snapshot {version_id} not found"}

        snapshot_dir = Path(snapshot["snapshot_path"])
        if not snapshot_dir.exists():
            logger.error("Snapshot directory %s does not exist", snapshot_dir)
            return {"success": False, "error": f"Snapshot directory {snapshot_dir} does not exist"}

        kb_path = Path(kb_storage_path)
        backup_dir = kb_path.parent / f"{kb_path.name}_backup_{int(time.time())}"

        logger.info("Rolling back kb %s to snapshot %s", kb_id, version_id)

        try:
            logger.info("Backing up current data to %s", backup_dir)
            shutil.move(str(kb_path), str(backup_dir))

            kb_path.mkdir(parents=True, exist_ok=True)

            snapshot_vectors = snapshot_dir / "vectors"
            if snapshot_vectors.exists():
                shutil.copytree(
                    str(snapshot_vectors),
                    str(kb_path / "vectors"),
                    copy_function=os.link,
                )
                logger.info("Restored vectors from snapshot")

            snapshot_metadata = snapshot_dir / "metadata.db"
            if snapshot_metadata.exists():
                shutil.copy2(str(snapshot_metadata), str(kb_path / "metadata.db"))
                logger.info("Restored metadata.db from snapshot")

            snapshot_bm25 = snapshot_dir / "bm25_index.db"
            if snapshot_bm25.exists():
                shutil.copy2(str(snapshot_bm25), str(kb_path / "bm25_index.db"))
                logger.info("Restored bm25_index.db from snapshot")

            shutil.rmtree(str(backup_dir), ignore_errors=True)
            logger.info("Removed backup directory after successful rollback")

            return {
                "success": True,
                "version_id": version_id,
                "kb_id": kb_id,
                "message": f"Rolled back to snapshot {version_id}",
            }
        except Exception as e:
            logger.error("Rollback failed: %s", e)
            if backup_dir.exists():
                logger.info("Attempting to restore from backup %s", backup_dir)
                try:
                    if kb_path.exists():
                        shutil.rmtree(str(kb_path), ignore_errors=True)
                    shutil.move(str(backup_dir), str(kb_path))
                    logger.info("Restored original data from backup")
                except Exception as restore_err:
                    logger.error("Failed to restore backup: %s", restore_err)
            return {"success": False, "error": str(e)}

    def delete_snapshot(self, kb_id: str, version_id: str) -> bool:
        snapshot = self.get_snapshot(kb_id, version_id)
        if snapshot is None:
            logger.warning("Snapshot %s not found for kb %s", version_id, kb_id)
            return False

        snapshot_dir = Path(snapshot["snapshot_path"])
        if snapshot_dir.exists():
            shutil.rmtree(str(snapshot_dir), ignore_errors=True)
            logger.info("Removed snapshot directory %s", snapshot_dir)

        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM kb_versions WHERE kb_id = ? AND version_id = ?",
                (kb_id, version_id),
            )
            deleted = cur.rowcount > 0

        if deleted:
            logger.info("Deleted snapshot record %s for kb %s", version_id, kb_id)
        return deleted

    def _count_records(self, kb_storage_path: str) -> tuple[int, int]:
        doc_count = 0
        chunk_count = 0

        metadata_db = Path(kb_storage_path) / "metadata.db"
        if metadata_db.exists():
            try:
                conn = sqlite3.connect(str(metadata_db))
                cur = conn.cursor()
                try:
                    cur.execute("SELECT COUNT(*) FROM documents")
                    row = cur.fetchone()
                    if row:
                        doc_count = row[0]
                except Exception:
                    logger.warning("Could not count documents in metadata.db")

                try:
                    cur.execute("SELECT COUNT(*) FROM chunks")
                    row = cur.fetchone()
                    if row:
                        chunk_count = row[0]
                except Exception:
                    logger.warning("Could not count chunks in metadata.db")

                conn.close()
            except Exception as e:
                logger.warning("Failed to read metadata.db for counts: %s", e)

        return doc_count, chunk_count
