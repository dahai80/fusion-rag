import logging
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .sqlite_base import SqliteBase, open_sqlite

logger = logging.getLogger(__name__)


class VersionManager(SqliteBase):
    # P3-6: TTL for the cached record counts. create_snapshot calls
    # _count_records which opens a readonly conn and runs COUNT(*) over the
    # documents + chunks tables — a full scan on a large KB. Snapshots are
    # infrequent but the admin also reads counts elsewhere; cache the result
    # briefly so back-to-back snapshots (or a snapshot after a list) don't
    # re-scan. 60s is short enough that a count goes stale only briefly.
    _COUNT_TTL = 60.0

    def __init__(self, db_path: str):
        self.db_path = db_path
        super().__init__()
        self._count_cache: dict[str, tuple[float, int, int]] = {}
        self._ensure_table()

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        with self._db_lock:
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
        # L8: second snapshot in the same second collided on the timestamp-only
        # version_id → same dir → second copytree clobbered the first's hard
        # links. Append a uuid8 so same-second snapshots get distinct dirs;
        # assert the dir does not exist before writing.
        version_id = f"v_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        # H1 fix: snapshot MUST live OUTSIDE kb_storage_path. The prior layout
        # (kb_storage_path/snapshots/) meant rollback's move-the-live-dir step
        # also moved the snapshots away, then the copy-back read from the
        # moved path (all .exists() False → skipped), then rmtree deleted the
        # backup that now contained both live data + snapshots → total loss.
        # New layout: sibling .snapshots dir next to the KB dir, never inside it.
        snapshot_root = Path(kb_storage_path).parent / ".snapshots" / kb_id
        snapshot_dir = snapshot_root / version_id
        if snapshot_dir.exists():
            logger.error("Snapshot dir already exists (unexpected): %s", snapshot_dir)
            raise FileExistsError(f"snapshot dir already exists: {snapshot_dir}")
        snapshot_dir.mkdir(parents=True, exist_ok=False)

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
        # H1 fix: in-place per-artifact restore. The prior code moved the entire
        # live dir to a backup, then tried to copy back from the snapshot — but
        # the snapshot was NESTED inside the moved live dir, so every
        # snapshot_X.exists() was False (skipped), then it rmtree'd the backup
        # (which now held live data + snapshots). rollback returned success:True
        # leaving an empty KB, data unrecoverable. The recovery feature WAS the
        # data loss.
        #
        # New approach: never move the live dir. For each artifact in the
        # snapshot (vectors/ metadata.db bm25_index.db), write the restored
        # copy to a sibling temp path, then os.replace (atomic on POSIX) into
        # the live dir. A failure mid-restore leaves the un-replaced artifacts
        # intact; the replaced ones already match the snapshot. We never rmtree
        # the live dir, so a crash can't widen the loss. Admin DBs (versions.db
        # /permissions.db/templates.db/audit.db) are NOT in snapshots and are
        # left untouched — rollback only restores searchable data.
        snapshot = self.get_snapshot(kb_id, version_id)
        if snapshot is None:
            logger.error("Snapshot %s not found for kb %s", version_id, kb_id)
            return {"success": False, "error": f"Snapshot {version_id} not found"}

        snapshot_dir = Path(snapshot["snapshot_path"])
        if not snapshot_dir.exists():
            logger.error("Snapshot directory %s does not exist", snapshot_dir)
            return {"success": False, "error": f"Snapshot directory {snapshot_dir} does not exist"}

        kb_path = Path(kb_storage_path)
        kb_path.mkdir(parents=True, exist_ok=True)

        logger.info("Rolling back kb %s to snapshot %s (in-place restore)", kb_id, version_id)

        # R4 tie-in: a pooled VectorStore holds an open LanceDB/HNSW handle on
        # kb_path/vectors. Swapping the vectors dir under it corrupts the handle.
        # Evict the pool entry first so the next request reopens the restored
        # data. Best-effort — the pool lives on app.state and may be absent in
        # unit tests / direct calls.
        self._evict_vec_store_pool(kb_path)

        # Re-evaluate snapshot location: legacy snapshots nested inside
        # kb_storage_path/snapshots/ are unsupported under in-place restore
        # (copying vectors/ onto itself). Detect and refuse loudly — operator
        # must migrate. New snapshots live outside (create_snapshot H1 fix).
        try:
            snapshot_dir.relative_to(kb_path)
            logger.error(
                "rollback: snapshot %s is nested inside kb_storage_path %s — "
                "legacy layout, refusing in-place restore (would copy onto self). "
                "Re-create snapshots under the new external layout.",
                snapshot_dir,
                kb_path,
            )
            return {
                "success": False,
                "error": "snapshot is nested inside kb_storage_path (legacy layout); "
                "delete and re-create snapshots to migrate",
            }
        except ValueError:
            pass  # snapshot outside live dir — correct, proceed.

        restored: list[str] = []
        try:
            for name, is_dir in (("vectors", True), ("metadata.db", False), ("bm25_index.db", False)):
                src = snapshot_dir / name
                if not src.exists():
                    logger.info("rollback: %s absent in snapshot, leaving live copy untouched", name)
                    continue
                dst = kb_path / name
                self._restore_artifact(src, dst, is_dir)
                restored.append(name)
                logger.info("rollback: restored %s from snapshot", name)
        except Exception as e:
            logger.error(
                "rollback FAILED mid-restore (restored so far: %s): %s. "
                "Live dir NOT deleted — replaced artifacts match snapshot, "
                "unreplaced ones retain prior state.",
                restored,
                e,
            )
            return {"success": False, "error": str(e), "restored": restored}

        return {
            "success": True,
            "version_id": version_id,
            "kb_id": kb_id,
            "restored": restored,
            "message": f"Rolled back to snapshot {version_id}",
        }

    @staticmethod
    def _evict_vec_store_pool(kb_path: Path) -> None:
        # R4: drop the pooled VectorStore handle for this KB's vector_path so the
        # in-place vectors/ swap isn't observed by a stale open handle. No-op
        # outside the server (no app.state bound). Import lazily to avoid a
        # cycle (app_state imports nothing from engine; keep it that way).
        try:
            from ..api.app_state import get_vec_store_pool, get_vec_store_pool_lock

            pool = get_vec_store_pool()
        except Exception:
            return
        lock = get_vec_store_pool_lock()
        vector_path = str(kb_path / "vectors")
        with lock:
            vs = pool.pop(vector_path, None)
        if vs is not None:
            try:
                vs.close()
            except Exception as e:
                logger.warning("rollback: pooled vec_store close failed for %s: %s", vector_path, e)
            logger.info("rollback: evicted pooled vec_store for %s", vector_path)

    @staticmethod
    def _restore_artifact(src: Path, dst: Path, is_dir: bool) -> None:
        # Atomic per-artifact restore: write to a sibling temp, os.replace
        # (atomic rename on POSIX) into the live path. A crash before replace
        # leaves the live artifact untouched.
        parent = dst.parent
        tmp = parent / f".{dst.name}.rollback_tmp_{uuid.uuid4().hex[:8]}"
        try:
            if is_dir:
                if dst.exists():
                    shutil.rmtree(str(dst), ignore_errors=True)
                shutil.copytree(str(src), str(tmp), copy_function=os.link)
                os.replace(str(tmp), str(dst))
            else:
                shutil.copy2(str(src), str(tmp))
                os.replace(str(tmp), str(dst))
        except Exception:
            if tmp.exists():
                if tmp.is_dir():
                    shutil.rmtree(str(tmp), ignore_errors=True)
                else:
                    tmp.unlink(missing_ok=True)
            raise

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
        # P3-6: serve from cache within TTL so back-to-back snapshots (or a
        # snapshot right after another count read) don't re-open a readonly
        # conn and full-scan documents + chunks. Cache is per kb_storage_path,
        # keyed by the metadata.db path so a KB never sees another's counts.
        cache_key = str(kb_storage_path)
        cached = self._count_cache.get(cache_key)
        now = time.time()
        if cached is not None and (now - cached[0]) < self._COUNT_TTL:
            return cached[1], cached[2]

        doc_count = 0
        chunk_count = 0

        metadata_db = Path(kb_storage_path) / "metadata.db"
        if metadata_db.exists():
            try:
                conn = open_sqlite(metadata_db, readonly=True)
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

        self._count_cache[cache_key] = (now, doc_count, chunk_count)
        return doc_count, chunk_count
