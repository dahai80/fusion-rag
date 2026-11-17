"""Issue #156: standalone attachment file store — KB-independent file objects.

Chat clients (fusion-console) need an upload -> bounded-segment-read ->
delete pipeline that does NOT force a throwaway knowledge base per
attachment. FileStore keeps blobs on disk under {data_dir}/files/{file_id}
and metadata in a SQLite sidecar ({data_dir}/files/files.db, SqliteBase WAL).

Segment reads are the server-side enforcement point of the fusion-harness
file.read_segment contract: offset + max_bytes in, continuation metadata out
(content / total / next_offset / eof). The per-call byte cap is applied HERE
(oversize clamps, never errors). Text files serve UTF-8 slices cut on char
boundaries (next_offset = consumed bytes, so concatenating segments
reconstructs the file byte-exactly); binary files serve base64.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .sqlite_base import SqliteBase

logger = logging.getLogger(__name__)

# file_id is always a server-generated uuid4 hex — anything else never came
# from this store (and must never reach the blob path).
_FILE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def is_valid_file_id(file_id: str) -> bool:
    return bool(_FILE_ID_RE.fullmatch(file_id or ""))


class FileStore(SqliteBase):
    """KB-independent file objects: blob on disk + metadata row in SQLite."""

    def __init__(self, files_dir: str):
        self.files_dir = Path(files_dir)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.files_dir / "files.db")
        super().__init__()
        self._init_db()

    def _init_db(self) -> None:
        conn = self._get_conn()
        with self._db_lock:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS files (
                       file_id TEXT PRIMARY KEY,
                       filename TEXT NOT NULL,
                       size INTEGER NOT NULL,
                       checksum TEXT NOT NULL,
                       media_kind TEXT NOT NULL,
                       created_at TEXT NOT NULL
                   )"""
            )
            conn.commit()

    def put(self, filename: str, data: bytes) -> dict:
        if not data:
            raise ValueError("empty upload")
        file_id = uuid.uuid4().hex
        media_kind = "text" if self._decodable(data) else "binary"
        blob_path = self.files_dir / file_id
        blob_path.write_bytes(data)
        row = {
            "file_id": file_id,
            "filename": filename or f"attachment-{file_id[:8]}.bin",
            "size": len(data),
            "checksum": hashlib.sha256(data).hexdigest(),
            "media_kind": media_kind,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        conn = self._get_conn()
        with self._db_lock:
            conn.execute(
                "INSERT INTO files (file_id, filename, size, checksum, media_kind, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    row["file_id"],
                    row["filename"],
                    row["size"],
                    row["checksum"],
                    row["media_kind"],
                    row["created_at"],
                ),
            )
            conn.commit()
        logger.info(
            "file stored: %s (%s, %d bytes, %s)",
            file_id, row["filename"], row["size"], media_kind,
        )
        return row

    @staticmethod
    def _decodable(data: bytes) -> bool:
        try:
            data.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    def get(self, file_id: str) -> dict | None:
        if not is_valid_file_id(file_id):
            return None
        conn = self._get_conn()
        row = conn.execute(
            "SELECT file_id, filename, size, checksum, media_kind, created_at"
            " FROM files WHERE file_id = ?",
            (file_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_files(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT file_id, filename, size, checksum, media_kind, created_at"
            " FROM files ORDER BY created_at DESC, file_id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, file_id: str) -> bool:
        row = self.get(file_id)
        if row is None:
            return False
        conn = self._get_conn()
        with self._db_lock:
            conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
            conn.commit()
        blob_path = self.files_dir / file_id
        try:
            blob_path.unlink()
        except OSError as e:
            logger.warning("file blob unlink failed for %s: %s", file_id, e)
        logger.info("file deleted: %s (%s)", file_id, row["filename"])
        return True

    def blob_path(self, file_id: str) -> Path | None:
        if self.get(file_id) is None:
            return None
        return self.files_dir / file_id

    def read_segment(self, file_id: str, offset: int, max_bytes: int, cap: int) -> dict:
        """Bounded byte-range read. Oversize max_bytes CLAMPS to cap.

        Text files: the slice is trimmed to a UTF-8 char boundary and
        next_offset advances by the CONSUMED byte count (a split multibyte
        char is re-served by the next segment — concatenated segments
        reconstruct the file byte-exactly). Binary files: base64 content.
        """
        row = self.get(file_id)
        if row is None:
            raise KeyError(file_id)
        total = row["size"]
        offset = max(0, int(offset))
        if offset >= total:
            return {
                "file_id": file_id,
                "offset": offset,
                "content": "",
                "total": total,
                "next_offset": offset,
                "eof": True,
                "media_kind": row["media_kind"],
            }
        max_bytes = int(max_bytes) if max_bytes and int(max_bytes) > 0 else cap
        max_bytes = min(max_bytes, cap)
        with open(self.files_dir / file_id, "rb") as f:
            f.seek(offset)
            raw = f.read(max_bytes)
        if row["media_kind"] == "text":
            raw, dropped = self._trim_partial_utf8(raw)
            if dropped:
                logger.debug("segment for %s trimmed %d trailing partial utf-8 bytes", file_id, dropped)
            next_offset = offset + len(raw)
            content = raw.decode("utf-8")
        else:
            next_offset = offset + len(raw)
            content = base64.b64encode(raw).decode("ascii")
        eof = next_offset >= total
        return {
            "file_id": file_id,
            "offset": offset,
            "content": content,
            "total": total,
            "next_offset": next_offset,
            "eof": eof,
            "media_kind": row["media_kind"],
        }

    @staticmethod
    def _trim_partial_utf8(raw: bytes) -> tuple[bytes, int]:
        # A read boundary can split a multibyte char. Trim up to 3 trailing
        # bytes until the prefix decodes; the trimmed bytes are re-served by
        # the next segment (next_offset advances only over consumed bytes).
        for cut in (0, 1, 2, 3):
            if cut >= len(raw):
                return b"", 0
            try:
                raw[: len(raw) - cut].decode("utf-8")
                return raw[: len(raw) - cut], cut
            except UnicodeDecodeError:
                continue
        return raw, 0
