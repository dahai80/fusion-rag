import fnmatch
import hashlib
import logging
import os

logger = logging.getLogger(__name__)


class IncrementalSyncEngine:
    def __init__(self):
        pass

    def compute_file_hash(self, file_path: str) -> str:
        hasher = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except OSError as e:
            logger.error("Failed to compute hash for %s: %s", file_path, e)
            return ""
        return hasher.hexdigest()

    def detect_changes(
        self,
        directory: str,
        existing_docs: list[dict],
        patterns: list[str] | None = None,
    ) -> dict:
        added = []
        modified = []
        deleted = []
        unchanged = []

        existing_map = {}
        for doc in existing_docs:
            fp = doc.get("file_path", "")
            existing_map[fp] = doc

        scanned_paths = set()

        if not os.path.isdir(directory):
            logger.warning("Directory does not exist: %s", directory)
            for doc in existing_docs:
                fp = doc.get("file_path", "")
                deleted.append(
                    {
                        "file_path": fp,
                        "file_hash": doc.get("file_hash", ""),
                        "doc_id": doc.get("doc_id", ""),
                    }
                )
            return {"added": added, "modified": modified, "deleted": deleted, "unchanged": unchanged}

        scan_root = os.path.realpath(directory)
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                full_path = os.path.join(root, fname)
                if patterns:
                    matched = False
                    for pat in patterns:
                        if fnmatch.fnmatch(fname, pat):
                            matched = True
                            break
                    if not matched:
                        continue

                real_full = os.path.realpath(full_path)
                if real_full != full_path and not os.path.realpath(full_path).startswith(scan_root + os.sep):
                    logger.warning("Skipping symlink that escapes scan root: %s -> %s", full_path, real_full)
                    continue

                scanned_paths.add(full_path)

                try:
                    stat = os.stat(full_path)
                    mtime = stat.st_mtime
                    size = stat.st_size
                except OSError as e:
                    logger.error("Failed to stat %s: %s", full_path, e)
                    continue

                file_hash = self.compute_file_hash(full_path)
                if not file_hash:
                    continue

                existing_doc = existing_map.get(full_path)
                if existing_doc is None:
                    added.append(
                        {
                            "file_path": full_path,
                            "file_hash": file_hash,
                            "mtime": mtime,
                            "size": size,
                        }
                    )
                    logger.debug("Detected new file: %s", full_path)
                else:
                    old_hash = existing_doc.get("file_hash", "")
                    if old_hash != file_hash:
                        modified.append(
                            {
                                "file_path": full_path,
                                "file_hash": file_hash,
                                "old_file_hash": old_hash,
                                "mtime": mtime,
                                "size": size,
                                "doc_id": existing_doc.get("doc_id", ""),
                            }
                        )
                        logger.debug("Detected modified file: %s (old=%s new=%s)", full_path, old_hash, file_hash)
                    else:
                        unchanged.append(
                            {
                                "file_path": full_path,
                                "file_hash": file_hash,
                                "mtime": mtime,
                                "size": size,
                                "doc_id": existing_doc.get("doc_id", ""),
                            }
                        )

        for fp, doc in existing_map.items():
            if fp not in scanned_paths:
                deleted.append(
                    {
                        "file_path": fp,
                        "file_hash": doc.get("file_hash", ""),
                        "doc_id": doc.get("doc_id", ""),
                    }
                )
                logger.debug("Detected deleted file: %s", fp)

        logger.info(
            "Change detection complete: added=%d modified=%d deleted=%d unchanged=%d",
            len(added),
            len(modified),
            len(deleted),
            len(unchanged),
        )

        return {"added": added, "modified": modified, "deleted": deleted, "unchanged": unchanged}

    def sync_directory(
        self,
        directory: str,
        existing_docs: list[dict],
        patterns: list[str] | None = None,
    ) -> dict:
        changes = self.detect_changes(directory, existing_docs, patterns)

        summary = {
            "added_count": len(changes["added"]),
            "modified_count": len(changes["modified"]),
            "deleted_count": len(changes["deleted"]),
            "unchanged_count": len(changes["unchanged"]),
            "added": changes["added"],
            "modified": changes["modified"],
            "deleted": changes["deleted"],
        }

        logger.info(
            "Sync summary for %s: added=%d modified=%d deleted=%d unchanged=%d",
            directory,
            summary["added_count"],
            summary["modified_count"],
            summary["deleted_count"],
            summary["unchanged_count"],
        )

        return summary
