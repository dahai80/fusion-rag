from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


class GitLoader:
    def __init__(self, work_dir: str = ""):
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="fusion_git_")

    def clone(self, repo_url: str, branch: str = "main", depth: int = 1) -> str:
        target = os.path.join(self.work_dir, repo_url.split("/")[-1].replace(".git", ""))
        if os.path.exists(target):
            logger.info("Git repo already cloned at %s, pulling latest", target)
            self._run_git(["pull", "--ff-only"], cwd=target)
            return target
        cmd = ["git", "clone", "--branch", branch, "--depth", str(depth), repo_url, target]
        self._run_git(cmd)
        logger.info("Cloned %s (branch=%s) to %s", repo_url, branch, target)
        return target

    def list_files(
        self, repo_path: str, patterns: list[str] | None = None, ignore_file: str = ".gitignore"
    ) -> list[str]:
        gitignore_path = os.path.join(repo_path, ignore_file)
        ignore_patterns = self._parse_gitignore(gitignore_path)
        all_files = []
        for root, dirs, files in os.walk(repo_path):
            if ".git" in dirs:
                dirs.remove(".git")
            rel_root = os.path.relpath(root, repo_path)
            if self._is_ignored(rel_root, ignore_patterns):
                dirs.clear()
                continue
            for f in files:
                rel_path = os.path.join(rel_root, f) if rel_root != "." else f
                if self._is_ignored(rel_path, ignore_patterns):
                    continue
                if patterns and not any(rel_path.endswith(p.lstrip("*")) for p in patterns):
                    continue
                all_files.append(os.path.join(root, f))
        return all_files

    def clone_and_index(
        self, repo_url: str, branch: str = "main", patterns: list[str] | None = None, depth: int = 1
    ) -> list[dict[str, Any]]:
        repo_path = self.clone(repo_url, branch=branch, depth=depth)
        files = self.list_files(repo_path, patterns=patterns)
        results = []
        from ..engine.document import DocumentParser

        parser = DocumentParser()
        for fpath in files:
            try:
                parsed = parser.parse(fpath)
                results.append(
                    {
                        "file_path": fpath,
                        "file_name": os.path.basename(fpath),
                        "content": parsed.content,
                        "doc_type": parsed.doc_type.value
                        if hasattr(parsed.doc_type, "value")
                        else str(parsed.doc_type),
                        "chars": parsed.chars,
                    }
                )
            except Exception as e:
                logger.warning("Failed to parse %s: %s", fpath, e)
        logger.info("Indexed %d/%d files from %s", len(results), len(files), repo_url)
        return results

    def _run_git(self, cmd: list[str], cwd: str | None = None) -> str:
        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=cwd,
            )
            if result.returncode != 0:
                logger.error("git command failed: %s — %s", " ".join(cmd), result.stderr.strip())
                raise RuntimeError(f"git failed: {result.stderr.strip()}")
            return result.stdout.strip()
        except FileNotFoundError:
            raise RuntimeError("git is not installed or not in PATH")

    def _parse_gitignore(self, path: str) -> list[str]:
        patterns = []
        if not os.path.exists(path):
            return patterns
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)
        return patterns

    def _is_ignored(self, rel_path: str, patterns: list[str]) -> bool:
        if not patterns:
            return False
        name = os.path.basename(rel_path)
        for p in patterns:
            if p.endswith("/"):
                if rel_path.startswith(p) or name == p.rstrip("/"):
                    return True
            elif p.startswith("*"):
                if name.endswith(p.lstrip("*")):
                    return True
            elif p in (name, rel_path):
                return True
        return False
