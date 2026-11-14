from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any

import pathspec

from .._validators import validate_git_url

logger = logging.getLogger(__name__)


class GitLoader:
    def __init__(self, work_dir: str = ""):
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="fusion_git_")
        self._owns_work_dir = not bool(work_dir)

    def clone(self, repo_url: str, branch: str = "main", depth: int = 1) -> str:
        # F3 fix: validate scheme + reject ext:: (RCE via git transport/hook).
        validate_git_url(repo_url, field="repo_url")
        target = os.path.join(self.work_dir, repo_url.split("/")[-1].replace(".git", ""))
        if os.path.exists(target):
            logger.info("Git repo already cloned at %s, pulling latest", target)
            self._run_git(["pull", "--ff-only"], cwd=target)
            return target
        # F3: disable hooks to block post-checkout RCE; set core.hooksPath=/dev/null.
        cmd = [
            "git",
            "clone",
            "-c",
            "core.hooksPath=/dev/null",
            "--no-install-hooks",
            "--branch",
            branch,
            "--depth",
            str(depth),
            repo_url,
            target,
        ]
        self._run_git(cmd)
        logger.info("Cloned %s (branch=%s) to %s", repo_url, branch, target)
        return target

    def list_files(
        self, repo_path: str, patterns: list[str] | None = None, ignore_file: str = ".gitignore"
    ) -> list[str]:
        gitignore_path = os.path.join(repo_path, ignore_file)
        ignore_spec = self._parse_gitignore(gitignore_path)
        all_files = []
        for root, dirs, files in os.walk(repo_path):
            if ".git" in dirs:
                dirs.remove(".git")
            rel_root = os.path.relpath(root, repo_path)
            if self._is_ignored(rel_root, ignore_spec):
                dirs.clear()
                continue
            for f in files:
                rel_path = os.path.join(rel_root, f) if rel_root != "." else f
                if self._is_ignored(rel_path, ignore_spec):
                    continue
                if patterns and not any(rel_path.endswith(p.lstrip("*")) for p in patterns):
                    continue
                all_files.append(os.path.join(root, f))
        return all_files

    async def clone_and_index(
        self, repo_url: str, branch: str = "main", patterns: list[str] | None = None, depth: int = 1
    ) -> list[dict[str, Any]]:
        # F4 fix: async + await parser.parse (was sync call to async -> coroutine,
        # AttributeError swallowed by broad except -> always returned []). M6: cleanup work_dir.
        from ..engine.document import DocumentParser

        owns = self._owns_work_dir
        parser = DocumentParser()
        try:
            repo_path = self.clone(repo_url, branch=branch, depth=depth)
            files = self.list_files(repo_path, patterns=patterns)
            results = []
            for fpath in files:
                try:
                    parsed = await parser.parse(fpath)
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
        finally:
            if owns:
                shutil.rmtree(self.work_dir, ignore_errors=True)
                logger.debug("clone_and_index cleaned work_dir=%s", self.work_dir)

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

    def _parse_gitignore(self, path: str) -> pathspec.GitIgnoreSpec:
        # M6: the old naive matcher used substring/basename compares and
        # ignored gitignore semantics — `!important.log` negation was dropped
        # (an un-ignored file stayed ignored), `**/glob` and anchored paths
        # were unsupported. Use pathspec.GitIgnoreSpec for full gitignore
        # semantics including negation.
        if not os.path.exists(path):
            return pathspec.GitIgnoreSpec.from_lines([])
        with open(path, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip() and not ln.strip().startswith("#")]
        return pathspec.GitIgnoreSpec.from_lines(lines)

    def _is_ignored(self, rel_path: str, ignore_spec: pathspec.GitIgnoreSpec) -> bool:
        # M6: normalize to forward slashes (pathspec matches git-style paths);
        # rel_root == "." for the repo root must not be treated as ignored.
        if not ignore_spec.patterns:
            return False
        norm = rel_path.replace(os.sep, "/")
        if norm in (".", ""):
            return False
        return ignore_spec.match_file(norm)
