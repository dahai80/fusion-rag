"""M6: git_loader gitignore semantics — pathspec-backed, not naive matching.

Covers negation (`!important.log`), `**` glob, and directory patterns that the
old substring/basename matcher got wrong. No git clone — exercises _parse_*
and _is_ignored directly against a temp repo dir.
"""
from __future__ import annotations

import os
import tempfile

from fusion_rag.connectors.git_loader import GitLoader


class TestGitLoaderIgnore:
    def _write_gitignore(self, repo: str, lines: list[str]) -> None:
        with open(os.path.join(repo, ".gitignore"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def test_negation_unignores(self):
        # M6 core: `!important.log` must re-include a file `*.log` ignored.
        loader = GitLoader(work_dir=tempfile.mkdtemp(prefix="fusion_git_test_"))
        with tempfile.TemporaryDirectory() as repo:
            self._write_gitignore(repo, ["*.log", "!important.log"])
            spec = loader._parse_gitignore(os.path.join(repo, ".gitignore"))
            assert loader._is_ignored("debug.log", spec) is True
            assert loader._is_ignored("important.log", spec) is False

    def test_double_star_glob(self):
        # M6: `**/temp` must match at any depth (old matcher missed nested).
        loader = GitLoader(work_dir=tempfile.mkdtemp(prefix="fusion_git_test_"))
        with tempfile.TemporaryDirectory() as repo:
            self._write_gitignore(repo, ["**/temp"])
            spec = loader._parse_gitignore(os.path.join(repo, ".gitignore"))
            assert loader._is_ignored("temp", spec) is True
            assert loader._is_ignored("src/temp", spec) is True
            assert loader._is_ignored("src/main.py", spec) is False

    def test_directory_pattern(self):
        # M6: `build/` ignores everything under build/; `node_modules` (no
        # trailing slash) matches the dir and its contents at any depth.
        loader = GitLoader(work_dir=tempfile.mkdtemp(prefix="fusion_git_test_"))
        with tempfile.TemporaryDirectory() as repo:
            self._write_gitignore(repo, ["build/", "node_modules"])
            spec = loader._parse_gitignore(os.path.join(repo, ".gitignore"))
            assert loader._is_ignored("build/output.o", spec) is True
            assert loader._is_ignored("build/sub/deep.txt", spec) is True
            assert loader._is_ignored("node_modules", spec) is True
            assert loader._is_ignored("node_modules/pkg/index.js", spec) is True
            assert loader._is_ignored("src/app.py", spec) is False

    def test_root_not_ignored(self):
        # M6: rel_root "." must never be treated as ignored (would prune all).
        loader = GitLoader(work_dir=tempfile.mkdtemp(prefix="fusion_git_test_"))
        with tempfile.TemporaryDirectory() as repo:
            self._write_gitignore(repo, ["*.tmp"])
            spec = loader._parse_gitignore(os.path.join(repo, ".gitignore"))
            assert loader._is_ignored(".", spec) is False

    def test_empty_patterns_never_ignored(self):
        # No .gitignore -> nothing ignored.
        loader = GitLoader(work_dir=tempfile.mkdtemp(prefix="fusion_git_test_"))
        with tempfile.TemporaryDirectory() as repo:
            spec = loader._parse_gitignore(os.path.join(repo, ".gitignore"))
            assert loader._is_ignored("anything.txt", spec) is False
