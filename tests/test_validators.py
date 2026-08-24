"""硬伤3 trust-boundary layer tests — lock the validation contracts.

Each helper must reject the attack vectors named in the audit (F12/F15/F3/F5/F6
+ sync symlink escape) and accept the legitimate inputs.
"""

from __future__ import annotations

import os

import pytest

from fusion_rag._validators import (
    ValidationError,
    validate_git_url,
    validate_identifier,
    validate_path_under_root,
    validate_sql_identifier,
    validate_url,
)


class TestIdentifier:
    def test_accepts_normal(self):
        assert validate_identifier("kb_1") == "kb_1"
        assert validate_identifier("My-KB-9") == "My-KB-9"

    @pytest.mark.parametrize("bad", ["../etc", "a/b", "a.b", "a b", "", "a" * 65, "a:b", "a;b"])
    def test_rejects_traversal_and_bad_charset(self, bad):
        with pytest.raises(ValidationError):
            validate_identifier(bad, field="kb_id")

    def test_rejects_dot_dot(self):
        with pytest.raises(ValidationError):
            validate_identifier("..", field="kb_id")


class TestSqlIdentifier:
    def test_accepts_normal(self):
        assert validate_sql_identifier("kb_1") == "kb_1"
        assert validate_sql_identifier("my_table.col") == "my_table.col"

    @pytest.mark.parametrize("bad", ["'; DROP--", "a b", "1abc", "", "a\"b", "a;b"])
    def test_rejects_injection(self, bad):
        with pytest.raises(ValidationError):
            validate_sql_identifier(bad)


class TestPathUnderRoot:
    def test_accepts_inside(self, tmp_path):
        root = tmp_path / "store"
        (root / "kb1").mkdir(parents=True)
        f = root / "kb1" / "doc.txt"
        f.write_text("x")
        result = validate_path_under_root(str(f), root=root, field="file_path")
        assert result == f.resolve()

    def test_rejects_traversal_outside(self, tmp_path):
        root = tmp_path / "store"
        root.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("x")
        attack = str(root / ".." / "secret.txt")
        with pytest.raises(ValidationError):
            validate_path_under_root(attack, root=root, field="file_path")

    def test_rejects_symlink_escape(self, tmp_path):
        root = tmp_path / "store"
        root.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("x")
        link = root / "link.txt"
        os.symlink(outside, link)
        with pytest.raises(ValidationError):
            validate_path_under_root(str(link), root=root, field="file_path")


class TestUrl:
    def test_accepts_https(self):
        assert validate_url("https://example.com/doc") == "https://example.com/doc"

    @pytest.mark.parametrize("bad", ["file:///etc/passwd", "ftp://x", "javascript:alert(1)", ""])
    def test_rejects_bad_scheme(self, bad):
        with pytest.raises(ValidationError):
            validate_url(bad)

    def test_rejects_loopback(self):
        with pytest.raises(ValidationError):
            validate_url("http://127.0.0.1/")

    def test_allow_private_flag(self):
        assert validate_url("http://127.0.0.1/", allow_private=True) == "http://127.0.0.1/"


class TestGitUrl:
    def test_accepts_https(self):
        assert validate_git_url("https://github.com/x/y.git") == "https://github.com/x/y.git"

    def test_rejects_ext_transport(self):
        with pytest.raises(ValidationError):
            validate_git_url("ext::ssh -o ProxyCommand=evil %s x")

    def test_rejects_newline_injection(self):
        with pytest.raises(ValidationError):
            validate_git_url("https://x.git\n--upload-pack=evil")


class TestSyncSymlinkEscape:
    def test_symlink_escape_skipped(self, tmp_path):
        from fusion_rag.engine.incremental_sync import IncrementalSyncEngine

        scan = tmp_path / "scan"
        scan.mkdir()
        (scan / "real.txt").write_text("ok")
        outside = tmp_path / "outside_secret.txt"
        outside.write_text("leaked")
        os.symlink(outside, scan / "evil.txt")

        engine = IncrementalSyncEngine()
        changes = engine.detect_changes(str(scan), [])
        added_paths = [d["file_path"] for d in changes["added"]]
        assert any(p.endswith("real.txt") for p in added_paths)
        assert not any("evil" in p for p in added_paths), "symlink escape must be skipped"
