"""Input validation helpers — single trust-boundary layer.

Centralizes all external-input confinement (硬伤3):
  - kb_id / table_name / identifier -> regex whitelist
  - filesystem path -> configured root + is_relative_to + symlink reject
  - URL -> scheme whitelist + private-network reject (SSRF guard)
  - git repo_url -> scheme whitelist + ext:: reject (RCE guard)

No business logic here. Pure validation + raise on violation. Every
caller that takes external input and feeds it to fs/SQL/network/git
must route through these helpers.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# identifier whitelist: letters/digits/_-. , 1-64 chars, no path separators
_IDENT_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
# SQL identifier: letters/digits/_., must start with letter/_
_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
_ALLOWED_GIT_SCHEMES = frozenset({"http", "https", "git", "ssh"})

# RFC1918 + loopback + link-local + unique-local v6 ranges
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


class ValidationError(ValueError):
    """Raised when external input violates trust-boundary rules."""


def validate_identifier(name: str, *, field: str = "identifier") -> str:
    """Validate a KB id / rule id / generic identifier.

    Rejects path separators, dots, and any non-whitelisted char.
    """
    if not name or not _IDENT_RE.match(name):
        logger.warning("validate_identifier reject: field=%s value=%r", field, name)
        raise ValidationError(f"invalid {field}: must match [A-Za-z0-9_-]{{1,64}}")
    if name in (".", ".."):
        raise ValidationError(f"invalid {field}: path traversal denied")
    return name


def validate_sql_identifier(name: str, *, field: str = "table_name") -> str:
    """Validate a SQL table/column identifier against injection.

    Identifiers cannot be bound as params, so we whitelist the charset
    and reject anything containing quotes / semicolons / path separators.
    Caller should still wrap the result in double quotes for SQLite PRAGMA.
    """
    if not name or not _SQL_IDENT_RE.match(name):
        logger.warning("validate_sql_identifier reject: field=%s value=%r", field, name)
        raise ValidationError(f"invalid {field}: must match [A-Za-z_][A-Za-z0-9_.]*")
    if '"' in name or ";" in name or "--" in name:
        raise ValidationError(f"invalid {field}: forbidden chars")
    return name


def validate_path_under_root(
    path: str,
    *,
    root: str | Path,
    field: str = "file_path",
    allow_root: bool = False,
) -> Path:
    """Confine a filesystem path to a configured root (LFI / path-traversal guard).

    Resolves symlinks, then asserts the resolved path is relative to root.
    Rejects symlink escapes and any path that leaves root.

    NOTE (P1-11, TOCTOU): this validates the path AT CALL TIME. resolve() follows
    symlinks, so a symlink that points inside root at check-time passes; if an
    attacker swaps it to point outside root between this check and the caller's
    open(), the guard is defeated (classic TOCTOU). A pure-path validator cannot
    close this — callers that read untrusted paths MUST open with O_NOFOLLOW (or
    os.open + lstat-vs-fstat) to make the check-then-use atomic. This function is
    the first line, not the only line.
    """
    if not path:
        raise ValidationError(f"invalid {field}: empty")
    if not root:
        raise ValidationError(f"invalid root for {field}: empty")
    root_resolved = Path(root).expanduser().resolve()
    target = Path(path).expanduser().resolve()
    if not allow_root and target == root_resolved:
        # P1-11: this branch was a dead `pass` — the docstring claimed it
        # disallowed operating on root itself, but it never raised, so the
        # promise was a lie. All current callers validate a file UNDER root
        # (never the root dir), so rejecting root-equal matches intent. If a
        # future caller legitimately needs the root, pass allow_root=True.
        logger.warning("validate_path_under_root reject: field=%s path equals root=%s", field, root_resolved)
        raise ValidationError(f"invalid {field}: path equals ingest root (not a file under it)")
    if not target.is_relative_to(root_resolved):
        logger.warning(
            "validate_path_under_root reject: field=%s path=%s resolved=%s root=%s",
            field,
            path,
            target,
            root_resolved,
        )
        raise ValidationError(f"invalid {field}: escapes ingest root")
    # Symlink check: if the raw path is a symlink, ensure it doesn't escape.
    raw = Path(path)
    if raw.is_symlink() and not Path(raw.readlink()).resolve().is_relative_to(root_resolved):
        raise ValidationError(f"invalid {field}: symlink escapes root")
    return target


def _host_is_private(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(socket.gethostbyname(host))
    except (OSError, ValueError):
        return False
    return any(addr in net for net in _PRIVATE_NETWORKS)


def validate_url(url: str, *, field: str = "url", allow_private: bool = False) -> str:
    """Validate an HTTP(S) URL — scheme whitelist + SSRF guard.

    Rejects non-http(s) schemes, and (unless allow_private) hosts resolving
    to private/loopback/link-local ranges.
    """
    if not url:
        raise ValidationError(f"invalid {field}: empty")
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        logger.warning("validate_url reject: field=%s scheme=%s", field, scheme)
        raise ValidationError(f"invalid {field}: scheme must be http/https")
    host = parsed.hostname or ""
    if not host:
        raise ValidationError(f"invalid {field}: no host")
    if not allow_private and _host_is_private(host):
        logger.warning("validate_url reject: field=%s host=%s is private", field, host)
        raise ValidationError(f"invalid {field}: private/loopback host denied")
    return url


def validate_git_url(url: str, *, field: str = "repo_url") -> str:
    """Validate a git clone URL — scheme whitelist + ext:: RCE guard.

    Rejects ext:: (and any non-whitelisted scheme) which git treats as an
    executable transport protocol (RCE via post-checkout hook chain).
    """
    if not url:
        raise ValidationError(f"invalid {field}: empty")
    # Reject ext:: outright — it is an executable transport.
    if url.startswith("ext::") or "\n" in url or "\0" in url:
        logger.warning("validate_git_url reject: field=%s ext/nul/newline", field)
        raise ValidationError(f"invalid {field}: ext:: transport denied")
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme and scheme not in _ALLOWED_GIT_SCHEMES:
        logger.warning("validate_git_url reject: field=%s scheme=%s", field, scheme)
        raise ValidationError(f"invalid {field}: scheme must be http/https/git/ssh")
    # For http/https also apply SSRF guard.
    if scheme in ("http", "https"):
        host = parsed.hostname or ""
        if host and _host_is_private(host):
            raise ValidationError(f"invalid {field}: private/loopback host denied")
    return url
