import enum
import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Role(enum.Enum):
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    API_ONLY = "api_only"


class PermissionAction(enum.Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SEARCH = "search"


@dataclass
class PermissionRule:
    id: str
    kb_id: str
    subject: str
    resource_type: str
    resource_path: str
    actions: list[str]
    created_at: float = 0.0


class ACL:
    def __init__(self, rules: list[PermissionRule]):
        self.rules = rules

    def check(self, subject: str, action: str, resource_path: str) -> bool:
        logger.debug("ACL check: subject=%s action=%s resource_path=%s", subject, action, resource_path)

        for rule in self.rules:
            if rule.subject != subject:
                continue
            if rule.subject == Role.ADMIN.value:
                logger.debug("ACL: ADMIN subject %s allowed by default", subject)
                return True

        for rule in self.rules:
            if rule.subject == subject and rule.subject == Role.ADMIN.value:
                break

        matching_rules = []
        for rule in self.rules:
            if rule.subject != subject:
                continue
            if rule.resource_path == resource_path or resource_path.startswith(rule.resource_path.rstrip("/") + "/"):
                matching_rules.append(rule)

        if not matching_rules:
            logger.debug("ACL: no matching rules for subject=%s resource_path=%s -> deny", subject, resource_path)
            return False

        for rule in matching_rules:
            if action in rule.actions:
                logger.debug("ACL: subject=%s action=%s allowed on %s", subject, action, rule.resource_path)
                return True

        logger.debug("ACL: subject=%s action=%s not in any matching rule -> deny", subject, action)
        return False


class PermissionManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        logger.info("PermissionManager init: db_path=%s", db_path)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS permissions (
                    id TEXT PRIMARY KEY,
                    kb_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_path TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_permissions_kb_subject ON permissions (kb_id, subject)")
            conn.commit()
            logger.info("PermissionManager: table permissions ensured")
        finally:
            conn.close()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _cursor(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        d["actions"] = json.loads(d["actions"])
        return d

    def add_rule(self, rule: PermissionRule) -> dict:
        if not rule.id:
            rule.id = uuid.uuid4().hex
        if not rule.created_at:
            rule.created_at = time.time()

        actions_json = json.dumps(rule.actions)
        logger.info(
            "add_rule: id=%s kb_id=%s subject=%s resource_path=%s actions=%s",
            rule.id,
            rule.kb_id,
            rule.subject,
            rule.resource_path,
            actions_json,
        )

        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO permissions (id, kb_id, subject, resource_type, resource_path, actions, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule.id,
                    rule.kb_id,
                    rule.subject,
                    rule.resource_type,
                    rule.resource_path,
                    actions_json,
                    rule.created_at,
                ),
            )

        return {
            "id": rule.id,
            "kb_id": rule.kb_id,
            "subject": rule.subject,
            "resource_type": rule.resource_type,
            "resource_path": rule.resource_path,
            "actions": rule.actions,
            "created_at": rule.created_at,
        }

    def delete_rule(self, rule_id: str) -> bool:
        logger.info("delete_rule: rule_id=%s", rule_id)
        with self._cursor() as cur:
            cur.execute("DELETE FROM permissions WHERE id = ?", (rule_id,))
            deleted = cur.rowcount > 0
        if deleted:
            logger.info("delete_rule: deleted rule_id=%s", rule_id)
        else:
            logger.warning("delete_rule: rule_id=%s not found", rule_id)
        return deleted

    def list_rules(self, kb_id: str) -> list[dict]:
        logger.debug("list_rules: kb_id=%s", kb_id)
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM permissions WHERE kb_id = ? ORDER BY created_at",
                (kb_id,),
            )
            rows = cur.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_rules_for_subject(self, kb_id: str, subject: str) -> list[dict]:
        logger.debug("get_rules_for_subject: kb_id=%s subject=%s", kb_id, subject)
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM permissions WHERE kb_id = ? AND subject = ? ORDER BY created_at",
                (kb_id, subject),
            )
            rows = cur.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def check_permission(self, kb_id: str, subject: str, action: str, resource_path: str) -> bool:
        logger.debug(
            "check_permission: kb_id=%s subject=%s action=%s resource_path=%s", kb_id, subject, action, resource_path
        )

        subject_rules = self.get_rules_for_subject(kb_id, subject)

        role_rules = []
        for role in Role:
            role_rules_for_kb = self.get_rules_for_subject(kb_id, role.value)
            role_rules.extend(role_rules_for_kb)

        all_rule_dicts = subject_rules + role_rules

        rules = []
        for rd in all_rule_dicts:
            rule = PermissionRule(
                id=rd["id"],
                kb_id=rd["kb_id"],
                subject=rd["subject"],
                resource_type=rd["resource_type"],
                resource_path=rd["resource_path"],
                actions=rd["actions"],
                created_at=rd["created_at"],
            )
            rules.append(rule)

        acl = ACL(rules)
        result = acl.check(subject, action, resource_path)

        if subject != Role.ADMIN.value:
            admin_result = acl.check(Role.ADMIN.value, action, resource_path)
            if admin_result:
                logger.debug("check_permission: subject=%s inherits ADMIN permission", subject)
                result = True

        logger.debug("check_permission result: %s", result)
        return result
