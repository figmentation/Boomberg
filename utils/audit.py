"""
utils/audit.py :: Append-only record of who did what, and when.

What is recorded
  * auth.session      a session's first run under an identity and role
  * auth.login / auth.logout
  * access.denied     a page the session's role does not include
  * command           every command-bar command: ok, rejected or denied
  * page.view         a page opened, with how long its render took
  * page.error        a page that crashed
  * portfolio.*, cache.*, audit.export   state-changing or privileged actions

Integrity
  Rows live in their own SQLite file (config.AUDIT_DB), not the cache, so
  PURGE cannot reach them. Triggers refuse UPDATE and DELETE, so no code path
  in the terminal can rewrite history. Each row also carries a SHA-256 over
  its own fields and the previous row's hash, and `verify()` recomputes that
  chain: an edit made to the file directly - which triggers cannot stop - is
  reported at the first row that no longer matches. Deleting the newest rows
  leaves a shorter chain that still verifies, which is why the audit page
  shows the head hash for anyone who needs to anchor it elsewhere.

  Privileged actions are fail-closed: the terminal writes the record first
  and refuses the action if the write fails. Views and commands are
  best-effort, so an unwritable log cannot take the whole terminal down.

What is not recorded
  Secrets. Every free-text field passes `redact()`, which removes the
  configured API credentials verbatim and anything shaped like a key or a
  token. The actor's account identifier (an email) IS recorded - a trail that
  cannot name the actor is not an audit trail - so the file belongs to whoever
  administers the deployment, the in-app viewer is admin-only, and the
  identifier never goes to the application log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import config

log = logging.getLogger("openterm.audit")

# --------------------------------------------------------------------------
# Event vocabulary
# --------------------------------------------------------------------------
AUTH_SESSION = "auth.session"
AUTH_LOGIN = "auth.login"
AUTH_LOGOUT = "auth.logout"
ACCESS_DENIED = "access.denied"
COMMAND = "command"
PAGE_VIEW = "page.view"
PAGE_ERROR = "page.error"
PORTFOLIO_SAVE = "portfolio.save"
WATCHLIST_ADD = "portfolio.watch"
BRIEF_REBUILD = "portfolio.rebuild_brief"
CACHE_REFRESH = "cache.refresh"
CACHE_PURGE = "cache.purge"
AUDIT_EXPORT = "audit.export"

EVENT_TYPES: Tuple[str, ...] = (
    AUTH_SESSION, AUTH_LOGIN, AUTH_LOGOUT, ACCESS_DENIED, COMMAND, PAGE_VIEW,
    PAGE_ERROR, PORTFOLIO_SAVE, WATCHLIST_ADD, BRIEF_REBUILD, CACHE_REFRESH,
    CACHE_PURGE, AUDIT_EXPORT,
)
OUTCOMES: Tuple[str, ...] = ("ok", "rejected", "denied", "error")

GENESIS = "0" * 64
REDACTED = "[REDACTED]"
_MAX_FIELD = 500

# The hashed fields, in hash order. Changing this order invalidates every
# existing chain, so append new columns rather than reordering.
_FIELDS: Tuple[str, ...] = ("ts", "event", "outcome", "actor_key", "actor",
                            "role", "session", "module", "subject", "detail")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    event      TEXT NOT NULL,
    outcome    TEXT NOT NULL,
    actor_key  TEXT NOT NULL,
    actor      TEXT NOT NULL,
    role       TEXT NOT NULL,
    session    TEXT NOT NULL,
    module     TEXT NOT NULL,
    subject    TEXT NOT NULL,
    detail     TEXT NOT NULL,
    prev_hash  TEXT NOT NULL,
    hash       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_event ON audit(event);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit(actor_key);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------
# "api_key=...", "token: ...", "password=..." - the value goes, the name stays
# so a reviewer can see that something was removed and what kind.
_KEYWORD_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|client[_-]?secret|token|secret|"
    r"password|passwd|bearer)(\s*[:=]\s*)(\S+)")
# Long unbroken runs of key-alphabet characters. No ticker, desk name, region
# or MMSI comes close to 32; most API keys and session tokens exceed it.
_OPAQUE_TOKEN = re.compile(r"[A-Za-z0-9_\-]{32,}")


def _configured_secrets() -> List[str]:
    values = (config.FRED_API_KEY, config.OPENSKY_CLIENT_ID,
              config.OPENSKY_CLIENT_SECRET, config.AISSTREAM_API_KEY)
    # Very short values would redact ordinary words; real keys are longer.
    return [value for value in values if value and len(value) >= 6]


def redact(value: Any) -> str:
    """Text with credentials removed, capped at a fixed length."""
    text = "" if value is None else str(value)
    for secret in _configured_secrets():
        text = text.replace(secret, REDACTED)
    text = _KEYWORD_SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    text = _OPAQUE_TOKEN.sub(REDACTED, text)
    return text[:_MAX_FIELD]


def _detail_json(detail: Optional[Dict[str, Any]]) -> str:
    cleaned = {
        str(key)[:64]: (value if isinstance(value, (int, float, bool)) or value is None
                        else redact(value))
        for key, value in (detail or {}).items()
    }
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


def _digest(prev_hash: str, row: Dict[str, str]) -> str:
    payload = json.dumps([prev_hash] + [row[name] for name in _FIELDS],
                         separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _actor(principal: Any) -> Tuple[str, str, str]:
    """(actor_key, actor, role). Duck-typed so this module imports no auth code."""
    if principal is None:
        return "system", "SYSTEM", ""
    user = principal.user
    return str(user.key), str(user.display), str(principal.role or "none")


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
class AuditLog:
    """
    The append-only table. One short-lived connection per operation: writes
    are rare next to reads of market data, and a fresh connection keeps the
    object safe to share across Streamlit's script threads.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = str(path or config.AUDIT_DB)
        self._lock = threading.Lock()
        self._ready = False
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None: transactions are opened explicitly below.
        conn = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        try:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA)
            finally:
                conn.close()
            self._ready = True
        except Exception as exc:
            log.error("Audit log unavailable (%s): %s", self.path,
                      type(exc).__name__)

    def record(self, event: str, principal: Any = None, *, outcome: str = "ok",
               session: str = "", module: str = "", subject: Any = "",
               detail: Optional[Dict[str, Any]] = None) -> bool:
        """Append one row. True if it was durably written; never raises."""
        actor_key, actor, role = _actor(principal)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": str(event),
            "outcome": str(outcome),
            "actor_key": actor_key,
            "actor": actor,
            "role": role,
            "session": str(session or ""),
            "module": str(module or ""),
            "subject": redact(subject),
            "detail": _detail_json(detail),
        }
        columns = ", ".join(_FIELDS)
        placeholders = ", ".join("?" for _ in range(len(_FIELDS) + 2))

        with self._lock:
            if not self._ready:
                self._ensure_schema()
            try:
                conn = self._connect()
                try:
                    # IMMEDIATE takes the write lock before reading the head,
                    # so two processes cannot both chain onto the same row.
                    conn.execute("BEGIN IMMEDIATE")
                    head = conn.execute(
                        "SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
                    prev_hash = head[0] if head else GENESIS
                    conn.execute(
                        f"INSERT INTO audit ({columns}, prev_hash, hash) "
                        f"VALUES ({placeholders})",
                        [row[name] for name in _FIELDS]
                        + [prev_hash, _digest(prev_hash, row)],
                    )
                    conn.execute("COMMIT")
                finally:
                    conn.close()   # an uncommitted transaction rolls back
                return True
            except Exception as exc:
                log.error("Audit write failed for %s: %s", event,
                          type(exc).__name__)
                return False

    def recent(self, limit: int = 500, events: Optional[Iterable[str]] = None,
               outcome: Optional[str] = None,
               actor: Optional[str] = None) -> List[Dict[str, Any]]:
        """Newest rows first, optionally filtered."""
        clauses: List[str] = []
        params: List[Any] = []
        events = [str(e) for e in (events or [])]
        if events:
            clauses.append(f"event IN ({', '.join('?' for _ in events)})")
            params.extend(events)
        if outcome:
            clauses.append("outcome = ?")
            params.append(outcome)
        if actor:
            clauses.append("actor LIKE ?")
            params.append(f"%{actor}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 10_000)))

        names = ("id",) + _FIELDS + ("hash",)
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"SELECT {', '.join(names)} FROM audit {where} "
                    f"ORDER BY id DESC LIMIT ?", params).fetchall()
            finally:
                conn.close()
        except Exception as exc:
            log.error("Audit read failed: %s", type(exc).__name__)
            return []
        return [dict(zip(names, row)) for row in rows]

    def verify(self) -> Tuple[bool, Optional[int], int]:
        """
        Recompute the hash chain from the first row.

        Returns (intact, first_bad_id, rows_checked). A log that cannot be
        read at all reports (False, None, 0).
        """
        prev_hash = GENESIS
        checked = 0
        try:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    f"SELECT id, {', '.join(_FIELDS)}, prev_hash, hash "
                    f"FROM audit ORDER BY id")
                for record in cursor:
                    checked += 1
                    row = dict(zip(_FIELDS, record[1:1 + len(_FIELDS)]))
                    stored_prev, stored_hash = record[-2], record[-1]
                    if stored_prev != prev_hash or stored_hash != _digest(prev_hash, row):
                        return False, record[0], checked
                    prev_hash = stored_hash
            finally:
                conn.close()
        except Exception as exc:
            log.error("Audit verify failed: %s", type(exc).__name__)
            return False, None, checked
        return True, None, checked

    def head_hash(self) -> Optional[str]:
        """The newest row's hash, for anchoring outside this file."""
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            finally:
                conn.close()
        except Exception:
            return None
        return row[0] if row else None


# --------------------------------------------------------------------------
# Module-level singleton
# --------------------------------------------------------------------------
_log_singleton: Optional[AuditLog] = None
_singleton_lock = threading.Lock()


def get_audit_log() -> AuditLog:
    global _log_singleton
    if _log_singleton is None:
        with _singleton_lock:
            if _log_singleton is None:
                _log_singleton = AuditLog()
    return _log_singleton


def record(event: str, principal: Any = None, **fields: Any) -> bool:
    """Append to the process-wide log. See AuditLog.record."""
    return get_audit_log().record(event, principal, **fields)


def session_id() -> str:
    """Streamlit's id for the current browser session, or "" outside one."""
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import (
            get_script_run_ctx,
        )

        ctx = get_script_run_ctx(suppress_warning=True)
        return str(ctx.session_id) if ctx else ""
    except Exception:
        return ""


__all__ = [
    "AuditLog", "get_audit_log", "record", "redact", "session_id",
    "EVENT_TYPES", "OUTCOMES", "GENESIS", "REDACTED",
    "AUTH_SESSION", "AUTH_LOGIN", "AUTH_LOGOUT", "ACCESS_DENIED", "COMMAND",
    "PAGE_VIEW", "PAGE_ERROR", "PORTFOLIO_SAVE", "WATCHLIST_ADD",
    "BRIEF_REBUILD", "CACHE_REFRESH", "CACHE_PURGE", "AUDIT_EXPORT",
]
