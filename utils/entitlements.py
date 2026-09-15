"""
utils/entitlements.py :: Who may do what on this terminal.

Three layers, each replaceable without touching the others:

  identity     who the session is             utils.identity (Streamlit OIDC)
  assignment   which role an account holds    config.ENTITLEMENTS_FILE (TOML)
  role table   what each role may do          config.ROLE_PERMISSIONS

The role table is reviewed with the code; the assignment file is deployment
data an administrator edits without a release. Both are short plain text, so
an access review reads two files rather than the application.

A terminal on your own machine (MULTI_USER off) has exactly one user, and that
user is admin, so a single-user install behaves as it always has. With
MULTI_USER on, every doubtful case fails closed: an anonymous session, an
unlisted account under default_role = "none", a misspelt role name, and a
missing or unparseable assignment file all resolve to no access - never to a
guessed role.

Assignment file:

    default_role = "none"            # signed-in accounts not listed below

    [users]                          # exact account; beats a domain grant
    "alice@example.com" = "admin"
    "mallory@example.com" = "none"   # revoke one person inside a domain

    [domains]                        # every verified address in the domain
    "example.com" = "viewer"
"""

from __future__ import annotations

import logging
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

import config
from utils import identity
from utils.identity import User

log = logging.getLogger("openterm.entitlements")

NO_ROLE = "none"


@dataclass(frozen=True)
class Principal:
    """A user, the role they resolved to, and that role's permissions."""

    user: User
    role: Optional[str]
    permissions: FrozenSet[str] = field(default_factory=frozenset)

    @property
    def has_access(self) -> bool:
        return self.role is not None

    @property
    def role_label(self) -> str:
        return (self.role or NO_ROLE).upper()

    def can(self, permission: str) -> bool:
        return permission in self.permissions


@dataclass(frozen=True)
class Assignments:
    users: Dict[str, str] = field(default_factory=dict)
    domains: Dict[str, str] = field(default_factory=dict)
    default_role: str = NO_ROLE
    error: Optional[str] = None


# --------------------------------------------------------------------------
# Assignment file
# --------------------------------------------------------------------------
# Parsed once per file version: Streamlit reruns the script on every click,
# and an administrator's edit should still apply without a restart.
_memo: Dict[str, Tuple[int, Assignments]] = {}
_memo_lock = threading.Lock()


def load_assignments(path: Optional[Path] = None) -> Assignments:
    path = Path(path or config.ENTITLEMENTS_FILE)
    try:
        stamp = path.stat().st_mtime_ns
    except FileNotFoundError:
        return Assignments(error=f"{path.name} not found")
    except OSError as exc:
        return Assignments(error=f"{path.name} unreadable ({type(exc).__name__})")

    key = str(path.resolve())
    with _memo_lock:
        memo = _memo.get(key)
        if memo and memo[0] == stamp:
            return memo[1]

    assignments = _parse(path)
    with _memo_lock:
        _memo[key] = (stamp, assignments)
    return assignments


def _parse(path: Path) -> Assignments:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        # The exception type only: a parse message can quote file content,
        # and the file is a list of accounts.
        log.error("Entitlements file %s could not be parsed (%s); refusing "
                  "every account until it is fixed", path.name,
                  type(exc).__name__)
        return Assignments(error=f"{path.name} could not be parsed")

    def table(name: str) -> Dict[str, str]:
        section = raw.get(name)
        if not isinstance(section, dict):
            return {}
        return {str(k).strip().lower(): str(v).strip().lower()
                for k, v in section.items()}

    return Assignments(
        users=table("users"),
        domains=table("domains"),
        default_role=str(raw.get("default_role", NO_ROLE)).strip().lower(),
    )


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------
def role_for(user: User, assignments: Assignments) -> Optional[str]:
    """The role `user` holds, or None for no access."""
    if not config.MULTI_USER:
        return "admin"
    if not user.authenticated or assignments.error:
        return None

    account = user.display.strip().lower()
    domain = account.rpartition("@")[2] if "@" in account else ""

    role = assignments.users.get(account)
    if role is None and domain:
        role = assignments.domains.get(domain)
    if role is None:
        role = assignments.default_role

    if role == NO_ROLE:
        return None
    if role not in config.ROLE_PERMISSIONS:
        # Not logged with the account, and not guessed at: a typo in the file
        # must lock the account out, not grant whatever it resembled.
        log.error("Entitlements name an unknown role %r; the account gets no "
                  "access", role)
        return None
    return role


def resolve(user: User, assignments: Optional[Assignments] = None) -> Principal:
    if assignments is None:
        assignments = load_assignments() if config.MULTI_USER else Assignments()
    role = role_for(user, assignments)
    permissions = config.ROLE_PERMISSIONS.get(role, frozenset()) if role else frozenset()
    return Principal(user=user, role=role, permissions=frozenset(permissions))


def current_principal() -> Principal:
    """The principal for this Streamlit session, resolved fresh each call."""
    return resolve(identity.current_user())


def portfolio_owner() -> Optional[User]:
    """The user whose book this session may open, or None."""
    principal = current_principal()
    return principal.user if principal.can("module:portfolio") else None


__all__ = ["NO_ROLE", "Principal", "Assignments", "load_assignments",
           "role_for", "resolve", "current_principal", "portfolio_owner"]
