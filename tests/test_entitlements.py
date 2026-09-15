"""
Authentication and role entitlements (F1).

The failure that matters is the quiet one: an account resolving to a role it
was never given. So most of these check that every doubtful case - anonymous,
unlisted, misspelt role, unverified email, missing or broken assignment file -
resolves to no access rather than to a plausible guess.
"""

from __future__ import annotations

import os
import textwrap

import pytest

import config
from utils import entitlements, identity
from utils.identity import User

ALICE = User(key=identity.user_key("alice@example.com"),
             display="alice@example.com", authenticated=True)
BOB = User(key=identity.user_key("bob@corp.example"),
           display="bob@corp.example", authenticated=True)


def principal(role: str, user: User = ALICE) -> entitlements.Principal:
    return entitlements.Principal(user, role, config.ROLE_PERMISSIONS[role])


@pytest.fixture
def assign(tmp_path, monkeypatch):
    """Multi-user mode with an assignment file the test writes."""
    path = tmp_path / "entitlements.toml"
    monkeypatch.setattr(config, "MULTI_USER", True)
    monkeypatch.setattr(config, "ENTITLEMENTS_FILE", path)
    version = {"n": 0}

    def write(text: str) -> None:
        path.write_text(textwrap.dedent(text), encoding="utf-8")
        # Filesystems stamp mtimes coarsely; move it on explicitly so the parse
        # memo sees each rewrite the way it would see real edits minutes apart.
        version["n"] += 1
        stamp = 1_700_000_000_000_000_000 + version["n"] * 1_000_000_000
        os.utime(path, ns=(stamp, stamp))

    return write


# ==========================================================================
# Resolution
# ==========================================================================
class TestResolution:
    def test_single_user_install_is_admin(self, monkeypatch):
        monkeypatch.setattr(config, "MULTI_USER", False)
        resolved = entitlements.resolve(identity.LOCAL)
        assert resolved.role == "admin"
        assert resolved.can("audit.read") and resolved.can("cache.purge")

    def test_anonymous_has_no_access(self, assign):
        assign('default_role = "viewer"')
        assert not entitlements.resolve(identity.LOCAL).has_access

    def test_listed_user(self, assign):
        assign("""
            [users]
            "alice@example.com" = "admin"
        """)
        assert entitlements.resolve(ALICE).role == "admin"

    def test_domain_grant(self, assign):
        assign("""
            [domains]
            "corp.example" = "analyst"
        """)
        assert entitlements.resolve(BOB).role == "analyst"
        assert not entitlements.resolve(ALICE).has_access

    def test_accounts_match_case_insensitively(self, assign):
        assign("""
            [users]
            "Alice@Example.com" = "Analyst"
        """)
        shouting = User(ALICE.key, "ALICE@example.COM", True)
        assert entitlements.resolve(shouting).role == "analyst"

    def test_user_entry_none_revokes_domain_grant(self, assign):
        assign("""
            [users]
            "alice@example.com" = "none"
            [domains]
            "example.com" = "admin"
        """)
        assert not entitlements.resolve(ALICE).has_access

    def test_user_entry_beats_domain_grant(self, assign):
        assign("""
            [users]
            "alice@example.com" = "viewer"
            [domains]
            "example.com" = "admin"
        """)
        assert entitlements.resolve(ALICE).role == "viewer"

    def test_default_role_for_unlisted(self, assign):
        assign('default_role = "viewer"')
        assert entitlements.resolve(ALICE).role == "viewer"

    def test_default_is_none_when_unset(self, assign):
        assign("[users]")
        assert not entitlements.resolve(ALICE).has_access

    def test_unknown_role_is_no_access(self, assign):
        assign("""
            [users]
            "alice@example.com" = "superuser"
        """)
        assert not entitlements.resolve(ALICE).has_access

    def test_missing_file_is_no_access(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MULTI_USER", True)
        monkeypatch.setattr(config, "ENTITLEMENTS_FILE", tmp_path / "absent.toml")
        assert not entitlements.resolve(ALICE).has_access

    def test_malformed_file_is_no_access(self, assign):
        """Including an account the broken file might have granted."""
        assign('default_role = "admin"\n[users\n"alice@example.com" = ')
        assert not entitlements.resolve(ALICE).has_access

    def test_edit_applies_without_restart(self, assign):
        assign('[users]\n"alice@example.com" = "admin"')
        assert entitlements.resolve(ALICE).role == "admin"
        assign('[users]\n"alice@example.com" = "viewer"')
        assert entitlements.resolve(ALICE).role == "viewer"


class TestIdentityClaims:
    @staticmethod
    def _login(monkeypatch, **claims):
        import streamlit

        class FakeUser(dict):
            is_logged_in = True

        monkeypatch.setattr(streamlit, "user", FakeUser(claims))

    def test_verified_email_identifies(self, monkeypatch):
        self._login(monkeypatch, email="alice@example.com",
                    email_verified=True, sub="1001")
        user = identity.current_user()
        assert user.authenticated and user.display == "alice@example.com"

    def test_provider_without_verified_claim_trusted(self, monkeypatch):
        self._login(monkeypatch, email="alice@example.com", sub="1001")
        assert identity.current_user().display == "alice@example.com"

    def test_unverified_email_cannot_match_a_grant(self, monkeypatch, assign):
        assign("""
            [users]
            "alice@example.com" = "admin"
            [domains]
            "example.com" = "admin"
        """)
        self._login(monkeypatch, email="alice@example.com",
                    email_verified=False, sub="attacker-7")
        user = identity.current_user()
        assert user.display == "attacker-7"
        assert not entitlements.resolve(user).has_access


# ==========================================================================
# Role table
# ==========================================================================
class TestRoleTable:
    def test_roles_are_nested(self):
        roles = config.ROLE_PERMISSIONS
        assert roles["viewer"] < roles["analyst"] < roles["admin"]

    def test_every_role_can_land_on_home(self):
        """main() sends a refused page to home; a role without it would loop."""
        for role, granted in config.ROLE_PERMISSIONS.items():
            assert "module:home" in granted, role

    def test_module_permissions_name_real_pages(self):
        import app

        for granted in config.ROLE_PERMISSIONS.values():
            for permission in granted:
                if permission.startswith("module:"):
                    assert permission[len("module:"):] in app.ROUTES, permission

    def test_admin_reaches_every_page(self):
        import app

        admin = config.ROLE_PERMISSIONS["admin"]
        for module in app.ROUTES:
            assert f"module:{module}" in admin, module

    def test_viewer_changes_nothing_shared(self):
        viewer = principal("viewer")
        for permission in ("portfolio.write", "cache.refresh", "cache.purge",
                           "audit.read"):
            assert not viewer.can(permission), permission


# ==========================================================================
# Enforcement in the command path
# ==========================================================================
class TestCommandAuthorization:
    @pytest.mark.parametrize("raw,role,allowed", [
        ("AAPL EQUITY", "viewer", True),
        ("SPX INDEX", "viewer", True),
        ("PF", "viewer", False),
        ("NVDA WATCH", "viewer", False),
        ("AUDIT", "viewer", False),
        ("PF", "analyst", True),
        ("NVDA WATCH", "analyst", True),
        ("AUDIT", "analyst", False),
        ("AUDIT", "admin", True),
    ])
    def test_matrix(self, raw, role, allowed):
        import app

        denial = app.authorize_command(principal(role), app.parse_command(raw))
        assert (denial is None) is allowed, denial

    def test_watch_route_refuses_viewer(self, tmp_path, monkeypatch):
        """route_subject checks again, for callers that skip authorize_command."""
        import app
        from data_fetchers import portfolio

        monkeypatch.setattr(config, "PORTFOLIO_FILE", tmp_path / "portfolio.json")
        monkeypatch.setattr(config, "USERS_DIR", tmp_path / "users")
        monkeypatch.setattr(entitlements, "current_principal",
                            lambda: principal("viewer"))

        state: dict = {}
        app.route_subject(state, "portfolio", "NVDA")
        assert "cannot edit" in state["command_error"]
        assert portfolio.watchlist(ALICE.key).empty
