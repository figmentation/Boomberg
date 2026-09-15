"""
Audit trail (F2).

The contract the rest of the terminal leans on:
  * rows cannot be changed or removed through SQLite,
  * a change made to the file anyway is detectable, and where,
  * secrets never reach a row,
  * a privileged action does not happen if its record cannot be written.
"""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

import config
from utils import audit, entitlements, identity
from utils.identity import User

ALICE = User(identity.user_key("alice@example.com"), "alice@example.com", True)


def principal(role: str) -> entitlements.Principal:
    return entitlements.Principal(ALICE, role, config.ROLE_PERMISSIONS[role])


# ==========================================================================
# Recording
# ==========================================================================
class TestRecording:
    def test_row_round_trip(self, isolated_audit_log):
        assert isolated_audit_log.record(
            audit.COMMAND, principal("analyst"), module="equity",
            subject="AAPL EQUITY", session="s-1")
        row = isolated_audit_log.recent(limit=1)[0]
        assert row["event"] == "command"
        assert row["outcome"] == "ok"
        assert row["actor"] == "alice@example.com"
        assert row["actor_key"] == ALICE.key
        assert row["role"] == "analyst"
        assert row["module"] == "equity"
        assert row["subject"] == "AAPL EQUITY"
        assert row["session"] == "s-1"

    def test_system_actor_without_principal(self, isolated_audit_log):
        isolated_audit_log.record(audit.CACHE_PURGE)
        assert isolated_audit_log.recent(limit=1)[0]["actor"] == "SYSTEM"

    def test_filters(self, isolated_audit_log):
        admin = principal("admin")
        isolated_audit_log.record(audit.COMMAND, admin, subject="A")
        isolated_audit_log.record(audit.CACHE_PURGE, admin)
        isolated_audit_log.record(audit.COMMAND, admin, outcome="denied")

        assert [r["event"] for r in
                isolated_audit_log.recent(events=[audit.CACHE_PURGE])] == ["cache.purge"]
        assert len(isolated_audit_log.recent(outcome="denied")) == 1
        assert len(isolated_audit_log.recent(actor="alice")) == 3
        assert isolated_audit_log.recent(actor="nobody") == []

    def test_newest_first(self, isolated_audit_log):
        for n in range(3):
            isolated_audit_log.record(audit.COMMAND, subject=f"CMD{n}")
        assert [r["subject"] for r in isolated_audit_log.recent()] == [
            "CMD2", "CMD1", "CMD0"]

    def test_unwritable_log_reports_failure(self, tmp_path):
        broken = audit.AuditLog(path=str(tmp_path))   # a directory, not a file
        assert broken.record(audit.CACHE_PURGE, principal("admin")) is False


# ==========================================================================
# Integrity
# ==========================================================================
class TestIntegrity:
    def test_update_and_delete_refused(self, isolated_audit_log):
        isolated_audit_log.record(audit.COMMAND, principal("viewer"), subject="AAPL")
        conn = sqlite3.connect(isolated_audit_log.path)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute("UPDATE audit SET actor = 'mallory'")
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute("DELETE FROM audit")
        finally:
            conn.close()
        assert isolated_audit_log.recent(limit=1)[0]["actor"] == "alice@example.com"

    def test_chain_verifies(self, isolated_audit_log):
        for n in range(5):
            isolated_audit_log.record(audit.COMMAND, subject=f"CMD{n}")
        assert isolated_audit_log.verify() == (True, None, 5)

    def test_empty_log_verifies(self, isolated_audit_log):
        assert isolated_audit_log.verify() == (True, None, 0)
        assert isolated_audit_log.head_hash() is None

    def test_direct_edit_detected_at_the_row(self, isolated_audit_log):
        for n in range(5):
            isolated_audit_log.record(audit.COMMAND, subject=f"CMD{n}")
        conn = sqlite3.connect(isolated_audit_log.path)
        conn.execute("DROP TRIGGER audit_no_update")
        conn.execute("UPDATE audit SET subject = 'TAMPERED' WHERE id = 3")
        conn.commit()
        conn.close()
        assert isolated_audit_log.verify() == (False, 3, 3)

    def test_removed_row_detected(self, isolated_audit_log):
        for n in range(5):
            isolated_audit_log.record(audit.COMMAND, subject=f"CMD{n}")
        conn = sqlite3.connect(isolated_audit_log.path)
        conn.execute("DROP TRIGGER audit_no_delete")
        conn.execute("DELETE FROM audit WHERE id = 2")
        conn.commit()
        conn.close()
        intact, first_bad, _ = isolated_audit_log.verify()
        assert not intact and first_bad == 3

    def test_concurrent_writers_keep_one_chain(self, isolated_audit_log):
        """Two log objects on one file stand in for two server processes."""
        second = audit.AuditLog(path=isolated_audit_log.path)
        logs = [isolated_audit_log, second]

        def write(worker: int) -> None:
            for n in range(25):
                logs[worker % 2].record(audit.COMMAND, subject=f"W{worker}-{n}")

        threads = [threading.Thread(target=write, args=(w,)) for w in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert isolated_audit_log.verify() == (True, None, 200)


# ==========================================================================
# Redaction
# ==========================================================================
class TestRedaction:
    def test_configured_secret_removed(self, monkeypatch):
        monkeypatch.setattr(config, "FRED_API_KEY", "fredkey12345")
        assert "fredkey12345" not in audit.redact("series with fredkey12345 set")

    @pytest.mark.parametrize("text,leaked", [
        ("api_key=sk_live_123", "sk_live_123"),
        ("password: hunter2", "hunter2"),
        ("token=abc.def.ghi", "abc.def.ghi"),
        ("pasted " + "Ab3x" * 10, "Ab3x" * 10),
    ])
    def test_credential_shapes_removed(self, text, leaked):
        cleaned = audit.redact(text)
        assert leaked not in cleaned
        assert audit.REDACTED in cleaned

    @pytest.mark.parametrize("text", [
        "BRK/B US EQUITY", "EURUSD CURNCY", "ENERGY / COMMODITY", "636019825",
        "USGG10YR INDEX", "600519 CH",
    ])
    def test_market_subjects_untouched(self, text):
        assert audit.redact(text) == text

    def test_detail_values_redacted_in_stored_row(self, isolated_audit_log,
                                                  monkeypatch):
        monkeypatch.setattr(config, "AISSTREAM_API_KEY", "aisstreamsecret9")
        isolated_audit_log.record(
            audit.PAGE_ERROR, detail={"message": "auth failed aisstreamsecret9",
                                      "render_ms": 12})
        row = isolated_audit_log.recent(limit=1)[0]
        detail = json.loads(row["detail"])
        assert "aisstreamsecret9" not in row["detail"]
        assert detail["render_ms"] == 12


# ==========================================================================
# Fail-closed privileged actions
# ==========================================================================
@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_FILE", tmp_path / "portfolio.json")
    monkeypatch.setattr(config, "BRIEF_DIR", tmp_path / "briefs")
    monkeypatch.setattr(config, "USERS_DIR", tmp_path / "users")
    return tmp_path


class TestFailClosed:
    def test_watch_recorded_then_applied(self, stores, isolated_audit_log,
                                         monkeypatch):
        import app
        from data_fetchers import portfolio

        monkeypatch.setattr(entitlements, "current_principal",
                            lambda: principal("analyst"))
        state: dict = {}
        app.route_subject(state, "portfolio", "nvda")

        assert list(portfolio.watchlist(ALICE.key)["ticker"]) == ["NVDA"]
        row = isolated_audit_log.recent(limit=1)[0]
        assert (row["event"], row["subject"]) == ("portfolio.watch", "NVDA")

    def test_watch_not_applied_when_log_unwritable(self, stores, tmp_path,
                                                   monkeypatch):
        import app
        from data_fetchers import portfolio

        monkeypatch.setattr(entitlements, "current_principal",
                            lambda: principal("analyst"))
        monkeypatch.setattr(audit, "_log_singleton",
                            audit.AuditLog(path=str(tmp_path)))
        state: dict = {}
        app.route_subject(state, "portfolio", "NVDA")

        assert "Audit log unavailable" in state["command_error"]
        assert portfolio.watchlist(ALICE.key).empty

    def test_denied_action_recorded(self, isolated_audit_log, monkeypatch):
        import app

        monkeypatch.setattr(app.ui, "alert", lambda *_a, **_k: None)
        monkeypatch.setattr(entitlements, "current_principal",
                            lambda: principal("analyst"))
        assert app._authorized("cache.purge", audit.CACHE_PURGE) is False
        row = isolated_audit_log.recent(limit=1)[0]
        assert (row["event"], row["outcome"]) == ("cache.purge", "denied")

    def test_permitted_action_recorded_first(self, isolated_audit_log,
                                             monkeypatch):
        import app

        monkeypatch.setattr(app.ui, "alert", lambda *_a, **_k: None)
        monkeypatch.setattr(entitlements, "current_principal",
                            lambda: principal("admin"))
        assert app._authorized("cache.purge", audit.CACHE_PURGE) is True
        row = isolated_audit_log.recent(limit=1)[0]
        assert (row["event"], row["outcome"], row["role"]) == (
            "cache.purge", "ok", "admin")

    def test_permitted_action_refused_when_log_unwritable(self, tmp_path,
                                                          monkeypatch):
        import app

        monkeypatch.setattr(app.ui, "alert", lambda *_a, **_k: None)
        monkeypatch.setattr(entitlements, "current_principal",
                            lambda: principal("admin"))
        monkeypatch.setattr(audit, "_log_singleton",
                            audit.AuditLog(path=str(tmp_path)))
        assert app._authorized("cache.purge", audit.CACHE_PURGE) is False
