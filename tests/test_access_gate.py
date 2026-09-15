"""
The access gate, end to end through Streamlit's own test runner.

Unit tests prove the resolution rules; this proves main() applies them before
anything else renders. In multi-user mode an anonymous session must see the
gate and nothing behind it - no tape, no sidebar, no page - and the refusal
must be on record.
"""

from __future__ import annotations

from pathlib import Path

import config

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_anonymous_session_sees_only_the_gate(monkeypatch, isolated_audit_log):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(config, "MULTI_USER", True)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()

    assert not at.exception, at.exception
    rendered = " ".join(str(block.value) for block in at.markdown)
    assert "ACCESS CONTROLLED" in rendered
    # No [auth] section exists in the test environment, so the gate says why
    # nobody can get in rather than offering a sign-in that cannot work.
    assert "no sign-in provider" in rendered
    # The class name itself appears in the theme's stylesheet; the element
    # is what must be absent.
    assert 'class="ot-tape"' not in rendered, "market tape rendered behind the gate"
    assert len(at.sidebar) == 0, "sidebar rendered behind the gate"

    rows = isolated_audit_log.recent()
    assert [(r["event"], r["outcome"]) for r in rows] == [("auth.session", "denied")]
