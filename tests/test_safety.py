"""
Link sanitisation and per-user portfolio isolation.

REGRESSION CONTEXT
------------------
  * S1 / I22. News, filing and social links went into href attributes with
    html.escape alone. Escaping does nothing to `javascript:alert(1)`.

  * B12. Headlines were truncated after escaping, so the cut could land
    inside an entity and print "AT&am" for "AT&T".

  * I23. Every session read and wrote one shared portfolio.json.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

import config
from data_fetchers import portfolio
from ui import components as ui
from utils import entitlements, identity


# ==========================================================================
# Links
# ==========================================================================
class TestSafeUrl:
    @pytest.mark.parametrize("url", [
        "https://www.sec.gov/Archives/edgar/data/320193/x.htm",
        "http://example.com/a?b=1&c=2",
        "HTTPS://EXAMPLE.COM/",
    ])
    def test_http_links_pass(self, url):
        assert ui.safe_url(url) == url

    @pytest.mark.parametrize("url", [
        "javascript:alert(1)",
        "JaVaScRiPt:alert(1)",
        "  javascript:alert(1)",
        "java\tscript:alert(1)",       # browsers drop the tab
        "java\nscript:alert(1)",
        "\x01javascript:alert(1)",
        "data:text/html;base64,PHNjcmlwdD4=",
        "vbscript:msgbox(1)",
        "file:///etc/passwd",
        "//evil.example/x",
        "https:evil.example",
        "/relative/path",
        "",
        None,
        float("nan"),
        42,
    ])
    def test_everything_else_rejected(self, url):
        assert ui.safe_url(url) == ""


class TestRenderedLinks:
    @staticmethod
    def _news_html(monkeypatch, title, link):
        captured: list = []
        monkeypatch.setattr(ui.st, "markdown",
                            lambda body, **_kwargs: captured.append(body))
        ui.news_feed(pd.DataFrame([{
            "title": title, "link": link, "source": "Feed",
            "published": pd.Timestamp.now(tz="UTC"),
            "sentiment": 0.1, "label": "NEUTRAL",
        }]))
        return "".join(captured)

    def test_javascript_news_link_not_rendered(self, monkeypatch):
        out = self._news_html(monkeypatch, "Click me", "javascript:alert(1)")
        assert "javascript" not in out.lower()
        assert "<a " not in out
        assert "Click me" in out, "headline dropped along with its link"

    def test_https_news_link_rendered(self, monkeypatch):
        out = self._news_html(monkeypatch, "Story", "https://example.com/s")
        assert 'href="https://example.com/s"' in out

    def test_truncation_never_splits_an_entity(self, monkeypatch):
        out = self._news_html(monkeypatch, "x" * 198 + "&T", "")
        assert "&amp;T" in out
        assert not re.search(r"&[a-z]{0,3}<", out)

    def test_social_links_drop_unsafe_urls(self):
        out = ui.social_links({
            "twitter": {"handle": "@x", "url": "javascript:alert(1)"},
            "website": {"handle": "", "url": "https://example.com"},
        })
        assert "javascript" not in out
        assert 'href="https://example.com"' in out


# ==========================================================================
# Portfolio isolation
# ==========================================================================
ALICE = identity.user_key("alice@example.com")
BOB = identity.user_key("bob@example.com")


def _book(*tickers: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": list(tickers),
                         "quantity": [1.0] * len(tickers),
                         "cost_basis": [None] * len(tickers),
                         "note": [""] * len(tickers)})


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_FILE", tmp_path / "portfolio.json")
    monkeypatch.setattr(config, "BRIEF_DIR", tmp_path / "briefs")
    monkeypatch.setattr(config, "USERS_DIR", tmp_path / "users")
    return tmp_path


class TestPortfolioIsolation:
    def test_users_do_not_see_each_other(self, stores):
        portfolio.save(_book("AAPL"), pd.DataFrame(), ALICE)
        portfolio.save(_book("TSLA"), pd.DataFrame(), BOB)

        assert list(portfolio.holdings(ALICE)["ticker"]) == ["AAPL"]
        assert list(portfolio.holdings(BOB)["ticker"]) == ["TSLA"]
        assert portfolio.holdings().empty

    def test_local_user_keeps_original_file(self, stores):
        """An existing single-user book must need no migration."""
        portfolio.save(_book("VOO"), pd.DataFrame())
        assert (stores / "portfolio.json").exists()
        assert not (stores / "users").exists()
        assert list(portfolio.holdings()["ticker"]) == ["VOO"]

    def test_watch_writes_only_own_list(self, stores):
        assert portfolio.add_to_watchlist("NVDA", user=ALICE)
        assert list(portfolio.watchlist(ALICE)["ticker"]) == ["NVDA"]
        assert portfolio.watchlist().empty
        assert portfolio.watchlist(BOB).empty

    def test_brief_editions_are_per_user(self, stores):
        edition = portfolio.edition_date()
        path = portfolio.brief_dir(ALICE) / f"{edition.isoformat()}.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"empty": true}', encoding="utf-8")

        assert portfolio.stored_editions(user=ALICE) == [edition]
        assert portfolio.stored_editions() == []
        assert portfolio.load_edition(edition, user=BOB) is None

    @pytest.mark.parametrize("key", ["../local", "..", "a/b", "a\\b", "",
                                     "x" * 65, None])
    def test_path_escaping_keys_rejected(self, stores, key):
        with pytest.raises(ValueError):
            portfolio.portfolio_path(key)

    def test_watch_command_refused_without_login(self, stores, monkeypatch):
        import app

        monkeypatch.setattr(config, "MULTI_USER", True)
        state: dict = {}
        app.route_subject(state, "portfolio", "NVDA")
        assert "Sign in" in state["command_error"]
        assert portfolio.watchlist().empty


class TestIdentity:
    def test_no_runtime_is_local(self):
        assert identity.current_user() == identity.LOCAL

    def test_key_is_a_hash_not_the_identifier(self):
        key = identity.user_key(" Alice@Example.com ")
        assert key == ALICE
        assert re.fullmatch(r"[0-9a-f]{32}", key)
        assert "alice" not in key

    def test_multi_user_without_login_has_no_owner(self, monkeypatch):
        monkeypatch.setattr(config, "MULTI_USER", True)
        assert entitlements.portfolio_owner() is None

    def test_single_user_owner_is_local(self, monkeypatch):
        monkeypatch.setattr(config, "MULTI_USER", False)
        assert entitlements.portfolio_owner() == identity.LOCAL
