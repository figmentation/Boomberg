"""
Shared pytest fixtures and path setup.

Open-Terminal is run as a script (`streamlit run app.py`), not installed as a
package, so the project root has to be on sys.path for `import config` and
friends to resolve the same way they do at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================================
# Cache isolation
# ==========================================================================
@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    """
    Redirect the cache singleton at a throwaway SQLite file.

    Note: `@cached` resolves get_cache() at *decoration* time, so anything
    decorated at import is already bound to the real cache. Tests that need
    this fixture must define their decorated function inside the test body,
    after the fixture has swapped the singleton.
    """
    from utils import cache as cache_mod

    temp = cache_mod.SQLiteCache(path=str(tmp_path / "test_cache.sqlite"))
    monkeypatch.setattr(cache_mod, "_cache_singleton", temp)
    return temp


@pytest.fixture(autouse=True)
def reset_circuits():
    """
    Circuit breakers live in a module-level registry that survives between
    tests. Without this, a breaker tripped in one test fails the next.
    """
    from utils.rate_limiter import reset_all_circuits

    reset_all_circuits()
    yield
    reset_all_circuits()


@pytest.fixture(autouse=True)
def isolated_audit_log(tmp_path, monkeypatch):
    """
    Audit writes made by code under test go to a throwaway file.

    The log is append-only by design, so a test that wrote to the real
    .openterm/audit.sqlite would leave rows there that nothing can remove.
    """
    from utils import audit

    log = audit.AuditLog(path=str(tmp_path / "audit.sqlite"))
    monkeypatch.setattr(audit, "_log_singleton", log)
    return log


@pytest.fixture(autouse=True)
def single_user_mode(monkeypatch):
    """
    Tests run as the single-user install unless they opt in to multi-user.
    Without this, OPENTERM_MULTI_USER=1 in a developer's .env would flip
    every entitlement and portfolio test.
    """
    import config

    monkeypatch.setattr(config, "MULTI_USER", False)


# ==========================================================================
# Reference data
# ==========================================================================
@pytest.fixture
def wilder_prices() -> pd.Series:
    """
    The worked example from Wilder's "New Concepts in Technical Trading
    Systems" (1978), the canonical RSI test vector.

    Published RSI values: 70.46 at index 14, 66.25 at index 15.
    """
    return pd.Series([
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
        46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41,
        46.22, 45.64, 46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03,
        44.18, 44.22, 44.57, 43.42, 42.66, 43.13,
    ])


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """
    Deterministic synthetic OHLCV frame with a fixed seed.

    The component Series must be built on the SAME DatetimeIndex the frame
    uses. Constructing them on a default RangeIndex and then passing
    `index=date_range(...)` to the DataFrame makes pandas *reindex* rather
    than relabel, silently producing an all-NaN frame - which quietly turns
    every indicator assertion into a vacuous pass.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    n = 260
    index = pd.date_range("2024-01-01", periods=n, freq="B")

    close = pd.Series(100 + np.cumsum(rng.normal(0, 1.2, n)), index=index)
    frame = pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close + rng.uniform(0.2, 2.0, n),
        "Low": close - rng.uniform(0.2, 2.0, n),
        "Close": close,
        "Volume": rng.integers(1e6, 5e7, n),
    }, index=index)

    # Guard the fixture itself - a silent NaN frame is worse than no fixture.
    assert frame.notna().all().all(), "ohlcv fixture is not fully populated"
    return frame
