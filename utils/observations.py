"""
utils/observations.py :: Append-only local time series.

The TTL cache in `utils/cache.py` answers "what is the value now?". This
answers "what is normal for this thing?" - which needs history the cache
deliberately throws away.

It exists because several readings are only meaningful relative to their own
past. A chokepoint holding 140 vessels is congested or quiet depending
entirely on what that corridor usually holds, and nobody publishes a free
authoritative "normal" for the Strait of Malacca. The honest alternative to
inventing one is to measure it here, and to say so when there is not yet
enough history to measure.

Design notes:
  * Same SQLite file as the object cache, separate table. WAL, one connection
    per thread, writes best-effort - identical constraints to `cache.py`.
  * `record()` self-throttles. The maritime page can be re-rendered a dozen
    times in a minute; without a floor between samples an afternoon of
    fiddling would dominate the baseline.
  * Baselines use the median, not the mean. A single AIS outage recording
    zero vessels should not drag the reference value down.
  * Nothing here ever extrapolates. Too few samples returns None, and the
    caller is expected to say "still measuring" rather than guess.
"""

from __future__ import annotations

import logging
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import config

log = logging.getLogger("openterm.observations")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    series      TEXT NOT NULL,
    observed_at REAL NOT NULL,
    value       REAL NOT NULL,
    PRIMARY KEY (series, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_obs_series ON observations(series, observed_at);
"""

# A series needs this many samples before a baseline is offered at all.
MIN_SAMPLES = 12

# Samples older than this stop counting: a corridor's normal shifts with
# season, war and canal draught limits, and a two-year-old reading is not
# evidence about today.
MAX_AGE_DAYS = 120.0

# Minimum gap between two recorded samples for one series.
MIN_INTERVAL_SECONDS = 1800.0

# Samples older than this are deleted outright.
RETENTION_DAYS = 400.0


@dataclass(frozen=True)
class Baseline:
    """A measured reference value, with the evidence behind it."""

    series: str
    median: float
    samples: int
    low: float           # 25th percentile
    high: float          # 75th percentile
    first_seen: float    # unix seconds
    last_seen: float

    @property
    def span_days(self) -> float:
        return max(0.0, (self.last_seen - self.first_seen) / 86400.0)

    def provenance(self) -> str:
        """One line the UI can print under a gauge to justify the number."""
        return (
            f"{self.samples} observations over "
            f"{self.span_days:.1f} days (IQR {self.low:.0f}-{self.high:.0f})"
        )


class ObservationStore:
    """Thread-safe append-only numeric series on SQLite."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = str(path or config.CACHE_DB)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        with self._init_lock:
            try:
                self._conn().executescript(_SCHEMA)
                self._conn().commit()
            except Exception as exc:  # pragma: no cover
                log.error("Observation schema init failed: %s", exc)

    # -- write -------------------------------------------------------------
    def record(
        self,
        series: str,
        value: float,
        at: Optional[float] = None,
        min_interval: float = MIN_INTERVAL_SECONDS,
    ) -> bool:
        """
        Append one sample. Returns True if it was stored.

        Skipped silently when the previous sample for this series is newer
        than `min_interval`, so repeated page renders do not stuff the
        history with correlated readings taken seconds apart.
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False

        now = float(at if at is not None else time.time())

        try:
            # Any neighbour inside the window blocks the write, not just the
            # newest one - otherwise backfilling an older sample would be
            # judged against a future reading and silently dropped.
            if min_interval > 0:
                row = self._conn().execute(
                    "SELECT 1 FROM observations WHERE series = ? "
                    "AND observed_at > ? AND observed_at < ? LIMIT 1",
                    (series, now - min_interval, now + min_interval),
                ).fetchone()
                if row is not None:
                    return False

            self._conn().execute(
                "INSERT OR REPLACE INTO observations (series, observed_at, value) "
                "VALUES (?, ?, ?)",
                (series, now, value),
            )
            self._conn().execute(
                "DELETE FROM observations WHERE series = ? AND observed_at < ?",
                (series, now - RETENTION_DAYS * 86400.0),
            )
            self._conn().commit()
            return True
        except Exception as exc:
            log.debug("Observation write failed for %s: %s", series, exc)
            return False

    # -- read --------------------------------------------------------------
    def history(
        self, series: str, max_age_days: float = MAX_AGE_DAYS
    ) -> List[Tuple[float, float]]:
        """[(observed_at, value)] oldest first, within the age window."""
        cutoff = time.time() - max_age_days * 86400.0
        try:
            rows = self._conn().execute(
                "SELECT observed_at, value FROM observations "
                "WHERE series = ? AND observed_at >= ? ORDER BY observed_at",
                (series, cutoff),
            ).fetchall()
        except Exception as exc:
            log.debug("Observation read failed for %s: %s", series, exc)
            return []
        return [(float(t), float(v)) for t, v in rows]

    def sample_count(self, series: str,
                     max_age_days: float = MAX_AGE_DAYS) -> int:
        return len(self.history(series, max_age_days=max_age_days))

    def baseline(
        self,
        series: str,
        min_samples: int = MIN_SAMPLES,
        max_age_days: float = MAX_AGE_DAYS,
    ) -> Optional[Baseline]:
        """
        The measured normal for `series`, or None if it is not yet known.

        None is a real answer and callers must handle it. Returning a guess
        here would put a fabricated number behind a status label.
        """
        points = self.history(series, max_age_days=max_age_days)
        if len(points) < min_samples:
            return None

        values = sorted(v for _, v in points)
        quartiles = statistics.quantiles(values, n=4, method="inclusive")

        return Baseline(
            series=series,
            median=float(statistics.median(values)),
            samples=len(values),
            low=float(quartiles[0]),
            high=float(quartiles[2]),
            first_seen=points[0][0],
            last_seen=points[-1][0],
        )

    def clear(self, series: Optional[str] = None) -> int:
        """Drop one series, or everything. Returns rows deleted."""
        try:
            if series:
                cursor = self._conn().execute(
                    "DELETE FROM observations WHERE series = ?", (series,))
            else:
                cursor = self._conn().execute("DELETE FROM observations")
            self._conn().commit()
            return cursor.rowcount or 0
        except Exception as exc:
            log.debug("Observation clear failed: %s", exc)
            return 0


_store_singleton: Optional[ObservationStore] = None
_singleton_lock = threading.Lock()


def get_store() -> ObservationStore:
    """Process-wide store, created on first use."""
    global _store_singleton
    if _store_singleton is None:
        with _singleton_lock:
            if _store_singleton is None:
                _store_singleton = ObservationStore()
    return _store_singleton


def record(series: str, value: float, **kwargs) -> bool:
    return get_store().record(series, value, **kwargs)


def baseline(series: str, **kwargs) -> Optional[Baseline]:
    return get_store().baseline(series, **kwargs)


def history(series: str, **kwargs) -> List[Tuple[float, float]]:
    return get_store().history(series, **kwargs)


__all__ = [
    "Baseline", "ObservationStore", "get_store",
    "record", "baseline", "history",
    "MIN_SAMPLES", "MAX_AGE_DAYS", "MIN_INTERVAL_SECONDS",
]
