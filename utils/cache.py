"""
utils/cache.py :: Local SQLite caching layer.

Two independent layers, both backed by SQLite in `config.DATA_DIR`:

  1. `SQLiteCache` / `@cached`  - object cache. Stores arbitrary Python values
     (DataFrames, dicts, lists) keyed by function + arguments, with per-entry
     TTL. This is what fetchers decorate themselves with.

  2. `get_session()`            - a `requests` Session wired through
     `requests-cache`, so even un-decorated raw HTTP hits are deduplicated.

Design notes:
  * Values are pickled. DataFrames round-trip losslessly and fast.
  * Writes are best-effort: a cache failure must never break a render.
  * A `stale` read path exists. When the network is down, `@cached` will
    happily serve an expired entry rather than show the user nothing - it
    tags the result so the UI can display an "AS OF" warning.
  * An empty result never replaces a non-empty one. Most fetchers turn an
    outage into an empty value rather than an exception, so without this
    rule the first refetch during an outage overwrote the last good quote
    with `{}` and the stale path above never got the chance to run.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import config
from utils.rate_limiter import looks_empty

log = logging.getLogger("openterm.cache")

F = TypeVar("F", bound=Callable[..., Any])

# How long an empty result is remembered when there is no earlier value to
# fall back on. Short, because an empty answer is as likely to be an outage
# as a fact, and a daily TTL would keep the page blank long after recovery.
EMPTY_RESULT_TTL = 120

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key         TEXT PRIMARY KEY,
    namespace   TEXT NOT NULL,
    value       BLOB NOT NULL,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_kv_expires   ON kv(expires_at);
CREATE INDEX IF NOT EXISTS idx_kv_namespace ON kv(namespace);
"""


@dataclass(frozen=True)
class CacheEntry:
    """A value plus the provenance metadata the UI needs."""

    value: Any
    created_at: float
    expires_at: float
    stale: bool
    # True when this value was handed back because a refresh failed, rather
    # than because it was still valid. `stale` only says the TTL has passed,
    # which for a 60-second quote happens a minute after every fetch and says
    # nothing about whether the upstream is healthy.
    served_stale: bool = False

    @property
    def as_of(self) -> datetime:
        """When this value was fetched, as an aware UTC datetime."""
        return datetime.fromtimestamp(self.created_at, timezone.utc)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def age_label(self) -> str:
        """Human-readable age, e.g. '4m ago'."""
        secs = self.age_seconds
        if secs < 60:
            return f"{int(secs)}s ago"
        if secs < 3600:
            return f"{int(secs // 60)}m ago"
        if secs < 86400:
            return f"{int(secs // 3600)}h ago"
        return f"{int(secs // 86400)}d ago"


class SQLiteCache:
    """
    Thread-safe TTL cache on SQLite.

    One connection per thread (sqlite3 objects are not shareable across
    threads by default, and Streamlit runs script threads liberally).
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = str(path or config.CACHE_DB)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        # Keys last served because a refetch failed. In-process and
        # deliberately not persisted: it describes what this server is
        # currently showing, and a restart re-establishes it on the first
        # fetch that fails.
        self._fallbacks: Dict[str, float] = {}
        self._fallback_lock = threading.Lock()
        self._ensure_schema()

    # -- fallback bookkeeping ----------------------------------------------
    def note_fallback(self, key: str) -> None:
        """Record that `key` was served after a refresh failed."""
        with self._fallback_lock:
            self._fallbacks[key] = time.time()

    def clear_fallback(self, key: str) -> None:
        """A good value replaced the old one; the entry is no longer a fallback."""
        with self._fallback_lock:
            self._fallbacks.pop(key, None)

    def served_as_fallback(self, key: str) -> bool:
        with self._fallback_lock:
            return key in self._fallbacks

    # -- connection management ---------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
            # WAL lets readers proceed while a writer holds the DB - important
            # because Streamlit can have several script runs in flight.
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
                log.error("Cache schema init failed: %s", exc)

    # -- core API ----------------------------------------------------------
    def get(self, key: str, allow_stale: bool = False) -> Optional[CacheEntry]:
        """
        Return the entry for `key`, or None.

        If `allow_stale` is True an expired entry is returned with
        `.stale=True` instead of being treated as a miss.
        """
        try:
            row = self._conn().execute(
                "SELECT value, created_at, expires_at FROM kv WHERE key = ?",
                (key,),
            ).fetchone()
        except Exception as exc:
            log.debug("Cache read error for %s: %s", key, exc)
            return None

        if row is None:
            return None

        blob, created_at, expires_at = row
        expired = time.time() >= expires_at

        if expired and not allow_stale:
            return None

        try:
            value = pickle.loads(blob)
        except Exception as exc:
            log.warning("Cache entry %s is corrupt, evicting: %s", key, exc)
            self.delete(key)
            return None

        # Fire-and-forget hit counter.
        try:
            self._conn().execute(
                "UPDATE kv SET hits = hits + 1 WHERE key = ?", (key,)
            )
            self._conn().commit()
        except Exception:
            pass

        return CacheEntry(value=value, created_at=created_at,
                          expires_at=expires_at, stale=expired)

    def metadata(self, key: str) -> Optional[CacheEntry]:
        """
        Age and expiry for `key` without unpickling the value.

        What the UI needs to render a freshness badge is only ever the
        timestamps, and a page can ask about a dozen fetchers per run.
        `value` comes back as None, `stale` means the TTL has passed, and
        `served_stale` means this value is on screen because a refresh
        failed - the two the badge keeps apart.
        """
        try:
            row = self._conn().execute(
                "SELECT created_at, expires_at FROM kv WHERE key = ?", (key,),
            ).fetchone()
        except Exception as exc:
            log.debug("Cache metadata error for %s: %s", key, exc)
            return None

        if row is None:
            return None

        created_at, expires_at = row
        return CacheEntry(value=None, created_at=created_at,
                          expires_at=expires_at,
                          stale=time.time() >= expires_at,
                          served_stale=self.served_as_fallback(key))

    def set(self, key: str, value: Any, ttl: int, namespace: str = "default") -> None:
        """Store `value` under `key` for `ttl` seconds. Never raises."""
        try:
            blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            log.debug("Value for %s is not picklable, skipping cache: %s", key, exc)
            return

        now = time.time()
        try:
            self._conn().execute(
                "INSERT OR REPLACE INTO kv "
                "(key, namespace, value, created_at, expires_at, hits) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (key, namespace, blob, now, now + ttl),
            )
            self._conn().commit()
        except Exception as exc:
            log.debug("Cache write error for %s: %s", key, exc)

    def delete(self, key: str) -> None:
        try:
            self._conn().execute("DELETE FROM kv WHERE key = ?", (key,))
            self._conn().commit()
        except Exception:
            pass

    def purge_expired(self) -> int:
        """Drop expired rows. Returns the number removed."""
        try:
            cur = self._conn().execute(
                "DELETE FROM kv WHERE expires_at < ?", (time.time(),)
            )
            self._conn().commit()
            return cur.rowcount or 0
        except Exception:
            return 0

    def clear(self, namespace: Optional[str] = None) -> int:
        """Wipe a namespace, or the whole cache when namespace is None."""
        try:
            if namespace:
                cur = self._conn().execute(
                    "DELETE FROM kv WHERE namespace = ?", (namespace,)
                )
            else:
                cur = self._conn().execute("DELETE FROM kv")
            self._conn().commit()
            self._conn().execute("VACUUM")
            return cur.rowcount or 0
        except Exception:
            return 0

    def stats(self) -> List[Dict[str, Any]]:
        """Per-namespace summary for the diagnostics panel."""
        try:
            rows = self._conn().execute(
                "SELECT namespace, COUNT(*), SUM(hits), "
                "       SUM(LENGTH(value)), MIN(created_at) "
                "FROM kv GROUP BY namespace ORDER BY COUNT(*) DESC"
            ).fetchall()
        except Exception:
            return []

        return [
            {
                "namespace": ns,
                "entries": count,
                "hits": hits or 0,
                "bytes": size or 0,
                "oldest": oldest,
            }
            for ns, count, hits, size, oldest in rows
        ]


# --------------------------------------------------------------------------
# Module-level singleton
# --------------------------------------------------------------------------
_cache_singleton: Optional[SQLiteCache] = None
_singleton_lock = threading.Lock()


def get_cache() -> SQLiteCache:
    global _cache_singleton
    if _cache_singleton is None:
        with _singleton_lock:
            if _cache_singleton is None:
                _cache_singleton = SQLiteCache()
    return _cache_singleton


# --------------------------------------------------------------------------
# Key construction
# --------------------------------------------------------------------------
def make_key(namespace: str, *args: Any, **kwargs: Any) -> str:
    """
    Deterministic cache key from a namespace plus call arguments.

    Arguments are repr'd and hashed, so unhashable values (lists, dicts) work
    fine and the key stays a fixed length regardless of payload size.
    """
    payload = repr(args) + "|" + repr(sorted(kwargs.items()))
    digest = hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:24]
    return f"{namespace}:{digest}"


# --------------------------------------------------------------------------
# @cached decorator
# --------------------------------------------------------------------------
def cached(
    ttl: int,
    namespace: Optional[str] = None,
    allow_stale_on_error: bool = True,
    key_fn: Optional[Callable[..., str]] = None,
    keep_last_good: bool = True,
) -> Callable[[F], F]:
    """
    Memoise a fetcher into SQLite.

    Args:
        ttl:                  Seconds the fresh value is valid for.
        namespace:            Grouping label. Defaults to the function name.
        allow_stale_on_error: If the wrapped call raises, serve the last known
                              value (however old) instead of propagating. This
                              is what keeps the terminal readable when an
                              upstream free API goes down mid-session.
        key_fn:               Custom key builder, receives the call args.
        keep_last_good:       If the wrapped call returns an empty value
                              ({}, [], empty frame, None) while a non-empty one
                              is on record, serve the recorded one, marked
                              stale, and leave it in place. An empty result
                              with nothing to protect is cached for at most
                              EMPTY_RESULT_TTL seconds.

    The wrapper exposes:
        fn.cache_clear()       Drop every entry in this namespace.
        fn.uncached(*a, **k)   Bypass the cache for one call.
        fn.cache_key(*a, **k)  The key one call reads and writes.
        fn.provenance(*a, **k) Age and expiry of that call's entry, or None
                               if nothing is stored yet. Pass the same
                               arguments the fetch was made with.

    Pass `_refresh=True` to skip the fresh read for one call. The existing
    entry stays in place until a good value replaces it, so a refresh during
    an outage still has something to fall back on.

    `fn.last_entry` records the *most recent* call anywhere in the process.
    Prefer `fn.provenance(...)`: it is keyed by the arguments, so a badge
    cannot end up describing another ticker's fetch, or another session's.
    """

    def decorator(func: F) -> F:
        ns = namespace or func.__name__
        cache = get_cache()

        def key_for(*args: Any, **kwargs: Any) -> str:
            return key_fn(*args, **kwargs) if key_fn else make_key(ns, *args, **kwargs)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            refresh = kwargs.pop("_refresh", False)
            key = key_for(*args, **kwargs)

            # Offline mode: cache is the only source of truth.
            if config.OFFLINE:
                entry = cache.get(key, allow_stale=True)
                wrapper.last_entry = entry  # type: ignore[attr-defined]
                if entry is None:
                    raise RuntimeError(
                        f"OFFLINE mode and no cached value for {ns}"
                    )
                return entry.value

            if not refresh:
                entry = cache.get(key)
                if entry is not None:
                    wrapper.last_entry = entry  # type: ignore[attr-defined]
                    return entry.value

            try:
                value = func(*args, **kwargs)
            except Exception as exc:
                if allow_stale_on_error:
                    stale = cache.get(key, allow_stale=True)
                    if stale is not None:
                        log.warning(
                            "%s failed (%s) - serving stale value from %s",
                            ns, exc, stale.age_label(),
                        )
                        cache.note_fallback(key)
                        wrapper.last_entry = replace(  # type: ignore[attr-defined]
                            stale, stale=True, served_stale=True)
                        return stale.value
                raise

            entry_ttl = ttl
            if keep_last_good and looks_empty(value):
                previous = cache.get(key, allow_stale=True)
                if previous is not None and not looks_empty(previous.value):
                    log.warning(
                        "%s returned nothing - keeping the value from %s",
                        ns, previous.age_label(),
                    )
                    cache.note_fallback(key)
                    wrapper.last_entry = replace(  # type: ignore[attr-defined]
                        previous, stale=True, served_stale=True)
                    return previous.value
                entry_ttl = min(ttl, EMPTY_RESULT_TTL)

            cache.set(key, value, ttl=entry_ttl, namespace=ns)
            cache.clear_fallback(key)
            now = time.time()
            wrapper.last_entry = CacheEntry(  # type: ignore[attr-defined]
                value=value, created_at=now,
                expires_at=now + entry_ttl, stale=False,
            )
            return value

        wrapper.last_entry = None            # type: ignore[attr-defined]
        wrapper.cache_clear = lambda: cache.clear(ns)   # type: ignore[attr-defined]
        wrapper.uncached = func              # type: ignore[attr-defined]
        wrapper.namespace = ns               # type: ignore[attr-defined]
        wrapper.cache_key = key_for          # type: ignore[attr-defined]
        wrapper.provenance = (               # type: ignore[attr-defined]
            lambda *a, **k: cache.metadata(key_for(*a, **k)))
        return wrapper  # type: ignore[return-value]

    return decorator


# --------------------------------------------------------------------------
# HTTP-level cache (requests-cache)
# --------------------------------------------------------------------------
_session_registry: Dict[Tuple[str, int], Any] = {}
_session_lock = threading.Lock()


def get_session(name: str = "default", expire_after: int = 900,
                user_agent: Optional[str] = None):
    """
    A `requests.Session` with transparent SQLite response caching.

    Falls back to a plain Session if `requests-cache` isn't installed, so the
    app still runs on a minimal environment.

    Args:
        name:         Segregates cache files by concern ("sec", "scrape", ...).
        expire_after: Seconds before a cached response is revalidated.
        user_agent:   Overrides the default browser UA.
    """
    key = (name, expire_after)
    with _session_lock:
        session = _session_registry.get(key)
        if session is not None:
            return session

        try:
            import requests_cache

            session = requests_cache.CachedSession(
                cache_name=f"{config.HTTP_CACHE_DB}_{name}",
                backend="sqlite",
                expire_after=expire_after,
                allowable_codes=(200, 203, 300, 301, 308),
                allowable_methods=("GET", "HEAD"),
                stale_if_error=True,   # serve stale on upstream 5xx
                match_headers=False,
            )
        except Exception as exc:
            log.info("requests-cache unavailable (%s), using plain Session", exc)
            import requests

            session = requests.Session()

        session.headers.update({
            "User-Agent": user_agent or config.BROWSER_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
        _session_registry[key] = session
        return session


def clear_all_caches() -> Dict[str, int]:
    """
    Nuke both layers. Wired to the sidebar's PURGE CACHE button.

    Deliberately does not touch the `observations` table that shares this
    database. Cached values are re-fetchable in seconds; measured baselines
    take weeks to accumulate, and clearing them would silently reset every
    congestion gauge to MEASURING.
    """
    removed = {"objects": get_cache().clear()}

    with _session_lock:
        for (name, _), session in list(_session_registry.items()):
            try:
                session.cache.clear()  # type: ignore[attr-defined]
                removed[f"http:{name}"] = 1
            except Exception:
                pass
    return removed


__all__ = [
    "SQLiteCache", "CacheEntry", "get_cache", "cached", "make_key",
    "get_session", "clear_all_caches",
]
