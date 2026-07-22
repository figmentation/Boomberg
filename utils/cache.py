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
"""

from __future__ import annotations

import functools
import hashlib
import logging
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import config

log = logging.getLogger("openterm.cache")

F = TypeVar("F", bound=Callable[..., Any])

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
        self._ensure_schema()

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

    The wrapper exposes two extras:
        fn.cache_clear()      Drop every entry in this namespace.
        fn.uncached(*a, **k)  Bypass the cache for one call.

    Provenance for the *most recent* call is recorded on
    `fn.last_entry` (a CacheEntry or None) so the UI can render an age badge.
    """

    def decorator(func: F) -> F:
        ns = namespace or func.__name__
        cache = get_cache()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if kwargs.pop("_refresh", False):
                key = key_fn(*args, **kwargs) if key_fn else make_key(ns, *args, **kwargs)
                cache.delete(key)
            else:
                key = key_fn(*args, **kwargs) if key_fn else make_key(ns, *args, **kwargs)

            # Offline mode: cache is the only source of truth.
            if config.OFFLINE:
                entry = cache.get(key, allow_stale=True)
                wrapper.last_entry = entry  # type: ignore[attr-defined]
                if entry is None:
                    raise RuntimeError(
                        f"OFFLINE mode and no cached value for {ns}"
                    )
                return entry.value

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
                        wrapper.last_entry = stale  # type: ignore[attr-defined]
                        return stale.value
                raise

            cache.set(key, value, ttl=ttl, namespace=ns)
            wrapper.last_entry = CacheEntry(  # type: ignore[attr-defined]
                value=value, created_at=time.time(),
                expires_at=time.time() + ttl, stale=False,
            )
            return value

        wrapper.last_entry = None            # type: ignore[attr-defined]
        wrapper.cache_clear = lambda: cache.clear(ns)   # type: ignore[attr-defined]
        wrapper.uncached = func              # type: ignore[attr-defined]
        wrapper.namespace = ns               # type: ignore[attr-defined]
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
    """Nuke both layers. Wired to the sidebar's PURGE CACHE button."""
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
