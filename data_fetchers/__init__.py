"""
Open-Terminal data fetchers.

Each submodule owns one domain and exposes plain functions returning pandas
objects. Import them directly (`from data_fetchers import equities`) rather
than pulling names through this package - several modules import heavy
optional dependencies and eager loading slows cold start noticeably.
"""

__all__ = ["equities", "maritime", "aviation", "macro", "news"]
