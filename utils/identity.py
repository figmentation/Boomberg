"""
utils/identity.py :: Who a session belongs to.

One seam for identity, so no page asks Streamlit, a header or an environment
variable directly. The source is Streamlit's built-in OIDC login
(`st.login()`, configured under [auth] in .streamlit/secrets.toml). What an
identity may do is decided separately, in utils.entitlements; swapping the
provider means replacing the body of `current_user`, not its callers.

Nothing here logs an account identifier. The storage key is a one-way hash,
so a listing of .openterm/users/ does not enumerate email addresses either.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# The single-user store this terminal has always had.
LOCAL_USER = "local"


@dataclass(frozen=True)
class User:
    key: str             # storage key: LOCAL_USER, or a hash of the account id
    display: str         # for the UI; never logged
    authenticated: bool


LOCAL = User(key=LOCAL_USER, display="LOCAL", authenticated=False)


def user_key(identifier: str) -> str:
    """Stable, filesystem-safe key for an account identifier."""
    normalised = str(identifier).strip().lower()
    digest = hashlib.sha256(f"openterm-user:{normalised}".encode("utf-8"))
    return digest.hexdigest()[:32]


def current_user() -> User:
    """
    The signed-in account, or LOCAL.

    LOCAL covers every case without a login: no Streamlit runtime (tests,
    scripts), auth not configured, or a visitor who has not signed in.

    The email is the identifier only when the provider has not said it is
    unverified. Entitlements grant roles by email and by email domain, so an
    unverified claim to alice@example.com must not be able to match either;
    such an account is identified by its opaque `sub` instead. A provider
    that sends no email_verified claim at all is taken at its word.
    """
    try:
        import streamlit as st

        account = st.user
        if not account.is_logged_in:
            return LOCAL
        email = account.get("email")
        if email and account.get("email_verified") is not False:
            identifier = email
        else:
            identifier = account.get("sub")
    except Exception:
        return LOCAL

    if not identifier:
        return LOCAL
    return User(key=user_key(identifier), display=str(identifier),
                authenticated=True)


def login_available() -> bool:
    """True when an OIDC provider is configured, so st.login() can work."""
    try:
        import streamlit as st

        return "auth" in st.secrets
    except Exception:
        return False


__all__ = ["LOCAL_USER", "LOCAL", "User", "user_key", "current_user",
           "login_available"]
