"""SQLite persistence for per-user and per-channel OAuth tokens and client credentials."""

import logging
import os
import sqlite3
import time

import config

logger = logging.getLogger(__name__)

_CREATE_USER_TABLE = """\
CREATE TABLE IF NOT EXISTS user_tokens (
    slack_user_id   TEXT PRIMARY KEY,
    client_id       TEXT NOT NULL,
    client_secret   TEXT NOT NULL,
    access_token    TEXT,
    refresh_token   TEXT,
    token_expires_at REAL,
    auth_server_url TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
"""

_CREATE_CHANNEL_TABLE = """\
CREATE TABLE IF NOT EXISTS channel_tokens (
    slack_channel_id TEXT PRIMARY KEY,
    connected_by     TEXT NOT NULL,
    client_id        TEXT NOT NULL,
    client_secret    TEXT NOT NULL,
    access_token     TEXT,
    refresh_token    TEXT,
    token_expires_at REAL,
    auth_server_url  TEXT,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);
"""


def _db_path() -> str:
    return config.SQLITE_DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


# -- Lifecycle --

def init_db() -> None:
    """Create the user and channel tokens tables if they don't already exist."""
    os.makedirs(os.path.dirname(_db_path()), exist_ok=True)
    with _connect() as conn:
        conn.execute(_CREATE_USER_TABLE)
        conn.execute(_CREATE_CHANNEL_TABLE)
    logger.info("SQLite DB initialized at %s", _db_path())


# -- Write operations --

def save_credentials(
    slack_user_id: str,
    client_id: str,
    client_secret: str,
    auth_server_url: str = "",
) -> None:
    """Store client credentials at the start of the OAuth flow."""
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_tokens
                (slack_user_id, client_id, client_secret, auth_server_url,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(slack_user_id) DO UPDATE SET
                client_id       = excluded.client_id,
                client_secret   = excluded.client_secret,
                auth_server_url = excluded.auth_server_url,
                access_token    = NULL,
                refresh_token   = NULL,
                token_expires_at = NULL,
                updated_at      = excluded.updated_at
            """,
            (slack_user_id, client_id, client_secret, auth_server_url, now, now),
        )


def save_tokens(
    slack_user_id: str,
    access_token: str,
    refresh_token: str | None = None,
    expires_in: int | None = None,
) -> None:
    """Store OAuth tokens after a successful code exchange or refresh."""
    now = time.time()
    expires_at = (now + expires_in) if expires_in else None
    with _connect() as conn:
        conn.execute(
            """
            UPDATE user_tokens
            SET access_token     = ?,
                refresh_token    = ?,
                token_expires_at = ?,
                updated_at       = ?
            WHERE slack_user_id  = ?
            """,
            (access_token, refresh_token, expires_at, now, slack_user_id),
        )


# -- Read operations --

def get_user_auth(slack_user_id: str) -> dict | None:
    """Return the full auth row as a dict, or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM user_tokens WHERE slack_user_id = ?",
            (slack_user_id,),
        ).fetchone()
    return dict(row) if row else None


def is_connected(slack_user_id: str) -> bool:
    """True if the user has a non-null access token."""
    auth = get_user_auth(slack_user_id)
    return bool(auth and auth.get("access_token"))


def is_token_expired(slack_user_id: str) -> bool:
    """True if the token's expiry time has passed (with 60s buffer)."""
    auth = get_user_auth(slack_user_id)
    if not auth or not auth.get("token_expires_at"):
        return True
    return time.time() >= (auth["token_expires_at"] - 60)


# -- Delete --

def delete_user_auth(slack_user_id: str) -> None:
    """Remove all credentials and tokens for a user."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM user_tokens WHERE slack_user_id = ?",
            (slack_user_id,),
        )
    logger.info("Deleted auth for user %s", slack_user_id)


# -- Channel token operations --

def save_channel_credentials(
    channel_id: str,
    connected_by: str,
    client_id: str,
    client_secret: str,
    auth_server_url: str = "",
) -> None:
    """Store client credentials for a channel connection (upsert)."""
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO channel_tokens
                (slack_channel_id, connected_by, client_id, client_secret,
                 auth_server_url, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slack_channel_id) DO UPDATE SET
                connected_by    = excluded.connected_by,
                client_id       = excluded.client_id,
                client_secret   = excluded.client_secret,
                auth_server_url = excluded.auth_server_url,
                access_token    = NULL,
                refresh_token   = NULL,
                token_expires_at = NULL,
                updated_at      = excluded.updated_at
            """,
            (channel_id, connected_by, client_id, client_secret,
             auth_server_url, now, now),
        )


def save_channel_tokens(
    channel_id: str,
    access_token: str,
    refresh_token: str | None = None,
    expires_in: int | None = None,
) -> None:
    """Store OAuth tokens for a channel after code exchange or refresh."""
    now = time.time()
    expires_at = (now + expires_in) if expires_in else None
    with _connect() as conn:
        conn.execute(
            """
            UPDATE channel_tokens
            SET access_token     = ?,
                refresh_token    = ?,
                token_expires_at = ?,
                updated_at       = ?
            WHERE slack_channel_id = ?
            """,
            (access_token, refresh_token, expires_at, now, channel_id),
        )


def get_channel_auth(channel_id: str) -> dict | None:
    """Return the full channel auth row as a dict, or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM channel_tokens WHERE slack_channel_id = ?",
            (channel_id,),
        ).fetchone()
    return dict(row) if row else None


def is_channel_connected(channel_id: str) -> bool:
    """True if the channel has a non-null access token."""
    auth = get_channel_auth(channel_id)
    return bool(auth and auth.get("access_token"))


def is_channel_token_expired(channel_id: str) -> bool:
    """True if the channel token's expiry time has passed (with 60s buffer)."""
    auth = get_channel_auth(channel_id)
    if not auth or not auth.get("token_expires_at"):
        return True
    return time.time() >= (auth["token_expires_at"] - 60)


def delete_channel_auth(channel_id: str) -> None:
    """Remove all credentials and tokens for a channel."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM channel_tokens WHERE slack_channel_id = ?",
            (channel_id,),
        )
    logger.info("Deleted auth for channel %s", channel_id)
