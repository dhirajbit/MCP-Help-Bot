"""OAuth 2.1 + PKCE helpers for MCP server connection.

Manual-code flow: the user opens an authorize URL in their browser,
authenticates, then copies the authorization code from the redirect
URL and pastes it back in Slack.
"""

import base64
import hashlib
import logging
import secrets
import urllib.parse

import httpx

import config

logger = logging.getLogger(__name__)

REDIRECT_URI = config.OAUTH_REDIRECT_URI


# -- PKCE --

def generate_pkce() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) for S256 PKCE."""
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


# -- OAuth discovery --

def discover_oauth_endpoints(mcp_url: str | None = None) -> dict:
    """Try MCP Protected Resource Metadata -> Authorization Server Metadata.

    Returns dict with keys ``authorization_endpoint`` and ``token_endpoint``.
    Raises ``RuntimeError`` if discovery fails entirely.
    """
    mcp_url = (mcp_url or config.MCP_SERVER_URL).rstrip("/")

    # Step 1: Protected Resource Metadata (RFC 9728)
    resource_url = f"{mcp_url}/.well-known/oauth-protected-resource"
    try:
        resp = httpx.get(resource_url, timeout=10, follow_redirects=True)
        if resp.status_code == 200:
            data = resp.json()
            auth_servers = data.get("authorization_servers") or []
            if auth_servers:
                as_url = auth_servers[0].rstrip("/")
                return _fetch_as_metadata(as_url)
            resource = data.get("resource")
            if resource:
                return _fetch_as_metadata(resource.rstrip("/"))
    except Exception as exc:
        logger.debug("Protected-resource discovery failed: %s", exc)

    # Step 2: Try common AS metadata path on the MCP host itself
    try:
        return _fetch_as_metadata(mcp_url)
    except Exception as exc:
        logger.debug("Direct AS metadata failed: %s", exc)

    raise RuntimeError(
        "Could not discover OAuth endpoints. "
        "Please provide authorize and token URLs manually."
    )


def _fetch_as_metadata(base_url: str) -> dict:
    """Fetch ``/.well-known/oauth-authorization-server`` from *base_url*."""
    url = f"{base_url}/.well-known/oauth-authorization-server"
    resp = httpx.get(url, timeout=10, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()

    auth_endpoint = data.get("authorization_endpoint")
    token_endpoint = data.get("token_endpoint")
    if not auth_endpoint or not token_endpoint:
        raise ValueError("Missing authorization_endpoint or token_endpoint in AS metadata")

    logger.info(
        "Discovered OAuth endpoints — authorize: %s, token: %s",
        auth_endpoint,
        token_endpoint,
    )
    return {
        "authorization_endpoint": auth_endpoint,
        "token_endpoint": token_endpoint,
        "registration_endpoint": data.get("registration_endpoint"),
        "scopes_supported": data.get("scopes_supported", []),
    }


# -- Authorize URL --

def build_authorize_url(
    authorization_endpoint: str,
    client_id: str,
    code_challenge: str,
    state: str,
    scope: str = "",
) -> str:
    """Build the full authorization URL the user should open in a browser."""
    params: dict[str, str] = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    if scope:
        params["scope"] = scope
    return f"{authorization_endpoint}?{urllib.parse.urlencode(params)}"


# -- Token exchange --

def exchange_code(
    token_endpoint: str,
    client_id: str,
    client_secret: str,
    code: str,
    code_verifier: str,
) -> dict:
    """Exchange an authorization code + PKCE verifier for tokens.

    Returns ``{"access_token": ..., "refresh_token": ..., "expires_in": ...}``.
    Raises ``RuntimeError`` on failure.
    """
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": client_id,
        "client_secret": client_secret,
        "code_verifier": code_verifier,
    }

    resp = httpx.post(token_endpoint, data=payload, timeout=15)
    if resp.status_code != 200:
        logger.error("Token exchange failed (%d): %s", resp.status_code, resp.text)
        raise RuntimeError(f"Token exchange failed: {resp.text}")

    data = resp.json()
    return {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token"),
        "expires_in": data.get("expires_in"),
    }


def refresh_access_token(
    token_endpoint: str,
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> dict:
    """Use a refresh token to obtain a new access token.

    Returns the same shape as :func:`exchange_code`.
    Raises ``RuntimeError`` on failure.
    """
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }

    resp = httpx.post(token_endpoint, data=payload, timeout=15)
    if resp.status_code != 200:
        logger.error("Token refresh failed (%d): %s", resp.status_code, resp.text)
        raise RuntimeError(f"Token refresh failed: {resp.text}")

    data = resp.json()
    return {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token", refresh_token),
        "expires_in": data.get("expires_in"),
    }
