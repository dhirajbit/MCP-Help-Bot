"""Slack Bolt app — listens for DMs and channel @mentions, routes to knowledge base RAG or MCP tools."""

import asyncio
import logging
import re
import secrets
import threading
import time

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import config
from auth import token_store
from auth.flow_state import FlowStep, OAuthFlowManager
from auth.oauth import (
    build_authorize_url,
    discover_oauth_endpoints,
    exchange_code,
    generate_pkce,
)
from bot.chat_history import ChatHistoryManager
from bot.mcp_handler import handle_mcp_query
from bot.rag import answer_question
from mcp_client.client import call_mcp_tool

logger = logging.getLogger(__name__)

# Module-level managers
flow_manager = OAuthFlowManager()
chat_history = ChatHistoryManager()

# -- Event deduplication --
_DEDUP_TTL = 60
_seen_events: dict[str, float] = {}
_seen_lock = threading.Lock()


def _is_duplicate_event(event: dict) -> bool:
    key = event.get("client_msg_id") or event.get("event_ts") or event.get("ts")
    if not key:
        return False

    now = time.monotonic()

    with _seen_lock:
        stale = [k for k, ts in _seen_events.items() if now - ts > _DEDUP_TTL]
        for k in stale:
            del _seen_events[k]

        if key in _seen_events:
            return True
        _seen_events[key] = now
        return False


# Keywords that signal a pure knowledge-base / "how does this work?" question.
_RAG_ONLY_KEYWORDS = {
    "how to",
    "how do i",
    "what is",
    "what are",
    "guide",
    "tutorial",
    "documentation",
    "help center",
    "setup",
    "set up",
    "configure",
    "integrate",
    "integration",
    "getting started",
    "explain",
}


# -- Command handlers --

def _handle_connect(user_id: str, say) -> None:
    if token_store.is_connected(user_id):
        say(
            f"You're already connected to {config.PRODUCT_NAME}!\n"
            "Type *disconnect* first if you want to reconnect with different credentials."
        )
        return

    existing = flow_manager.get_state(user_id)
    if existing:
        pending = flow_manager.get_pending(user_id)
        if pending.get("target_channel_id"):
            say(
                "You have a channel connection in progress. "
                "Type *cancel* first, then try *connect* again."
            )
            return

    flow_manager.start_flow(user_id)

    help_link = ""
    if config.MCP_HELP_URL:
        help_link = f"\n\n<{config.MCP_HELP_URL}|See the full setup guide>\n"

    say(
        f"Let's connect your {config.PRODUCT_NAME} workspace!\n\n"
        "I need your *OAuth Client Secret*.\n"
        f"{help_link}\n"
        "Paste your secret below:"
    )


def _handle_disconnect(user_id: str, say) -> None:
    flow_manager.cancel(user_id)
    token_store.delete_user_auth(user_id)
    chat_history.clear(user_id)
    say(
        f"Disconnected from {config.PRODUCT_NAME}.\n"
        "Type *connect* whenever you want to link your workspace again."
    )


def _handle_status(user_id: str, say) -> None:
    if token_store.is_connected(user_id):
        say(f"*Connected* to {config.PRODUCT_NAME}.\nType *disconnect* to unlink.")
    elif flow_manager.get_state(user_id):
        say("Connection in progress — follow the steps above to finish.")
    else:
        say(f"Not connected. Type *connect* to link your {config.PRODUCT_NAME} workspace.")


def _handle_cancel(user_id: str, say) -> None:
    flow_manager.cancel(user_id)
    say("Connection flow cancelled. Type *connect* to start again anytime.")


def _handle_help(say) -> None:
    say(
        "*Available commands:*\n\n"
        f"* *connect* — Link your {config.PRODUCT_NAME} workspace\n"
        "* *disconnect* — Unlink your workspace\n"
        "* *status* — Check your connection status\n"
        "* *cancel* — Cancel an in-progress connection\n"
        "* *help* — Show this message\n\n"
        "Or just ask me any question — "
        "I'll look it up in the knowledge base!"
    )


# -- Helpers --

def _clean_slack_input(text: str) -> str:
    raw = text.strip()
    raw = raw.strip("`")
    return raw.strip()


def _extract_auth_code(text: str) -> str:
    import urllib.parse

    raw = _clean_slack_input(text)

    m = re.search(r"<(https?://[^|>]+)", raw)
    if m:
        raw = m.group(1)

    raw = raw.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")

    if "code=" in raw and ("http://" in raw or "https://" in raw):
        parsed = urllib.parse.urlparse(raw)
        params = urllib.parse.parse_qs(parsed.query)
        code_list = params.get("code", [])
        if code_list:
            return code_list[0].strip()

    return raw.strip()


# -- Workspace name helper --

def _fetch_workspace_name(access_token: str) -> str:
    """Try to get the connected workspace/account name from the MCP server."""
    import json as _json

    for tool_name in ("list_connected_accounts", "get_authorized_connected_account"):
        try:
            raw = asyncio.run(call_mcp_tool(access_token, tool_name, {}))
            data = _json.loads(raw) if isinstance(raw, str) else raw

            if isinstance(data, list):
                for acct in data:
                    if isinstance(acct, dict):
                        name = (
                            acct.get("workspace_name")
                            or acct.get("company_name")
                            or acct.get("name")
                            or ""
                        )
                        if isinstance(name, str) and name.strip():
                            return name.strip()
            elif isinstance(data, dict):
                name = (
                    data.get("workspace_name")
                    or data.get("company_name")
                    or data.get("name")
                    or ""
                )
                if isinstance(name, str) and name.strip():
                    return name.strip()
        except Exception:
            logger.debug("%s failed", tool_name, exc_info=True)

    return ""


def _post_channel_connected_message(
    client, channel_id: str, workspace_name: str, connected_by: str,
) -> None:
    if workspace_name:
        headline = f"This channel is now connected to *{workspace_name}* on {config.PRODUCT_NAME}!"
    else:
        headline = f"This channel is now connected to {config.PRODUCT_NAME}!"

    client.chat_postMessage(
        channel=channel_id,
        text=(
            f"{headline}\n\n"
            f"Set up by <@{connected_by}>. Anyone here can now @mention me to "
            "use the available tools.\n\n"
            f"Just tag me like `@{config.BOT_NAME} <your question>` and I'll handle it!"
        ),
    )


# -- OAuth step-by-step handler --

def _handle_oauth_step(user_id: str, text: str, say, client, channel: str) -> None:
    step = flow_manager.get_state(user_id)

    if step == FlowStep.AWAITING_CLIENT_SECRET:
        client_secret = _clean_slack_input(text)
        logger.info(
            "Received client secret for user %s: len=%d",
            user_id, len(client_secret),
        )
        if not client_secret or len(client_secret) < 10:
            say("That doesn't look like a valid secret. Please try again:")
            return

        say("Discovering OAuth endpoints...")
        try:
            endpoints = discover_oauth_endpoints()
        except Exception as exc:
            logger.warning("OAuth discovery failed: %s", exc)
            say(
                "I couldn't auto-discover the OAuth endpoints.\n"
                "Please contact support for help connecting."
            )
            flow_manager.cancel(user_id)
            return

        verifier, challenge = generate_pkce()
        state_nonce = secrets.token_urlsafe(16)

        flow_manager.set_client_secret(
            user_id,
            client_secret,
            code_verifier=verifier,
            code_challenge=challenge,
            state_nonce=state_nonce,
            authorization_endpoint=endpoints["authorization_endpoint"],
            token_endpoint=endpoints["token_endpoint"],
        )

        pending = flow_manager.get_pending(user_id)
        target_channel = pending.get("target_channel_id", "")

        if target_channel:
            token_store.save_channel_credentials(
                channel_id=target_channel,
                connected_by=user_id,
                client_id=pending["client_id"],
                client_secret=client_secret,
                auth_server_url=endpoints["token_endpoint"],
            )
        else:
            token_store.save_credentials(
                user_id,
                pending["client_id"],
                client_secret,
                auth_server_url=endpoints["token_endpoint"],
            )

        auth_url = build_authorize_url(
            authorization_endpoint=endpoints["authorization_endpoint"],
            client_id=pending["client_id"],
            code_challenge=challenge,
            state=state_nonce,
            scope=config.OAUTH_SCOPES,
        )

        say(
            "Great! Click the link below to authorize:\n\n"
            f"<{auth_url}|*Connect to {config.PRODUCT_NAME}*>\n\n"
            "After you sign in, your browser will redirect.\n\n"
            "Look at your browser's *address bar*. The URL will contain a `code=` parameter.\n\n"
            "Copy the *entire URL* from the address bar and paste it here "
            "(I'll extract the code automatically). Or just copy the `code=` value."
        )

    elif step == FlowStep.AWAITING_AUTH_CODE:
        code = _extract_auth_code(text)
        logger.info(
            "Extracted auth code for user %s: len=%d, first8=%s",
            user_id, len(code), code[:8],
        )

        if not code or len(code) < 5:
            say("That doesn't look like a valid code. Please try again:")
            return

        pending = flow_manager.get_pending(user_id)
        target_channel = pending.get("target_channel_id", "")
        say("Exchanging code for tokens...")

        try:
            tokens = exchange_code(
                token_endpoint=pending["token_endpoint"],
                client_id=pending["client_id"],
                client_secret=pending["client_secret"],
                code=code,
                code_verifier=pending["code_verifier"],
            )
        except Exception as exc:
            logger.warning("Token exchange failed for user %s: %s", user_id, exc)
            say(
                "Token exchange failed — the code may have expired.\n"
                "Type *connect* to try again."
            )
            flow_manager.cancel(user_id)
            return

        if target_channel:
            token_store.save_channel_tokens(
                target_channel,
                tokens["access_token"],
                tokens.get("refresh_token"),
                tokens.get("expires_in"),
            )
            flow_manager.complete(user_id)

            try:
                info = client.conversations_info(channel=target_channel)
                channel_name = f"#{info['channel']['name']}"
            except Exception:
                channel_name = f"channel {target_channel}"

            workspace_name = _fetch_workspace_name(tokens["access_token"])
            workspace_label = f" (*{workspace_name}*)" if workspace_name else ""

            say(
                f"*{channel_name} is now connected to {config.PRODUCT_NAME}{workspace_label}!*\n\n"
                "Anyone in that channel can now @mention me to use the available tools.\n\n"
                f"To disconnect, use `@{config.BOT_NAME} disconnect` in {channel_name}."
            )

            _post_channel_connected_message(
                client, target_channel, workspace_name, user_id,
            )
        else:
            token_store.save_tokens(
                user_id,
                tokens["access_token"],
                tokens.get("refresh_token"),
                tokens.get("expires_in"),
            )
            flow_manager.complete(user_id)

            say(
                f"*Connected to {config.PRODUCT_NAME}!*\n\n"
                "You can now use MCP tools via this bot.\n"
                "Type *disconnect* to unlink, or just ask me anything!"
            )


# -- Routing --

def _looks_like_rag_only(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _RAG_ONLY_KEYWORDS)


def _route_to_mcp_or_rag(
    id_type: str,
    id_value: str,
    question: str,
    history: list[dict],
) -> str:
    try:
        return handle_mcp_query(id_type, id_value, question, history=history)
    except Exception as exc:
        logger.warning("MCP query failed, falling back to RAG: %s", exc)
        return answer_question(question, history=history)


# -- Shared question-answering logic --

def _strip_bot_mention(text: str) -> str:
    return re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()


def _answer_question_flow(
    user_id: str,
    question: str,
    channel: str,
    client,
    thread_ts: str | None = None,
    is_channel: bool = False,
) -> None:
    thinking = client.chat_postMessage(
        channel=channel,
        text=":hourglass_flowing_sand: Looking into this for you...",
        thread_ts=thread_ts,
    )

    history_key = channel if is_channel else user_id
    history = chat_history.get_history(history_key)

    try:
        if is_channel and token_store.is_channel_connected(channel):
            answer = _route_to_mcp_or_rag("channel", channel, question, history)
        elif token_store.is_connected(user_id):
            answer = _route_to_mcp_or_rag("user", user_id, question, history)
        else:
            answer = answer_question(question, history=history)

        chat_history.add_user_message(history_key, question)
        chat_history.add_assistant_message(history_key, answer)

        client.chat_update(
            channel=channel, ts=thinking["ts"], text=answer,
        )
    except Exception:
        logger.exception("Error answering question")
        client.chat_update(
            channel=channel,
            ts=thinking["ts"],
            text=(
                "Sorry, something went wrong while processing your question. "
                "Please try again or contact support."
            ),
        )


# -- Channel command handlers --

def _handle_channel_connect(
    user_id: str, channel_id: str, client, thread_ts: str,
) -> None:
    existing_state = flow_manager.get_state(user_id)
    if existing_state:
        pending = flow_manager.get_pending(user_id)
        if pending.get("target_channel_id"):
            client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=(
                    "You already have a connection in progress for another channel. "
                    "DM me *cancel* first, then try again."
                ),
            )
        else:
            client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=(
                    "You have a personal connection in progress. "
                    "DM me *cancel* first, then try again."
                ),
            )
        return

    if token_store.is_channel_connected(channel_id):
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=(
                f"This channel is already connected to {config.PRODUCT_NAME}!\n"
                f"Use `@{config.BOT_NAME} disconnect` to remove the connection first."
            ),
        )
        return

    client.chat_postMessage(
        channel=channel_id,
        thread_ts=thread_ts,
        text="I'll send you a DM to set up the connection for this channel.",
    )

    dm = client.conversations_open(users=[user_id])
    dm_channel = dm["channel"]["id"]

    try:
        info = client.conversations_info(channel=channel_id)
        channel_name = f"#{info['channel']['name']}"
    except Exception:
        channel_name = f"channel {channel_id}"

    flow_manager.start_flow(user_id, target_channel_id=channel_id)

    help_link = ""
    if config.MCP_HELP_URL:
        help_link = f"\n\n<{config.MCP_HELP_URL}|See the full setup guide>\n"

    client.chat_postMessage(
        channel=dm_channel,
        text=(
            f"Let's connect *{channel_name}* to {config.PRODUCT_NAME}!\n\n"
            "I need the *OAuth Client Secret* for that workspace.\n"
            f"{help_link}\n"
            "Paste the secret below:"
        ),
    )


def _handle_channel_disconnect(
    user_id: str, channel_id: str, client, thread_ts: str,
) -> None:
    if not token_store.is_channel_connected(channel_id):
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f"This channel isn't connected to {config.PRODUCT_NAME}.",
        )
        return

    token_store.delete_channel_auth(channel_id)
    chat_history.clear(channel_id)
    client.chat_postMessage(
        channel=channel_id,
        thread_ts=thread_ts,
        text=(
            f"Disconnected this channel from {config.PRODUCT_NAME}.\n"
            f"Use `@{config.BOT_NAME} connect` to link it again."
        ),
    )


def _handle_channel_status(
    channel_id: str, client, thread_ts: str,
) -> None:
    auth = token_store.get_channel_auth(channel_id)
    if auth and auth.get("access_token"):
        connected_by = auth.get("connected_by", "unknown")
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=(
                f"This channel is *connected* to {config.PRODUCT_NAME} "
                f"(set up by <@{connected_by}>).\n"
                f"Use `@{config.BOT_NAME} disconnect` to unlink."
            ),
        )
    else:
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=(
                f"This channel is not connected to {config.PRODUCT_NAME}.\n"
                f"Use `@{config.BOT_NAME} connect` to link a workspace."
            ),
        )


# -- Main listener --

def _register_listeners(app: App):

    @app.event("message")
    def handle_dm(event: dict, say, client):
        if event.get("channel_type") != "im":
            return
        subtype = event.get("subtype")
        if event.get("bot_id"):
            return
        if subtype and subtype != "file_share":
            return

        if _is_duplicate_event(event):
            return

        question = event.get("text", "").strip()
        if not question:
            return

        user_id = event.get("user", "")
        channel = event["channel"]
        text_lower = question.lower().strip()

        logger.info("DM from user %s: %s", user_id, question[:80])

        if text_lower in ("connect", "link"):
            return _handle_connect(user_id, say)
        if text_lower in ("disconnect", "unlink"):
            return _handle_disconnect(user_id, say)
        if text_lower == "status":
            return _handle_status(user_id, say)
        if text_lower == "cancel":
            return _handle_cancel(user_id, say)
        if text_lower == "help":
            return _handle_help(say)

        if flow_manager.get_state(user_id):
            return _handle_oauth_step(user_id, question, say, client, channel)

        _answer_question_flow(user_id, question, channel, client)

    @app.event("app_mention")
    def handle_mention(event: dict, say, client):
        if event.get("bot_id"):
            return

        if _is_duplicate_event(event):
            return

        raw_text = event.get("text", "")
        question = _strip_bot_mention(raw_text).strip()
        if not question:
            return

        user_id = event.get("user", "")
        channel = event.get("channel", "")
        thread_ts = event.get("thread_ts") or event.get("ts")

        logger.info("Channel mention from user %s in %s: %s", user_id, channel, question[:80])

        text_lower = question.lower().strip()

        if text_lower in ("connect", "link"):
            return _handle_channel_connect(user_id, channel, client, thread_ts)
        if text_lower in ("disconnect", "unlink"):
            return _handle_channel_disconnect(user_id, channel, client, thread_ts)
        if text_lower == "status":
            return _handle_channel_status(channel, client, thread_ts)
        if text_lower == "cancel":
            flow_manager.cancel(user_id)
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=f"Connection flow cancelled. Use `@{config.BOT_NAME} connect` to start again.",
            )
            return

        if text_lower == "help":
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=(
                    f"I'm *{config.BOT_NAME}* — tag me with any question!\n\n"
                    f"* `@{config.BOT_NAME} connect` — link this channel to {config.PRODUCT_NAME}\n"
                    f"* `@{config.BOT_NAME} disconnect` — remove the connection\n"
                    f"* `@{config.BOT_NAME} status` — check the connection\n"
                    "* I'll reply in this thread so the channel stays clean"
                ),
            )
            return

        _answer_question_flow(
            user_id, question, channel, client,
            thread_ts=thread_ts, is_channel=True,
        )


def start_bot():
    """Create the Slack Bolt app and start in Socket Mode (blocking)."""
    app = App(token=config.SLACK_BOT_TOKEN)
    _register_listeners(app)

    handler = SocketModeHandler(app, config.SLACK_APP_TOKEN)
    logger.info("Slack bot starting in Socket Mode...")
    handler.start()
