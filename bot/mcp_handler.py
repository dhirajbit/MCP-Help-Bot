"""Claude agent loop with MCP tool use for connected users.

When a user who has linked their workspace asks a question that needs
MCP tools, this module orchestrates the conversation between Claude and
the remote MCP server in an iterative tool-call loop.

Before calling Claude, it also fetches relevant knowledge from the BM25
search index so that tool calls are informed by best practices.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

import anthropic

import config
from auth import token_store
from auth.oauth import refresh_access_token
from mcp_client.client import call_mcp_tool, list_mcp_tools
from sync.embedder import get_collection

logger = logging.getLogger(__name__)

_anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

# -- System prompt (knowledge section injected at runtime) --

_MCP_SYSTEM_BASE = f"""\
You are {config.BOT_NAME}, a hands-on assistant for {config.PRODUCT_NAME}. \
You DO things for users — you don't just explain how. You have live access \
to their workspace via tools.

TOOLS-FIRST RULE (CRITICAL — ALWAYS FOLLOW):
* *ALWAYS call tools BEFORE answering.* Never answer from memory or guess. \
Your tools are your source of truth — use them on every question.
* If multiple tools could help, call them. It's always better to call a tool \
and get real data than to explain something from memory.

AFTER ANSWERING — ALWAYS OFFER ACTION:
* Don't just explain — offer to do it.
* If you listed items, offer to inspect or modify one.

ANTI-HALLUCINATION (CRITICAL):
* *NEVER invent* features, types, or settings. If it's not returned by a \
tool or in the KNOWLEDGE BASE, it does not exist.
* *If a tool call fails or returns an error, do NOT guess what the data \
would have been.* Say the tool call failed and offer to retry or suggest \
the user check directly. Never fabricate tool results.
* When in doubt, say you're not sure — a wrong answer is far worse than \
"I don't know — please contact support for help."

DO NOT:
* Never list your capabilities or tools.
* Never explain how you work internally (OAuth, tool calls, etc.).
* Never give long introductions. Get straight to the point.

FORMATTING (Slack, NOT Markdown):
* *bold* with single asterisks. _italic_ with underscores.
* NEVER use ## headings or **double asterisks**.
* Use bullets for lists. <URL|text> for links.
* *NEVER use Markdown tables* (pipes and dashes). Slack cannot render them.
* For tabular data, use a code block with aligned columns.
* Keep responses under 300 words. Be concise — lead with results.\
"""

_KNOWLEDGE_SECTION = """

KNOWLEDGE BASE — THIS IS YOUR ONLY SOURCE OF PRODUCT TRUTH:
Below are relevant excerpts from the knowledge base. These are the \
ONLY facts you may state about features and capabilities.

Use this knowledge to give informed recommendations instead of generic ones.

CRITICAL: If a feature or setting is NOT mentioned below, do NOT claim it \
exists. Say you don't have that info and suggest contacting support.

---
{knowledge}
---\
"""

MAX_TOOL_ROUNDS = 10
RAG_CHUNKS_FOR_MCP = 10
MAX_TOOL_RESULT_CHARS = 60_000
MAX_TOTAL_TOOL_CHARS = 300_000


# -- Markdown -> Slack --

def _markdown_to_slack(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    text = re.sub(r"^- ", "* ", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)
    return text


# -- Knowledge retrieval --

def _get_relevant_knowledge(question: str) -> str:
    try:
        collection = get_collection()
        if collection.count() == 0:
            return ""

        results = collection.query(
            query_texts=[question],
            n_results=min(RAG_CHUNKS_FOR_MCP, collection.count()),
            include=["documents", "metadatas"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return ""

        sections: list[str] = []
        for doc, meta in zip(documents, metadatas):
            title = meta.get("article_title", "Unknown")
            url = meta.get("url", "")
            header = f"[{title}]"
            if url:
                header += f" ({url})"
            sections.append(f"{header}\n{doc}")

        return "\n\n".join(sections)

    except Exception:
        logger.debug("Failed to retrieve knowledge", exc_info=True)
        return ""


def _build_system_prompt(question: str) -> str:
    from datetime import date

    today = date.today().isoformat()
    date_line = f"\nToday's date is *{today}*. Use this to calculate relative dates " \
                f"like \"past 3 months\", \"last week\", \"this quarter\", etc. " \
                f"Always convert relative time references to exact date ranges " \
                f"before passing them to tools.\n"

    base = _MCP_SYSTEM_BASE + date_line

    knowledge = _get_relevant_knowledge(question)
    if knowledge:
        logger.info(
            "Injected %d chars of knowledge into MCP prompt",
            len(knowledge),
        )
        return base + _KNOWLEDGE_SECTION.format(knowledge=knowledge)
    return base


# -- Token refresh helper --

def _ensure_valid_token(id_type: str, id_value: str) -> str:
    """Return a valid access token, refreshing if expired."""
    if id_type == "channel":
        auth = token_store.get_channel_auth(id_value)
        label = f"channel {id_value}"
    else:
        auth = token_store.get_user_auth(id_value)
        label = f"user {id_value}"

    if not auth or not auth.get("access_token"):
        raise RuntimeError(f"Not connected to {config.PRODUCT_NAME}.")

    expires_at = auth.get("token_expires_at")
    if expires_at and time.time() >= (expires_at - 60):
        refresh_tok = auth.get("refresh_token")
        token_ep = auth.get("auth_server_url", "")
        if not refresh_tok or not token_ep:
            raise RuntimeError(
                "The session has expired. "
                "Please type *disconnect* and then *connect* again."
            )
        logger.info("Refreshing expired token for %s", label)
        tokens = refresh_access_token(
            token_endpoint=token_ep,
            client_id=auth["client_id"],
            client_secret=auth["client_secret"],
            refresh_token=refresh_tok,
        )
        if id_type == "channel":
            token_store.save_channel_tokens(
                id_value,
                tokens["access_token"],
                tokens.get("refresh_token"),
                tokens.get("expires_in"),
            )
        else:
            token_store.save_tokens(
                id_value,
                tokens["access_token"],
                tokens.get("refresh_token"),
                tokens.get("expires_in"),
            )
        return tokens["access_token"]

    return auth["access_token"]


# -- Agent loop --

async def _async_handle_mcp_query(
    access_token: str,
    question: str,
    history: list[dict] | None = None,
) -> str:
    """Async core: Claude + MCP tool-call loop."""

    tools = await list_mcp_tools(access_token)
    if not tools:
        return (
            f"I connected to {config.PRODUCT_NAME} but couldn't find any available tools. "
            "Please try again or contact support for help."
        )

    system_prompt = _build_system_prompt(question)

    messages: list[dict] = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    response = _anthropic_client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=messages,
        tools=tools,
    )

    rounds = 0
    total_tool_chars = 0
    while response.stop_reason == "tool_use" and rounds < MAX_TOOL_ROUNDS:
        rounds += 1

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_results = []
        for tc in tool_use_blocks:
            logger.info("MCP tool call: %s(%s)", tc.name, tc.input)
            try:
                result_text = await call_mcp_tool(access_token, tc.name, tc.input)
            except Exception as exc:
                logger.exception("MCP tool %s failed", tc.name)
                result_text = (
                    f"[TOOL CALL FAILED] {tc.name} returned an error: {exc}\n"
                    f"IMPORTANT: You have NO data from this tool call. "
                    f"Do NOT guess or fabricate what the result would have been. "
                    f"Tell the user the tool call failed and suggest they check "
                    f"directly, or offer to retry."
                )

            if len(result_text) > MAX_TOOL_RESULT_CHARS:
                original_len = len(result_text)
                result_text = (
                    result_text[:MAX_TOOL_RESULT_CHARS]
                    + f"\n\n[TRUNCATED — result was {original_len:,} chars, "
                    f"showing first {MAX_TOOL_RESULT_CHARS:,}. "
                    f"Ask the user to narrow the query if more detail is needed.]"
                )

            total_tool_chars += len(result_text)
            if total_tool_chars > MAX_TOTAL_TOOL_CHARS:
                overage = total_tool_chars - MAX_TOTAL_TOOL_CHARS
                result_text = result_text[: max(500, len(result_text) - overage)]
                result_text += (
                    "\n\n[TRUNCATED — conversation tool results exceeded "
                    f"{MAX_TOTAL_TOOL_CHARS:,} chars total. "
                    "Summarize what you have and ask the user to narrow the query.]"
                )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_text,
                }
            )

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        response = _anthropic_client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
            tools=tools,
        )

    text_parts = [b.text for b in response.content if hasattr(b, "text")]
    return _markdown_to_slack("\n".join(text_parts))


# -- Public sync entry point --

def handle_mcp_query(
    id_type: str,
    id_value: str,
    question: str,
    history: list[dict] | None = None,
) -> str:
    """Sync wrapper — called from the Slack handler."""
    access_token = _ensure_valid_token(id_type, id_value)
    return asyncio.run(_async_handle_mcp_query(access_token, question, history))
