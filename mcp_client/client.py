"""Async MCP client wrapper for any remote MCP server.

Each public function creates a *fresh* session, calls the MCP server, and
returns the result — no long-lived connections to manage.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import config

logger = logging.getLogger(__name__)

# Timeout for any single MCP operation (connect + init + call).
_MCP_TIMEOUT_SECS = 30


# -- helpers --

def _mcp_url() -> str:
    """Streamable-HTTP endpoint on the MCP server."""
    base = config.MCP_SERVER_URL.rstrip("/")
    return f"{base}/mcp"


def _auth_headers(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


def _mcp_tool_to_anthropic(tool) -> dict:
    """Convert an MCP ``Tool`` object to Anthropic tool-use format."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema,
    }


# -- public API (all async) --

async def list_mcp_tools(access_token: str) -> list[dict]:
    """Connect to the MCP server and return available tools in Anthropic format."""
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession

    url = _mcp_url()
    headers = _auth_headers(access_token)

    try:
        async with asyncio.timeout(_MCP_TIMEOUT_SECS):
            async with streamablehttp_client(url=url, headers=headers) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    tools = [_mcp_tool_to_anthropic(t) for t in result.tools]
                    logger.info("Listed %d MCP tools", len(tools))
                    return tools
    except Exception:
        logger.exception("Failed to list MCP tools via Streamable HTTP, trying SSE")

    # Fallback: SSE transport
    from mcp.client.sse import sse_client

    sse_url = config.MCP_SERVER_URL.rstrip("/") + "/sse"
    try:
        async with asyncio.timeout(_MCP_TIMEOUT_SECS):
            async with sse_client(url=sse_url, headers=headers) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    tools = [_mcp_tool_to_anthropic(t) for t in result.tools]
                    logger.info("Listed %d MCP tools (SSE fallback)", len(tools))
                    return tools
    except Exception:
        logger.exception("Failed to list MCP tools via SSE as well")
        raise


async def call_mcp_tool(
    access_token: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Execute a single tool call on the MCP server and return the result as a string."""
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession

    url = _mcp_url()
    headers = _auth_headers(access_token)

    try:
        async with asyncio.timeout(_MCP_TIMEOUT_SECS):
            async with streamablehttp_client(url=url, headers=headers) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments or {})
                    return _format_tool_result(result)
    except Exception:
        logger.exception(
            "Streamable HTTP call failed for tool %s, trying SSE", tool_name
        )

    # Fallback: SSE transport
    from mcp.client.sse import sse_client

    sse_url = config.MCP_SERVER_URL.rstrip("/") + "/sse"
    try:
        async with asyncio.timeout(_MCP_TIMEOUT_SECS):
            async with sse_client(url=sse_url, headers=headers) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments or {})
                    return _format_tool_result(result)
    except TimeoutError:
        logger.error("MCP tool %s timed out after %ds", tool_name, _MCP_TIMEOUT_SECS)
        raise
    except Exception:
        logger.exception("SSE fallback also failed for tool %s", tool_name)
        raise


def _format_tool_result(result) -> str:
    """Serialize an MCP CallToolResult to a string suitable for Claude."""
    if result.isError:
        return f"[MCP Error] {_content_to_str(result.content)}"
    return _content_to_str(result.content)


def _content_to_str(content) -> str:
    """Flatten MCP content list into a string."""
    parts: list[str] = []
    for item in content:
        if hasattr(item, "text"):
            parts.append(item.text)
        else:
            parts.append(json.dumps(item.model_dump(), default=str))
    return "\n".join(parts) if parts else "(empty response)"


async def test_connection(access_token: str) -> bool:
    """Quick connectivity check — tries to list tools."""
    try:
        tools = await list_mcp_tools(access_token)
        return len(tools) > 0
    except Exception:
        return False
