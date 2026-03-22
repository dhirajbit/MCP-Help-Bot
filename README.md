# MCP Help Bot

A generalized Slack bot that connects to **any MCP server** with a **Notion knowledge base** for customer support. Users can ask questions (answered via RAG from your Notion docs) or connect their workspace via OAuth to use MCP tools directly from Slack.

## Features

- **RAG-powered Q&A** — Syncs articles from a Notion database and answers questions using BM25 search + Claude
- **MCP tool integration** — Connected users can execute MCP tools (search leads, manage workflows, etc.) directly from Slack
- **OAuth 2.1 + PKCE** — Secure workspace connection flow with automatic token refresh
- **Channel & DM support** — Works in DMs and channel @mentions with per-channel connections
- **Conversation history** — Maintains context for follow-up questions
- **Daily auto-sync** — Keeps the knowledge base fresh with scheduled Notion syncs

## Quick Start

### 1. Create a Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app
2. Use the `manifest.yaml` in this repo as a starting point (update the display name)
3. Install the app to your workspace
4. Copy the **Bot Token** (`xoxb-...`) and **App-Level Token** (`xapp-...`)

### 2. Set Up Notion

1. Create a [Notion integration](https://www.notion.so/my-integrations)
2. Share your knowledge base database with the integration
3. Copy the **Integration Token** and **Database ID**

Your Notion database should have this structure:
```
Top-level Database
  └── Collection Pages (with optional "Published" checkbox)
       └── Child Databases or Child Pages (articles)
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

| Variable | Required | Description |
|----------|----------|-------------|
| `SLACK_BOT_TOKEN` | Yes | Slack bot token (`xoxb-...`) |
| `SLACK_APP_TOKEN` | Yes | Slack app-level token (`xapp-...`) |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `NOTION_API_KEY` | Yes | Notion integration token |
| `NOTION_DATABASE_ID` | Yes | Notion database ID for your knowledge base |
| `MCP_SERVER_URL` | Yes | Your MCP server URL (e.g. `https://mcp.yourproduct.com`) |
| `BOT_NAME` | No | Bot display name (default: `Help Bot`) |
| `PRODUCT_NAME` | No | Your product name used in messages (default: `your product`) |
| `OAUTH_CLIENT_ID` | No | OAuth client ID (default: `claude`) |
| `OAUTH_REDIRECT_URI` | No | OAuth redirect URI (default: `https://claude.ai/oauth/callback`) |
| `OAUTH_SCOPES` | No | OAuth scopes (default: `mcp.read mcp.write`) |
| `MCP_HELP_URL` | No | Link to setup docs shown during OAuth flow |
| `CLAUDE_MODEL` | No | Claude model to use (default: `claude-sonnet-4-5-20250929`) |
| `RAG_TOP_K` | No | Number of chunks to retrieve (default: `8`) |

### 4. Run

```bash
pip install -r requirements.txt
python main.py
```

## How It Works

1. **On startup**: Fetches all articles from your Notion database, chunks them, and builds a BM25 search index
2. **DM questions**: Users DM the bot with questions — it searches the knowledge base and answers using Claude
3. **MCP connection**: Users type `connect` to link their workspace via OAuth. Once connected, questions are routed to the MCP server's tools with knowledge base context injected
4. **Channel support**: @mention the bot in a channel. Admins can `connect` a channel to share one MCP connection with everyone in that channel

## Slack Commands

| Command | Description |
|---------|-------------|
| `connect` | Start OAuth flow to link your workspace |
| `disconnect` | Remove workspace connection |
| `status` | Check connection status |
| `cancel` | Cancel an in-progress connection |
| `help` | Show available commands |

## Deployment

### Render

Use the included `render.yaml` for one-click deployment on [Render](https://render.com). Set the environment variables in the Render dashboard.

### Docker

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

## Architecture

```
main.py                  # Entry point — sync + scheduler + bot
config.py                # All configuration from env vars
auth/
  oauth.py               # OAuth 2.1 + PKCE helpers
  token_store.py          # SQLite persistence for tokens
  flow_state.py           # In-memory OAuth flow state machine
bot/
  slack_handler.py        # Slack event handlers and routing
  rag.py                  # RAG pipeline (BM25 search + Claude)
  mcp_handler.py          # Claude agent loop with MCP tools
  chat_history.py         # Per-user conversation history
mcp_client/
  client.py               # Async MCP client (Streamable HTTP + SSE)
sync/
  notion_sync.py          # Notion database crawler
  embedder.py             # BM25 search index
```

## License

MIT
