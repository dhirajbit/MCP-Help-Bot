import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Required credentials ─────────────────────────────────────────────
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
NOTION_API_KEY = os.environ["NOTION_API_KEY"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]

# ── MCP configuration ────────────────────────────────────────────────
MCP_SERVER_URL = os.environ["MCP_SERVER_URL"]

# ── Branding / product identity ──────────────────────────────────────
# These customize the bot's personality — change them to match your product.
BOT_NAME = os.environ.get("BOT_NAME", "Help Bot")
PRODUCT_NAME = os.environ.get("PRODUCT_NAME", "your product")

# ── OAuth settings ───────────────────────────────────────────────────
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "claude")
OAUTH_REDIRECT_URI = os.environ.get(
    "OAUTH_REDIRECT_URI", "https://claude.ai/oauth/callback"
)
OAUTH_SCOPES = os.environ.get("OAUTH_SCOPES", "mcp.read mcp.write")

# Optional: URL to a help article explaining how to find the OAuth secret
MCP_HELP_URL = os.environ.get("MCP_HELP_URL", "")

# ── Persistent data directory ────────────────────────────────────────
# On Render, set DATA_DIR to the persistent disk mount (e.g. /var/data).
# Locally, defaults to the project directory.
_DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(__file__))

# Search index
SEARCH_INDEX_DIR = os.path.join(_DATA_DIR, "search_data")

# RAG settings
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "8"))
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

# SQLite token store
SQLITE_DB_PATH = os.path.join(_DATA_DIR, "data", "tokens.db")
