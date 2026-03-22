"""Fetch all articles from a Notion knowledge base database.

Structure:
  Top-level DB  ->  Collection pages  ->  each contains a child_database  ->  articles
  (some collections may also have direct child_page blocks)

We find every child_database and child_page inside each collection,
query the sub-databases to get article pages, then recursively extract text.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from notion_client import Client, APIResponseError

import config

logger = logging.getLogger(__name__)

notion = Client(auth=config.NOTION_API_KEY)


# -- Retry helper for Notion 429 / 5xx errors --

_MAX_RETRIES = 5
_BASE_DELAY = 1.0  # seconds


def _notion_retry(fn, *args, **kwargs):
    """Call *fn* with retry + exponential backoff on 429 and 5xx errors."""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except APIResponseError as exc:
            status = getattr(exc, "status", 0)
            if status == 429 or status >= 500:
                if attempt == _MAX_RETRIES:
                    raise
                delay = _BASE_DELAY * (2 ** attempt)
                retry_after = getattr(exc, "headers", {}).get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except (ValueError, TypeError):
                        pass
                logger.warning(
                    "Notion %d — retrying in %.1fs (attempt %d/%d)",
                    status, delay, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(delay)
            else:
                raise


@dataclass
class Article:
    id: str
    title: str
    collection: str
    content: str
    url: str


# -- Block text extraction --


def _rich_text_to_str(rich_texts: list[dict]) -> str:
    """Concatenate Notion rich-text objects into a plain string."""
    return "".join(rt.get("plain_text", "") for rt in rich_texts)


def _extract_block_text(block: dict) -> str:
    """Return the plain-text content of a single Notion block."""
    btype = block.get("type", "")
    data = block.get(btype, {})

    if "rich_text" in data:
        return _rich_text_to_str(data["rich_text"])
    if btype == "child_page":
        return data.get("title", "")
    if btype == "child_database":
        return data.get("title", "")
    if btype == "table_row":
        cells = data.get("cells", [])
        cell_texts = [_rich_text_to_str(cell) for cell in cells]
        return " | ".join(cell_texts)
    return ""


# -- Recursive block fetcher --


def _fetch_blocks_text(block_id: str, depth: int = 0) -> str:
    """Recursively fetch all child blocks and return concatenated text."""
    if depth > 5:
        return ""

    texts: list[str] = []
    cursor: Optional[str] = None

    while True:
        kwargs: dict = {"block_id": block_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor

        response = _notion_retry(notion.blocks.children.list, **kwargs)

        for block in response.get("results", []):
            line = _extract_block_text(block)
            if line:
                texts.append(line)

            if block.get("has_children") and block.get("type") != "child_database":
                child_text = _fetch_blocks_text(block["id"], depth + 1)
                if child_text:
                    texts.append(child_text)

        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")

    return "\n".join(texts)


# -- Helpers --


def _page_title(page: dict) -> str:
    """Extract the title from a Notion page object."""
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            return _rich_text_to_str(prop.get("title", []))
    return "Untitled"


def _page_url(page: dict) -> str:
    return page.get("url", "")


def _is_published(page: dict) -> bool:
    """Check if a Notion page has Published = True."""
    props = page.get("properties", {})
    published = props.get("Published", {})
    if published.get("type") == "checkbox":
        return published.get("checkbox", False) is True
    return True  # if no Published property, assume published


def _find_child_databases(parent_page_id: str) -> list:
    """Return IDs of all child_database blocks under a page."""
    db_ids: list = []
    cursor: Optional[str] = None

    while True:
        kwargs: dict = {"block_id": parent_page_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor

        response = _notion_retry(notion.blocks.children.list, **kwargs)

        for block in response.get("results", []):
            if block.get("type") == "child_database":
                db_ids.append(block["id"])
            elif block.get("type") == "child_page":
                db_ids.append(("child_page", block["id"], block.get("child_page", {}).get("title", "Untitled")))

        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")

    return db_ids


def _query_all_pages_in_database(database_id: str) -> list[dict]:
    """Query a Notion database and return all page objects."""
    pages: list[dict] = []
    cursor: Optional[str] = None

    while True:
        kwargs: dict = {"database_id": database_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor

        response = _notion_retry(notion.databases.query, **kwargs)
        pages.extend(response.get("results", []))

        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")

    return pages


# -- Public API --


def fetch_all_articles() -> list[Article]:
    """Query the knowledge base database, walk every collection, return articles."""
    logger.info("Starting Notion sync — querying database %s", config.NOTION_DATABASE_ID)

    articles: list[Article] = []
    cursor: Optional[str] = None

    # 1. Get all collection pages from the top-level database
    while True:
        kwargs: dict = {"database_id": config.NOTION_DATABASE_ID, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor

        response = _notion_retry(notion.databases.query, **kwargs)

        for collection_page in response.get("results", []):
            collection_name = _page_title(collection_page)
            collection_id = collection_page["id"]

            if not _is_published(collection_page):
                logger.info("  Collection: %s (SKIPPED — not published)", collection_name)
                continue

            logger.info("  Collection: %s", collection_name)

            # 2. Find child_databases and child_pages inside this collection
            children = _find_child_databases(collection_id)

            for child in children:
                # Handle direct child_page blocks
                if isinstance(child, tuple) and child[0] == "child_page":
                    _, article_id, article_title = child

                    try:
                        page_obj = _notion_retry(notion.pages.retrieve, page_id=article_id)
                        url = _page_url(page_obj)
                    except Exception:
                        url = f"https://notion.so/{article_id.replace('-', '')}"

                    content = _fetch_blocks_text(article_id)
                    if content.strip():
                        articles.append(Article(
                            id=article_id, title=article_title,
                            collection=collection_name, content=content, url=url,
                        ))
                        logger.info("    Article (child_page): %s (%d chars)", article_title, len(content))
                    continue

                # Handle child_database blocks
                sub_db_id = child
                try:
                    sub_pages = _query_all_pages_in_database(sub_db_id)
                except Exception:
                    logger.warning("    Failed to query child database %s — skipping", sub_db_id)
                    continue

                logger.info("    Sub-database %s: %d pages", sub_db_id[:8], len(sub_pages))

                for article_page in sub_pages:
                    if not _is_published(article_page):
                        continue

                    article_id = article_page["id"]
                    article_title = _page_title(article_page)
                    url = _page_url(article_page)

                    content = _fetch_blocks_text(article_id)

                    if not content.strip():
                        logger.debug("    Skipping empty article: %s", article_title)
                        continue

                    articles.append(Article(
                        id=article_id, title=article_title,
                        collection=collection_name, content=content, url=url,
                    ))
                    logger.info("    Article: %s (%d chars)", article_title, len(content))

        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")

    logger.info("Notion sync complete — %d articles fetched", len(articles))
    return articles
