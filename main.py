"""Entry point: init DB, run initial Notion sync, schedule daily re-sync, start Slack bot."""

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from auth.token_store import init_db
from sync.notion_sync import fetch_all_articles
from sync.embedder import rebuild_vector_store
from bot.slack_handler import start_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_sync():
    """Fetch articles from Notion and rebuild the search index."""
    try:
        articles = fetch_all_articles()
        count = rebuild_vector_store(articles)
        logger.info("Sync complete — %d chunks indexed", count)
    except Exception:
        logger.exception("Notion sync failed — will serve from last successful sync")


def main():
    # 0. Initialise auth database
    init_db()

    # 1. Initial sync on startup
    logger.info("Running initial Notion sync...")
    run_sync()

    # 2. Schedule daily sync at 2:00 AM UTC
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_sync, "cron", hour=2, minute=0, timezone="UTC")
    scheduler.start()
    logger.info("Scheduled daily sync at 02:00 UTC")

    # 3. Start the Slack bot (blocking call)
    try:
        start_bot()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
