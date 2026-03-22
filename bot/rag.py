"""RAG pipeline: query BM25 index for relevant chunks and ask Claude."""

import logging
import re

import anthropic

import config
from sync.embedder import get_collection

logger = logging.getLogger(__name__)

_anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

# System prompt uses PRODUCT_NAME for branding
SYSTEM_PROMPT = f"""\
You are {config.BOT_NAME}, a help assistant for {config.PRODUCT_NAME}. \
Answer questions using the knowledge base content provided below.

RESPONSE STYLE:
* *Be concise.* Answer in 2-5 sentences. No walls of text.
* Lead with the direct answer, then give brief supporting details.
* For how-to questions, use a short numbered list of steps.
* Skip preamble like "Based on the documentation..." or "Great question!"

FORMATTING (Slack, NOT Markdown):
* *bold* with single asterisks. _italic_ with underscores.
* NEVER use ## headings or **double asterisks**.
* Use bullets for lists. <URL|text> for links.

RULES:
* Only use the provided knowledge base content — never make things up.
* If greeted, reply briefly and offer to help.
* If you don't know, say: "I don't have that in my docs — please contact \
support for help."
* End with one relevant article link if a URL is in the chunks.
* Keep total response under 200 words.\
"""


def _markdown_to_slack(text: str) -> str:
    """Convert Markdown formatting to Slack mrkdwn formatting."""
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    text = re.sub(r"^- ", "* ", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)
    return text


def _build_context(results: dict) -> str:
    """Format search results into a context string for Claude."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return ""

    sections: list[str] = []
    for doc, meta in zip(documents, metadatas):
        title = meta.get("article_title", "Unknown")
        collection = meta.get("collection", "")
        url = meta.get("url", "")
        header = f"[{collection} > {title}]" if collection else f"[{title}]"
        if url:
            header += f"\nURL: {url}"
        sections.append(f"{header}\n{doc}")

    return "\n\n---\n\n".join(sections)


def answer_question(
    question: str,
    history: list[dict] | None = None,
) -> str:
    """Run the full RAG pipeline: retrieve chunks, call Claude, return answer."""
    collection = get_collection()

    if collection.count() == 0:
        return (
            "The knowledge base hasn't been synced yet. "
            "Please try again in a few minutes."
        )

    results = collection.query(
        query_texts=[question],
        n_results=min(config.RAG_TOP_K, collection.count()),
    )

    context = _build_context(results)

    if not context:
        return (
            "I couldn't find any relevant content for your question. "
            "Please contact support for further help."
        )

    user_message = (
        f"KNOWLEDGE BASE CONTENT:\n\n{context}\n\n---\n\nUSER QUESTION: {question}"
    )

    logger.info("Calling Claude with %d context chunks", len(results.get("documents", [[]])[0]))

    messages: list[dict] = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = _anthropic_client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    text_parts = [b.text for b in response.content if hasattr(b, "text")]
    if not text_parts:
        return (
            "I wasn't able to generate an answer. "
            "Please try rephrasing, or contact support."
        )
    return _markdown_to_slack("\n".join(text_parts))
