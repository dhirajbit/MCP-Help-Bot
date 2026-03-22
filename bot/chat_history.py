"""Per-user in-memory chat history for DM conversations.

Stores the last N exchanges so Claude has context of what was just
discussed — prevents hallucination and enables follow-up questions.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MAX_MESSAGES = 40  # 20 user + 20 assistant turns
_DEFAULT_TTL_SECONDS = 30 * 60  # 30 minutes of inactivity


@dataclasses.dataclass
class _ConversationEntry:
    messages: list[dict]
    last_activity: float


class ChatHistoryManager:
    """Thread-safe (enough) per-user conversation history."""

    def __init__(
        self,
        max_messages: int = _DEFAULT_MAX_MESSAGES,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._histories: dict[str, _ConversationEntry] = {}
        self._max_messages = max_messages
        self._ttl_seconds = ttl_seconds

    def get_history(self, user_id: str) -> list[dict]:
        entry = self._histories.get(user_id)
        if not entry:
            return []

        if time.time() - entry.last_activity > self._ttl_seconds:
            self._histories.pop(user_id, None)
            return []

        return list(entry.messages)

    def add_user_message(self, user_id: str, text: str) -> None:
        self._ensure_entry(user_id)
        entry = self._histories[user_id]
        entry.messages.append({"role": "user", "content": text})
        entry.last_activity = time.time()
        self._trim(entry)

    def add_assistant_message(self, user_id: str, text: str) -> None:
        self._ensure_entry(user_id)
        entry = self._histories[user_id]
        entry.messages.append({"role": "assistant", "content": text})
        entry.last_activity = time.time()
        self._trim(entry)

    def clear(self, user_id: str) -> None:
        self._histories.pop(user_id, None)

    def _ensure_entry(self, user_id: str) -> None:
        if user_id not in self._histories:
            self._histories[user_id] = _ConversationEntry(
                messages=[], last_activity=time.time()
            )
        else:
            entry = self._histories[user_id]
            if time.time() - entry.last_activity > self._ttl_seconds:
                self._histories[user_id] = _ConversationEntry(
                    messages=[], last_activity=time.time()
                )

    def _trim(self, entry: _ConversationEntry) -> None:
        if len(entry.messages) > self._max_messages:
            excess = len(entry.messages) - self._max_messages
            entry.messages = entry.messages[excess:]
            while entry.messages and entry.messages[0]["role"] != "user":
                entry.messages.pop(0)
