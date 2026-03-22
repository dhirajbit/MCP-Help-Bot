"""In-memory state machine for the multi-step OAuth conversation.

Each Slack user goes through:
    AWAITING_CLIENT_SECRET  ->  AWAITING_AUTH_CODE
before the flow completes.

Temporary data (secret, PKCE verifier/challenge, state nonce,
discovered endpoints) is held here until the exchange succeeds, then
persisted via :mod:`auth.token_store`.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Optional

import config

# Default OAuth client ID — configurable via OAUTH_CLIENT_ID env var
DEFAULT_CLIENT_ID = config.OAUTH_CLIENT_ID


class FlowStep(str, Enum):
    AWAITING_CLIENT_SECRET = "AWAITING_CLIENT_SECRET"
    AWAITING_AUTH_CODE = "AWAITING_AUTH_CODE"


@dataclasses.dataclass
class _PendingFlow:
    step: FlowStep
    client_id: str = DEFAULT_CLIENT_ID
    client_secret: str = ""
    code_verifier: str = ""
    code_challenge: str = ""
    state_nonce: str = ""
    authorization_endpoint: str = ""
    token_endpoint: str = ""
    target_channel_id: str = ""  # non-empty = channel flow, empty = personal DM flow


class OAuthFlowManager:
    """Thread-safe (enough) manager for in-progress OAuth flows.

    One instance lives at module-level in ``slack_handler.py``.
    """

    def __init__(self) -> None:
        self._flows: dict[str, _PendingFlow] = {}

    # -- queries --

    def get_state(self, user_id: str) -> Optional[FlowStep]:
        flow = self._flows.get(user_id)
        return flow.step if flow else None

    def get_pending(self, user_id: str) -> dict:
        flow = self._flows.get(user_id)
        if not flow:
            return {}
        return dataclasses.asdict(flow)

    # -- mutations --

    def start_flow(self, user_id: str, target_channel_id: str = "") -> None:
        """Start a new flow — goes straight to AWAITING_CLIENT_SECRET."""
        self._flows[user_id] = _PendingFlow(
            step=FlowStep.AWAITING_CLIENT_SECRET,
            client_id=DEFAULT_CLIENT_ID,
            target_channel_id=target_channel_id,
        )

    def set_client_secret(
        self,
        user_id: str,
        client_secret: str,
        *,
        code_verifier: str = "",
        code_challenge: str = "",
        state_nonce: str = "",
        authorization_endpoint: str = "",
        token_endpoint: str = "",
    ) -> None:
        flow = self._flows.get(user_id)
        if flow:
            flow.client_secret = client_secret
            flow.code_verifier = code_verifier
            flow.code_challenge = code_challenge
            flow.state_nonce = state_nonce
            flow.authorization_endpoint = authorization_endpoint
            flow.token_endpoint = token_endpoint
            flow.step = FlowStep.AWAITING_AUTH_CODE

    def complete(self, user_id: str) -> None:
        self._flows.pop(user_id, None)

    def cancel(self, user_id: str) -> None:
        self._flows.pop(user_id, None)
