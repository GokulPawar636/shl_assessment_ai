"""
Conversation Memory: per-session state.

Tracks turn history, confirmed/inferred filters (role, job level, language,
etc.), and the most recently shown recommendation set, so later turns
("add AWS and Docker", "drop OPQ", "keep the shortlist as-is") can be
resolved against prior context instead of starting from scratch.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal

Role = Literal["user", "assistant"]


@dataclass
class Turn:
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)
    # Recommendations attached to this turn, if the agent produced any.
    recommendations: list[dict[str, Any]] | None = None


@dataclass
class SessionState:
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    # Slots the planner has inferred/confirmed over the conversation, e.g.
    # {"role": "senior rust engineer", "job_level": "Professional Individual
    # Contributor", "language": "English (USA)", "purpose": "selection"}
    slots: dict[str, Any] = field(default_factory=dict)
    # Entity IDs of the last recommendation set shown to the user, in order.
    last_recommendation_ids: list[str] = field(default_factory=list)
    # Products explicitly excluded/dropped by the user during the conversation.
    excluded_ids: set[str] = field(default_factory=set)
    ended: bool = False
    created_at: float = field(default_factory=time.time)

    def add_user_turn(self, content: str) -> None:
        self.turns.append(Turn(role="user", content=content))

    def add_agent_turn(self, content: str, recommendations: list[dict[str, Any]] | None) -> None:
        self.turns.append(Turn(role="assistant", content=content, recommendations=recommendations))

    def history_as_messages(self) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def last_recommendations(self) -> list[dict[str, Any]] | None:
        for t in reversed(self.turns):
            if t.role == "assistant" and t.recommendations:
                return t.recommendations
        return None


class ConversationMemory:
    """Thread-safe in-memory session store.

    Swap out with a Redis/DB-backed implementation for production; the
    interface (get_or_create / save) is what the rest of the app depends on.
    """

    def __init__(self):
        self._sessions: dict[str, SessionState] = {}
        self._lock = Lock()

    def get_or_create(self, session_id: str | None) -> SessionState:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            new_id = session_id or str(uuid.uuid4())
            state = SessionState(session_id=new_id)
            self._sessions[new_id] = state
            return state

    def save(self, state: SessionState) -> None:
        with self._lock:
            self._sessions[state.session_id] = state

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

