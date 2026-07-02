"""
Conversation Manager.

Top-level orchestrator matching the architecture diagram:

    User -> Conversation Manager -> {Conversation Memory, LLM Planner}
         -> Tool Decision Engine -> {Metadata, FAISS, Compare tools}
         -> Recommendation Agent -> Response Generator -> User

The public assignment API is stateless: each /chat call supplies the full
conversation history. `handle_history()` rebuilds the useful transient state
from that history, while `handle_turn()` remains for local/debug session use.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.core.catalog import Catalog
from app.core.llm_client import LLMClient, get_llm_client
from app.core.memory import ConversationMemory, SessionState
from app.core.planner import LLMPlanner
from app.core.recommendation_agent import RecommendationAgent
from app.core.response_generator import ResponseGenerator
from app.core.tool_engine import ToolDecisionEngine, ToolRequest
from app.tools.faiss_tool import FaissTool


@dataclass
class TurnResult:
    session_id: str
    reply: str
    recommendations: list[dict] | None
    end_of_conversation: bool


class ConversationManager:
    def __init__(
        self,
        catalog: Catalog | None = None,
        llm: LLMClient | None = None,
        memory: ConversationMemory | None = None,
        faiss_tool: FaissTool | None = None,
    ):
        self.catalog = catalog or Catalog()
        self.llm = llm or get_llm_client()
        self.memory = memory or ConversationMemory()

        faiss_tool = faiss_tool or FaissTool(self.catalog)
        self.tool_engine = ToolDecisionEngine(self.catalog, faiss_tool=faiss_tool)
        self.planner = LLMPlanner(self.llm)
        self.recommendation_agent = RecommendationAgent(self.catalog)
        self.response_generator = ResponseGenerator(self.llm)

    def handle_history(self, messages: list[dict[str, Any]]) -> TurnResult:
        """Handle the assignment's stateless chat contract.

        The evaluator sends the full history on every call. We rebuild a
        temporary SessionState from all earlier turns, recover the last shown
        shortlist from catalog URLs in assistant messages, then process the
        final user message without saving anything server-side.
        """
        if not messages:
            raise ValueError("messages must contain at least one user message")
        last = messages[-1]
        if last.get("role") != "user" or not str(last.get("content", "")).strip():
            raise ValueError("the final message must be a non-empty user message")

        state = SessionState(session_id="stateless")
        for msg in messages[:-1]:
            role = msg.get("role")
            content = str(msg.get("content", ""))
            if role == "user":
                state.add_user_turn(content)
                self._absorb_explicit_exclusions(state, content)
            elif role == "assistant":
                rows = self._rows_from_assistant_content(content)
                state.add_agent_turn(content, recommendations=rows or None)
                if rows:
                    state.last_recommendation_ids = [r["entity_id"] for r in rows if r.get("entity_id")]
            else:
                raise ValueError("message roles must be 'user' or 'assistant'")

        # Preserve the last shortlist for confirmation turns even when the
        # assistant message did not include a table or the history is stateless.
        if state.last_recommendation_ids:
            state.last_recommendation_ids = [eid for eid in state.last_recommendation_ids if eid]

        return self._handle_state_turn(state, str(last["content"]), persist=False)

    def handle_turn(self, session_id: str | None, user_message: str) -> TurnResult:
        state = self.memory.get_or_create(session_id)
        return self._handle_state_turn(state, user_message, persist=True)

    def _handle_state_turn(self, state: SessionState, user_message: str, persist: bool) -> TurnResult:
        state.add_user_turn(user_message)

        self._absorb_explicit_exclusions(state, user_message)

        decision = self.planner.decide(state, user_message)
        state.slots.update(decision.updated_slots)

        if decision.action == "clarify":
            reply = self.response_generator.for_clarify(decision.clarifying_question or "Could you tell me more?")
            state.add_agent_turn(reply, recommendations=None)
            self._save_if_needed(state, persist)
            return TurnResult(state.session_id, reply, None, end_of_conversation=False)

        if decision.action == "refuse":
            reply = self.response_generator.for_refuse(
                decision.refusal_note or "I can only help with SHL assessment recommendations."
            )
            state.add_agent_turn(reply, recommendations=None)
            self._save_if_needed(state, persist)
            return TurnResult(state.session_id, reply, None, end_of_conversation=False)

        if decision.action == "compare":
            tool_result = self.tool_engine.run(
                [ToolRequest(tool="compare_products", args={"product_names": decision.compare_names})]
            )[0]
            reply = self.response_generator.for_compare(state, user_message, tool_result.comparison)
            state.add_agent_turn(reply, recommendations=None)
            self._save_if_needed(state, persist)
            return TurnResult(state.session_id, reply, None, end_of_conversation=False)

        if decision.action == "finalize":
            recommendation = self.recommendation_agent.carry_forward(state)
            if not recommendation.products:
                last_rows = state.last_recommendations() or []
                if last_rows:
                    product_ids = [row.get("entity_id") for row in last_rows if row.get("entity_id")]
                    products = [self.catalog.get_by_id(pid) for pid in product_ids if self.catalog.get_by_id(pid)]
                    recommendation = Recommendation(recommendation.products or products)
            prose, full = self.response_generator.for_recommendation(
                state, user_message, recommendation, decision.reasoning, [], finalized=True
            )
            rows = recommendation.to_table_rows()
            state.add_agent_turn(full, recommendations=rows)
            state.ended = True
            self._save_if_needed(state, persist)
            return TurnResult(state.session_id, full, rows, end_of_conversation=True)

        # action == "tool_call"
        tool_results = self.tool_engine.run(decision.tool_requests, exclude_ids=state.excluded_ids)
        recommendation = self.recommendation_agent.build(
            tool_results, decision.default_additions, state
        )
        prose, full = self.response_generator.for_recommendation(
            state, user_message, recommendation, decision.reasoning, decision.default_additions, finalized=False
        )
        rows = recommendation.to_table_rows()
        state.add_agent_turn(full, recommendations=rows)
        state.last_recommendation_ids = recommendation.entity_ids()
        self._save_if_needed(state, persist)
        return TurnResult(state.session_id, full, rows, end_of_conversation=False)

    def _save_if_needed(self, state: SessionState, persist: bool) -> None:
        if persist:
            self.memory.save(state)

    def _rows_from_assistant_content(self, content: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        urls = re.findall(r"https://www\.shl\.com/products/product-catalog/view/[^\s)>|]+", content)
        for url in urls:
            normalized = url.rstrip("./") + "/"
            product = next((p for p in self.catalog.products if p.link.rstrip("/") + "/" == normalized), None)
            if not product or product.entity_id in seen:
                continue
            row = product.to_row(len(rows) + 1)
            row["entity_id"] = product.entity_id
            rows.append(row)
            seen.add(product.entity_id)

        if rows:
            return rows

        table_lines = [line.strip() for line in content.splitlines() if "|" in line]
        for line in table_lines:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            if cells[0].startswith("#") or cells[0].lower() in {"#", "---", "name"}:
                continue
            if re.fullmatch(r"\d+", cells[0].strip()) is None:
                continue
            name = cells[1].strip()
            if not name or name.lower() in {"name", "test type", "keys", "duration", "languages", "url"}:
                continue
            product = self.catalog.get_by_name(name)
            if not product or product.entity_id in seen:
                continue
            row = product.to_row(len(rows) + 1)
            row["entity_id"] = product.entity_id
            rows.append(row)
            seen.add(product.entity_id)

        return rows

    def _absorb_explicit_exclusions(self, state: SessionState, user_message: str) -> None:
        """Cheap heuristic safety net for turns like 'drop/remove/exclude X'."""
        msg = user_message.lower()
        if not any(kw in msg for kw in ("drop ", "remove ", "exclude ", "without ")):
            return

        last = state.last_recommendations() or []
        for row in last:
            name = row.get("name", "")
            if not name:
                continue
            candidates = {name.lower(), name.lower().split(" (")[0]}
            candidates |= {m.lower() for m in re.findall(r"\(([^)]+)\)", name)}
            if any(c in msg for c in candidates if len(c) > 2):
                product = self.catalog.get_by_name(name)
                if product:
                    state.excluded_ids.add(product.entity_id)


from app.core.planner import LLMPlanner


