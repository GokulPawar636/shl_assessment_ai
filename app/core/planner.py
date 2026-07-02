"""
LLM Planner.

Given conversation memory + the new user message, decides ONE of:
  - "clarify"  -> ask a single clarifying question, no tool calls, no
                  recommendations (mirrors C1 turn 1-2, C9 turn 1-2, C3 turn 1-2)
  - "tool_call"-> emit 1+ ToolRequests for the Tool Decision Engine
  - "compare"  -> a compare_products tool call answering a "what's the
                  difference between X and Y" question (C3 t4, C5 t2, C6 t2)
  - "refuse"   -> out-of-scope (legal/compliance) question; answer directly
                  without tools or recommendations (C7 turn 3)
  - "finalize" -> user confirmed/locked the shortlist; re-render the last
                  recommendation set as final, end_of_conversation=True
                  (seen at the end of every example transcript)

This module owns the *decision*, not catalog access â€” it never reads the
catalog directly, only reasons over the conversation and hands structured
tool requests to the engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.core.catalog import KEY_SHORT_CODE, KNOWN_JOB_LEVELS, KNOWN_KEYS
from app.core.llm_client import LLMClient
from app.core.memory import SessionState
from app.core.tool_engine import ToolRequest

Action = Literal["clarify", "tool_call", "compare", "refuse", "finalize"]

PLANNER_SYSTEM_PROMPT = f"""You are the planning module of an SHL assessment recommendation \
agent. You do not talk to the user directly and you do not know the product \
catalog's contents â€” you only decide what should happen next in the \
conversation and hand off structured requests to tools that DO know the \
catalog.

Valid catalog "keys" (categories) you may filter on: {", ".join(KNOWN_KEYS)}
Valid "job_levels" you may filter on: {", ".join(KNOWN_JOB_LEVELS)}

Behavioral rules, learned from prior expert-handled conversations:
1. If the request is broad or the role/purpose is unclear (e.g. "we need a \
   solution for senior leadership" with no context on selection vs. \
   development, or a JD with multiple unclear priorities), choose "clarify" \
   and ask exactly ONE focused, human question. Do not stack multiple \
   questions. Good clarifying questions sound like the sample conversations: \
   "Who is this meant for?", "What language are the calls in?", "Is this a \
   backend-leaning role or balanced full-stack?", or "Is this for selection \
   or development feedback?" Keep the tone warm, conversational, and \
   consultative rather than rigid or survey-like.
2. Always tailor the recommendation to the user’s exact role, context, and \
   constraints. Read for specifics such as seniority, language needs, safety \
   requirements, volume, selection vs. development use, and any explicit \
   tradeoffs the user raises. Do not give a generic answer when the user has \
   provided concrete context.
2. Once purpose, level, and language (if relevant) are known, choose \
   "tool_call" and emit one or more tool requests:
   - "metadata_filter" for exact structured constraints (job level, \
     language, category, duration ceiling, adaptive).
   - "semantic_search" for fuzzy/topical asks (competencies, behavioural \
     traits, role themes) using a short natural-language query.
   You may request both in the same turn if the ask has both a hard \
   constraint and a fuzzy topic.
3. If the user asks how two or more NAMED products differ, or asks you to \
   justify/re-confirm a choice between named products, choose "compare" and \
   list the product names.
4. If the user asks a legal/compliance/regulatory question (e.g. "are we \
   legally required to...", "does this satisfy HIPAA law") choose "refuse" \
   â€” this agent recommends assessments, it does not give legal advice.
5. If the user is confirming, locking in, or accepting the current \
   shortlist ("that's what we need", "confirmed", "keep as-is", "locking \
   it in") with no further changes requested, choose "finalize".
6. Default recommendation: OPQ32r (Occupational Personality Questionnaire) \
   is commonly added as a personality baseline for hiring-decision \
   shortlists unless the user has explicitly excluded personality \
   assessments. Mention this as an assumption when you add it, don't \
   silently include it without flagging it to the user in your reasoning.
7. Respect explicit exclusions the user has stated earlier in the \
   conversation (e.g. "drop OPQ") â€” do not re-add excluded products unless \
   the user re-requests them.
8. Prefer the smallest number of tool calls that fully answers the turn. \
   Don't re-run tools for information already available in memory/slots.
9. Keep the conversation consultative, not questionnaire-like. Ask only for \
   information that changes the shortlist: role ownership, seniority, \
   language/accent, selection vs. development, volume/time constraints, or \
   must-have skills. If the user has already given enough, recommend instead \
   of asking another question. Use natural, human-friendly phrasing that \
   feels like a knowledgeable consultant helping the client think through \
   the decision. When the user provides a specific scenario, anchor the answer \
   to that scenario rather than repeating a generic template.

Return your decision as JSON with this shape:
{{
  "action": "clarify" | "tool_call" | "compare" | "refuse" | "finalize",
  "reasoning": "<one sentence, internal, not shown to user>",
  "clarifying_question": "<only if action=clarify>",
  "refusal_note": "<only if action=refuse â€” what to tell the user>",
  "tool_requests": [
     {{"tool": "metadata_filter", "args": {{...}}}},
     {{"tool": "semantic_search", "args": {{"query": "...", "top_k": 10}}}}
  ],
  "compare_names": ["<only if action=compare>"],
  "updated_slots": {{"role": "...", "job_level": "...", "language": "...", "purpose": "..."}},
  "default_additions": ["OPQ32r"]  // products added by default rule 6, empty list if none
}}
"""


@dataclass
class PlannerDecision:
    action: Action
    reasoning: str = ""
    clarifying_question: str | None = None
    refusal_note: str | None = None
    tool_requests: list[ToolRequest] = None
    compare_names: list[str] = None
    updated_slots: dict[str, Any] = None
    default_additions: list[str] = None

    def __post_init__(self):
        self.tool_requests = self.tool_requests or []
        self.compare_names = self.compare_names or []
        self.updated_slots = self.updated_slots or {}
        self.default_additions = self.default_additions or []




def fallback_clarifying_question(user_message: str, state: SessionState) -> str:
    """Human-friendly backup when the planner omits a clarifying question."""
    msg = user_message.lower()
    known = state.slots or {}

    if any(term in msg for term in ("leadership", "executive", "cxo", "director")):
        return "Is this for selecting new leaders, or for development feedback for leaders already in role?"
    if any(term in msg for term in ("developer", "engineer", "java", "python", "sql", "cloud", "aws")):
        return "What seniority level is this role, and which skills matter most for the shortlist?"
    if "assessment" in msg and not known.get("role"):
        return "What role are you hiring for, and what should the assessment help you measure?"
    if not known.get("purpose"):
        return "Is this for hiring selection, internal development, or benchmarking existing employees?"
    return "What role, seniority level, and must-have skills should I optimize the SHL shortlist for?"

class LLMPlanner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def decide(self, state: SessionState, user_message: str) -> PlannerDecision:
        context_lines = [
            f"Known slots so far: {state.slots or '{}'}",
            f"Products excluded by user so far: {sorted(state.excluded_ids) or '[]'}",
            f"Last recommendation set shown (entity_ids): {state.last_recommendation_ids or '[]'}",
        ]
        messages = state.history_as_messages()
        messages.append({
            "role": "user",
            "content": "\n".join(context_lines) + f"\n\nNew user message: {user_message}",
        })

        if self._looks_like_confirmation(user_message, state):
            return PlannerDecision(
                action="finalize",
                reasoning="User confirmed the current shortlist.",
                updated_slots={},
                default_additions=[],
            )

        try:
            raw = self.llm.complete_json(PLANNER_SYSTEM_PROMPT, messages, max_tokens=800)
        except Exception:
            raw = {}

        tool_requests = [
            ToolRequest(tool=t["tool"], args=t.get("args", {}))
            for t in raw.get("tool_requests", [])
        ]

        action = raw.get("action", "clarify")
        if action == "clarify" and self._should_finalize_from_context(state, user_message):
            action = "finalize"

        return PlannerDecision(
            action=action,
            reasoning=raw.get("reasoning", ""),
            clarifying_question=raw.get("clarifying_question") or raw.get("question") or fallback_clarifying_question(user_message, state),
            refusal_note=raw.get("refusal_note"),
            tool_requests=tool_requests,
            compare_names=raw.get("compare_names", []),
            updated_slots=raw.get("updated_slots", {}),
            default_additions=raw.get("default_additions", []),
        )

    def _looks_like_confirmation(self, user_message: str, state: SessionState) -> bool:
        msg = user_message.lower().strip()
        if any(term in msg for term in ("should i", "can i", "would you", "should we", "do we")):
            return False
        if not any(term in msg for term in ("perfect", "confirmed", "confirm", "keep as-is", "keep it", "that’s what we need", "that's what we need", "locking it in", "lock it in", "sounds good", "that works", "thanks", "thank you")):
            return False
        if msg.endswith("?"):
            return False
        return bool(state.last_recommendation_ids)

    def _should_finalize_from_context(self, state: SessionState, user_message: str) -> bool:
        msg = user_message.lower().strip()
        if any(term in msg for term in ("should i", "can i", "would you", "should we", "do we")):
            return False
        if any(term in msg for term in ("perfect", "confirmed", "confirm", "keep as-is", "that’s what we need", "that's what we need", "locking it in", "lock it in", "sounds good", "that works", "thanks", "thank you")):
            if msg.endswith("?"):
                return False
            return bool(state.last_recommendation_ids)
        return False

    def _has_prior_recommendation_context(self, state: SessionState) -> bool:
        for turn in state.turns:
            if turn.role != "assistant":
                continue
            content = turn.content.lower()
            if any(marker in content for marker in ("instrument", "report", "recommend", "shortlist", "battery", "assessment", "selection", "development", "benchmark")):
                return True
        return False


