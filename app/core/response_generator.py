"""
Response Generator.

Turns the Recommendation Agent's product list (or a Compare/clarify/refuse
decision) into the final user-facing reply: short conversational prose +
a markdown table, matching the format used throughout the example
transcripts. The LLM only writes the prose; the table is rendered
deterministically from Product data so numbers/URLs are never hallucinated.
"""
from __future__ import annotations

from app.core.llm_client import LLMClient
from app.core.memory import SessionState
from app.core.recommendation_agent import Recommendation
from app.tools.compare_tool import ComparisonResult

RESPONSE_SYSTEM_PROMPT = """You are the user-facing voice of an SHL assessment \
recommendation agent. Write like a knowledgeable SHL solutions consultant in \
the sample conversations: warm, direct, practical, and human. You write ONLY \
the short conversational prose that precedes or follows a recommendation \
table; product names, durations, languages, test types, and URLs come from \
the data verbatim.

Tone and style:
- Sound like a helpful consultant, not a scripted bot.
- Keep the phrasing warm, conversational, and natural, with short sentences
  that feel easy to read.
- Acknowledge the user's situation plainly and move the conversation forward
  without sounding stiff or overly formal.
- Anchor the answer to the user’s specific context, role, constraints, and
  intent rather than giving a generic response.

Style rules:
- Keep prose to 1-4 sentences.
- Start by acknowledging the user's situation or latest change when useful \
  ("Understood", "Updated", "For this role", "Good two-stage design").
- Explain the reason for the stack in plain language, especially tradeoffs \
  like knowledge vs. simulation, technical skill vs. reasoning, or instrument \
  vs. report.
- Prefer natural, human-friendly phrasing over robotic checklist language.
- If a catalog constraint matters, say it plainly instead of pretending a \
  perfect test exists.
- If OPQ32r or another default is included, mention the assumption briefly.
- On final confirmation, use concise confirmation language and summarize the \
  battery's purpose.
- Do not invent catalog facts. Do not repeat every table row in prose.
"""


def render_table(rows: list[dict]) -> str:
    if not rows:
        return "_No matching products found in the catalog for this request._"
    header = "| # | Name | Test Type | Keys | Duration | Languages | URL |"
    sep = "|---|------|-----------|------|----------|-----------|-----|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['#']} | {r['name']} | {r['test_type']} | {r['keys']} | "
            f"{r['duration']} | {r['languages']} | <{r['url']}> |"
        )
    return "\n".join(lines)


class ResponseGenerator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def for_clarify(self, question: str) -> str:
        return question

    def for_refuse(self, note: str) -> str:
        return note

    def for_recommendation(
        self,
        state: SessionState,
        user_message: str,
        recommendation: Recommendation,
        planner_reasoning: str,
        default_additions: list[str],
        finalized: bool,
    ) -> tuple[str, str]:
        """Returns (prose, full_markdown_with_table)."""
        rows = recommendation.to_table_rows()
        context = (
            f"User's latest message: {user_message}\n"
            f"Planner reasoning: {planner_reasoning}\n"
            f"Products being recommended (name â€” keys): "
            + "; ".join(f"{p.name} ({', '.join(p.keys)})" for p in recommendation.products)
            + "\n"
            f"Default additions applied this turn: {default_additions or 'none'}\n"
            f"Is this the final/confirmed shortlist: {finalized}\n"
        )
        messages = state.history_as_messages() + [{"role": "user", "content": context}]
        resp = self.llm.complete(RESPONSE_SYSTEM_PROMPT, messages, max_tokens=300, temperature=0.4)
        prose = resp.text.strip()
        table = render_table(rows)
        full = f"{prose}\n\n{table}" if prose else table
        return prose, full

    def for_compare(self, state: SessionState, user_message: str, comparison: ComparisonResult) -> str:
        field_lines = []
        for f in comparison.fields:
            vals = "; ".join(f"{name}: {val}" for name, val in f.values.items())
            field_lines.append(f"- {f.label} â€” {vals}")
        context = (
            f"User's latest message: {user_message}\n"
            f"Structured comparison data (do not invent anything beyond this):\n"
            + "\n".join(field_lines)
            + (f"\nNot found in catalog: {comparison.not_found}" if comparison.not_found else "")
        )
        messages = state.history_as_messages() + [{"role": "user", "content": context}]
        resp = self.llm.complete(
            RESPONSE_SYSTEM_PROMPT
            + " For comparisons, explain the practical difference in 2-4 sentences; "
              "no table needed.",
            messages,
            max_tokens=350,
            temperature=0.4,
        )
        return resp.text.strip()

