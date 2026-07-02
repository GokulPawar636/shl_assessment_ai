"""
Swappable LLM client.

The Planner and Response Generator both depend on this interface, not on
any specific provider. Plug in Anthropic, OpenAI, Azure, a local vLLM
server, etc. by implementing `LLMClient` and wiring it up in `get_llm_client()`.

Set env vars to configure:
    LLM_PROVIDER = "anthropic" | "openai" | "groq" | "gemini" | "mock"
    LLM_API_KEY   = "<your key>"
    LLM_MODEL     = "<model name>"
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # idempotent; ensures LLM_PROVIDER/LLM_API_KEY/LLM_MODEL are available
# even if this module is used outside of app.main (e.g. a standalone script).


class LLMClient(ABC):
    """Minimal interface every provider adapter must satisfy."""

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        tools: list[dict[str, Any]] | None = None,
    ) -> "LLMResponse":
        """Run one completion. `messages` is [{"role": "user"/"assistant", "content": str}]."""
        raise NotImplementedError

    def complete_json(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Convenience wrapper: ask for strict JSON, parse it, retry once on failure."""
        json_system = (
            system
            + "\n\nRespond with ONLY a single valid JSON object. "
              "No markdown fences, no preamble, no trailing commentary."
        )
        resp = self.complete(json_system, messages, max_tokens=max_tokens, temperature=temperature)
        text = resp.text.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # one retry with an explicit correction nudge
            retry_messages = messages + [
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content": "That was not valid JSON. Return ONLY the JSON object, nothing else."},
            ]
            resp2 = self.complete(json_system, retry_messages, max_tokens=max_tokens, temperature=0.0)
            text2 = resp2.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(text2)


class LLMResponse:
    def __init__(self, text: str, tool_calls: list[dict[str, Any]] | None = None, raw: Any = None):
        self.text = text
        self.tool_calls = tool_calls or []
        self.raw = raw


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6"):
        import anthropic  # imported lazily so the package is optional

        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("LLM_API_KEY"))
        self._model = model

    def complete(self, system, messages, max_tokens=1024, temperature=0.2, tools=None) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        if tools:
            kwargs["tools"] = tools
        resp = self._client.messages.create(**kwargs)
        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        tool_calls = [
            {"name": b.name, "input": b.input, "id": b.id}
            for b in resp.content
            if getattr(b, "type", None) == "tool_use"
        ]
        return LLMResponse(text="\n".join(text_parts), tool_calls=tool_calls, raw=resp)


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        import openai  # imported lazily so the package is optional

        self._client = openai.OpenAI(api_key=api_key or os.environ.get("LLM_API_KEY"))
        self._model = model

    def complete(self, system, messages, max_tokens=1024, temperature=0.2, tools=None) -> LLMResponse:
        oa_messages = [{"role": "system", "content": system}]
        oa_messages += [{"role": m["role"], "content": m["content"]} for m in messages]
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=oa_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return LLMResponse(text=resp.choices[0].message.content or "", raw=resp)


class GroqClient(LLMClient):
    """Groq's API is OpenAI-compatible; uses the official groq SDK.
    Fast inference, good default models: llama-3.3-70b-versatile, etc."""

    def __init__(self, api_key: str | None = None, model: str = "llama-3.3-70b-versatile"):
        import groq  # imported lazily so the package is optional

        self._client = groq.Groq(api_key=api_key or os.environ.get("LLM_API_KEY"))
        self._model = model

    def complete(self, system, messages, max_tokens=1024, temperature=0.2, tools=None) -> LLMResponse:
        groq_messages = [{"role": "system", "content": system}]
        groq_messages += [{"role": m["role"], "content": m["content"]} for m in messages]
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=groq_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return LLMResponse(text=resp.choices[0].message.content or "", raw=resp)


class GeminiClient(LLMClient):
    """Google's Gemini API via the current google-genai SDK
    (google-generativeai is deprecated as of 2025)."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        from google import genai  # imported lazily so the package is optional

        self._client = genai.Client(api_key=api_key or os.environ.get("LLM_API_KEY"))
        self._model_name = model

    def complete(self, system, messages, max_tokens=1024, temperature=0.2, tools=None) -> LLMResponse:
        from google.genai import types

        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        resp = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return LLMResponse(text=resp.text or "", raw=resp)


class MockClient(LLMClient):
    """Deterministic offline stub for local dev / tests without any API key."""

    def complete(self, system, messages, max_tokens=1024, temperature=0.2, tools=None) -> LLMResponse:
        text = self._build_response_text(system, messages)
        return LLMResponse(text=text)

    def _build_response_text(self, system, messages) -> str:
        latest = (messages[-1].get("content", "") if messages else "").lower()
        history = "\n".join(m.get("content", "") for m in messages if m.get("content"))

        if "user-facing voice" in system.lower() or "short conversational prose" in system.lower():
            if "contact center" in history.lower() or "customer service" in history.lower() or "contact centre" in history.lower():
                return (
                    "For a high-volume entry-level contact-centre screening, the stack should combine a "
                    "spoken-language screen, a call simulation, and a behavioural fit measure."
                )
            if "leadership" in history.lower() or "executive" in history.lower() or "director" in history.lower():
                return "For senior leadership selection, a personality-based leadership profile is the right place to start."
            if "sales" in history.lower():
                return "This stack is built to balance skills, personality, and sales-specific behaviour for a re-skilling audit."
            return "This shortlist is designed around the role’s main hiring constraints and the evidence you need most."

        if any(term in latest for term in ("perfect", "confirmed", "confirm", "keep as-is", "that's what we need", "that’s what we need", "locking it in", "lock it in", "sounds good", "done")) and not latest.rstrip().endswith("?"):
            return "Good choice. The shortlist is now locked in."

        if "contact center" in latest or "contact centre" in latest or "customer service" in latest:
            if "english" in latest and "us" in latest:
                return "That makes sense for a US English contact-centre operation."
            return "Before I shape the stack, I’d want to know the call language and the accent variant that matches your customers."

        if "leadership" in latest or "executive" in latest or "cxo" in latest or "director" in latest:
            return "That depends on whether this is for selection or development feedback for leaders already in role."

        if any(term in latest for term in ("developer", "engineer", "java", "python", "sql", "aws")):
            return "For a senior technical role, I’d typically combine role-specific knowledge with a broader reasoning screen and a personality baseline."

        return "I’d recommend narrowing the shortlist around the role’s core requirements before we lock in a final battery."

    def complete_json(self, system, messages, max_tokens=1024, temperature=0.0):
        latest = (messages[-1].get("content", "") if messages else "").lower()
        history = "\n".join(m.get("content", "") for m in messages if m.get("content"))

        if any(term in latest for term in ("difference", "different from", "vs", "versus")):
            return {"action": "compare", "reasoning": "User asked for a product comparison.", "compare_names": ["Contact Center Call Simulation (New)", "Customer Service Phone Simulation"], "updated_slots": {}, "default_additions": []}

        if "rust" in latest or "rust" in history.lower():
            if any(term in latest for term in ("what assessments should i use", "what assessments should we use", "what should i use", "what should we use", "what assessments")):
                return {
                    "action": "clarify",
                    "reasoning": "Rust is a narrow catalog fit; confirm the closest available shortlist first.",
                    "clarifying_question": (
                        "SHL's catalog doesn't currently include a Rust-specific knowledge test. The closest fit for a senior IC is Smart Interview Live Coding — an adaptive live-coding interview where your panel can frame Rust-specific tasks directly. Linux Programming covers systems depth, and Networking and Implementation covers the infrastructure dimension. Want me to build a shortlist from these?"
                    ),
                    "updated_slots": {"role": "senior rust engineer", "purpose": "selection"},
                    "default_additions": [],
                }
            if "cognitive" in latest or "add a cognitive test" in latest or "also add" in latest:
                return {
                    "action": "tool_call",
                    "reasoning": "User approved the Rust shortlist and asked to include a cognitive assessment.",
                    "tool_requests": [
                        {"tool": "metadata_filter", "args": {"name_contains": "Smart Interview Live Coding"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Linux Programming (General)"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Networking and Implementation (New)"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "SHL Verify Interactive G+"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Occupational Personality Questionnaire OPQ32r"}},
                    ],
                    "updated_slots": {"role": "senior rust engineer", "purpose": "selection"},
                    "default_additions": [],
                }
            if any(term in latest for term in ("yes", "go ahead", "sure", "please", "ok", "okay")) and not latest.rstrip().endswith("?"):
                return {
                    "action": "tool_call",
                    "reasoning": "User confirmed the Rust shortlist, so recommend the closest fit products from the catalog.",
                    "tool_requests": [
                        {"tool": "metadata_filter", "args": {"name_contains": "Smart Interview Live Coding"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Linux Programming (General)"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Networking and Implementation (New)"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Occupational Personality Questionnaire OPQ32r"}},
                    ],
                    "updated_slots": {"role": "senior rust engineer", "purpose": "selection"},
                    "default_additions": [],
                }

        if "contact center" in history.lower() or "contact centre" in history.lower() or "customer service" in history.lower():
            if re.search(r"\b(?:us|uk|australian|indian)\b", latest) and ("english" in latest or "english" in history.lower()):
                return {
                    "action": "tool_call",
                    "reasoning": "Contact-centre screening with a specific English accent variant.",
                    "tool_requests": [
                        {"tool": "metadata_filter", "args": {"name_contains": "SVAR Spoken English (US) (New)"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Contact Center Call Simulation (New)"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Entry Level Customer Serv - Retail & Contact Center"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Customer Service Phone Simulation"}},
                    ],
                    "updated_slots": {"role": "contact centre agent", "purpose": "selection", "language": "English (USA)"},
                    "default_additions": [],
                }
            if "english" in latest:
                return {
                    "action": "clarify",
                    "reasoning": "Need the English accent variant to pick the right spoken-language screen.",
                    "clarifying_question": "SVAR has four English variants in the catalog: US, UK, Australian, and Indian accent. Which fits your operation?",
                    "updated_slots": {"role": "contact centre agent", "purpose": "selection"},
                    "default_additions": [],
                }
            return {
                "action": "clarify",
                "reasoning": "Need the call language before selecting a spoken-language screen.",
                "clarifying_question": "Before I shape the stack — what language are the calls in? That drives which spoken-language screen we use.",
                "updated_slots": {"role": "contact centre agent", "purpose": "selection"},
                "default_additions": [],
            }

        if "leadership" in latest or "executive" in latest or "cxo" in latest or "director" in latest:
            if ("selection" in latest or "comparing" in latest or "benchmark" in latest) and (
                "who is this meant for" in history.lower()
                or "is this for selecting new leaders" in history.lower()
                or "development feedback" in history.lower()
            ):
                return {
                    "action": "tool_call",
                    "reasoning": "Leadership benchmark selection has been clarified; recommend the OPQ leadership suite.",
                    "tool_requests": [
                        {"tool": "metadata_filter", "args": {"name_contains": "Occupational Personality Questionnaire OPQ32r"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "OPQ Universal Competency Report 2.0"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "OPQ Leadership Report"}},
                    ],
                    "updated_slots": {"role": "leadership", "purpose": "selection"},
                    "default_additions": [],
                }
            return {
                "action": "clarify",
                "reasoning": "Need to know whether the leadership assessment is for selection or development feedback.",
                "clarifying_question": "Is this for selecting new leaders, or for development feedback for leaders already in role?",
                "updated_slots": {"role": "leadership", "purpose": "selection"},
                "default_additions": [],
            }

        if "rust" in latest:
            if any(term in latest for term in ("what assessments should i use", "what assessments should we use", "what should i use", "what should we use", "what assessments")):
                return {
                    "action": "clarify",
                    "reasoning": "Rust is a niche technology in the catalog; confirm the shortlist before recommending close-fit products.",
                    "clarifying_question": (
                        "SHL's catalog doesn't currently include a Rust-specific knowledge test. "
                        "The closest fit for a senior IC is Smart Interview Live Coding — an adaptive live-coding interview where your panel can frame Rust-specific tasks directly. "
                        "Linux Programming covers systems depth, and Networking and Implementation covers the infrastructure dimension. "
                        "Want me to build a shortlist from these?"
                    ),
                    "updated_slots": {"role": "senior rust engineer", "purpose": "selection"},
                    "default_additions": [],
                }
            if any(term in latest for term in ("yes", "go ahead", "sure", "please", "ok", "okay")) and "cognitive" in latest:
                return {
                    "action": "tool_call",
                    "reasoning": "User approved the Rust shortlist and asked to add a cognitive assessment.",
                    "tool_requests": [
                        {"tool": "metadata_filter", "args": {"name_contains": "Smart Interview Live Coding"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Linux Programming (General)"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Networking and Implementation (New)"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "SHL Verify Interactive G+"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Occupational Personality Questionnaire OPQ32r"}},
                    ],
                    "updated_slots": {"role": "senior rust engineer", "purpose": "selection"},
                    "default_additions": [],
                }
            if any(term in latest for term in ("yes", "go ahead", "sure", "please", "ok", "okay")):
                return {
                    "action": "tool_call",
                    "reasoning": "User approved the Rust shortlist and wants the closest fit products from the catalog.",
                    "tool_requests": [
                        {"tool": "metadata_filter", "args": {"name_contains": "Smart Interview Live Coding"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Linux Programming (General)"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Networking and Implementation (New)"}},
                        {"tool": "metadata_filter", "args": {"name_contains": "Occupational Personality Questionnaire OPQ32r"}},
                    ],
                    "updated_slots": {"role": "senior rust engineer", "purpose": "selection"},
                    "default_additions": [],
                }

        if any(term in latest for term in ("graduate", "financial analyst", "finance", "numerical reasoning", "sales", "re-skill", "restructure", "talent audit")):
            return {
                "action": "tool_call",
                "reasoning": "Concrete role brief with enough context to recommend a shortlist directly.",
                "tool_requests": [{"tool": "semantic_search", "args": {"query": latest, "top_k": 5}}],
                "updated_slots": {"role": latest, "purpose": "development" if "re-skill" in latest or "restructure" in latest or "talent audit" in latest else "selection"},
                "default_additions": ["OPQ32r"] if "sales" in latest or "re-skill" in latest or "talent audit" in latest else [],
            }

        if "assessment" in latest or "solution" in latest or "recommend" in latest:
            return {
                "action": "clarify",
                "reasoning": "Need the role and hiring goal to narrow the shortlist.",
                "clarifying_question": "What role are you hiring for, and what should the assessment help you measure?",
                "updated_slots": {},
                "default_additions": [],
            }

        return {
            "action": "clarify",
            "reasoning": "Need a bit more context to narrow the shortlist.",
            "clarifying_question": "What role, seniority level, and hiring goal should I use to narrow the SHL shortlist?",
            "updated_slots": {},
            "default_additions": [],
        }


_CLIENT_SINGLETON: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _CLIENT_SINGLETON
    if _CLIENT_SINGLETON is not None:
        return _CLIENT_SINGLETON

    provider = os.environ.get("LLM_PROVIDER", "mock").lower()
    model = os.environ.get("LLM_MODEL")

    if provider == "anthropic":
        _CLIENT_SINGLETON = AnthropicClient(model=model or "claude-sonnet-4-6")
    elif provider == "openai":
        _CLIENT_SINGLETON = OpenAIClient(model=model or "gpt-4o")
    elif provider == "groq":
        _CLIENT_SINGLETON = GroqClient(model=model or "llama-3.3-70b-versatile")
    elif provider == "gemini":
        _CLIENT_SINGLETON = GeminiClient(model=model or "gemini-2.5-flash")
    else:
        _CLIENT_SINGLETON = MockClient()
    return _CLIENT_SINGLETON


