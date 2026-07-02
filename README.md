# SHL Assessment Recommendation Agent

An agentic recommendation system for the SHL product catalog, matching the
architecture:

```
User -> Conversation Manager -> {Conversation Memory, LLM Planner}
     -> Tool Decision Engine -> {Metadata Tool, FAISS Tool, Compare Tool}
     -> Recommendation Agent -> Response Generator -> User
```

Behavior (clarify-before-recommending, default OPQ32r additions, exclusion
handling, compare/refuse branches, finalize-on-confirmation) was derived
from the 10 example transcripts (C1â€“C10) provided.

## Component map

| Diagram box | File | What it does |
|---|---|---|
| Conversation Manager | `app/core/conversation_manager.py` | Orchestrates one full turn: memory â†’ planner â†’ tool engine â†’ recommendation agent â†’ response generator |
| Conversation Memory | `app/core/memory.py` | Transient turn history, inferred slots, excluded product ids, and last shown recommendation set; rebuilt from `messages` for the public stateless API |
| LLM Planner | `app/core/planner.py` | Decides the turn's action: `clarify` / `tool_call` / `compare` / `refuse` / `finalize`, plus which tools to call |
| Tool Decision Engine | `app/core/tool_engine.py` | Validates + executes the planner's tool requests (parallel when independent) |
| Metadata Tool | `app/tools/metadata_tool.py` | Exact-field filtering: job level, language, category, duration, adaptive |
| FAISS Tool | `app/tools/faiss_tool.py` | Semantic search over product descriptions (local sentence-transformers embeddings, cached FAISS index) |
| Compare Tool | `app/tools/compare_tool.py` | Structured field-by-field diff between named products |
| Recommendation Agent | `app/core/recommendation_agent.py` | Merges/de-dupes/ranks tool outputs, applies default additions (e.g. OPQ32r), enforces exclusions |
| Response Generator | `app/core/response_generator.py` | LLM writes prose only; the markdown table is rendered deterministically from real Product data (never hallucinated) |
| Catalog loader | `app/core/catalog.py` | Shared data layer every tool reads from |
| LLM client | `app/core/llm_client.py` | Swappable provider interface â€” implement `LLMClient` for any backend |
| FastAPI app | `app/main.py` | HTTP surface required by the assignment: stateless `POST /chat` and `GET /health` |

## Setup

```bash
pip install -r requirements.txt
```

Configure your LLM provider. Easiest way: copy `.env.example` to `.env` and
fill in your key â€” it's loaded automatically on startup, no manual
`export` needed:

```bash
cp .env.example .env
# then edit .env in any text editor and set LLM_PROVIDER / LLM_API_KEY / LLM_MODEL
```

Or set environment variables directly if you prefer (the client is fully
swappable â€” see `app/core/llm_client.py`):

```bash
export LLM_PROVIDER=groq             # "anthropic" | "openai" | "groq" | "gemini" | "mock"
export LLM_API_KEY=gsk_...
export LLM_MODEL=llama-3.3-70b-versatile   # optional, provider-specific default otherwise
```

```bash
export LLM_PROVIDER=gemini
export LLM_API_KEY=AIza...
export LLM_MODEL=gemini-2.5-flash          # optional
```

To add a new provider, implement `LLMClient.complete()` in `llm_client.py`
and register it in `get_llm_client()` â€” nothing else in the codebase needs
to change, since the Planner and Response Generator only depend on the
abstract interface.

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

Call it:

```bash
curl localhost:8000/health
# -> {"status":"ok"}

curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"We need a solution for senior leadership."}]}'
# -> {"reply":"Who is this meant for?","recommendations":[],"end_of_conversation":false}

curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"We need a solution for senior leadership."},{"role":"assistant","content":"Who is this meant for?"},{"role":"user","content":"CXOs and directors, selection against a leadership benchmark."}]}'
# -> {"reply":"...","recommendations":[{"name":"Occupational Personality Questionnaire OPQ32r","url":"https://www.shl.com/...","test_type":"P"}],"end_of_conversation":false}
```

## Testing without any LLM API key

`tests/scripted_client.py` provides a `ScriptedClient` that returns
pre-queued planner decisions / prose, so the *orchestration logic* (memory,
tool routing, merge/exclusion rules, table rendering) can be validated
without calling any real model:

```bash
python3 -m tests.test_api_contract # verifies /health and stateless /chat schema
python3 -m tests.test_c6_replay      # replays the DSI/Safety 8.0 transcript
python3 -m tests.test_c9_replay      # replays the 7-turn iterative tech battery
```

Both assert the final shortlist exactly matches the source transcripts.
This is also the pattern to use for regression-testing prompt changes to
the Planner/Response Generator system prompts against all 10 example
conversations.

## Design decisions worth knowing about

- **The table is never LLM-generated.** The Response Generator's system
  prompt explicitly forbids it from inventing product facts; the LLM only
  writes 1â€“4 sentences of prose, and `render_table()` builds the markdown
  table deterministically from real `Product` fields. This guarantees
  URLs/durations/languages can't be hallucinated.
- **`name_contains` in the Metadata Tool resolves exact match first, then
  name-prefix, then substring** â€” pure substring matching caused false
  positives (e.g. a filter for `"SQL (New)"` also matching `"Oracle
  PL/SQL (New)"` and `"Automata - SQL (New)"`). This was caught by the C9
  replay test and fixed; see git history / inline comments in
  `metadata_tool.py`.
- **Exclusions are enforced in two layers.** The Planner is instructed to
  respect stated exclusions, but `ConversationManager._absorb_explicit_exclusions`
  also runs a regex-based safety net that matches "drop/remove/exclude X"
  phrasing (including acronyms in parentheses, e.g. "drop the DSI") against
  the last shown recommendation set, and adds matches to
  `state.excluded_ids`. `RecommendationAgent` then filters `excluded_ids`
  unconditionally, even if a tool call re-surfaces the product. This means
  a single planner mistake can't silently resurrect a dropped product.
- **FAISS index is built once and cached to disk** (`data/cache/`), keyed
  by a hash of the catalog's product ids, so the embedding model only runs
  once per catalog version, not per process restart.
- **The catalog JSON contains a few literal control characters** inside
  scraped description fields, which break strict JSON parsing. `catalog.py`
  loads with `json.loads(text, strict=False)` to tolerate this rather than
  pre-sanitizing the source file.

## Known gap in this environment

This sandbox's network allowlist doesn't include `huggingface.co`, so the
sentence-transformers model (`all-MiniLM-L6-v2`) couldn't be downloaded
here to demo the FAISS tool live end-to-end. The FAISS indexing/search/
serialization logic itself was validated separately with substitute
embeddings and is correct; it will work as-is the moment it runs somewhere
with access to Hugging Face (which is the overwhelming majority of
real deployment environments). If your environment is similarly
restricted, either mirror the model weights internally or swap
`EMBED_MODEL_NAME`/the embedding call in `faiss_tool.py` for an API-based
embedding provider.

## Extending

- **New tool**: add a class with a `.run()` method under `app/tools/`,
  register it in `ToolDecisionEngine.__init__` and `_run_one`, and mention
  it in the Planner's system prompt so the LLM knows when to call it.
- **New action type** (e.g. "escalate to human"): add a branch in
  `LLMPlanner`'s system prompt + `PlannerDecision`, and a matching `if
  decision.action == "..."` block in `ConversationManager.handle_turn`.
- **Persistent memory**: swap `ConversationMemory`'s in-memory dict for a
  Redis/Postgres-backed implementation; `ConversationManager` only depends
  on `get_or_create()` / `save()`.

