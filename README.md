# Customer Service Data Analyst Agent

A LangGraph ReAct agent that answers questions about the
[Bitext Customer Service Tagged Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).
It handles three kinds of questions:

- **Structured** — concrete, data-driven (`"How many refund requests?"`).
- **Unstructured** — open-ended summarization (`"Summarize the FEEDBACK category"`).
- **Out-of-scope** — unrelated to the dataset, politely declined.

It has two kinds of persistent memory (Task 2):

- **Episodic** — per-session conversation history via a `SqliteSaver`
  checkpointer (`--session <id>` resumes prior conversations across
  restarts).
- **Semantic** — a per-user markdown profile of distilled facts
  (name, recurring interests, preferences) maintained in
  `data/profiles/<user>.md` and answered from the agent's prompt
  directly. Pass `--user <id>` to scope the profile.

This README covers **Tasks 1 and 2**.

---

## Setup (≈5 minutes)

### 1. Clone and create a virtual environment

```bash
git clone <this-repo>
cd <this-repo>
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Provide your Nebius Token Factory API key

```bash
cp .env.example .env
# Then edit .env and paste your NEBIUS_API_KEY.
```

You can get a key at <https://studio.nebius.com>. The defaults in
`.env.example` point at the Nebius OpenAI-compatible endpoint and use the
`Qwen/Qwen3-235B-A22B` model (see *Model choice* below).

### 4. Run the agent

```bash
python main.py
```

The first run downloads the Bitext dataset from HuggingFace (~27k rows,
a few MB) and caches it under `data/bitext.parquet`. Subsequent runs are
fully offline for the dataset.

Useful flags:

```bash
python main.py                                # default session + user = "default"
python main.py --session refunds              # named session, resumes on next run
python main.py --session refunds --user adam  # session bound to a named user profile
python main.py --quiet                        # hide per-step reasoning
```

In-REPL commands: `:profile` prints the current user's stored profile,
`:reset` starts a fresh session (old one stays on disk), `exit` quits.

State on disk:

| Path | Contents |
|---|---|
| `data/checkpoints.sqlite` | LangGraph checkpoints (one row per turn per session). |
| `data/profiles/<user>.md` | Per-user distilled profile. |
| `data/bitext.parquet` | Cached dataset (rebuilt on first run). |

All of these are under `data/` and gitignored.

---

## Example session

```
you › How many refund requests did we get?
[router] structured — direct count query about an intent
[agent → tool] list_intents({"category": "REFUND"})
[tool ← list_intents]
  ["check_refund_policy", "get_refund", "track_refund"]
[agent → tool] count_records({"intent": "get_refund"})
[tool ← count_records]
  {"count": 997, "filters": {"intent": "get_refund"}}

agent › There are 997 rows tagged with the `get_refund` intent.
```

Try also:

- `What categories exist in the dataset?`
- `Show me 5 examples of the SHIPPING category.`
- `Summarize how agents respond to complaint intents.`
- `Show me examples of people wanting their money back.`
- `What is the distribution of intents in the ACCOUNT category?`
- `Who is the president of France?` *(out-of-scope, politely declined)*

---

## Architecture

```
              ┌────────┐
              │ router │ ─── classifies query ───┐
              └────┬───┘                         │
                   │                             ▼
                   │                       ┌────────────┐
                   │  out_of_scope ──────▶ │  decline   │──▶ END
                   │                       └────────────┘
                   │  structured / unstructured
                   ▼
              ┌────────┐    tool_calls    ┌────────┐
              │ agent  │ ───────────────▶ │ tools  │
              │ (LLM)  │ ◀─────────────── │ (exec) │
              └───┬────┘                  └────────┘
       no calls   │   iter ≥ max
                  ▼                       ┌──────────────────┐
        ┌──────────────────┐         ───▶ │ max_iterations   │──▶ END
        │  update_profile  │              └──────────────────┘
        └────────┬─────────┘
                 ▼
                END
```

A `SqliteSaver` checkpointer wraps the whole graph so `messages` and
`user_id` survive process restarts under a given `thread_id` (i.e.
`--session`). The `update_profile` node persists distilled facts about
the user separately, in `data/profiles/<user>.md`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for a node-by-node walkthrough.

Key files:

| File | Role |
|---|---|
| [main.py](main.py) | CLI entry point with `argparse` |
| [src/cli.py](src/cli.py) | Streams graph updates and prints reasoning; opens SqliteSaver |
| [src/agent.py](src/agent.py) | LangGraph wiring + max-iterations fallback + profile injection |
| [src/router.py](src/router.py) | Structured-output router (Pydantic) |
| [src/tools.py](src/tools.py) | Tools with descriptions + Pydantic schemas |
| [src/profile.py](src/profile.py) | Per-user profile load/save + LLM updater |
| [src/dataset.py](src/dataset.py) | HuggingFace loader with parquet cache |
| [src/llm.py](src/llm.py) | Nebius Token Factory chat-model factory |
| [src/config.py](src/config.py) | Env-var configuration |

### Model choice

Single model: **`Qwen/Qwen3-235B-A22B`** via Nebius Token Factory.

Why one model rather than a router/agent split:

- The router task is *trivially* an OpenAI-style function call with three
  enum options. A 235B MoE model handles it instantly; the latency cost
  is dominated by network round-trip, not parameters.
- Qwen3-235B has strong tool-calling and instruction-following, which is
  what the ReAct loop needs.
- Keeping one model keeps the README, the `.env`, and the failure modes
  simple. To switch (e.g. `meta-llama/Llama-3.3-70B-Instruct` for a
  cheaper run), set `NEBIUS_MODEL` in `.env` — no code changes.

If you wanted a split, swapping `build_router()` in `src/router.py` to
build its own `ChatOpenAI` with a smaller model is a 2-line change.

### Tools

All tools live in [src/tools.py](src/tools.py) and use Pydantic input
schemas. Each tool's docstring is what the LLM sees when picking it —
they're written to make the *when-to-use* signal explicit.

| Tool | Purpose |
|---|---|
| `list_categories` | Enumerate all categories. |
| `list_intents` | Enumerate intents (optionally scoped to a category). |
| `count_records` | Count rows under optional category/intent filters. |
| `get_examples` | Fetch N example rows; also feeds unstructured summarization. |
| `search_examples` | Free-text substring search on customer instructions — for paraphrased asks ("people wanting their money back"). |
| `intent_distribution` | Per-intent row counts (optionally per category). |

### Router

A dedicated node (`src/router.py`) classifies the question via
`with_structured_output(RouteDecision)` into `structured`,
`unstructured`, or `out_of_scope`. Out-of-scope queries short-circuit to a
polite refusal — the agent **never** answers them from the LLM's general
knowledge.

### Max iterations

`AGENT_MAX_ITERATIONS` (default **12**) bounds the ReAct loop. If the
agent has not produced a final answer within that many tool-using turns,
the graph routes to a `max_iterations` node that returns a graceful
fallback message rather than spinning.

### Multi-step reasoning

The example `"How many refund requests did we get?"` typically chains:
`list_intents(category="REFUND")` → `count_records(intent="get_refund")`.
`"Summarize the FEEDBACK category"` chains `list_intents("FEEDBACK")`
(optional) → `get_examples(category="FEEDBACK", n=20)` → final
summarization by the LLM.

---

## Memory (Task 2)

### Episodic — conversation across turns and restarts

Persistence is provided by
`langgraph.checkpoint.sqlite.SqliteSaver` against
`data/checkpoints.sqlite`. The CLI opens it once per process and passes
`config={"configurable": {"thread_id": <session_id>}}` on every
`stream` call, so the messages list is restored automatically.

Demo:

```bash
$ python main.py --session demo --user adam
agent › ...
you › Show me 3 examples from REFUND
you › exit

$ python main.py --session demo --user adam     # different process
(resumed: 4 prior messages in this session)
you › Show me 3 more
```

The "3 more" is resolved against the prior turn that lives in the
checkpoint — the LLM sees the full message history when it picks tool
arguments.

### Semantic — per-user profile

A small markdown file lives at `data/profiles/<user>.md`. It holds
distilled facts only (name, interests, preferences, notes) — never a
transcript replay. After every final agent answer, the
`update_profile` node calls an LLM that either rewrites the profile or
returns it unchanged.

The profile is injected into the agent's **system prompt** on every
turn, so questions like *"What do you remember about me?"* are
answered directly without tool calls.

Inspect the current profile from inside the REPL with `:profile`, or
just open the markdown file.

## Project layout

```
.
├── main.py
├── requirements.txt
├── .env.example
├── data/                       # all runtime state, gitignored
│   ├── bitext.parquet          # cached dataset
│   ├── checkpoints.sqlite      # episodic memory (Task 2a)
│   └── profiles/<user>.md      # semantic memory (Task 2b)
└── src/
    ├── agent.py
    ├── cli.py
    ├── config.py
    ├── dataset.py
    ├── llm.py
    ├── profile.py
    ├── router.py
    └── tools.py
```
