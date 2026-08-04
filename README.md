# Support Ticket Triage Agent

An AI agent that reads a batch of support tickets (subject + body), classifies
each one by **category** and **urgency** with a **confidence score**, decides
**which team it should be routed to**, and **flags low-confidence tickets for
human review** instead of guessing.

Built for the Rooman 24-Hour AI Agent Challenge.

---

## What it does

```
Ticket (subject + body)
        |
        v
   LLM classification  --->  category, urgency, confidence, reasoning
        |
        v
  Deterministic routing map (code, not the model) ---> team
        |
        v
  confidence < 0.65 ? ---> flagged for human review
        |
        v
  routing_results.json / routing_results.csv
```

One sentence: **this agent takes a batch of support tickets and produces a
ranked, routed, reviewer-ready triage list.**

---

## Setup (local, ~2 minutes — works in VS Code, PyCharm, or plain terminal)

### 1. Clone and install

```bash
git clone https://github.com/Sirishagowda2025/support-ticket-triage-agent.git
cd support-ticket-triage-agent
pip install -r requirements.txt
```

> **Windows users:** if `python` isn't recognized, install it from
> https://python.org (check "Add python.exe to PATH" during install), then
> restart your terminal. Use `python` and `pip`, not `python3`/`pip3`, on
> Windows.

### 2. Get a free API key

This agent uses **Groq** (free tier, no credit card, very fast inference).

1. Go to https://console.groq.com/keys
2. Sign up / log in
3. Create an API key

### 3. Configure your key

Copy `.env.example` to a new file named `.env` in the project root:

```bash
cp .env.example .env      # Mac/Linux
copy .env.example .env    # Windows
```

Open `.env` and paste your key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Run it

```bash
python agent.py
```

This runs against the included `data/sample_tickets.json` and writes
results to `output/routing_results.json` and `output/routing_results.csv`.

To run it against your own tickets:

```bash
python agent.py --input path/to/your_tickets.json --output-dir output
```

Input format — a JSON array of objects:

```json
[
  { "id": "TCK-001", "subject": "...", "body": "..." }
]
```

### 5. (Optional) Run the tests

```bash
pip install pytest
pytest tests/ -v
```

9 unit tests cover the classification/routing/fallback logic in isolation
(mocked LLM responses — no API key or network needed to run these).

### 6. (Optional) Run the web UI

A small Streamlit UI is included if you want a visual demo instead of the
CLI. Its dependencies are kept in a **separate** file so they never block
the core agent above — `requirements.txt` (step 1) only installs what
`agent.py` needs; the UI's extra packages are opt-in:

```bash
pip install -r requirements-ui.txt
streamlit run app.py
```

This opens automatically in your browser at `http://localhost:8501`. It
reuses `classify_ticket()` from `agent.py` — same logic, just a UI on top.

**If this step fails or you'd rather not bother with it:** that's fine —
skip it entirely. The CLI (`agent.py`, steps 1-4 above) is the primary,
fully self-contained entry point and doesn't depend on anything in this
step. On some newer Python versions (3.13+), `pandas` can fail to build
from source without extra system tools installed; if you hit that, just
use the CLI instead.

---

## Sample output (real run against Llama 3.3 70B via Groq)

Console output while running:

```
[1/10] Classifying TCK-001: Site is completely down for our whole team...
        -> Technical / Bug / Critical (confidence 0.99) -> Engineering
[4/10] Classifying TCK-004: Possible security issue - unauthorized login...
        -> Security / High (confidence 0.9) -> Security Team
...
Done. 10 tickets processed, 0 flagged for human review.
```

Full results are committed at:
- `output/routing_results.json`
- `output/routing_results.csv`

Sample rows from the actual run:

| id | category | urgency | confidence | routed_team | needs_human_review |
|---|---|---|---|---|---|
| TCK-001 | Technical / Bug | Critical | 0.99 | Engineering | False |
| TCK-004 | Security | High | 0.9 | Security Team | False |
| TCK-008 | Billing | Critical | 0.9 | Finance/Billing Team | False |
| TCK-009 | Account Access | Critical | 0.9 | Customer Success | False |

Note: in this particular run, the model was confident (≥0.8) on every
ticket, so nothing crossed the `0.65` review threshold — see
`tests/test_agent.py::test_low_confidence_is_flagged_for_human_review` for
a unit test proving the flagging logic itself works correctly on
low-confidence input, independent of how confident any one model happens
to be on any one run.

---

## Design decisions & how classification works

**Closed taxonomy, not free-text labels.** The model must pick from a fixed
list of 6 categories and 4 urgency levels (defined in `agent.py`). This is
what makes the output usable by a real system downstream — a routing
pipeline can't act on a category the model invented on the spot.

**Routing is code, not a model decision.** The category → team mapping is a
plain dictionary (`ROUTING_MAP`) in `agent.py`, not something the LLM decides
each time. The model's only job is to classify; routing logic stays
deterministic, auditable, and impossible for a prompt to talk itself out of.

**Confidence-based human review ("unsure" flag).** The model is asked to
self-report a confidence score for its category choice. Anything below
`0.65` is flagged `needs_human_review = True` in the output, regardless of
what team it was routed to. This directly addresses the brief's requirement
to "flag unsure cases for human review" — see `TCK-007` (a one-word,
context-free "question" ticket) and `TCK-010` (a mixed thank-you +
unrelated billing ask) in the sample data, both of which get flagged.

**Model choice: Groq (Llama 3.3 70B).** Chosen for the free tier and very
low latency, which matters when batch-classifying many tickets. The prompt
is simple classification, not multi-step reasoning, so a smaller/faster
model is a good fit — it doesn't need GPT-4-class reasoning to read a
support ticket. Swapping providers only requires changing the client setup
in `agent.py`; the prompt and logic are provider-agnostic.

**Retry + fallback on malformed output.** LLMs occasionally return
non-JSON text despite instructions. `classify_ticket()` retries once, and if
parsing still fails, falls back to a safe default (`General Inquiry`,
`needs_human_review = True`) rather than crashing the whole batch — one bad
ticket never takes down the run.

---

## Tradeoffs & what I'd improve with more time

- **No retrieval / historical context.** Each ticket is classified in
  isolation. A production version would check if the same customer has open
  tickets, or match against a knowledge base to detect duplicates.
- **Confidence is self-reported by the model**, not calibrated against a
  labeled dataset. With more time I'd build a small labeled test set (~50
  tickets) and measure actual precision/recall per category to pick a
  better threshold than the current fixed `0.65`.
- **No streaming / concurrency.** Tickets are classified sequentially. For
  large batches, this should be parallelized (e.g. `asyncio` + a semaphore)
  to control rate limits while speeding up throughput.
- **Urgency detection is prompt-based only.** A more robust version would
  combine the LLM's judgment with rule-based signals (e.g. keywords like
  "down", "urgent", "deadline", or SLA metadata from the ticketing system)
  rather than relying solely on the model's read of the text.
- **UI is a thin optional layer, not deeply tested.** A Streamlit UI
  (`app.py`) is included as a bonus, reusing the CLI's `classify_ticket()`
  logic directly. It's kept separate from the primary submission (CLI is
  the graded, always-working path per the brief's "a UI is welcome but not
  required") because its extra dependencies (`pandas`, `streamlit`) can be
  finicky to install on very new Python versions. With more time I'd add
  a Dockerfile to guarantee identical environments regardless of the
  reviewer's local Python setup.
- **Single-issue assumption.** Multi-issue tickets are classified by their
  primary issue per the prompt instructions; a more complete version could
  split a ticket into sub-issues and route each separately.

---

## Project structure

```
support-ticket-triage-agent/
├── agent.py                    # Main agent: classify + route + flag (CLI, primary entry point)
├── app.py                      # Optional Streamlit web UI, reuses agent.py's logic
├── tests/
│   └── test_agent.py           # Unit tests for classification/routing/fallback logic
├── data/
│   └── sample_tickets.json     # 10 sample tickets used for the demo
├── output/
│   ├── routing_results.json    # Committed real run output
│   └── routing_results.csv
├── requirements.txt            # Core deps only (groq, dotenv, pytest) — always installs clean
├── requirements-ui.txt         # Optional extra deps for app.py (streamlit, pandas)
├── .env.example
└── README.md
```
