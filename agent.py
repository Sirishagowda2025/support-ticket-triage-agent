"""
Support Ticket Triage Agent
----------------------------
Reads a batch of support tickets (subject + body), asks an LLM to classify
each one by category and urgency, decides which team it should be routed to,
and flags low-confidence cases for human review.

Usage:
    python agent.py --input data/sample_tickets.json --output-dir output
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# Fixed taxonomy. Keeping this closed (rather than letting the model invent
# categories) is what makes routing deterministic and the output usable by
# a real ticketing system downstream.
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Billing",
    "Technical / Bug",
    "Account Access",
    "Security",
    "Feature Request",
    "General Inquiry",
]

URGENCY_LEVELS = ["Critical", "High", "Medium", "Low"]

# category -> team. Security and outages get an extra bump handled in code,
# not left to the model, so it can't be talked out of an escalation.
ROUTING_MAP = {
    "Billing": "Finance/Billing Team",
    "Technical / Bug": "Engineering",
    "Account Access": "Customer Success",
    "Security": "Security Team",
    "Feature Request": "Product Team",
    "General Inquiry": "Support Team",
}

# Below this confidence, we don't trust the model's own call -> human review.
CONFIDENCE_THRESHOLD = 0.65

MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = f"""You are a support ticket triage assistant for a SaaS company.

Given a ticket's subject and body, classify it and respond with ONLY a raw
JSON object (no markdown fences, no commentary) with exactly these fields:

{{
  "category": one of {CATEGORIES},
  "urgency": one of {URGENCY_LEVELS},
  "confidence": a number between 0.0 and 1.0 representing how confident you
    are in the category classification,
  "reasoning": a one-sentence explanation of why you chose this category
    and urgency
}}

Guidance:
- "Critical" urgency = active outage, security breach, or something blocking
  a paying customer from working right now, or a hard deadline within hours.
- "High" = broken functionality affecting the user's ability to work, no
  immediate deadline.
- "Medium" = real issue, no urgency signal, or a request tied to a routine
  task.
- "Low" = questions, feature requests, or compliments with a minor ask.
- If the ticket is vague, off-topic, or could plausibly fit two categories,
  lower your confidence score accordingly instead of guessing.
- Multi-issue tickets: classify by the ticket's PRIMARY/first-mentioned issue.
"""


def classify_ticket(client: Groq, ticket: dict, retries: int = 1) -> dict:
    """Call the LLM to classify a single ticket. Returns a result dict.
    Falls back to a safe 'needs manual review' result if the model output
    can't be parsed after retries, rather than crashing the batch.
    """
    user_prompt = f"Subject: {ticket['subject']}\n\nBody: {ticket['body']}"

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if the model adds them despite instructions
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)

            category = parsed.get("category", "General Inquiry")
            if category not in CATEGORIES:
                category = "General Inquiry"

            urgency = parsed.get("urgency", "Medium")
            if urgency not in URGENCY_LEVELS:
                urgency = "Medium"

            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            reasoning = parsed.get("reasoning", "").strip()

            needs_review = confidence < CONFIDENCE_THRESHOLD

            return {
                "id": ticket["id"],
                "subject": ticket["subject"],
                "category": category,
                "urgency": urgency,
                "confidence": round(confidence, 2),
                "routed_team": ROUTING_MAP.get(category, "Support Team"),
                "needs_human_review": needs_review,
                "reasoning": reasoning,
                "error": None,
            }
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            last_error = str(e)
            time.sleep(0.5)
            continue

    # Fallback after retries exhausted: never silently drop a ticket.
    return {
        "id": ticket["id"],
        "subject": ticket["subject"],
        "category": "General Inquiry",
        "urgency": "Medium",
        "confidence": 0.0,
        "routed_team": "Support Team",
        "needs_human_review": True,
        "reasoning": f"Model output could not be parsed after retries: {last_error}",
        "error": last_error,
    }


def run_batch(input_path: str, output_dir: str) -> list:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found. Copy .env.example to .env and add your key.")
        sys.exit(1)

    client = Groq(api_key=api_key)

    with open(input_path, "r") as f:
        tickets = json.load(f)

    print(f"Loaded {len(tickets)} tickets from {input_path}")
    print(f"Using model: {MODEL}\n")

    results = []
    for i, ticket in enumerate(tickets, 1):
        print(f"[{i}/{len(tickets)}] Classifying {ticket['id']}: {ticket['subject'][:50]}...")
        result = classify_ticket(client, ticket)
        flag = "  <-- FLAGGED FOR REVIEW" if result["needs_human_review"] else ""
        print(f"        -> {result['category']} / {result['urgency']} "
              f"(confidence {result['confidence']}) -> {result['routed_team']}{flag}")
        results.append(result)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    json_path = Path(output_dir) / "routing_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = Path(output_dir) / "routing_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    flagged = sum(1 for r in results if r["needs_human_review"])
    print(f"\nDone. {len(results)} tickets processed, {flagged} flagged for human review.")
    print(f"Results written to:\n  {json_path}\n  {csv_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Support Ticket Triage Agent")
    parser.add_argument("--input", default="data/sample_tickets.json",
                         help="Path to input JSON file of tickets")
    parser.add_argument("--output-dir", default="output",
                         help="Directory to write routing_results.json/.csv")
    args = parser.parse_args()

    run_batch(args.input, args.output_dir)


if __name__ == "__main__":
    main()
