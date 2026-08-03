"""
Optional web UI for the Support Ticket Triage Agent.
Run with: streamlit run app.py

Lets a reviewer upload a JSON file of tickets (or use the sample set),
watch them get classified, and see the routed results as a color-coded
table with a one-click CSV/JSON download.
"""

import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from agent import classify_ticket

load_dotenv()

st.set_page_config(page_title="Support Ticket Triage Agent", layout="wide")
st.title("🎫 Support Ticket Triage Agent")
st.caption("Classifies tickets by category & urgency, routes them to a team, "
           "and flags low-confidence cases for human review.")

# --- API key ---
api_key = os.environ.get("GROQ_API_KEY") or st.text_input(
    "Groq API key", type="password",
    help="Get a free key at https://console.groq.com/keys"
)

if not api_key:
    st.info("Enter your Groq API key above to get started (or set GROQ_API_KEY in a .env file).")
    st.stop()

client = Groq(api_key=api_key)

# --- Input ---
st.subheader("1. Load tickets")
source = st.radio("Ticket source", ["Use sample tickets", "Upload a JSON file"], horizontal=True)

if source == "Upload a JSON file":
    uploaded = st.file_uploader("Upload a JSON array of {id, subject, body}", type="json")
    tickets = json.load(uploaded) if uploaded else None
else:
    with open("data/sample_tickets.json") as f:
        tickets = json.load(f)
    st.success(f"Loaded {len(tickets)} sample tickets.")

if tickets:
    st.dataframe(pd.DataFrame(tickets), use_container_width=True, height=200)

# --- Run ---
st.subheader("2. Run triage")
if tickets and st.button("🚀 Classify & route all tickets", type="primary"):
    results = []
    progress = st.progress(0, text="Starting...")
    for i, ticket in enumerate(tickets):
        progress.progress((i + 1) / len(tickets), text=f"Classifying {ticket['id']}...")
        results.append(classify_ticket(client, ticket))
    progress.empty()

    df = pd.DataFrame(results)
    st.subheader("3. Results")

    flagged = int(df["needs_human_review"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Tickets processed", len(df))
    c2.metric("Flagged for review", flagged)
    c3.metric("Avg. confidence", f"{df['confidence'].mean():.2f}")

    def highlight_review(row):
        color = "background-color: #ffe3e3" if row["needs_human_review"] else ""
        return [color] * len(row)

    st.dataframe(
        df.style.apply(highlight_review, axis=1),
        use_container_width=True,
        height=400,
    )

    st.download_button("⬇️ Download CSV", df.to_csv(index=False), "routing_results.csv", "text/csv")
    st.download_button("⬇️ Download JSON", json.dumps(results, indent=2), "routing_results.json", "application/json")
