"""
Unit tests for agent.py's classification and routing logic.

These tests mock the Groq client so they run instantly with no API key,
no network call, and no cost — they verify OUR code (parsing, validation,
routing, confidence flagging, fallback behavior), not the LLM's judgment.

Run with:
    pytest tests/
"""

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent  # noqa: E402


def make_fake_client(response_text: str) -> MagicMock:
    """Build a mock Groq client whose chat.completions.create() returns
    a single canned response string, mimicking the real SDK's shape."""
    fake_choice = MagicMock()
    fake_choice.message.content = response_text
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]

    client = MagicMock()
    client.chat.completions.create.return_value = fake_response
    return client


SAMPLE_TICKET = {
    "id": "TCK-TEST",
    "subject": "Test subject",
    "body": "Test body",
}


def test_valid_response_is_parsed_and_routed_correctly():
    response = json.dumps({
        "category": "Technical / Bug",
        "urgency": "High",
        "confidence": 0.9,
        "reasoning": "Clear bug report.",
    })
    client = make_fake_client(response)

    result = agent.classify_ticket(client, SAMPLE_TICKET)

    assert result["category"] == "Technical / Bug"
    assert result["urgency"] == "High"
    assert result["confidence"] == 0.9
    assert result["routed_team"] == "Engineering"
    assert result["needs_human_review"] is False
    assert result["error"] is None


def test_low_confidence_is_flagged_for_human_review():
    response = json.dumps({
        "category": "General Inquiry",
        "urgency": "Low",
        "confidence": 0.3,
        "reasoning": "Vague ticket.",
    })
    client = make_fake_client(response)

    result = agent.classify_ticket(client, SAMPLE_TICKET)

    assert result["confidence"] == 0.3
    assert result["needs_human_review"] is True


def test_confidence_exactly_at_threshold_is_not_flagged():
    # CONFIDENCE_THRESHOLD is 0.65; equal to it should count as confident enough.
    response = json.dumps({
        "category": "Billing",
        "urgency": "Medium",
        "confidence": agent.CONFIDENCE_THRESHOLD,
        "reasoning": "Borderline case.",
    })
    client = make_fake_client(response)

    result = agent.classify_ticket(client, SAMPLE_TICKET)

    assert result["needs_human_review"] is False


def test_every_category_maps_to_a_known_team():
    for category, team in agent.ROUTING_MAP.items():
        assert category in agent.CATEGORIES
        assert isinstance(team, str) and team


def test_unknown_category_falls_back_to_general_inquiry():
    response = json.dumps({
        "category": "Not A Real Category",
        "urgency": "Low",
        "confidence": 0.9,
        "reasoning": "Model invented a category.",
    })
    client = make_fake_client(response)

    result = agent.classify_ticket(client, SAMPLE_TICKET)

    assert result["category"] == "General Inquiry"
    assert result["routed_team"] == agent.ROUTING_MAP["General Inquiry"]


def test_unknown_urgency_falls_back_to_medium():
    response = json.dumps({
        "category": "Billing",
        "urgency": "Apocalyptic",
        "confidence": 0.9,
        "reasoning": "Model invented an urgency level.",
    })
    client = make_fake_client(response)

    result = agent.classify_ticket(client, SAMPLE_TICKET)

    assert result["urgency"] == "Medium"


def test_malformed_json_triggers_safe_fallback_not_a_crash():
    client = make_fake_client("this is not json at all")

    result = agent.classify_ticket(client, SAMPLE_TICKET, retries=0)

    assert result["needs_human_review"] is True
    assert result["error"] is not None
    assert result["category"] == "General Inquiry"


def test_markdown_fenced_json_is_still_parsed():
    response = "```json\n" + json.dumps({
        "category": "Security",
        "urgency": "Critical",
        "confidence": 0.95,
        "reasoning": "Model wrapped its output in a code fence despite instructions.",
    }) + "\n```"
    client = make_fake_client(response)

    result = agent.classify_ticket(client, SAMPLE_TICKET)

    assert result["category"] == "Security"
    assert result["routed_team"] == "Security Team"


def test_confidence_is_clamped_between_zero_and_one():
    response = json.dumps({
        "category": "Feature Request",
        "urgency": "Low",
        "confidence": 1.5,  # out-of-range value the model might return
        "reasoning": "Overconfident model.",
    })
    client = make_fake_client(response)

    result = agent.classify_ticket(client, SAMPLE_TICKET)

    assert result["confidence"] == 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
