"""FakeLLM rules that reproduce the refund bug offline.

When the stale 60-day policy is among the search results, the model quotes it 90% of the time.
The model usually (80%) searches for the *current* policy, which never returns the stale doc;
the failing runs are the ones where it searched for "refund window". The documented fix (a
system-prompt rule to trust the newest effective date) appears in `unless_any`.
"""

from __future__ import annotations

from agentrelay.providers.fake import Rule, text, tool_use

FIX_TEXT = "trust the document with the latest effective date"
# Present only once search results are in the context (every policy doc has an effective date).
SEARCHED = "(effective "

WRONG_ANSWER = (
    "Good news! Our policy allows refunds within 60 days of purchase, so your order "
    "(placed 45 days ago) qualifies for a full refund."
)
RIGHT_ANSWER = (
    "I'm sorry, refunds are available within 30 days of purchase, and your order was placed "
    "45 days ago, so it no longer qualifies. I can offer store credit or a warranty repair."
)

RULES = [
    Rule(
        name="look-up-order",
        when_all=["refund"],
        unless_any=["Order #A1001"],
        respond=tool_use("get_order", {"order_id": "A1001"}),
    ),
    Rule(
        name="search-current-policy",
        when_all=["Order #A1001"],
        unless_any=[SEARCHED],
        respond=tool_use("search_policy", {"query": "current refund policy"}),
        probability=0.8,
    ),
    Rule(
        name="search-refund-window",
        when_all=["Order #A1001"],
        unless_any=[SEARCHED],
        respond=tool_use("search_policy", {"query": "refund window"}),
    ),
    Rule(
        name="quote-stale-policy",
        when_all=[SEARCHED, "within 60 days"],
        unless_any=[FIX_TEXT],
        respond=text(WRONG_ANSWER),
        probability=0.9,
    ),
]
DEFAULT = text(RIGHT_ANSWER)
