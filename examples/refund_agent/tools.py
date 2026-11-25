"""Tools for the refund agent. `search_policy` returns five documents; for some queries one of
them is an outdated policy that allowed refunds within 60 days (the current window is 30)."""

from __future__ import annotations

import agentrelay as relay

ORDERS = {
    "A1001": "Order #A1001: wireless headphones, $89.00, purchased 45 days ago, delivered, "
    "unopened.",
}

CURRENT_POLICY = (
    "Policy doc 1 — Refund Policy (effective 2026-09-01, current). Customers may request a "
    "refund within 30 days of purchase. Items must be unused and in the original packaging. "
    "Refunds go back to the original payment method within 5-7 business days. After 30 days, "
    "we can offer store credit or a repair under warranty, but not a refund."
)
STALE_POLICY = (
    "Policy doc 4 — Refund Policy (effective 2024-01-01). Customers may request a refund "
    "within 60 days of purchase. Items must be unused and in the original packaging. Refunds go "
    "back to the original payment method within 10 business days. This policy applies to all "
    "orders placed through the online store and the mobile app."
)
SHIPPING = (
    "Policy doc 2 — Shipping. Standard shipping takes 3-5 business days and is free for orders "
    "over $50. Express shipping takes 1-2 business days and costs $15. We ship to every US state "
    "and to Canada. Tracking numbers are emailed as soon as the order leaves the warehouse."
)
WARRANTY = (
    "Policy doc 3 — Warranty. Electronics carry a one-year limited warranty against "
    "manufacturing defects. The warranty does not cover accidental damage, water damage, or "
    "normal wear. To make a claim, contact support with your order number and a description of "
    "the problem; we will send a prepaid shipping label."
)
EXCHANGES = (
    "Policy doc 5 — Exchanges. Unused items can be exchanged for a different size or color "
    "within 30 days of purchase at no cost. Exchanges for a different product are treated as a "
    "return and a new purchase. Gift recipients can exchange items using the gift receipt."
)
PRICE_MATCH = (
    "Policy doc 4 — Price Match. If you find a lower advertised price from a major US retailer "
    "within 14 days of purchase, we will refund the difference. The item must be identical and "
    "in stock at the competitor. Marketplace sellers and clearance sales are not eligible."
)


@relay.tool
def get_order(order_id: str) -> str:
    """Look up an order by id."""
    if order_id not in ORDERS:
        raise KeyError(f"no order {order_id}")
    return ORDERS[order_id]


@relay.tool
def search_policy(query: str) -> str:
    """Search the policy knowledge base and return the top five documents."""
    if "current" in query.lower():
        docs = [CURRENT_POLICY, SHIPPING, WARRANTY, PRICE_MATCH, EXCHANGES]
    else:
        docs = [CURRENT_POLICY, SHIPPING, WARRANTY, STALE_POLICY, EXCHANGES]
    return "\n\n".join(docs)
