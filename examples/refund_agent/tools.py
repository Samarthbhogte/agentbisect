"""Tools for the refund agent. `search_policy` returns five documents; for some queries one of
them is an outdated policy that allowed refunds within 60 days (the current window is 30)."""

from __future__ import annotations

import agentbisect as relay

ORDERS = {
    "A1001": "Order #A1001: wireless headphones, $89.00, purchased 45 days ago, delivered, "
    "unopened.",
}

CURRENT_POLICY = (
    "Policy doc 1 - Refund Policy (effective 2026-09-01, current). Customers may request a refund"
    " within 30 days of purchase. Items must be unused and in the original packaging, with all "
    "accessories, manuals and gifts that came with them. Refunds go back to the original payment "
    "method within 5-7 business days after the warehouse receives and inspects the item. Shipping"
    " costs are refunded only when the item arrived damaged or was not what you ordered. After 30"
    " days, we can offer store credit or a repair under warranty, but not a refund."
)
STALE_POLICY = (
    "Policy doc 4 - Refund Policy (effective 2024-01-01). Customers may request a refund within "
    "60 days of purchase. Items must be unused and in the original packaging. Refunds go back to "
    "the original payment method within 10 business days of receipt. Opened software, gift cards "
    "and personalised items cannot be refunded. This policy applies to all orders placed through "
    "the online store and the mobile app, and to orders placed by phone with a customer service "
    "agent."
)
SHIPPING = (
    "Policy doc 2 - Shipping (effective 2025-03-01). Standard shipping takes 3-5 business days "
    "and is free for orders over $50. Express shipping takes 1-2 business days and costs $15. We "
    "ship to every US state and to Canada; orders to Alaska, Hawaii and Canada may take two extra"
    " days. Tracking numbers are emailed as soon as the order leaves the warehouse. If a package "
    "is marked delivered but you can't find it, contact support within 7 days and we will open an"
    " investigation with the carrier."
)
WARRANTY = (
    "Policy doc 3 - Warranty (effective 2025-06-15). Electronics carry a one-year limited "
    "warranty against manufacturing defects, starting on the delivery date. The warranty does not"
    " cover accidental damage, water damage, or normal wear. To make a claim, contact support "
    "with your order number and a description of the problem; we will send a prepaid shipping "
    "label. Repaired or replaced items keep the remaining warranty of the original item, or 90 "
    "days, whichever is longer."
)
EXCHANGES = (
    "Policy doc 5 - Exchanges (effective 2025-01-10). Unused items can be exchanged for a "
    "different size or color within 30 days of purchase at no cost. Exchanges for a different "
    "product are treated as a return and a new purchase, so the refund policy applies to the "
    "original item. Gift recipients can exchange items using the gift receipt without the "
    "purchaser being notified. Exchanges are shipped free of charge once we receive the original "
    "item."
)
PRICE_MATCH = (
    "Policy doc 4 - Price Match (effective 2025-11-01). If you find a lower advertised price from"
    " a major US retailer within 14 days of purchase, we will refund the difference. The item "
    "must be identical and in stock at the competitor. Marketplace sellers, clearance sales and "
    "membership-only prices are not eligible. Send support a link or a photo of the advertisement"
    " with your order number; approved differences are refunded to the original payment method."
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
