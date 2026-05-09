"""Deterministic fallback pipeline.

Mirrors the LLM-driven workflow without making any model calls. Used by
``run_eval.py`` when ``PAPER_USE_DETERMINISTIC=1`` (e.g. when no OpenAI
API key is available, or when reproducible offline evaluation is desired).

The structure parallels the agent code so reviewers can compare the two
side by side: parser → inventory → quoting → fulfill/decline → reply.
"""

from __future__ import annotations

import re
from datetime import datetime

import project_starter as ps
from models.schemas import (
    InventoryLine,
    InventoryReport,
    LineItem,
    ParsedItems,
    Quote,
    QuoteLine,
    QuoteRequest,
    QuoteResponse,
    SalesOutcome,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

#: Synonym → canonical SKU. Order matters: more specific phrases first.
_SKU_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bglossy\s+photo|photo\s+paper|glossy\s+print", re.I), "glossy_photo_50"),
    (re.compile(r"\bposter|24x36|large\s+print", re.I),                  "poster_24x36"),
    (re.compile(r"\binvitation\s+envelope|invitation\s+env|wedding\s+envelope", re.I),
                                                                          "envelope_invitation"),
    (re.compile(r"\b#?10\s+envelope|standard\s+envelope|business\s+envelope", re.I),
                                                                          "envelope_10_box"),
    (re.compile(r"\bcolored?\s+cardstock|color\s+cardstock|colou?red\s+card", re.I),
                                                                          "cardstock_colored_250"),
    (re.compile(r"\bwhite\s+cardstock|cardstock|cover\s+stock|card\s+stock", re.I),
                                                                          "cardstock_white_250"),
    (re.compile(r"\bsticky\s+notes?|post-?its?", re.I),                  "sticky_notes_pack"),
    (re.compile(r"\bnotebooks?|composition\s+book", re.I),               "notebook_college"),
    (re.compile(r"\ba4\s+paper|a4\s+ream|a4\s+sheet|a4", re.I),          "A4_paper_500"),
    (re.compile(r"\bletter\s+paper|letter\s+ream|letter\s+sheet|copy\s+paper|printer\s+paper",
                re.I),                                                    "letter_paper_500"),
    (re.compile(r"\benvelopes?", re.I),                                   "envelope_10_box"),
]

#: Splitter: break a request into per-item chunks on " and " / commas / semicolons.
_CHUNK_SPLITTER = re.compile(r"\s+and\s+|[,;]+", re.I)
#: Quantity capture inside a single chunk: "200 reams of A4", "5x posters", "30 packs".
_QTY_PATTERN = re.compile(r"(?P<qty>\d{1,5})\s+(?P<rest>.+)")


def parse_request(text: str) -> ParsedItems:
    """Best-effort parser: split on conjunctions, then regex per chunk."""
    items: dict[str, int] = {}
    notes_parts: list[str] = []

    for chunk in _CHUNK_SPLITTER.split(text):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = _QTY_PATTERN.search(chunk)
        if not match:
            continue
        qty = int(match.group("qty"))
        if qty <= 0 or qty > 100_000:
            continue
        rest = match.group("rest")
        sku = _match_sku(rest)
        if sku is None:
            continue
        items[sku] = items.get(sku, 0) + qty
        notes_parts.append(f"matched '{chunk}' → {sku} x{qty}")

    if not items:
        return ParsedItems(
            items=[],
            parse_confidence=0.0,
            notes="could not extract any line items from request",
        )

    parsed = [LineItem(sku=k, quantity=v) for k, v in items.items()]
    confidence = 0.9 if len(parsed) == len(notes_parts) else 0.7
    return ParsedItems(items=parsed, parse_confidence=confidence, notes="; ".join(notes_parts))


def _match_sku(phrase: str) -> str | None:
    for pattern, sku in _SKU_PATTERNS:
        if pattern.search(phrase):
            return sku
    return None


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def check_inventory(items: list[LineItem], request_date: str, deadline: str) -> InventoryReport:
    on_hand_map = ps.get_all_inventory(request_date)
    deadline_dt = datetime.strptime(deadline, "%Y-%m-%d")
    lines: list[InventoryLine] = []
    restock_needed = False
    all_available = True

    for item in items:
        on_hand = on_hand_map.get(item.sku, 0)
        shortfall = max(item.quantity - on_hand, 0)
        eta: str | None = None
        viable = True
        if shortfall > 0:
            restock_needed = True
            all_available = False
            eta = ps.get_supplier_delivery_date(request_date, shortfall)
            viable = datetime.strptime(eta, "%Y-%m-%d") <= deadline_dt
        lines.append(
            InventoryLine(
                sku=item.sku,
                needed=item.quantity,
                on_hand=on_hand,
                shortfall=shortfall,
                eta_if_reordered=eta,
                viable_by_deadline=viable,
            )
        )

    return InventoryReport(lines=lines, all_available=all_available, restock_needed=restock_needed)


# ---------------------------------------------------------------------------
# Quoting
# ---------------------------------------------------------------------------

def _bulk_discount(qty: int) -> float:
    if qty >= 500:
        return 15.0
    if qty >= 200:
        return 10.0
    if qty >= 50:
        return 5.0
    return 0.0


def _urgency_premium_pct(urgency_days: int) -> float:
    if urgency_days <= 3:
        return 8.0
    if urgency_days <= 7:
        return 4.0
    return 0.0


def build_quote(
    items: list[LineItem],
    catalog_prices: dict[str, float],
    event_type: str,
    job_description: str,
    urgency_days: int,
) -> Quote:
    quote_lines: list[QuoteLine] = []
    subtotal = 0.0
    bulk_discount_total = 0.0

    for item in items:
        unit_price = catalog_prices.get(item.sku, 0.0)
        line_subtotal = round(unit_price * item.quantity, 2)
        discount_pct = _bulk_discount(item.quantity)
        line_total = round(line_subtotal * (1 - discount_pct / 100.0), 2)
        subtotal += line_subtotal
        bulk_discount_total += line_subtotal - line_total
        quote_lines.append(
            QuoteLine(
                sku=item.sku,
                quantity=item.quantity,
                unit_price=unit_price,
                line_subtotal=line_subtotal,
                discount_pct=discount_pct,
                line_total=line_total,
            )
        )

    discounted_subtotal = subtotal - bulk_discount_total
    urgency_pct = _urgency_premium_pct(urgency_days)
    urgency_total = round(discounted_subtotal * urgency_pct / 100.0, 2)
    total = round(discounted_subtotal + urgency_total, 2)

    history = ps.search_quote_history(
        [w for w in (event_type + " " + job_description).split() if len(w) > 3][:4],
        limit=3,
    )
    rationale_parts = []
    discounted_lines = [l for l in quote_lines if l.discount_pct > 0]
    if discounted_lines:
        tiers = sorted({l.discount_pct for l in discounted_lines})
        rationale_parts.append(
            "Bulk discounts applied: " + ", ".join(f"{t:.0f}%" for t in tiers) + "."
        )
    if urgency_pct > 0:
        rationale_parts.append(f"Urgency premium of {urgency_pct:.0f}% added for the tight deadline.")
    if history:
        rationale_parts.append(
            f"Pricing benchmarked against {len(history)} similar past order(s)."
        )
    if not rationale_parts:
        rationale_parts.append("Standard list pricing applied with no bulk or urgency adjustments.")

    return Quote(
        lines=quote_lines,
        subtotal=round(subtotal, 2),
        bulk_discount_total=round(bulk_discount_total, 2),
        urgency_premium_pct=urgency_pct,
        urgency_premium_total=urgency_total,
        total_price=total,
        pricing_rationale=" ".join(rationale_parts),
    )


#: Customer-facing display names for SKUs. Internal SKU codes still appear in
#: ``SalesOutcome.reasoning`` for ops/audit purposes, but never in customer text.
_DISPLAY_NAMES: dict[str, str] = {
    "A4_paper_500":          "A4 paper (500-sheet ream)",
    "letter_paper_500":      "letter paper (500-sheet ream)",
    "cardstock_white_250":   "white cardstock (250-sheet pack)",
    "cardstock_colored_250": "colored cardstock (250-sheet pack)",
    "glossy_photo_50":       "glossy photo paper (50-sheet pack)",
    "poster_24x36":          "24x36 poster",
    "envelope_10_box":       "#10 envelope (box)",
    "envelope_invitation":   "invitation envelope (box)",
    "notebook_college":      "college-ruled notebook",
    "sticky_notes_pack":     "sticky notes pack",
}


def _display(sku: str) -> str:
    return _DISPLAY_NAMES.get(sku, sku.replace("_", " "))


# ---------------------------------------------------------------------------
# Sales
# ---------------------------------------------------------------------------

def finalize_sale(
    decision: str,
    quote: Quote,
    inventory: InventoryReport,
    request_date: str,
    deadline: str = "",
    decline_kind: str = "",
    internal_reason: str = "",
) -> SalesOutcome:
    """Record transactions on fulfilled orders and assemble the response.

    ``decline_kind`` selects the customer-facing language for declines:

    * ``"viable_restock"`` — stock would arrive in time but is not on hand;
      we offer to re-quote at the supplier ETA.
    * ``"not_viable"``     — restock ETA falls after the customer's deadline.
    * ``""`` (default)     — generic apology.

    ``internal_reason`` is preserved in ``SalesOutcome.reasoning`` for
    ops/audit purposes but never echoed into the customer reply.
    """
    cash_before = ps.get_cash_balance(request_date)
    tx_ids: list[int] = []

    if decision == "fulfilled":
        for line in quote.lines:
            tx_id = ps.create_transaction(
                item_name=line.sku,
                transaction_type="sale",
                units=line.quantity,
                price=line.line_total,
                date=request_date,
            )
            tx_ids.append(tx_id)
        cash_after = ps.get_cash_balance(request_date)
        reasoning = "All items in stock by deadline; sale recorded."
        customer_reply = _build_fulfilled_reply(quote)
    else:
        cash_after = cash_before
        reasoning = internal_reason or "Order could not be fulfilled."
        customer_reply = _build_declined_reply(inventory, deadline, decline_kind)

    return SalesOutcome(
        decision=decision,  # type: ignore[arg-type]
        customer_reply=customer_reply,
        recorded_transaction_ids=tx_ids,
        cash_balance_before=round(cash_before, 2),
        cash_balance_after=round(cash_after, 2),
        reasoning=reasoning,
    )


def _build_fulfilled_reply(quote: Quote) -> str:
    parts = ["Thank you for your order from Beaver's Choice. Here is your itemized quote:"]
    for line in quote.lines:
        discount_note = f" ({line.discount_pct:.0f}% bulk discount)" if line.discount_pct > 0 else ""
        parts.append(
            f"  - {line.quantity} x {_display(line.sku)}: ${line.line_total:,.2f}{discount_note}"
        )
    if quote.urgency_premium_pct > 0:
        parts.append(
            f"An urgency premium of {quote.urgency_premium_pct:.0f}% (${quote.urgency_premium_total:,.2f}) "
            f"was added for the expedited timeline."
        )
    parts.append(f"Total: ${quote.total_price:,.2f}. Your order is confirmed and being prepared.")
    return "\n".join(parts)


def _build_declined_reply(inventory: InventoryReport, deadline: str, decline_kind: str) -> str:
    """Customer-facing decline message in plain language.

    Uses friendly product names, surfaces ETAs factually, and never echoes
    internal logic strings like "the customer prefers immediate fulfillment".
    """
    short_lines = [l for l in inventory.lines if l.shortfall > 0]
    latest_eta = max(
        (l.eta_if_reordered for l in short_lines if l.eta_if_reordered),
        default=None,
    )

    parts = ["Thank you for your interest in Beaver's Choice."]

    if decline_kind == "not_viable":
        deadline_phrase = f"by {deadline}" if deadline else "by your requested deadline"
        parts.append(
            f"We are unable to fulfill this order {deadline_phrase}: we do not have "
            "sufficient stock on hand and a restock would not arrive in time."
        )
    elif decline_kind == "viable_restock":
        deadline_phrase = f"by {deadline}" if deadline else "by your requested deadline"
        eta_phrase = f"by {latest_eta}" if latest_eta else "shortly after that date"
        parts.append(
            f"We do not currently have enough stock on hand to fulfill your order "
            f"{deadline_phrase}. The next available stock is expected {eta_phrase}."
        )
    else:
        parts.append("Unfortunately, we are unable to fulfill this order at this time.")

    if short_lines:
        parts.append("Items affected:")
        for l in short_lines:
            parts.append(
                f"  - {_display(l.sku)}: {l.shortfall} more unit(s) needed"
            )

    if decline_kind == "viable_restock":
        parts.append(
            "If you would like a quote for delivery once the restock arrives, please "
            "let us know and we will gladly re-quote."
        )
    else:
        parts.append(
            "Please feel free to adjust quantities or extend the deadline and we will "
            "gladly re-quote."
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _items_summary(quote: Quote) -> str:
    return "; ".join(f"{l.sku} x{l.quantity}" for l in quote.lines)


def run_request(request: QuoteRequest, catalog_prices: dict[str, float]) -> QuoteResponse:
    """Run one quote request through the deterministic pipeline."""
    parsed = parse_request(request.request_text)
    request_dt = datetime.strptime(request.request_date, "%Y-%m-%d")
    deadline_dt = datetime.strptime(request.deadline, "%Y-%m-%d")
    urgency_days = (deadline_dt - request_dt).days

    if not parsed.items or parsed.parse_confidence == 0:
        cash = ps.get_cash_balance(request.request_date)
        return QuoteResponse(
            request_id=request.request_id,
            decision="declined",
            total_price=0.0,
            customer_reply=(
                "Thank you for your interest in Beaver's Choice. We were unable to "
                "interpret the items requested — could you list specific products "
                "(e.g. 'A4 paper', 'cardstock', 'envelopes') with quantities?"
            ),
            reasoning="parser could not extract any catalog items from the request",
            cash_balance_before=cash,
            cash_balance_after=cash,
            items_summary="",
        )

    if urgency_days < 0:
        cash = ps.get_cash_balance(request.request_date)
        return QuoteResponse(
            request_id=request.request_id,
            decision="declined",
            total_price=0.0,
            customer_reply=(
                "Thank you for your interest. The deadline you provided is before the "
                "request date, so we are unable to schedule this order."
            ),
            reasoning="deadline before request_date",
            cash_balance_before=cash,
            cash_balance_after=cash,
            items_summary="; ".join(f"{i.sku} x{i.quantity}" for i in parsed.items),
        )

    inventory = check_inventory(parsed.items, request.request_date, request.deadline)
    quote = build_quote(parsed.items, catalog_prices, request.event_type,
                        request.job_description, urgency_days)

    short_lines = [l for l in inventory.lines if l.shortfall > 0]
    if short_lines:
        viable = all(l.viable_by_deadline for l in short_lines)
        if viable:
            decline_kind = "viable_restock"
            internal_reason = (
                "viable_restock: shortfalls within supplier ETA — "
                + ", ".join(
                    f"{l.sku} short by {l.shortfall} (ETA {l.eta_if_reordered})"
                    for l in short_lines
                )
            )
        else:
            decline_kind = "not_viable"
            internal_reason = (
                "not_viable: shortfalls past deadline — "
                + ", ".join(
                    f"{l.sku} short by {l.shortfall} (ETA {l.eta_if_reordered})"
                    for l in short_lines
                )
            )
        outcome = finalize_sale(
            "declined",
            quote,
            inventory,
            request.request_date,
            deadline=request.deadline,
            decline_kind=decline_kind,
            internal_reason=internal_reason,
        )
    else:
        outcome = finalize_sale(
            "fulfilled",
            quote,
            inventory,
            request.request_date,
            deadline=request.deadline,
        )

    return QuoteResponse(
        request_id=request.request_id,
        decision=outcome.decision,
        total_price=quote.total_price if outcome.decision == "fulfilled" else 0.0,
        customer_reply=outcome.customer_reply,
        reasoning=outcome.reasoning,
        cash_balance_before=outcome.cash_balance_before,
        cash_balance_after=outcome.cash_balance_after,
        items_summary=_items_summary(quote),
    )
