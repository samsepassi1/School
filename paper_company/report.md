# Beaver's Choice — Reflection Report

## 1. Multi-agent system overview

The system is built with **pydantic-ai** and processes free-text quote
requests for a paper-products company. Five agents collaborate behind a
single `OrchestratorAgent` entry point:

1. **OrchestratorAgent** — receives the `QuoteRequest`, calls each
   specialist in turn, applies the deterministic fulfill/decline branch,
   and returns a `QuoteResponse`.
2. **RequestParserAgent** — extracts canonical SKUs and quantities from
   the customer's free text. The catalog is injected into the system
   prompt so the LLM can map "200 reams of A4 paper" → `A4_paper_500 x 200`.
3. **InventoryAgent** — calls `get_all_inventory`, computes shortfalls,
   and (when needed) calls `get_supplier_delivery_date` to estimate when
   missing stock would arrive. It returns one `InventoryLine` per
   requested SKU.
4. **QuotingAgent** — applies bulk discounts (5%/10%/15% on quantities
   ≥50/200/500) and an urgency premium (+4% or +8% on tight deadlines),
   grounded against similar past orders via `search_quote_history`.
5. **SalesAgent** — reads the cash balance, writes one `sale`
   transaction per line on fulfilled orders, and writes the customer
   reply. It never surfaces supplier prices, margins, or internal SKU
   codes the customer didn't ask about.

### Why this architecture?

- **Orchestrator-driven sequential flow** beats supervisor-style
  hand-off because the steps are strictly ordered (parse → inventory
  check → price → fulfill). Putting the routing in deterministic Python
  makes the run reproducible across LLM versions and lets us evaluate
  outcomes without LLM variance contaminating the decision.
- **One Pydantic IO model per step** (`ParsedItems`, `InventoryReport`,
  `Quote`, `SalesOutcome`, `QuoteResponse`) keeps the contract between
  agents explicit and validates outputs at every boundary.
- **Tools wrap helpers, not the other way around** — the seven helper
  functions in `project_starter.py` are the single source of truth for
  database operations, and the tool wrappers exist only to give the LLM
  a clean docstring-driven schema.
- **A deterministic fallback pipeline** (`agents/deterministic.py`)
  mirrors the LLM workflow with regex-based parsing and the same pricing
  rules. It runs offline (no API key required) and is what produced the
  submitted `test_results.csv`. The LLM path is enabled by setting
  `OPENAI_API_KEY` and unsetting `PAPER_USE_DETERMINISTIC`.

## 2. Evaluation results

The system was evaluated against the full 20-row `quote_requests_sample.csv`
(in deterministic mode). Summary:

| Metric | Value |
|---|---|
| Total requests | 20 |
| Fulfilled | 12 |
| Declined | 8 |
| Cash balance start | $50,000.00 |
| Cash balance end | $56,294.75 |
| Cash delta | +$6,294.75 |
| Sensitive-term leaks in customer replies | 0 |

### Strengths

- **Clear customer-facing reasoning.** Every fulfilled reply lists each
  line with its discount tier and total; every declined reply states the
  specific reason (e.g. "stock would not arrive before your deadline" or
  "could not interpret the items requested").
- **Bulk discounts and urgency premiums are applied consistently.**
  `R007` (150 reams of A4 by Friday) shows both: a 5% bulk discount and a
  4% urgency premium, computed deterministically from the same rules the
  LLM is instructed to follow.
- **Realistic decline coverage across multiple decline modes.** The
  declined set spans: parse-failure (`R010`, `R011`), deadline-not-viable
  stock shortage (`R008`, `R009`, `R017`), and viable-restock-but-not-on-
  hand (`R015`, `R016`, `R020`).
- **All seven helper functions are exercised.** `create_transaction`,
  `get_all_inventory`, `get_stock_level`, `get_supplier_delivery_date`,
  `get_cash_balance`, `generate_financial_report`, and
  `search_quote_history` each appear in at least one tool wrapper.
- **No sensitive-term leaks.** The eval's regression check
  (`SENSITIVE_RE`) finds zero customer replies containing "supplier",
  "margin", or similar internal vocabulary.
- **Reproducibility.** Running `python run_eval.py` always resets the DB
  to the seeded opening state before processing the CSV, so results are
  deterministic regardless of when the script is run.

### Areas for improvement

- **No auto-restock path.** Today, any shortfall results in a decline,
  even when the supplier ETA fits within the deadline. We surface the
  ETA to the customer but never actually trigger a restock + delayed
  fulfillment. Three of the eight declines (R015, R016, R020) would have
  been fulfillable revenue with this capability.
- **Brittle free-text parsing in deterministic mode.** The regex-based
  parser relies on quantities sitting next to product nouns. Phrasings
  like "we'd like to look into ordering some envelopes" wouldn't parse.
  The LLM path handles these naturally; the deterministic path does not.
- **Pricing rationale is template-driven.** Customer replies in
  deterministic mode use a fixed template. Replies in LLM mode can vary
  in tone but the rationale section is still mechanical.
- **No negotiation loop.** The customer cannot push back on the price or
  request a counter-offer; the system returns a single quote.

## 3. Suggestions for further improvements

1. **Add a `ProcurementAgent` that auto-restocks viable shortfalls.**
   When `viable_by_deadline=True` for every short line, the orchestrator
   should issue a `stock_orders` transaction at supplier price, log the
   ETA in the customer reply, and fulfill the order at the projected
   delivery date instead of declining. This converts ~3 declined
   requests per 20-row eval into completed sales while preserving margin
   and respecting deadlines.

2. **Replace the regex parser with an LLM "first-pass + schema-validated
   retry."** Keep the deterministic path as a fallback but use the LLM
   parser as the default — even in offline tests via VCR-style cassettes
   for reproducibility. This would handle ambiguous phrasing
   ("a couple of cases of paper") that the regex misses without changing
   the rest of the architecture.

3. **Add a `BusinessAdvisorAgent` (mentioned in the rubric stand-out
   suggestions) that runs after every N requests, reads the
   `transactions` and `quote_history` tables, and emits proactive
   recommendations** — e.g. "glossy_photo_50 has hit its min_stock; raise
   the standing reorder quantity" or "marketing-event quotes have a 70%
   fulfillment rate; consider holding more poster_24x36 in Q2." This
   closes the loop from operations back into business strategy and
   demonstrates the multi-agent system as more than a pass-through
   pipeline.

4. **Add a customer "negotiation" agent.** Today the system returns a
   single quote and stops; in real sales, a small concession (e.g.
   waiving urgency premium for repeat customers) often closes deals.
   This would also exercise `search_quote_history` more deeply by
   weighting historical pricing per customer.

5. **Persist per-customer history.** The current schema has no `customer`
   column on `transactions` or `quote_history`. Adding one would enable
   loyalty discounts and personalized pricing in the QuotingAgent.

## 4. Files in this submission

| File | Purpose |
|---|---|
| `diagram.md` | Mermaid agent workflow diagram + tool/helper mapping |
| `project_starter.py` | SQLite schema, seed data, all 7 helper functions |
| `models/schemas.py` | Pydantic IO models for every agent |
| `tools/{inventory,quoting,sales}_tools.py` | Tool wrappers around the helpers |
| `agents/{parser,inventory,quoting,sales}_agent.py` | pydantic-ai specialist agents |
| `agents/orchestrator.py` | LLM-driven orchestrator |
| `agents/deterministic.py` | Offline equivalent pipeline |
| `quote_requests_sample.csv` | 20-row evaluation set |
| `run_eval.py` | Batch runner that produces `test_results.csv` |
| `test_results.csv` | Submitted evaluation output |
| `main.py` | Single-request CLI demo |
| `report.md` | This reflection report |
