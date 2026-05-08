# Beaver's Choice — Multi-Agent Workflow

This diagram shows the agents, their tools (mapped to the helper functions
in `project_starter.py`), and the data flow for a single quote request.

```mermaid
flowchart TD
    Customer([Customer Request<br/>free text + dates])
    Orchestrator{{OrchestratorAgent<br/>sequential pipeline + decision branch}}
    Parser[/RequestParserAgent/]
    Inventory[/InventoryAgent/]
    Quoting[/QuotingAgent/]
    Sales[/SalesAgent/]

    %% Tools — one per helper function
    T_GetAll[(tool_get_all_inventory<br/>→ get_all_inventory)]
    T_GetStock[(tool_get_stock_level<br/>→ get_stock_level)]
    T_ETA[(tool_get_supplier_eta<br/>→ get_supplier_delivery_date)]
    T_History[(tool_search_quote_history<br/>→ search_quote_history)]
    T_Tx[(tool_create_transaction<br/>→ create_transaction)]
    T_Cash[(tool_get_cash_balance<br/>→ get_cash_balance)]
    T_Report[(tool_financial_report<br/>→ generate_financial_report)]

    %% Database
    DB[(SQLite<br/>inventory + transactions<br/>+ quote_history)]

    Customer --> Orchestrator
    Orchestrator -->|raw text| Parser
    Parser -->|ParsedItems| Orchestrator

    Orchestrator -->|items + dates| Inventory
    Inventory --> T_GetAll
    Inventory --> T_GetStock
    Inventory --> T_ETA
    Inventory -->|InventoryReport| Orchestrator

    Orchestrator -->|items + inventory + urgency| Quoting
    Quoting --> T_History
    Quoting --> T_GetStock
    Quoting -->|Quote| Orchestrator

    Orchestrator -- "fulfill or decline<br/>(deterministic)" --> Sales
    Sales --> T_Cash
    Sales --> T_Tx
    Sales --> T_Report
    Sales -->|SalesOutcome| Orchestrator

    Orchestrator -->|QuoteResponse<br/>customer reply + status| Customer

    T_GetAll  --- DB
    T_GetStock --- DB
    T_ETA --- DB
    T_History --- DB
    T_Tx --- DB
    T_Cash --- DB
    T_Report --- DB
```

## Agent responsibilities

| Agent | Responsibility | Tools |
|---|---|---|
| **OrchestratorAgent** | Receives the customer request, sequences the four specialists, applies the deterministic fulfill/decline branch, and returns the final `QuoteResponse`. | (delegates to other agents) |
| **RequestParserAgent** | Maps free-text customer language to canonical SKUs and integer quantities (`ParsedItems`). Returns `parse_confidence=0` for unparseable requests. | none — catalog injected into prompt |
| **InventoryAgent** | For each requested SKU computes on-hand stock, shortfall, and supplier ETA if a restock would be needed (`InventoryReport`). | `tool_get_all_inventory`, `tool_get_stock_level`, `tool_get_supplier_eta` |
| **QuotingAgent** | Builds line items with bulk discounts (5/10/15% tiers) and an urgency premium (4 or 8% on tight deadlines), grounded against similar past quotes (`Quote`). | `tool_search_quote_history`, `tool_get_stock_level` |
| **SalesAgent** | Reads cash balance, writes `sale` transactions on fulfilled orders, and produces the customer-facing reply (`SalesOutcome`). | `tool_create_transaction`, `tool_get_cash_balance`, `tool_financial_report` |

## Tool ↔ helper-function mapping (rubric coverage)

| Helper function (`project_starter.py`) | Tool wrapper | Used by agent |
|---|---|---|
| `get_all_inventory` | `tool_get_all_inventory` | Inventory |
| `get_stock_level` | `tool_get_stock_level` | Inventory, Quoting |
| `get_supplier_delivery_date` | `tool_get_supplier_eta` | Inventory |
| `search_quote_history` | `tool_search_quote_history` | Quoting |
| `create_transaction` | `tool_create_transaction` | Sales |
| `get_cash_balance` | `tool_get_cash_balance` | Sales |
| `generate_financial_report` | `tool_financial_report` | Sales |

All seven helper functions are wired into at least one tool used by at
least one agent.

## Decision branch (deterministic, in `agents/orchestrator.py`)

```
if parsed.items is empty           → declined ("could not interpret")
elif deadline < request_date        → declined ("deadline before request_date")
elif any line has shortfall > 0
   and not viable_by_deadline       → declined ("stock won't arrive in time")
elif any line has shortfall > 0
   and     viable_by_deadline       → declined ("restock viable; offer alt timeline")
else                                → fulfilled (write transactions)
```

The decision is computed in plain Python so the eval is reproducible —
the LLM is responsible only for parsing free text, calling tools, and
writing the customer-friendly reply.
