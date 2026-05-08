# Data model

Two SQLite databases live under `data/`:

- `data/external/cultpass.db` — owned by **CultPass** (the customer). UDA-Hub reads it via tools to enrich tickets.
- `data/core/uda_hub.db` — owned by **UDA-Hub**. Contains the rubric-required tables.

## External (CultPass)

| Table              | Columns                                                                                  |
|--------------------|------------------------------------------------------------------------------------------|
| `CultPassMember`   | `member_id` PK, `full_name`, `email` UQ, `tier` (classic/plus/elite), `status`, `city`, `joined_at` |
| `CultPassEvent`    | `event_id` PK, `title`, `venue`, `city`, `starts_at`, `capacity`, `price_cents`, `category` |
| `CultPassBooking`  | `booking_id` PK, `member_id` FK, `event_id` FK, `status`, `plus_one`, `booked_at`        |
| `CultPassPayment`  | `payment_id` PK, `member_id` FK, `amount_cents`, `kind` (subscription/event/refund), `status`, `created_at` |

## Core (UDA-Hub)

| Table             | Columns                                                                                                                                            |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `Account`         | `account_id` PK, `name`, `plan`, `status`, `balance_cents`, `external_member_id` UQ → `CultPassMember.member_id`, `created_at`                     |
| `User`            | `user_id` PK, `account_id` FK, `email` UQ, `full_name`, `locale`, `created_at`                                                                     |
| `Ticket`          | `ticket_id` PK, `user_id` FK, `subject`, `body`, `status` (open/in_progress/resolved/escalated/closed), `created_at`, `updated_at`                  |
| `TicketMetadata`  | `ticket_id` PK FK, `channel`, `urgency`, `category`, `sentiment`, `confidence`, `routed_to`, `extra_json`                                          |
| `TicketMessage`   | `message_id` PK auto, `ticket_id` FK, `role` (customer/agent/system/tool), `author`, `content`, `created_at`                                       |
| `Knowledge`       | `article_id` PK, `title`, `category`, `tags`, `body`, `updated_at` — populated from `data/external/cultpass_articles.jsonl` (16 articles)         |

The `Account.external_member_id` foreign-id link is what lets the
`cultpass_member_lookup` tool join from a UDA-Hub user to their CultPass
member profile and payments.
