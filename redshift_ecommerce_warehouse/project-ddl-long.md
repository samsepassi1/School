# Redshift Data Warehouse DDL

This DDL defines the **staging** tables (raw landing zone) and the **warehouse star schema** (dimensions and facts) for the e-commerce analytics data warehouse.

## Source Data Mapping

| Source CSV | Staging Table | Target Fact Table |
|------------|---------------|-------------------|
| `ecom_orders_postgres.csv` | `stg.orders_raw` | `dw.fact_orders` |
| `ecom_events_cassandra.csv` | `stg.events_raw` | `dw.fact_events` |
| `ecom_graph_edges_neo4j.csv` | `stg.edges_raw` | `dw.fact_graph_edges` |

---

# 1) Schemas

```sql
CREATE SCHEMA IF NOT EXISTS stg;
CREATE SCHEMA IF NOT EXISTS dw;
```

---

# 2) Staging Tables

> Staging is deliberately permissive: wide VARCHARs, minimal constraints. Timestamps as TIMESTAMP, numerics as DECIMAL where obvious.

```sql
-- ========== ORDERS (from PostgreSQL) ==========
DROP TABLE IF EXISTS stg.orders_raw;
CREATE TABLE stg.orders_raw (
  order_id              VARCHAR(32),
  customer_id           VARCHAR(32),
  order_datetime        TIMESTAMP,
  ship_datetime         TIMESTAMP,
  channel               VARCHAR(32),
  device_type           VARCHAR(16),
  browser               VARCHAR(16),
  country               VARCHAR(8),
  state                 VARCHAR(8),
  payment_method        VARCHAR(16),
  campaign              VARCHAR(32),
  primary_category      VARCHAR(32),
  num_distinct_items    INTEGER,
  subtotal_usd          DECIMAL(12,2),
  discount_rate         DECIMAL(5,3),
  discount_amount_usd   DECIMAL(12,2),
  shipping_method       VARCHAR(16),
  shipping_cost_usd     DECIMAL(12,2),
  tax_rate              DECIMAL(6,4),
  tax_amount_usd        DECIMAL(12,2),
  order_total_usd       DECIMAL(12,2),
  order_weight_kg       DECIMAL(10,2),
  delivery_days         INTEGER,
  on_time_delivery      BOOLEAN,
  authorization_approved BOOLEAN,
  returned              BOOLEAN
);

-- ========== EVENTS (from Cassandra) ==========
DROP TABLE IF EXISTS stg.events_raw;
CREATE TABLE stg.events_raw (
  event_id          VARCHAR(32),
  customer_id       VARCHAR(32),
  session_id        VARCHAR(32),
  event_type        VARCHAR(32),
  event_ts          TIMESTAMP,
  device_type       VARCHAR(16),
  browser           VARCHAR(16),
  os                VARCHAR(16),
  referrer          VARCHAR(32),
  country           VARCHAR(8),
  state             VARCHAR(8),
  ab_variant        VARCHAR(4),
  is_logged_in      BOOLEAN,
  page_depth        INTEGER,
  latency_ms        INTEGER,
  dwell_seconds     INTEGER,
  cart_value_usd    DECIMAL(12,2),
  discount_rate     DECIMAL(5,3),
  fraud_score       DECIMAL(6,3),
  payment_outcome   VARCHAR(16),
  sequence_num      INTEGER,
  product_id        VARCHAR(32),
  category          VARCHAR(32),
  promo_code        VARCHAR(32)
);

-- ========== GRAPH EDGES (from Neo4j) ==========
DROP TABLE IF EXISTS stg.edges_raw;
CREATE TABLE stg.edges_raw (
  edge_id            VARCHAR(32),
  from_node_id       VARCHAR(32),
  from_node_type     VARCHAR(16),
  to_node_id         VARCHAR(32),
  to_node_type       VARCHAR(16),
  relationship       VARCHAR(32),
  timestamp          TIMESTAMP,
  order_id           VARCHAR(32),
  category           VARCHAR(32),
  customer_segment   VARCHAR(16),
  edge_strength      DECIMAL(6,3),
  price_bucket       VARCHAR(16),
  region             VARCHAR(16),
  state              VARCHAR(8),
  campaign           VARCHAR(32),
  same_household     BOOLEAN,
  prior_interactions INTEGER,
  dwell_seconds      INTEGER,
  product_id         VARCHAR(32),
  unit_price_usd     DECIMAL(12,2),
  quantity           INTEGER,
  returned_flag      BOOLEAN,
  auth_approved      BOOLEAN
);
```

---

# 3) Dimension Tables

> Surrogate keys use `IDENTITY`. Small lookup dimensions use `DISTSTYLE ALL` for broadcast joins. Larger dimensions (customer, product) use `DISTKEY` on the business key for collocated joins.

```sql
-- ========== DATE DIMENSION ==========
DROP TABLE IF EXISTS dw.dim_date;
CREATE TABLE dw.dim_date (
  date_key        INTEGER   NOT NULL,  -- yyyymmdd format
  date_actual     DATE      NOT NULL,
  year            SMALLINT  NOT NULL,
  quarter         SMALLINT  NOT NULL,
  month           SMALLINT  NOT NULL,
  day             SMALLINT  NOT NULL,
  week_of_year    SMALLINT  NOT NULL,
  day_of_week     SMALLINT  NOT NULL,
  is_weekend      BOOLEAN   NOT NULL,
  PRIMARY KEY (date_key)
)
DISTSTYLE ALL
SORTKEY (date_key);

-- ========== CUSTOMER DIMENSION (SCD Type 2 ready) ==========
DROP TABLE IF EXISTS dw.dim_customer;
CREATE TABLE dw.dim_customer (
  customer_sk      BIGINT IDENTITY(1,1),
  customer_id      VARCHAR(32) ENCODE zstd,
  country          VARCHAR(8)  ENCODE zstd,
  state            VARCHAR(8)  ENCODE zstd,
  customer_segment VARCHAR(16) ENCODE zstd,
  is_logged_in     BOOLEAN     ENCODE zstd,
  effective_from   TIMESTAMP   ENCODE zstd,
  effective_to     TIMESTAMP   ENCODE zstd,
  is_current       BOOLEAN     ENCODE zstd,
  PRIMARY KEY (customer_sk)
)
DISTKEY(customer_id)
SORTKEY(customer_id);

-- ========== PRODUCT DIMENSION (SCD Type 2 ready) ==========
DROP TABLE IF EXISTS dw.dim_product;
CREATE TABLE dw.dim_product (
  product_sk             BIGINT IDENTITY(1,1),
  product_id             VARCHAR(32) ENCODE zstd,
  category               VARCHAR(32) ENCODE zstd,
  price_bucket           VARCHAR(16) ENCODE zstd,
  current_unit_price_usd DECIMAL(12,2) ENCODE zstd,
  effective_from         TIMESTAMP   ENCODE zstd,
  effective_to           TIMESTAMP   ENCODE zstd,
  is_current             BOOLEAN     ENCODE zstd,
  PRIMARY KEY (product_sk)
)
DISTKEY(product_id)
SORTKEY(product_id);

-- ========== CAMPAIGN DIMENSION ==========
DROP TABLE IF EXISTS dw.dim_campaign;
CREATE TABLE dw.dim_campaign (
  campaign_sk  BIGINT IDENTITY(1,1),
  campaign     VARCHAR(32) ENCODE zstd,
  PRIMARY KEY (campaign_sk)
)
DISTSTYLE ALL;

-- ========== CHANNEL DIMENSION ==========
DROP TABLE IF EXISTS dw.dim_channel;
CREATE TABLE dw.dim_channel (
  channel_sk  BIGINT IDENTITY(1,1),
  channel     VARCHAR(32) ENCODE zstd,
  PRIMARY KEY (channel_sk)
) DISTSTYLE ALL;

-- ========== DEVICE DIMENSION ==========
DROP TABLE IF EXISTS dw.dim_device;
CREATE TABLE dw.dim_device (
  device_sk   BIGINT IDENTITY(1,1),
  device_type VARCHAR(16) ENCODE zstd,
  PRIMARY KEY (device_sk)
) DISTSTYLE ALL;

-- ========== BROWSER DIMENSION ==========
DROP TABLE IF EXISTS dw.dim_browser;
CREATE TABLE dw.dim_browser (
  browser_sk  BIGINT IDENTITY(1,1),
  browser     VARCHAR(16) ENCODE zstd,
  PRIMARY KEY (browser_sk)
) DISTSTYLE ALL;

-- ========== OS DIMENSION ==========
DROP TABLE IF EXISTS dw.dim_os;
CREATE TABLE dw.dim_os (
  os_sk  BIGINT IDENTITY(1,1),
  os     VARCHAR(16) ENCODE zstd,
  PRIMARY KEY (os_sk)
) DISTSTYLE ALL;

-- ========== REFERRER DIMENSION ==========
DROP TABLE IF EXISTS dw.dim_referrer;
CREATE TABLE dw.dim_referrer (
  referrer_sk BIGINT IDENTITY(1,1),
  referrer    VARCHAR(32) ENCODE zstd,
  PRIMARY KEY (referrer_sk)
) DISTSTYLE ALL;

-- ========== SHIPPING METHOD DIMENSION ==========
DROP TABLE IF EXISTS dw.dim_shipping_method;
CREATE TABLE dw.dim_shipping_method (
  shipping_method_sk BIGINT IDENTITY(1,1),
  shipping_method    VARCHAR(16) ENCODE zstd,
  PRIMARY KEY (shipping_method_sk)
) DISTSTYLE ALL;

-- ========== PAYMENT METHOD DIMENSION ==========
DROP TABLE IF EXISTS dw.dim_payment_method;
CREATE TABLE dw.dim_payment_method (
  payment_method_sk BIGINT IDENTITY(1,1),
  payment_method    VARCHAR(16) ENCODE zstd,
  PRIMARY KEY (payment_method_sk)
) DISTSTYLE ALL;

-- ========== A/B VARIANT DIMENSION ==========
DROP TABLE IF EXISTS dw.dim_ab_variant;
CREATE TABLE dw.dim_ab_variant (
  ab_variant_sk BIGINT IDENTITY(1,1),
  ab_variant    VARCHAR(4) ENCODE zstd,
  PRIMARY KEY (ab_variant_sk)
) DISTSTYLE ALL;
```

---

# 4) Fact Tables

> **Grain definitions:**
> - `fact_orders`: One row per order (order_id is unique)
> - `fact_events`: One row per event (event_id is unique)
> - `fact_graph_edges`: One row per graph edge (edge_id is unique)
>
> **Distribution strategy:**
> - Orders and events distributed by `customer_sk` for customer-centric analysis
> - Graph edges distributed by `to_product_sk` for product-centric analysis
> - All facts sorted by date key for time-range query optimization

```sql
-- ========== FACT: ORDERS ==========
DROP TABLE IF EXISTS dw.fact_orders;
CREATE TABLE dw.fact_orders (
  order_sk              BIGINT IDENTITY(1,1),
  order_id              VARCHAR(32) ENCODE zstd,
  customer_sk           BIGINT      ENCODE zstd,
  order_date_key        INTEGER     ENCODE zstd,
  ship_date_key         INTEGER     ENCODE zstd,
  channel_sk            BIGINT      ENCODE zstd,
  device_sk             BIGINT      ENCODE zstd,
  browser_sk            BIGINT      ENCODE zstd,
  campaign_sk           BIGINT      ENCODE zstd,
  payment_method_sk     BIGINT      ENCODE zstd,
  shipping_method_sk    BIGINT      ENCODE zstd,
  primary_category      VARCHAR(32) ENCODE zstd,

  num_distinct_items    INTEGER       ENCODE zstd,
  subtotal_usd          DECIMAL(12,2) ENCODE zstd,
  discount_rate         DECIMAL(5,3)  ENCODE zstd,
  discount_amount_usd   DECIMAL(12,2) ENCODE zstd,
  shipping_cost_usd     DECIMAL(12,2) ENCODE zstd,
  tax_rate              DECIMAL(6,4)  ENCODE zstd,
  tax_amount_usd        DECIMAL(12,2) ENCODE zstd,
  order_total_usd       DECIMAL(12,2) ENCODE zstd,
  order_weight_kg       DECIMAL(10,2) ENCODE zstd,
  delivery_days         INTEGER       ENCODE zstd,
  on_time_delivery      BOOLEAN       ENCODE zstd,
  authorization_approved BOOLEAN      ENCODE zstd,
  returned              BOOLEAN       ENCODE zstd
)
DISTKEY (customer_sk)
SORTKEY (order_date_key);

-- ========== FACT: EVENTS ==========
DROP TABLE IF EXISTS dw.fact_events;
CREATE TABLE dw.fact_events (
  event_sk          BIGINT IDENTITY(1,1),
  event_id          VARCHAR(32) ENCODE zstd,
  customer_sk       BIGINT      ENCODE zstd,
  product_sk        BIGINT      ENCODE zstd,
  event_date_key    INTEGER     ENCODE zstd,
  session_id        VARCHAR(32) ENCODE zstd,

  event_type        VARCHAR(32) ENCODE zstd,
  channel_sk        BIGINT      ENCODE zstd,
  device_sk         BIGINT      ENCODE zstd,
  browser_sk        BIGINT      ENCODE zstd,
  os_sk             BIGINT      ENCODE zstd,
  referrer_sk       BIGINT      ENCODE zstd,
  ab_variant_sk     BIGINT      ENCODE zstd,

  page_depth        INTEGER       ENCODE zstd,
  latency_ms        INTEGER       ENCODE zstd,
  dwell_seconds     INTEGER       ENCODE zstd,
  cart_value_usd    DECIMAL(12,2) ENCODE zstd,
  discount_rate     DECIMAL(5,3)  ENCODE zstd,
  fraud_score       DECIMAL(6,3)  ENCODE zstd,
  payment_outcome   VARCHAR(16)   ENCODE zstd,
  sequence_num      INTEGER       ENCODE zstd,
  category          VARCHAR(32)   ENCODE zstd,
  promo_code        VARCHAR(32)   ENCODE zstd
)
DISTKEY (customer_sk)
SORTKEY (event_date_key);

-- ========== FACT: GRAPH EDGES ==========
DROP TABLE IF EXISTS dw.fact_graph_edges;
CREATE TABLE dw.fact_graph_edges (
  edge_sk            BIGINT IDENTITY(1,1),
  edge_id            VARCHAR(32) ENCODE zstd,
  event_date_key     INTEGER     ENCODE zstd,
  relationship       VARCHAR(32) ENCODE zstd,

  from_customer_sk   BIGINT      ENCODE zstd,
  to_customer_sk     BIGINT      ENCODE zstd,
  from_product_sk    BIGINT      ENCODE zstd,
  to_product_sk      BIGINT      ENCODE zstd,

  order_id           VARCHAR(32) ENCODE zstd,
  category           VARCHAR(32) ENCODE zstd,
  campaign_sk        BIGINT      ENCODE zstd,
  customer_segment   VARCHAR(16) ENCODE zstd,
  region             VARCHAR(16) ENCODE zstd,
  state              VARCHAR(8)  ENCODE zstd,

  edge_strength      DECIMAL(6,3)  ENCODE zstd,
  price_bucket       VARCHAR(16)   ENCODE zstd,
  prior_interactions INTEGER       ENCODE zstd,
  dwell_seconds      INTEGER       ENCODE zstd,
  unit_price_usd     DECIMAL(12,2) ENCODE zstd,
  quantity           INTEGER       ENCODE zstd,
  returned_flag      BOOLEAN       ENCODE zstd,
  auth_approved      BOOLEAN       ENCODE zstd
)
DISTKEY (to_product_sk)
SORTKEY (event_date_key);
```

---

# 5) Lookup Views (Optional Helpers)

> These views simplify dimension lookups during fact table population.

```sql
CREATE OR REPLACE VIEW dw.v_lookup_customer AS
SELECT customer_sk, customer_id FROM dw.dim_customer WHERE is_current = TRUE;

CREATE OR REPLACE VIEW dw.v_lookup_product AS
SELECT product_sk, product_id FROM dw.dim_product WHERE is_current = TRUE;
```

---

# Schema Design Rationale

## Distribution Keys
- **Customer-centric facts** (`fact_orders`, `fact_events`): Distributed by `customer_sk` to collocate customer data for analytics queries
- **Product-centric facts** (`fact_graph_edges`): Distributed by `to_product_sk` for product relationship analysis
- **Small dimensions**: Use `DISTSTYLE ALL` for broadcast joins (no shuffling needed)

## Sort Keys
- All fact tables sorted by date key for efficient time-range queries
- Dimension tables sorted by business key for merge joins

## Encoding
- `ENCODE zstd` on most columns for optimal compression
- Redshift auto-tunes encoding during COPY operations

## SCD Support
- Customer and product dimensions include `effective_from`, `effective_to`, and `is_current` columns for Type 2 slowly changing dimension support
