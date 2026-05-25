# https://www.mermaidchart.com/app/projects/b63a9376-3040-4666-a51d-2dafa52b02a7/diagrams/fa5469ac-68bf-49d9-ac86-462d84a67a4d/share/invite/eyJhbG...3DVI

erDiagram
    %% =========================
    %% DIMENSIONS
    %% =========================
    dw_dim_date {
      INT date_key PK "yyyymmdd"
      DATE date_actual
      SMALLINT year
      SMALLINT quarter
      SMALLINT month
      SMALLINT day
      SMALLINT week_of_year
      SMALLINT day_of_week
      BOOLEAN is_weekend
    }

    dw_dim_customer {
      BIGINT customer_sk PK
      VARCHAR customer_id
      VARCHAR country
      VARCHAR state
      VARCHAR customer_segment
      BOOLEAN is_logged_in
      TIMESTAMP effective_from
      TIMESTAMP effective_to
      BOOLEAN is_current
    }

    dw_dim_product {
      BIGINT product_sk PK
      VARCHAR product_id
      VARCHAR category
      VARCHAR price_bucket
      DECIMAL current_unit_price_usd
      TIMESTAMP effective_from
      TIMESTAMP effective_to
      BOOLEAN is_current
    }

    dw_dim_campaign    { BIGINT campaign_sk PK  VARCHAR campaign }
    dw_dim_channel     { BIGINT channel_sk  PK  VARCHAR channel }
    dw_dim_device      { BIGINT device_sk   PK  VARCHAR device_type }
    dw_dim_browser     { BIGINT browser_sk  PK  VARCHAR browser }
    dw_dim_os          { BIGINT os_sk       PK  VARCHAR os }
    dw_dim_referrer    { BIGINT referrer_sk PK  VARCHAR referrer }
    dw_dim_shipmethod  { BIGINT shipping_method_sk PK VARCHAR shipping_method }
    dw_dim_paymethod   { BIGINT payment_method_sk  PK VARCHAR payment_method }
    dw_dim_ab_variant  { BIGINT ab_variant_sk PK VARCHAR ab_variant }

    %% =========================
    %% FACTS
    %% =========================
    dw_fact_orders {
      BIGINT order_sk PK
      VARCHAR order_id
      BIGINT customer_sk FK
      INT    order_date_key FK
      INT    ship_date_key  FK
      BIGINT channel_sk FK
      BIGINT device_sk  FK
      BIGINT browser_sk FK
      BIGINT campaign_sk FK
      BIGINT payment_method_sk FK
      BIGINT shipping_method_sk FK
      VARCHAR primary_category
      INT    num_distinct_items
      DECIMAL subtotal_usd
      DECIMAL discount_rate
      DECIMAL discount_amount_usd
      DECIMAL shipping_cost_usd
      DECIMAL tax_rate
      DECIMAL tax_amount_usd
      DECIMAL order_total_usd
      DECIMAL order_weight_kg
      INT    delivery_days
      BOOLEAN on_time_delivery
      BOOLEAN authorization_approved
      BOOLEAN returned
    }

    dw_fact_events {
      BIGINT event_sk PK
      VARCHAR event_id
      BIGINT customer_sk FK
      BIGINT product_sk  FK
      INT    event_date_key FK
      VARCHAR session_id
      VARCHAR event_type
      BIGINT channel_sk  FK
      BIGINT device_sk   FK
      BIGINT browser_sk  FK
      BIGINT os_sk       FK
      BIGINT referrer_sk FK
      BIGINT ab_variant_sk FK
      INT    page_depth
      INT    latency_ms
      INT    dwell_seconds
      DECIMAL cart_value_usd
      DECIMAL discount_rate
      DECIMAL fraud_score
      VARCHAR payment_outcome
      INT    sequence_num
      VARCHAR category
      VARCHAR promo_code
    }

    dw_fact_graph_edges {
      BIGINT edge_sk PK
      VARCHAR edge_id
      INT    event_date_key FK
      VARCHAR relationship

      BIGINT from_customer_sk FK
      BIGINT to_customer_sk   FK
      BIGINT from_product_sk  FK
      BIGINT to_product_sk    FK

      VARCHAR order_id
      VARCHAR category
      BIGINT  campaign_sk FK
      VARCHAR customer_segment
      VARCHAR region
      VARCHAR state

      DECIMAL edge_strength
      VARCHAR price_bucket
      INT     prior_interactions
      INT     dwell_seconds
      DECIMAL unit_price_usd
      INT     quantity
      BOOLEAN returned_flag
      BOOLEAN auth_approved
    }

    %% =========================
    %% RELATIONSHIPS
    %% =========================
    dw_dim_date ||--o{ dw_fact_orders : "order_date_key"
    dw_dim_date ||--o{ dw_fact_orders : "ship_date_key"
    dw_dim_date ||--o{ dw_fact_events : "event_date_key"
    dw_dim_date ||--o{ dw_fact_graph_edges : "event_date_key"

    dw_dim_customer ||--o{ dw_fact_orders : "customer_sk"
    dw_dim_customer ||--o{ dw_fact_events : "customer_sk"
    dw_dim_customer ||--o{ dw_fact_graph_edges : "from_customer_sk"
    dw_dim_customer ||--o{ dw_fact_graph_edges : "to_customer_sk"

    dw_dim_product ||--o{ dw_fact_events : "product_sk"
    dw_dim_product ||--o{ dw_fact_graph_edges : "from_product_sk"
    dw_dim_product ||--o{ dw_fact_graph_edges : "to_product_sk"

    dw_dim_campaign ||--o{ dw_fact_orders : "campaign_sk"
    dw_dim_campaign ||--o{ dw_fact_graph_edges : "campaign_sk"

    dw_dim_channel ||--o{ dw_fact_orders : "channel_sk"
    dw_dim_channel ||--o{ dw_fact_events : "channel_sk"

    dw_dim_device ||--o{ dw_fact_orders : "device_sk"
    dw_dim_device ||--o{ dw_fact_events : "device_sk"

    dw_dim_browser ||--o{ dw_fact_orders : "browser_sk"
    dw_dim_browser ||--o{ dw_fact_events : "browser_sk"

    dw_dim_os ||--o{ dw_fact_events : "os_sk"
    dw_dim_referrer ||--o{ dw_fact_events : "referrer_sk"
    dw_dim_ab_variant ||--o{ dw_fact_events : "ab_variant_sk"

    dw_dim_shipmethod ||--o{ dw_fact_orders : "shipping_method_sk"
    dw_dim_paymethod  ||--o{ dw_fact_orders : "payment_method_sk"
