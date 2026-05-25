-- Bronze row counts and duplicate primary key validation
SELECT 'orders' table_name, count(*) rows FROM ecommerce_analytics_bronze_dev.orders
UNION ALL SELECT 'users', count(*) FROM ecommerce_analytics_bronze_dev.users
UNION ALL SELECT 'products', count(*) FROM ecommerce_analytics_bronze_dev.products
UNION ALL SELECT 'events', count(*) FROM ecommerce_analytics_bronze_dev.events;
SELECT order_id, count(*) FROM ecommerce_analytics_bronze_dev.orders GROUP BY order_id HAVING count(*) > 1;
SELECT event_type, count(*) FROM ecommerce_analytics_bronze_dev.events GROUP BY event_type;
-- Silver and gold examples require catalog=s3tablescatalog
SELECT customer_segment, count(*) orders, avg(order_total) avg_value FROM ecommerce_silver.order_details GROUP BY customer_segment;
SELECT category, sum(page_views) views, sum(purchases) purchases, avg(conversion_rate) conversion_rate FROM ecommerce_silver.product_performance GROUP BY category;
SELECT * FROM ecommerce_gold.customer_analytics ORDER BY month, country LIMIT 50;
SELECT * FROM ecommerce_gold.realtime_metrics WHERE revenue > 0 ORDER BY event_hour LIMIT 50;
SELECT snapshot_id, committed_at, operation, summary FROM ecommerce_analytics_bronze_dev."orders$snapshots" ORDER BY committed_at;
SELECT * FROM ecommerce_analytics_bronze_dev."orders$history" ORDER BY made_current_at;
-- Example: SELECT count(*) FROM ecommerce_analytics_bronze_dev.orders FOR TIMESTAMP AS OF TIMESTAMP '2025-01-01 00:00:00';
