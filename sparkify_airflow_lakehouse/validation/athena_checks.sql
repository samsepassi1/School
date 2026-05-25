SELECT count(*) FROM sparkify_raw.logs;
SELECT count(*) FROM sparkify_raw.songs;
SELECT count(*) FROM sparkify_transactions.events;
SELECT count(*) FROM sparkify_analytics.songplay_facts;
SELECT * FROM sparkify_analytics.user_activity_daily ORDER BY activity_date LIMIT 20;
