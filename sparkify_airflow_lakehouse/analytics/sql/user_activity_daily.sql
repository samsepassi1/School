SELECT date(from_unixtime(ts/1000)) activity_date,user_id,count(*) events,count(distinct session_id) sessions FROM sparkify_transactions.events GROUP BY 1,2
