SELECT a.artist_id,a.artist_name,count(*) plays FROM sparkify_transactions.events e JOIN sparkify_transactions.artists a ON e.artist_id=a.artist_id GROUP BY 1,2
