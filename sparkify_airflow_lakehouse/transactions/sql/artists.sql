SELECT DISTINCT artist_id, artist_name, artist_location, artist_latitude, artist_longitude FROM sparkify_raw.songs WHERE data_interval='{{ data_interval }}' AND artist_id IS NOT NULL
