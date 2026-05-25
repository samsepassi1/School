SELECT song_id, title, artist_id, year, duration, current_timestamp() version_loaded_at FROM sparkify_raw.songs WHERE data_interval='{{ data_interval }}'
