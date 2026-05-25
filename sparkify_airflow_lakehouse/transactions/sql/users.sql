SELECT DISTINCT userId user_id, firstName first_name, lastName last_name, gender, level FROM sparkify_raw.logs WHERE data_interval='{{ data_interval }}' AND userId IS NOT NULL
