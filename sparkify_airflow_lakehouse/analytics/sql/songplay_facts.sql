SELECT e.ts,e.user_id,e.session_id,e.song_id,e.artist_id,e.level FROM sparkify_transactions.events e WHERE e.page='NextSong'
