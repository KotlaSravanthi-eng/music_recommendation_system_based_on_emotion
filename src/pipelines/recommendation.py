import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from src.pipelines.mood_extraction import extract_mood_artist

def get_recommendations(user_input, 
                        data, 
                        final_matrix, 
                        scaled_audio, 
                        tfidf, 
                        tfidf_similar,
                        similar_vectors, 
                        similar_song_dict, 
                        known_artists, 
                        top_n = 5):
    # 1. Extract mood and artist
    mood, artist = extract_mood_artist(user_input, known_artists)

    # 2. Handle activity context (like "Good for Party")
    activity_map = {
        "party": "Good for Party", 
        "reading": "Good for Work/Study", 
        "study": "Good for Work/Study",
        "exercise": "Good for Exercise", 
        "running": "Good for Running", 
        "driving": "Good for Driving",
        "morning": "Good for Morning Routine", 
        "relaxation": "Good for Relaxation/Yoga",
        "yoga": "Good for Relaxation/Yoga", 
        "meditation": "Good for Relaxation/Yoga"
    }

    activity_col = ""
    for key, col in activity_map.items():
        if key in user_input.lower():
            activity_col = col
            break

    # 3. Prepare search text
    search_text = f"{mood} {user_input}" if mood else user_input
    mood_vector = tfidf.transform([search_text])

    # 4. Prepare similarity context
    sim_context_input = artist if artist else mood or user_input
    context_vector = tfidf_similar.transform([sim_context_input])

    # 5. Build hybrid user vector (text + mood + average audio features)
    avg_audio_vector = np.mean(scaled_audio, axis=0).reshape(1, -1)
    user_vector = hstack([mood_vector, context_vector, avg_audio_vector])

    # 6. Content similarity
    content_scores = cosine_similarity(user_vector, final_matrix).flatten()

    # Apply optional activity column filter
    if activity_col:
        activity_mask = data[activity_col] == 1
        content_scores *= activity_mask.astype(int).values

    # Boost scores of songs from the mentioned artist
    if artist:
        artist_mask = data['Artist(s)'].str.lower() == artist.lower()
        content_scores += artist_mask.astype(int).values * 0.75

    # Get top N recommended songs (main DataFrame)
    top_indices = content_scores.argsort()[::-1][:top_n]
    results = data.iloc[top_indices][['song', 'Artist(s)', 'emotion', 'Genre_str']].copy()
    results['similarity_score'] = content_scores[top_indices]

    # 8. Extract 5 similar songs + artists (combined)
    all_similar_songs = set()
    all_similar_artists = set()

    for _, row in results.iterrows():
        song = row['song']
        original_artist = row['Artist(s)'].strip().lower()

        # Get similar song
        song_key = song.strip().lower()

        sim_song = list(similar_song_dict.get(song_key, {}).keys())[:3]
        all_similar_songs.update(sim_song)


        # Similar artists via genre
        song_genres = row['Genre_str'].split()
        genre_mask = data['Genre_str'].apply(lambda x: any(g in x.split() for g in song_genres))

        exclude_artists = {original_artist}
        if artist:
            exclude_artists.add(artist.strip().lower())

        genre_artists = data[genre_mask]['Artist(s)'].str.lower().unique().tolist()
        filtered_artists = [a for a in genre_artists if a not in exclude_artists]
        all_similar_artists.update(filtered_artists)  # Collect more

    # Final lists
    final_similar_songs = list(all_similar_songs)[:5]
    final_similar_artists = list(all_similar_artists)[:5]

    return results, final_similar_songs, final_similar_artists