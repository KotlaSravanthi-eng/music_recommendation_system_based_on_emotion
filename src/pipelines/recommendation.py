import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from src.pipelines.mood_extraction import extract_mood_artist


def get_recommendations(
    user_input, data, final_matrix, scaled_audio, tfidf, tfidf_similar,
    similar_vectors, similar_song_dict, similar_artist_dict, known_artists
):
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
    sim_context_input = artist or mood or user_input
    context_vector = tfidf_similar.transform([sim_context_input])

    # 5. Build hybrid user vector (text + mood + average audio features)
    avg_audio_vector = np.mean(scaled_audio, axis=0).reshape(1, -1)
    user_vector = hstack([mood_vector, context_vector, avg_audio_vector])

    # 6. Content similarity
    content_scores = cosine_similarity(user_vector, final_matrix).flatten()

    # 7. Similar artist/song similarity
    sim_scores = cosine_similarity(context_vector, similar_vectors).flatten()

    # 8. Final hybrid score
    final_scores = 0.7 * content_scores + 0.3 * sim_scores

    # 9. Apply activity mask if matched
    if activity_col and activity_col in data.columns:
        mask = data[activity_col] == 1
        final_scores *= mask.astype(int).values

    # 10. Pick top results
    top_indices = final_scores.argsort()[::-1][:5]
    results = data.iloc[top_indices][['song', 'Artist(s)', 'emotion', 'Genre_str']].copy()

    # 11. Expand similar songs and artists per recommendation
    similar_rows = []
    for _, row in results.iterrows():
        song = row['song']
        sim_songs = list(similar_song_dict.get(song, {}).keys())[:5]
        sim_artists = list(similar_artist_dict.get(song, []))[:5]
        
        if not sim_artists: sim_artists = ["N/A"]
        for sim_artist in sim_artists:
            similar_rows.append({
                'song': row['song'],
                'Artist(s)': row['Artist(s)'],
                'emotion': row['emotion'],
                'Genre_str': row['Genre_str'],
                'Similar Artist': sim_artist,
                'Similar Songs': ', '.join(sim_songs)
            })

    return pd.DataFrame(similar_rows)
