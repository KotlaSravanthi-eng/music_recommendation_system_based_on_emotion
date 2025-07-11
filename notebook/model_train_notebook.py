# %%
#### import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.preprocessing import PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
data = pd.read_csv('cleaned_data.csv')

# %%
data.info()

# %%
# Fully numeric artist names
fully_numeric = data['Artist(s)'][data['Artist(s)'].str.fullmatch(r'\d+')]
print("🎤 Fully numeric artist names:", fully_numeric.unique().tolist())

# %%
data['Artist(s)'] = data['Artist(s)'].astype(str).str.lower().str.strip()

# %%
data['Artist(s)'].value_counts()

# %%
data['Artist(s)'].isnull().sum()

# %%
data.dropna(inplace=True)
data.reset_index(drop = True, inplace=True)

# %%
data.info()

# %% [markdown]
# #### 4.2 **Power Transformer for Skewed Audio Features**

# %% [markdown]
# * we need to apply power transformer to the heavily skewed features like `instrumentalness`, `liveness`, `speechiness`, `loudness (db)` and `acousticness`.

# %%
#### Applying power transformer to the skewed audio features.
pt = PowerTransformer(method='yeo-johnson')

cols_to_transform = ['Liveness', 'Speechiness', 'Instrumentalness', 'Acousticness', 'Loudness (db)']

scaled_audio = pt.fit_transform(data[cols_to_transform])
data[cols_to_transform] = scaled_audio

cols_to_transform = ['Liveness', 'Speechiness', 'Instrumentalness', 'Acousticness', 'Loudness (db)']
for feature in cols_to_transform:
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# %%
# apply log1p to instrumentalness first 
data[['Instrumentalness']] = np.log1p(data[['Instrumentalness']])

pt = PowerTransformer()
data[['Instrumentalness']] = pt.fit_transform(data[['Instrumentalness']])


sns.histplot(data['Instrumentalness'], kde=True)
plt.title(f'Distribution of Instrumentalness')
plt.show()

# %% [markdown]
# * Even after applying `log1p` and `power transformer` it doesn't change anything. so i will keep `Instrumentalness` as it is.

# %% [markdown]
# #### 4.3 **TF-IDF on text Columns**

# %%
### Applying tf-idf on text columns 
data['combined_text'] = (
    data['text'] + " " +
    data['emotion'] + " " +
    data['Genre_str'] + " " +
    data['Artist(s)']
)

tfidf = TfidfVectorizer(max_features=300, stop_words='english') 
tfidf_matrix = tfidf.fit_transform(data['combined_text'])

# %% [markdown]
# #### 4.4 **Combine TF-IDF & Scaled Audio features**

# %%
### Checking the shape od the tfidf matrix and scaled audio 
print(tfidf_matrix.shape[0], scaled_audio.shape[0]) 

# %%
from scipy.sparse import hstack

# Combine into one final feature set for cosine similarity
final_matrix = hstack([tfidf_matrix, scaled_audio])

# %%
data.info()

# %% [markdown]
# #### **4.5 Handle User Query Input**

# %%
data['emotion'].unique()

# %%
import re
import spacy
nlp = spacy.load("en_core_web_sm")

# Define allowed dataset moods
available_moods = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']

# Mapping user moods to dataset moods
mood_map = {
    'happy': 'joy',
    'calm': 'love',
    'relaxed': 'joy',
    'peaceful': 'joy',
    'romantic': 'love',
    'chill': 'joy',
    'energetic': 'surprise',
    'excited': 'surprise',
    'angry': 'anger',
    'depressed': 'sadness',
    'bored': 'sadness',
    'sad': 'sadness',
    'fearful': 'fear'
}

def extract_mood_artist_spacy(user_input, known_artists):
    doc = nlp(user_input.lower())
    user_input_lower = user_input.lower()

    # 1. Detect mood
    mood = ""
    for word in user_input_lower.split():
        if word in mood_map:
            mood = mood_map[word]
            break
        elif word in available_moods:
            mood = word
            break

    # 2. Detect artist
    artist = ""
    for known_artist in known_artists:
        pattern = r'\b' + re.escape(known_artist) + r'\b'
        if re.search(pattern, user_input_lower):
            artist = known_artist
            break

    return mood, artist


# %%
known_artists = data['Artist(s)'].str.lower().str.strip().unique().tolist()

user_input = "I'm feeling sad give me songs from mayday artist"
mood, artist = extract_mood_artist_spacy(user_input, known_artists)

print("Mapped mood:", mood)    
print("Artist:", artist) 

# %%
# Fully numeric artist names
fully_numeric = data['Artist(s)'][data['Artist(s)'].str.fullmatch(r'\d+')]
print("🎤 Fully numeric artist names:", fully_numeric.unique().tolist())

# %%
data['Artist(s)'].unique()

# %%
data['Artist(s)'] = data['Artist(s)'].str.lower().str.strip()

# %% [markdown]
# #### **5.Building Recommendation Logic Using Cosine Similarity**

# %%
### Create Similar Artist Dictionary
from collections import defaultdict

# create song dict
song_similar_artist = defaultdict(list)
for idx, row in data.iterrows():
    base_song = row['song']
    for i in range(1, 4):
        sim_artist = row[f'Similar Artist {i}']
        if pd.notnull(sim_artist):
            song_similar_artist[base_song].append(sim_artist.strip().lower())

# %%
### Create Similar Song Dictionary
similar_song = defaultdict(dict)

for idx, row in data.iterrows():
    base_song = row['song']
    for i in range(1,4):
        sim_song = row[f'Similar Song {i}']
        sim_score = row[f'Similarity Score {i}']
        if pd.notnull(sim_song):
            similar_song[base_song][sim_song] = sim_score

# %%
similar_context_series = (
    data['Similar Artist 1']+ ' ' +
    data['Similar Artist 2']+ ' ' +
    data['Similar Artist 3']+ ' ' +
    data['Similar Song 1']+ ' ' +
    data['Similar Song 2']+ ' ' +
    data['Similar Song 3']
)

tfidf_similar = TfidfVectorizer(stop_words="english", max_features=5000)
similar_vectors = tfidf_similar.fit_transform(similar_context_series)

# %%
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


def get_recommendations(user_input, top_n=5):
    # Extract mood and artist using spaCy
    mood, artist = extract_mood_artist_spacy(user_input, known_artists)

    # Detect activity 
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

    # TF-IDF vector for mood + input text
    search_text = f"{mood} {user_input}" if mood else user_input
    mood_vector = tfidf.transform([search_text])

    # Combine with average audio vector
    avg_audio_vec = np.mean(scaled_audio, axis=0).reshape(1, -1)
    user_vector = hstack([mood_vector, avg_audio_vec])

    # Compute content-based similarity
    content_scores = cosine_similarity(user_vector, final_matrix).flatten()

    # Compute similar artist/song scores
    sim_context_input = artist if artist else mood
    sim_context_vector = tfidf_similar.transform([sim_context_input])
    sim_context_scores = cosine_similarity(sim_context_vector, similar_vectors).flatten()

    # Combine into final hybrid scores
    final_scores = 0.7 * content_scores + 0.3 * sim_context_scores

    # Apply optional activity column filter
    if activity_col:
        activity_mask = data[activity_col] == 1
        final_scores *= activity_mask.astype(int).values

    # Boost scores of songs from the mentioned artist
    if artist:
        artist_mask = data['Artist(s)'].str.lower() == artist.lower()
        final_scores += artist_mask.astype(int).values * 0.25  

    # Get top results
    top_indices = final_scores.argsort()[::-1][:top_n]
    results =  data.iloc[top_indices][['song', 'Artist(s)', 'emotion', 'Genre_str']].copy()

    # Add similar songs & artist from dictionaries
    similar_songs_list = []
    similar_artists_list = []

    for song in results['song']:
        similar_songs = list(similar_song.get(song, {}).keys())[:1]
        
        # Get original artist for the current song
        original_artist = data.loc[data['song'] == song, 'Artist(s)'].values[0].strip().lower()

        # Step 1: Get genres of this song
        song_genres = data.loc[data['song'] == song, 'Genre_str'].values[0].split()

        # Step 2: Filter dataset to songs having ANY of these genres
        genre_mask = data['Genre_str'].apply(lambda x: any(g in x.split() for g in song_genres))

        # Step 3: Get artists in those genres, excluding original/user artist
        exclude_artists = {original_artist}
        if artist:
            exclude_artists.add(artist.strip().lower())

        similar_artist_candidates = data[genre_mask]['Artist(s)'].str.lower().unique().tolist()
        filtered_similar_artists = [a for a in similar_artist_candidates if a not in exclude_artists]

        # Step 4: Pick top 1 or top N
        similar_artists = filtered_similar_artists[:5] if filtered_similar_artists else ""

        similar_songs_list.append(", ".join(similar_songs) if similar_songs else "")
        similar_artists_list.append(similar_artists if similar_artists else [])

    results['Similar Songs'] = similar_songs_list 
    results['Similar Artists'] = similar_artists_list 
    results['similarity_score'] = final_scores[top_indices]

    # Convert Similar Artists from string to list for exploding
    #results['Similar Artists'] = results['Similar Artists'].apply(lambda x: x.split(', ') if x else [])

    # 🔥 Explode similar artists so each one appears in a separate row
    results = results.explode('Similar Artists').reset_index(drop=True)

    return results

# %%
data['Artist(s)'].value_counts()

# %%
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


def get_recommendations(user_input, top_n=5):
    # Extract mood and artist using spaCy
    mood, artist = extract_mood_artist_spacy(user_input, known_artists)

    # Detect activity 
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

    # TF-IDF vector for mood + input text
    search_text = f"{mood} {user_input}" if mood else user_input
    mood_vector = tfidf.transform([search_text])

    # Combine with average audio vector
    avg_audio_vec = np.mean(scaled_audio, axis=0).reshape(1, -1)
    user_vector = hstack([mood_vector, avg_audio_vec])

    # Compute content-based similarity
    content_scores = cosine_similarity(user_vector, final_matrix).flatten()

    # Compute similar artist/song scores
    sim_context_input = artist if artist else mood
    sim_context_vector = tfidf_similar.transform([sim_context_input])
    sim_context_scores = cosine_similarity(sim_context_vector, similar_vectors).flatten()

    # Combine into final hybrid scores
    final_scores = 0.7 * content_scores + 0.3 * sim_context_scores

    # Apply optional activity column filter
    if activity_col:
        activity_mask = data[activity_col] == 1
        final_scores *= activity_mask.astype(int).values

    # Boost scores of songs from the mentioned artist
    if artist:
        artist_mask = data['Artist(s)'].str.lower() == artist.lower()
        final_scores += artist_mask.astype(int).values * 0.25  

    # Get top results
    top_indices = final_scores.argsort()[::-1][:top_n]
    results =  data.iloc[top_indices][['song', 'Artist(s)', 'emotion', 'Genre_str']].copy()

    # Add similar songs & artist from dictionaries
    similar_songs_list = []
    similar_artists_list = []

    for song in results['song']:
        similar_songs = list(similar_song.get(song, {}).keys())[:1]
        
        # Get original artist for the current song
        original_artist = data.loc[data['song'] == song, 'Artist(s)'].values[0].strip().lower()

        # Step 1: Get genres of this song
        song_genres = data.loc[data['song'] == song, 'Genre_str'].values[0].split()

        # Step 2: Filter dataset to songs having ANY of these genres
        genre_mask = data['Genre_str'].apply(lambda x: any(g in x.split() for g in song_genres))

        # Step 3: Get artists in those genres, excluding original/user artist
        exclude_artists = {original_artist}
        if artist:
            exclude_artists.add(artist.strip().lower())

        similar_artist_candidates = data[genre_mask]['Artist(s)'].str.lower().unique().tolist()
        filtered_similar_artists = [a for a in similar_artist_candidates if a not in exclude_artists]

        # Step 4: Pick top 1 or top N
        similar_artists = filtered_similar_artists[:5] if filtered_similar_artists else ""

        similar_songs_list.append(", ".join(similar_songs) if similar_songs else "")
        similar_artists_list.append(similar_artists if similar_artists else [])

    results['similarity_score'] = final_scores[top_indices]
    
    # Build fully expanded rows: each similar artist in a new row
    expanded_rows = []

    for i, row in results.iterrows():
        song = row['song']
        original_artist = row['Artist(s)'].strip().lower()
        similar_songs = list(similar_song.get(song, {}).keys())[:1]

        # Step 1: Get genres of this song
        song_genres = data.loc[data['song'] == song, 'Genre_str'].values[0].split()

        # Step 2: Filter dataset to songs with same genre
        genre_mask = data['Genre_str'].apply(lambda x: any(g in x.split() for g in song_genres))
        exclude_artists = {original_artist}
        if artist:
            exclude_artists.add(artist.strip().lower())

        similar_artist_candidates = data[genre_mask]['Artist(s)'].str.lower().unique().tolist()
        filtered_similar_artists = [a for a in similar_artist_candidates if a not in exclude_artists]

        for sim_artist in filtered_similar_artists[:5]:  # Limit to 5
            expanded_rows.append({
                'song': song,
                'Artist(s)': row['Artist(s)'],
                'emotion': row['emotion'],
                'Genre_str': row['Genre_str'],
                'Similar Songs': similar_songs[0] if similar_songs else "",
                'Similar Artists': sim_artist,
                'similarity_score': row['similarity_score']
            })

    # Return exploded format as new DataFrame
    return pd.DataFrame(expanded_rows)

# %%
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


def get_recommendations(user_input, top_n=5):
    # Extract mood and artist using spaCy
    mood, artist = extract_mood_artist_spacy(user_input, known_artists)

    # Detect activity 
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

    # TF-IDF vector for mood + input text
    search_text = f"{mood} {user_input}" if mood else user_input
    mood_vector = tfidf.transform([search_text])

    # Combine with average audio vector
    avg_audio_vec = np.mean(scaled_audio, axis=0).reshape(1, -1)
    user_vector = hstack([mood_vector, avg_audio_vec])

    # Compute content-based similarity
    content_scores = cosine_similarity(user_vector, final_matrix).flatten()

    # Compute similar artist/song scores
    sim_context_input = artist if artist else mood
    sim_context_vector = tfidf_similar.transform([sim_context_input])
    sim_context_scores = cosine_similarity(sim_context_vector, similar_vectors).flatten()

    # Combine into final hybrid scores
    final_scores = 0.7 * content_scores + 0.3 * sim_context_scores

    # Apply optional activity column filter
    if activity_col:
        activity_mask = data[activity_col] == 1
        final_scores *= activity_mask.astype(int).values

    # Boost scores of songs from the mentioned artist
    if artist:
        artist_mask = data['Artist(s)'].str.lower() == artist.lower()
        final_scores += artist_mask.astype(int).values * 0.25  

   # Get top N recommended songs (main DataFrame)
    top_indices = final_scores.argsort()[::-1][:top_n]
    results = data.iloc[top_indices][['song', 'Artist(s)', 'emotion', 'Genre_str']].copy()
    results['similarity_score'] = final_scores[top_indices]

    # Collect similar songs and artists from top recommendations
    all_similar_songs = set()
    all_similar_artists = set()

    for _, row in results.iterrows():
        song = row['song']
        original_artist = row['Artist(s)'].strip().lower()

        # Get similar song
        sim_song = list(similar_song.get(song, {}).keys())[:3]
        all_similar_songs.update(sim_song)

        # Genre-based similar artists
        song_genres = row['Genre_str'].split()
        genre_mask = data['Genre_str'].apply(lambda x: any(g in x.split() for g in song_genres))

        exclude_artists = {original_artist}
        if artist:
            exclude_artists.add(artist.strip().lower())

        candidate_artists = data[genre_mask]['Artist(s)'].str.lower().unique().tolist()
        filtered_artists = [a for a in candidate_artists if a not in exclude_artists]

        all_similar_artists.update(filtered_artists[:10])  # Collect more, filter later

    # Final top 5 unique similar songs and artists
    final_similar_songs = list(all_similar_songs)[:5]
    final_similar_artists = list(all_similar_artists)[:5]

    return results, final_similar_songs, final_similar_artists


# %%
user_input = "I'm feeling very joy broo give me some songs from kora."
get_recommendations(user_input)

# %%
recommendations_df, similar_songs, similar_artists = get_recommendations(user_input)

print("\n Final Recommendations:")
print(recommendations_df)

print("\n Top 5 Similar Songs:")
for s in similar_songs:
    print(f" * {s}")

print("\n Top 5 Similar Artists:")
for a in similar_artists:
    print(f" * {a}")

# %% [markdown]
# * Instead of strictly relying on emotion labels (e.g., joy, sadness), similar songs are recommended based on lyrical similarity using TF-IDF and cosine similarity. This allows the system to capture songs whose lyrics align more closely with the user’s mood, even if the labeled emotion differs.


