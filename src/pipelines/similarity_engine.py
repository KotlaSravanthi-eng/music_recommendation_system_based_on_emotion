import pandas as pd
from collections import defaultdict

def build_similarity_dicts(data):
    similar_song = defaultdict(dict)

    for _, row in data.iterrows():
        song = row['song'].strip().lower()
        for i in range(1, 4):
            sim_song = row.get(f'Similar Song {i}')
            sim_score = row.get(f'Similarity Score {i}')
            if pd.notnull(sim_song):
                similar_song[song][sim_song.strip().lower()] = sim_score

    return similar_song
