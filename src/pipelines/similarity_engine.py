import pandas as pd
from collections import defaultdict

def build_similarity_dicts(data):
    similar_song = defaultdict(dict)
    similar_artist = defaultdict(list)

    for idx, row in data.iterrows():
        song = row['song']
        for i in range(1, 4):
            sim_song = row.get(f'Similar Song {i}')
            sim_score = row.get(f'Similarity Score {i}')
            if pd.notnull(sim_song):
                similar_song[song][sim_song] = sim_score

            sim_artist = row.get(f'Similar Artist {i}')
            if pd.notnull(sim_artist):
                similar_artist[song].append(sim_artist.strip().lower())
    return similar_song, similar_artist
