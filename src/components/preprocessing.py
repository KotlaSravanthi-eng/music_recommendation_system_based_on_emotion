import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib
import os


def preprocess_and_save(data, output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Clean artist names
    data['Artist(s)'] = data['Artist(s)'].astype(str).str.lower().str.strip()

    # 2. Drop nulls and reset index
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # 3. Fix genre column (convert list to string)
    data['Genre'] = data['Genre'].apply(ast.literal_eval)
    data['Genre_str'] = data['Genre'].apply(lambda x: ' '.join(x))

    # 4. Power transform selected audio columns
    audio_cols = ['Liveness', 'Speechiness', 'Instrumentalness', 'Acousticness', 'Loudness (db)']

    pt = PowerTransformer()
    scaled_audio = pt.fit_transform(data[audio_cols])

    # 5. TF-IDF for lyrics + emotion + genre + artist
    data['combined_text'] = (
        data['text'] + " " +
        data['emotion'] + " " +
        data['Genre_str'] + " " +
        data['Artist(s)']
    )

    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_text'])

    # 6. TF-IDF on similar artist + similar song context
    data['similar_context'] = (
        data['Similar Artist 1'].fillna('') + ' ' +
        data['Similar Artist 2'].fillna('') + ' ' +
        data['Similar Artist 3'].fillna('') + ' ' +
        data['Similar Song 1'].fillna('') + ' ' +
        data['Similar Song 2'].fillna('') + ' ' +
        data['Similar Song 3'].fillna('')
    )

    tfidf_sim = TfidfVectorizer(max_features=5000, stop_words='english')
    sim_matrix = tfidf_sim.fit_transform(data['similar_context'])

    # 7. Combine everything into final feature matrix
    final_matrix = hstack([tfidf_matrix, sim_matrix, scaled_audio])

    # 8. Save everything to disk
    joblib.dump({
        "final_matrix": final_matrix,
        "scaled_audio": scaled_audio,
        "power_transformer": pt,
        "tfidf": tfidf,
        "tfidf_similar": tfidf_sim
    }, os.path.join(output_dir, "preprocessed_bundle.pkl"))

    return final_matrix, scaled_audio, tfidf, tfidf_sim
