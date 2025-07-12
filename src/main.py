import os
import joblib
import pandas as pd
from components.data_ingestion import DataIngestion
from components.preprocessing import preprocess_and_save
from pipelines.similarity_engine import build_similarity_dicts
from pipelines.recommendation import get_recommendations

from logger import logger

if __name__ == "__main__":

    # Step 1: Ingest Cleaned data
    ingestion = DataIngestion()
    cleaned_path = ingestion.initiate_data_ingestion()

    data = pd.read_csv(cleaned_path)

    # Step 2: Load or Create Preprocessing Bundle
    bundle_path = "artifacts/preprocessed_bundle.pkl"
    if os.path.exists(bundle_path):
        logger.info("Loading existing preprocessing bundle...")
        bundle = joblib.load(bundle_path)
        final_matrix = bundle['final_matrix']
        scaled_audio = bundle['scaled_audio']
        tfidf = bundle['tfidf']
        tfidf_sim = bundle['tfidf_similar']
    else:
        logger.info("Bundle not found. Creating preprocessing bundle...")
        final_matrix, scaled_audio, tfidf, tfidf_sim = preprocess_and_save(data)
        bundle = {
            'final_matrix': final_matrix,
            'scaled_audio': scaled_audio,
            'tfidf': tfidf,
            'tfidf_similar': tfidf_sim
        }

        joblib.dump(bundle, bundle_path)

    # step 3: Preprocess data and save transformers

    final_matrix = bundle['final_matrix']
    scaled_audio = bundle['scaled_audio']
    tfidf_sim = bundle['tfidf_similar']
    tfidf = bundle['tfidf']

    # Step 4: prepare similarity dictionaries
    similar_song_dict = build_similarity_dicts(data)

    # step 5: Add similar context 
    data['similar_context'] = (
        data['Similar Artist 1'] + ' '+
        data['Similar Artist 2'] + ' '+
        data['Similar Artist 3'] + ' '+
        data['Similar Song 1'] + ' '+
        data['Similar Song 2'] + ' '+
        data['Similar Song 3'] 

    )
    similar_vectors = tfidf_sim.transform(data['similar_context'])

    # step 6: Test recommendation
    known_artists = set(data['Artist(s)'].str.lower().unique())
    user_input = input("\nEnter your music request (mood, artist, or activity): ")

    top_df, final_similar_songs, final_similar_artists = get_recommendations(
        user_input=user_input,
        data=data,
        final_matrix= final_matrix,
        scaled_audio=scaled_audio,
        tfidf=tfidf,
        tfidf_similar= tfidf_sim,
        similar_vectors=similar_vectors,
        similar_song_dict=similar_song_dict,
        known_artists=known_artists

    )
    # Final output formatting
    print("\nFinal Recommendations:")
    print(top_df[['song', 'Artist(s)', 'emotion', 'Genre_str', 'similarity_score']])

    print("\nTop 5 Similar Songs:")
    for song in final_similar_songs:
        print(f" * {song}")

    print("\nTop 5 Similar Artists:")
    for artist in final_similar_artists:
        print(f" * {artist}")