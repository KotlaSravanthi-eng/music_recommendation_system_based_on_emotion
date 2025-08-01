from flask import Flask, render_template, request, url_for
import pandas as pd
import joblib
from src.pipelines.recommendation import get_recommendations

import spacy
import subprocess
import importlib.util

model_name = "en_core_web_sm"

if importlib.util.find_spec(model_name) is None:
    subprocess.run(["python", "-m", "spacy", "download", model_name])

nlp = spacy.load(model_name)

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load cleaned data
data = pd.read_csv("artifacts/music_cleaned_data.csv")  

# Load preprocessing bundle
bundle_path = "artifacts/preprocessed_bundle.pkl"
bundle = joblib.load(bundle_path)
final_matrix = bundle['final_matrix']
scaled_audio = bundle['scaled_audio']
tfidf = bundle['tfidf']
tfidf_similar = bundle['tfidf_similar']

# Build similarity dictionary
similar_song_dict = joblib.load("artifacts/similar_song_dict.pkl") 

# Rebuild similar context vectors
data['similar_context'] = (
    data['Similar Artist 1'] + ' ' +
    data['Similar Artist 2'] + ' ' +
    data['Similar Artist 3'] + ' ' +
    data['Similar Song 1'] + ' ' +
    data['Similar Song 2'] + ' ' +
    data['Similar Song 3']
)
similar_vectors = tfidf_similar.transform(data['similar_context'])

# Known artists
known_artists = set(data['Artist(s)'].str.lower().unique())


# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.form['user_input']

        top_df, final_similar_songs, final_similar_artists = get_recommendations(
            user_input=user_input,
            data=data,
            final_matrix=final_matrix,
            scaled_audio=scaled_audio,
            tfidf=tfidf,
            tfidf_similar=tfidf_similar,
            similar_vectors=similar_vectors,
            similar_song_dict=similar_song_dict,
            known_artists=known_artists
        )
        print("Top songs:", top_df)
        print("similar songs:", final_similar_songs)
        print("similar artist:", final_similar_artists)

        # similar songs 
        final_similar_songs = [{'song': song} for song in final_similar_songs]

        # similar artists
        similar_artists_info = []
        for artist in final_similar_artists:
            similar_artists_info.append({
                'name': artist,
                'image': url_for('static', filename='logo/default_artist.png')
            })
        final_similar_artists = similar_artists_info
        
        # results data
        results_data = []
        for _, row in top_df.iterrows():
            results_data.append({
                'song': row['song'],
                'artist': row.get('Artist(s)', 'Unknown') 
            })

        return render_template(
            'index.html',
            user_input=user_input,
            results=results_data,
            similar_songs=final_similar_songs,
            similar_artists=final_similar_artists
        )
    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7860)

