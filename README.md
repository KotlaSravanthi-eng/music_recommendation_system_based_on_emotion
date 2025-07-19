# 🎵 Music Mind - Mood-Based Music Recommendation System

Let your **mood pick the music** 🎧  
Music Mind is a hybrid music recommendation system that suggests songs based on the user's mood, activity, or preferred artist, using both audio features and lyrics analysis.

## 💡What It Does
    * If the user enters a mood (e.g., "happy", "sad"), the system recommends songs matching that emotion using audio and lyrical cues.

    * If an artist name is entered, it recommends songs from that artist.

    * If the input contains both a mood + artist, it filters songs from that artist matching the given mood.

    * If the user selects a predefined activity like party, study, or driving (via buttons), it treats the activity as the mood and suggests songs accordingly.

The system also uses collaborative filtering to suggest similar songs and similar artists, enhancing the user experience with relevant and diverse recommendations.

---

## 🔗 Live Demo

[👉 Try the App Here](https://your-deployment-link.com)  
📹 *(Optional: Add demo video if available)*

---

## 🚀 Features

- 🔍 Recommend songs based on **mood**, **activity**, or **artist** input
- 🎶 Hybrid recommendation using **lyrics**, **genre**, and **audio features**
- 📊 Audio preprocessing using **PowerTransformer**
- Using Lematization and stopwords for **text** column
- 🧠 NLP-based mood & artist extraction with **spaCy**
- 💡 Shows **top 5 similar songs** and **similar artists**
- 🖼️ Clean and attractive **web UI** with activity buttons
- 🔄 Modular Python codebase ready for scaling

---

## 📥 Data Ingestion from Google Cloud
    * The raw dataset was stored and retrieved from Google Cloud Storage.  
    However, a cleaned version (`cleaned_data.csv`) is included for offline development.


## 🧱 Tech Stack

- BAckend: Python, Flask
- ML & NLP: pandas, scikit-learn, spaCy
- Vectorization: TF-IDF
- Audio Prep: PowerTransformer
- Frontend: HTML + CSS 
- Deployement: Render
- Storage: Google Cloud Storage

---

## 📁 Project Structure
music_recommendation_system/
├── artifacts/ 
|   ├── preprocessed_bundle.pkl
|   └── similar_song_dict.pkl
├── deployment/
| ├── static
| |   ├──css/style.css
| |   ├──logo/  
| |      ├──default_artist
| |      ├──music_mind_logo
| |      └── play_button
| ├──templates/index.html
│ └── app.py
| └──Procfile
| └──render.yaml
| └──runtime.txt
├── notebook/
|   ├── Data_cleaning 
|   ├── Model_Training 
|   └── cleaned_data
├── src/
│ ├── components/
| |   ├── Data ingestion 
| |   └── preprocessing
│ ├── pipelines/
| |   ├── Mood extraction
| |   ├── similarity engine
| |   └── recommendations
│ └── main.py 
| └──logger.py
| └──exception.py
| └──utils.py
├── requirements.txt
└── README.md

## 💻 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/KotlaSravanthi-eng/music_recommendation_system_based_on_emotion.git
cd music-mind

# 2. Create virtual env
python -m venv music_env
source music_env/Scripts/activate   

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python -m deploymen.app
--
```

## Contact 
Created by **Sravanthi Kotla**
📧 Email: kotlasravanthi229@gmail.com 
🔗 GitHub Profile : https://github.com/KotlaSravanthi-eng