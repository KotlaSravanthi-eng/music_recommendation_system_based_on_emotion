# %% [markdown]
# ## Data Exploration and Cleaning of project

# %% [markdown]
# ### Project Work flow

# %% [markdown]
# * Problem Statement
# * Data Collection
# * EDA
# * Preprocessing 
# * Emotion Detection 
# * Music Recommendation Logic
# * Model evaluation

# %% [markdown]
# #### 1) Problem Statement

# %% [markdown]
# * This is an emotion-aware music recommendation system that classifies and recommends songs not only based on their musical and lyrical features, but also by incorporating the emotional context behind them. 
# 
# * The main goal of this project is to enhance the effectiveness of music recommendations by categorizing songs into four core emotions — such as `happy`, `sad`, `angry`, and `relaxed` — and leveraging similarities in audio features within these emotion groups.
# 
# * By integrating emotional weighting into the recommendation logic, it is aims to provide more personalized and mood-relevant suggestions to users, helping them discover songs that resonate with their current feelings or desired emotional state.

# %% [markdown]
# #### 2) Data Collection

# %% [markdown]
# * This dataset has been downloaded from `500k+ Spotify Songs with Lyrics, Emotions & More`
# * Dataset Link - https://www.kaggle.com/datasets/devdope/900k-spotify

# %%
### Importing required packages
import numpy as np
import gdown
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %%
file_id = "1Hg3ENUzwsb3rLSFqEW9zcHPB98VVApXi"
gdown.download(f"https://drive.google.com/uc?id={file_id}", "raw_data.csv", quiet=False)

# %%
data = pd.read_csv('raw_data.csv')

# %%
data.head()

# %% [markdown]
# ### 3) Exploratory Data Analysis

# %%
data.columns

# %%
data.info()

# %%
data.shape

# %% [markdown]
# #### Insights

# %% [markdown]
# * This dataset contain nearly 5,50,000 rows with 39 columns. 
# * It has totally 18 categorical columns, 18 integer columns and 3 float columns.

# %% [markdown]
# ### Checking for null values

# %%
data.isnull().sum()

# %% [markdown]
# #### Insights

# %% [markdown]
# * The dataset contains some missing values. Since it's quite large, I chose to remove the rows with null values instead of filling them with placeholder values, to maintain data quality.

# %%
data.dropna(inplace=True)

# %%
data.shape

# %% [markdown]
# ### Checking for duplicated values

# %%
data.duplicated().sum()

# %%
data = data.drop_duplicates().reset_index(drop = True)

# %%
data.shape

# %%
data.drop(['Length', 'Album', 'Release Date', 'Key', 'Time signature', 'Explicit', 'Popularity'], axis = 1, inplace=True)

# %%
data.info()

# %%
data['emotion'].value_counts().plot(kind='bar', color='orange')

# %%
data['emotion'].value_counts()['True']

# %%
data['emotion'].value_counts()['Love']

# %%
data['emotion'].value_counts()['angry']

# %%
data['emotion'].unique()

# %% [markdown]
# #### Insights 

# %% [markdown]
# * Labels like `True`, `thirst`, `interest`, `confusion`, `angry`, `Love` and `pink` are very less compared to other. So i just want to remove them for better data quality.

# %%
data = data[~data['emotion'].isin(['True', 'thirst', 'pink', 'interest', 'angry','Love','confusion'])]

# %%
data.shape

# %%
artist_count = data['Artist(s)'].value_counts()
artist_count

# %%
popular_artist = artist_count[artist_count >= 25].index
data = data[data['Artist(s)'].isin(popular_artist)].reset_index(drop=True)

# %%
data.shape

# %%
data['Artist(s)'].value_counts()

# %%
data.info()

# %% [markdown]
# #### Insights

# %% [markdown]
# ***Filtering Artists by Frequency***

# %% [markdown]
# * To reduce noise and focus on more relevant data, we remove artists that appear fewer than 50 times in the dataset. 
# 
# * Before filtering: 127,308 unique artists
# 
# * After filtering: 1578 unique artists
# * This step ensures that the model trains on artists with enough representation, leading to more reliable recommendations.

# %%
data['emotion'].value_counts().plot(kind='bar', color='orange')

# %%
data.Genre.value_counts()

# %%
data.columns

# %% [markdown]
# #### Audio Feature Distributions

# %%
import seaborn as sns
import matplotlib.pyplot as plt

audio_features = ['Positiveness', 'Energy', 'Danceability', 'Acousticness', 'Tempo']
for feature in audio_features:
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# %% [markdown]
# #### Insights

# %% [markdown]
# * `Energy`: Most songs have moderate to high energy (40-100 range), showing a strong presence of upbeat tracks and biased toward energetic songs. Useful for identifying emotions like joy, anger and motivation.
# * `Acousticness`: Sharp peak near 0, meaning most songs are highly electronic and Helpful in classifying calm emotions like love, sadness, or relaxation.
# * `Positiveness`: Distributed fairly evenly but slightly skewed toward mid range(30-70). Indicates a mix of both happy and sad songs.
# * `Danceability`: Normally distributed peaking around 50-60 very few songs are undanceable and few songs are extremely danceable.
# * `Tempo`: Most songs lie between 80-150 with a peak near 120. have some outliers at very low or very high tempos. it can influence mood recognitio like faster means energetic, slower means emotional.

# %%
audio_features = ['Speechiness', 'Liveness', 'Instrumentalness']
for feature in audio_features:
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# %%
sns.histplot(data['Loudness (db)'], kde=True)
plt.title('Distribution of Loudness')
plt.show()

# %%
data.columns

# %% [markdown]
# #### Insights

# %% [markdown]
# * `Speechiness` is heavily right skewed, peaking at very low values, meaning most songs have little to no speech.
# * `Liveness` is somewhat normally distributed with a peak around 10-15 low stage presence detection.
# * `Instrumentalness` is almost zero for most tracks, highly right skewed. Most songs aren't instrumental.

# %%
from collections import Counter


all_similar_artists = (
    data['Similar Artist 1'].tolist() + data['Similar Artist 2'].tolist() + data['Similar Artist 3'].tolist()
    )

artist_counts = Counter(all_similar_artists)
top_artists = dict(artist_counts.most_common(20))  

# Plot
plt.figure(figsize=(10, 8))
plt.barh(list(top_artists.keys()), list(top_artists.values()), color='teal')
plt.xlabel("Frequency")
plt.title("Top 20 Similar Artists")
plt.gca().invert_yaxis()
plt.show()

# %% [markdown]
# #### Relation Between Emotion and Audio

# %%
#### Relation between emotion and valence
sns.boxplot(x='emotion', y='Positiveness', data=data)

# %%
#### Relation between emotion and energy
sns.boxplot(x='emotion', y='Energy', data=data)

# %%
#### Relation between emotion and danceability
sns.boxplot(x='emotion', y='Danceability', data=data)

# %%
#### Relation between emotion and tempo
sns.boxplot(x='emotion', y='Tempo', data=data)

# %%
#### Relation between emotion and acousticness
sns.boxplot(x='emotion', y='Acousticness', data=data)

# %% [markdown]
# #### Insights 

# %% [markdown]
# * `Positiveness VS Emotion`: All emotions have a similar spread in positiveness (0-100), joy and love show a slightly higher median positiveness. sadness and fear have slightly lower medians.
# * `Energy VS Emotion`: Anger, Fear and Sadness show high energy levels and Anger has many outliers on the lower end, indicating some angry tracks that are low in energy. Love and Joy have slightly lower medians but still high.
# * `Danceability VS Emotion`: All emotions show similar medians and distributions. Anger has a higher median and slightly narrower IQR implying consistent danceability in angry tracks. Sadness shows more variability with some lower outliers.
# * `Tempo VS Emotion`: Tempo distribution is very similar across all emotions and medians around 120-130.
# * `Acousticness VS Emotion`: Love, Sadness and Joy have higher acousticness, Anger and Fear show low median acousticness with lots of low outliers, Anger and Fear also have many outliers.

# %% [markdown]
# #### Correlation Heatmap

# %%
plt.figure(figsize=(20,18))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')

plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels
plt.yticks(rotation=0, fontsize=10)   # Rotate y-axis labels
plt.title("Correlation between Audio Features", fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Insights

# %% [markdown]
# **`Strong Positive Correlations`** 
# * Energy and Danceability (0.62) Enegetic songs tend to be more danceable.
# * Positiveness and Danceability (0.41) Happier songs are also more danceable.
# * Good for party correlates highly with Energy(0.58), Danceability(0.50), Positiveness(0.52).
# * Good for Exercise Highly correlation with Energy(0.62) and also with Danceability(0.49).

# %% [markdown]
# **`Strong Negative correlations`**
# * Acousticness has negative correlations with Energy(-0.61), Danceability(-0.18), Good for Excercise(-0.38), Good for Party(-0.41), Good for Running(-0.35).
# * Acoustic songs tend to be calm, less energetic, and not suitable for workouts or parties.
# * Good for Relaxation/Meditation is Negative correlation with energy(-0.48) and Tempo(-0.25) but Positively correlative with Acousticness(0.43) Because Meditation tracks are slow, calm and acoustic-heavy.

# %% [markdown]
# #### UX Tags Analysis

# %%
ux_cols = [col for col in data.columns if 'Good for' in col]
data[ux_cols].sum().sort_values().plot(kind='barh', color='teal')
plt.title("Frequency of UX Tags")
plt.show()

# %% [markdown]
# #### Insights

# %% [markdown]
# * `Good for Exercise`: Most frequent tag by a large margin — over 30,000 tracks. This shows a strong bias in the dataset towards high-energy, motivational music.
# * `Good for Party & Good for Work/Study` has similar volume of data with nearly 12,000 tracks.
# * `Good for Morning Routine & Good for Running` have moderate usage nearly 10k and 9k respectively.
# * `Good for Driving` Around 8k which is less frequent.

# %% [markdown]
# * `Good for Relaxation/Meditation & Good for Yoga/Stretching` are around 5k and 4k nearly. They are typically calm, slow tempo and possibly acoustic tracks.
# * So i need to merge these columns for into a new tag like `Relaxation/Yoga` for improving model generalization for calm/emotional music.
# * Also want to remove `Good for Social Gatherings` which may introduce noise or imbalance to model.

# %%
data['Good for Relaxation/Yoga'] = ((data['Good for Relaxation/Meditation'] == 1) | (data['Good for Yoga/Stretching'] == 1)).astype(int)
data.drop(['Good for Relaxation/Meditation', 'Good for Yoga/Stretching'], axis = 1, inplace=True)

# %%
data.drop('Good for Social Gatherings', axis = 1, inplace=True)

# %% [markdown]
# ### 4) Preprocessing and Cleaning data

# %%
### Converting objective type into numerical type
data['Loudness (db)'] = data['Loudness (db)'].str.replace('db','')
data['Loudness (db)'] = pd.to_numeric(data['Loudness (db)'], errors='coerce')

# %%
data['Loudness (db)'].dtype

# %%
suspicious = data[data['Artist(s)'].str.len() < 2]
print(suspicious[['Artist(s)']])

# %%
# Remove only unwanted symbols and extra spaces — keep useful names like 'u2', 'd major'
import re
def clean_artist(artist):
    artist = str(artist).strip().lower()
    artist = re.sub(r'[^\w\s]', '', artist)  
    artist = re.sub(r'\s+', ' ', artist)     
    return artist.strip()

data['Artist(s)'] = data['Artist(s)'].apply(clean_artist)

# %%
# Now remove entries with artist names too short
data = data[data['Artist(s)'].str.len() > 1]

# %%
print("Remaining short artist names:", data['Artist(s)'][data['Artist(s)'].str.len() < 2].unique())

# %%
short_artists = data['Artist(s)'][data['Artist(s)'].str.lower().str.len() < 2].unique()
print("Short artist names:", short_artists)

# %%
data['Artist(s)']

# %%
data.text.value_counts()

# %% [markdown]
# #### Remove Artist(s) with invalid name 

# %%
### Cleaning Text column
import re
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# %%
data['song_clean'] = data['song'].apply(clean_text)
data['artist_clean'] = data['Artist(s)'].apply(clean_text)
data['text_clean'] = data['text'].apply(clean_text)

# %%
data = data.drop_duplicates().reset_index(drop=True)

# %%
data['Artist(s)'].value_counts()

# %%
data.drop(columns=['song_clean', 'artist_clean', 'text_clean'], inplace=True)

# %%
data.info()

# %%
#### Cleaning Genre Column
data.Genre

# %%
#### Cleaning Genre Column
data['Genre'] = data['Genre'].str.lower().str.replace(' ', '').str.replace('-', '')
data['Genre'] = data['Genre'].str.split(',')

# Join list into a space-separated string
data['Genre_str'] = data['Genre'].apply(lambda x: ' '.join(x))

# %%
top_genres = data['Genre'].explode().value_counts().nlargest(20)
top_genres.plot(kind='bar', figsize=(12, 6))
plt.title('Top 20 Genres')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

top_genres.sort_values().plot(kind='barh', figsize=(10, 8))

# %%
data.Genre_str

# %%
data = data.drop('Genre', axis=1)

# %%
data.info()

# %%
data.to_csv("cleaned_data.csv", index = False)


