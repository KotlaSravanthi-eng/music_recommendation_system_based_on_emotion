import spacy
import re

nlp = spacy.load("en_core_web_sm")

# Map user mood keywords to dataset mood labels
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
    'fearful': 'fear',
}

def extract_mood_artist(user_input, known_artists):
    user_input = user_input.lower()
    doc = nlp(user_input)

    # Extract mood from predefined map
    raw_moods = [token.text for token in doc if token.text in mood_map]
    mood = mood_map[raw_moods[0]] if raw_moods else ""

    # Fallback: try regex to catch mood if spaCy misses it
    if not mood:
        for word in mood_map:
            if re.search(rf"\b{word}\b", user_input):
                mood = mood_map[word]
                break

    # Extract artist by fuzzy matching user input with known artists
    artist = ""
    for known in known_artists:
        if known.lower() in user_input:
            artist = known.lower()
            break

    return mood, artist
