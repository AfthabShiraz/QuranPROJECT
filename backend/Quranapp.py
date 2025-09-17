import re
import whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests


def normalize_arabic(text):
    # Remove harakat (diacritics)
    harakat = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = harakat.sub('', text)
    # Remove tatweel (kashida)
    text = text.replace('ـ', '')
    # Normalize alif forms
    text = re.sub('[إأآا]', 'ا', text)
    # Normalize yaa and ta marbuta
    text = text.replace('ى', 'ي')
    text = text.replace('ة', 'ه')
    # Remove common pause marks or unusual signs if present
    text = re.sub(r'[۩۞۝]', '', text)
    # Trim spaces
    return text.strip()

ayah_index = {}

#splits the file into ayah surah and text
with open('/Users/afthabshiraz/Downloads/quran-uthmani.txt', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) == 3:
            surah = int(parts[0])
            ayah = int(parts[1])
            text = parts[2]
            norm_text = normalize_arabic(text)
            ayah_index[(surah, ayah)] = norm_text

print(f"Indexed {len(ayah_index)} ayahs.")

#Loads pretrained whisper model to transcribe recording
model = whisper.load_model("medium")
audiopath=""
audio_file = audiopath
result = model.transcribe(audio_file, language='ar')
raw_text = result['text']
normalized_transcript = normalize_arabic(raw_text)

#Build TF-IDF matrix of all ayahs
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2))
ayah_texts = list(ayah_index.values())
tfidf_matrix = vectorizer.fit_transform(ayah_texts)

# Encode transcript and compare
query_vec = vectorizer.transform([normalized_transcript])
cosines = cosine_similarity(query_vec, tfidf_matrix).flatten()

# Get the index of the highest cosine similarity
best_idx = np.argmax(cosines)
best_score = cosines[best_idx]
predicted_surah_ayah = list(ayah_index.keys())[best_idx]

#uses quran.com api to fetch corresponding tafseer
def get_tafseer(surah, ayah, tafseer_id=169):
    url = f"https://api.quran.com/api/v4/tafsirs/{tafseer_id}/by_ayah/{surah}:{ayah}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        text = data['tafsir']['text']
        return text
    else:
        return "Tafseer not available."

surah, ayah = predicted_surah_ayah

# Fetch tafseer
tafsir_text = get_tafseer(surah, ayah)
