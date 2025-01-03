import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Load merged song data
with open("updated_song_data.json", "r") as f:
    song_data = json.load(f)

# Extract features for the model
def extract_features(song):
    audio_features = song["audio_features"]
    lyrical_analysis = song["lyrical_analysis"]

    # Extract normalized audio features
    audio_vector = np.array([
        audio_features.get("tempo", 0),
        audio_features.get("chroma_stft_mean", 0),
        audio_features.get("rmse_mean", 0),
        audio_features.get("spectral_centroid_mean", 0),
        audio_features.get("spectral_bandwidth_mean", 0),
        audio_features.get("spectral_contrast_mean", 0),
        audio_features.get("spectral_flatness_mean", 0),
        audio_features.get("zero_crossing_rate_mean", 0),
    ] + [audio_features.get(f"mfcc_mean_{i}", 0) for i in range(1, 21)])

    # Extract normalized lyrical features
    lyrical_vector = np.array([
        lyrical_analysis.get("polarity_score", 0),
        lyrical_analysis["emotions"].get("joy", 0),
        lyrical_analysis["emotions"].get("sadness", 0),
        lyrical_analysis["emotions"].get("anger", 0),
    ])

    # Concatenate audio and lyrical features
    return np.concatenate((audio_vector, lyrical_vector))

# Precompute feature vectors for all songs
song_vectors = {}
for song in song_data:
    try:
        song_vectors[song["track_name"]] = extract_features(song)
    except KeyError:
        # Handle songs with missing lyrical or audio data
        pass

def generate_prompt_vector(user_prompt):
    client = Groq(
        api_key="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"
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant. When provided with a user prompt, generate a list of 32 comma-separated floating-point numbers to represent audio-lyrical metrics. Do not include any text or explanation, only the numbers. NOTE: GENERATE EXACTLY 32 NUMBERS, NOT MORE OR LESS."},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=512
    )
    response = chat_completion.choices[0].message.content.strip()

    # Parse the response into numerical metrics
    try:
        metrics = np.array([float(x) for x in response.split(",")])
        if len(metrics) != 32:  # Ensure we have 32 dimensions
            raise ValueError("Unexpected number of metrics.")
        return metrics
    except Exception as e:
        st.error(f"Error parsing LLM response: {response}. Falling back to default metrics. Error: {e}")
        # Return default metrics if parsing fails
        return np.zeros(32)

# Match songs to user prompt
def match_songs_to_prompt(user_prompt, top_n=10):
    # Generate user vector
    user_vector = generate_prompt_vector(user_prompt)

    # Compute similarities
    song_similarities = []
    for track_name, song_vector in song_vectors.items():
        similarity = cosine_similarity([user_vector], [song_vector])[0, 0]
        song_similarities.append((track_name, similarity))

    # Sort songs by similarity
    song_similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top N songs
    return [(track, next(song["spotify_link"] for song in song_data if song["track_name"] == track)) for track, _ in song_similarities[:top_n]]

# Streamlit UI
st.title("Song Matcher Based on User Prompts")

# Input for user prompt
user_prompt = st.text_input("Enter your prompt for the type of song you want:")

if user_prompt:
    st.write("Generating your playlist...")
    playlist = match_songs_to_prompt(user_prompt, top_n=10)
    st.subheader("Generated Playlist")
    for idx, (song, link) in enumerate(playlist, 1):
        st.markdown(f"{idx}. [{song}]({link})")
