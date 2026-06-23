
import streamlit as st
import tempfile
import pickle
import librosa
import numpy as np

from scipy.ndimage import maximum_filter
from collections import defaultdict

# Load fingerprint lookup database
with open("fingerprint_lookup.pkl", "rb") as f:
    fingerprint_lookup = pickle.load(f)


def identify_song(query_file):
    y, sr = librosa.load(query_file, sr=None)

    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)))

    local_max = maximum_filter(S_db, size=20) == S_db
    threshold = np.percentile(S_db, 98)

    peaks = np.where(local_max & (S_db > threshold))
    freq_idx, time_idx = peaks[0], peaks[1]

    sort_order = np.argsort(time_idx)
    freq_idx = freq_idx[sort_order]
    time_idx = time_idx[sort_order]

    query_fps = []

    for i in range(len(time_idx)):
        for j in range(i + 1, min(i + 6, len(time_idx))):

            dt = int(time_idx[j] - time_idx[i])

            if dt > 0:
                query_fps.append(
                    (
                        int(freq_idx[i]),
                        int(freq_idx[j]),
                        dt,
                        int(time_idx[i])
                    )
                )

    votes = defaultdict(int)

    for f1, f2, dt, t_query in query_fps:

        key = (f1, f2, dt)

        if key in fingerprint_lookup:

            for song, t_db in fingerprint_lookup[key]:
                votes[song] += 1

    if len(votes) == 0:
        return "No Match"

    return max(votes, key=votes.get)


# ---------------- Streamlit UI ---------------- #

st.set_page_config(page_title="EE200 Audio Fingerprinting")

st.title("EE200 Audio Fingerprinting")

st.write("Upload a song clip to identify it.")

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("Identifying song..."):

        prediction = identify_song(temp_path)

    st.success(f"Detected Song: {prediction}")
