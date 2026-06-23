import streamlit as st
import tempfile
import pickle
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.ndimage import maximum_filter
from collections import defaultdict

# =========================
# LOAD DATABASE
# =========================

with open("fingerprint_lookup.pkl", "rb") as f:
    fingerprint_lookup = pickle.load(f)

# =========================
# IDENTIFY SONG
# =========================

def identify_song(query_file):

    y, sr = librosa.load(query_file, sr=None)

    S_db = librosa.amplitude_to_db(
        np.abs(librosa.stft(y))
    )

    local_max = maximum_filter(
        S_db,
        size=20
    ) == S_db

    threshold = np.percentile(
        S_db,
        98
    )

    peaks = np.where(
        local_max & (S_db > threshold)
    )

    freq_idx = peaks[0]
    time_idx = peaks[1]

    sort_order = np.argsort(time_idx)

    freq_idx = freq_idx[sort_order]
    time_idx = time_idx[sort_order]

    query_fps = []

    for i in range(len(time_idx)):

        for j in range(
            i + 1,
            min(i + 6, len(time_idx))
        ):

            dt = int(
                time_idx[j] - time_idx[i]
            )

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
        return "No Match", {}

    best_song = max(
        votes,
        key=votes.get
    )

    return best_song, votes

# =========================
# PLOTS
# =========================

def plot_spectrogram(audio_file):

    y, sr = librosa.load(
        audio_file,
        sr=None
    )

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y)),
        ref=np.max
    )

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    ax.imshow(
        D,
        origin="lower",
        aspect="auto"
    )

    ax.set_title(
        "Spectrogram"
    )

    return fig


def plot_constellation(audio_file):

    y, sr = librosa.load(
        audio_file,
        sr=None
    )

    S_db = librosa.amplitude_to_db(
        np.abs(librosa.stft(y))
    )

    local_max = maximum_filter(
        S_db,
        size=20
    ) == S_db

    threshold = np.percentile(
        S_db,
        98
    )

    peaks = np.where(
        local_max & (S_db > threshold)
    )

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    ax.scatter(
        peaks[1],
        peaks[0],
        s=5
    )

    ax.set_title(
        "Constellation Map"
    )

    return fig


def plot_histogram(votes):

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    ax.bar(
        list(votes.keys()),
        list(votes.values())
    )

    ax.set_title(
        "Offset Histogram"
    )

    plt.xticks(rotation=90)

    return fig

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="EE200 Audio Fingerprinting"
)

st.title(
    "EE200 Audio Fingerprinting"
)

mode = st.sidebar.selectbox(
    "Mode",
    [
        "Single Clip",
        "Batch Mode"
    ]
)

# =========================
# SINGLE CLIP MODE
# =========================

if mode == "Single Clip":

    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["wav","mp3"]
    )

    if uploaded_file is not None:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(
                uploaded_file.read()
            )

            temp_path = tmp.name

        prediction, votes = identify_song(
            temp_path
        )

        st.success(
            f"Detected Song: {prediction}"
        )

        st.subheader(
            "Spectrogram"
        )

        st.pyplot(
            plot_spectrogram(temp_path)
        )

        st.subheader(
            "Constellation Map"
        )

        st.pyplot(
            plot_constellation(temp_path)
        )

        st.subheader(
            "Offset Histogram"
        )

        st.pyplot(
            plot_histogram(votes)
        )

# =========================
# BATCH MODE
# =========================

elif mode == "Batch Mode":

    files = st.file_uploader(
        "Upload Query Clips",
        type=["wav","mp3"],
        accept_multiple_files=True
    )

    if files:

        results = []

        for file in files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as tmp:

                tmp.write(
                    file.read()
                )

                temp_path = tmp.name

            prediction, _ = identify_song(
                temp_path
            )

            prediction = os.path.splitext(
                prediction
            )[0]

            results.append(
                [
                    file.name,
                    prediction
                ]
            )

        df = pd.DataFrame(
            results,
            columns=[
                "filename",
                "prediction"
            ]
        )

        st.dataframe(df)

        csv = df.to_csv(
            index=False
        )

        st.download_button(
            "Download results.csv",
            csv,
            "results.csv",
            "text/csv"
        )
