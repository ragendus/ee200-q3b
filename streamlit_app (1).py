
import streamlit as st
import tempfile

st.title("EE200 Audio Fingerprinting")

uploaded_file = st.file_uploader(
    "Upload a song clip",
    type=["mp3", "wav"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    prediction = identify_song(temp_path)

    st.success(f"Detected Song: {prediction}")
