import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tempfile
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from audio_utils import audio_to_text
from video_utils import extract_audio

@st.cache_resource
def load_resources():

    embedding_model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    routing_model = joblib.load(
        r"C:\Users\chitt\complaint-auto-routing-system\models\routing.pkl"
    )

    priority_model = joblib.load(
        r"C:\Users\chitt\complaint-auto-routing-system\models\priority.pkl"
    )

    eta_model = joblib.load(
        r"C:\Users\chitt\complaint-auto-routing-system\models\eta.pkl"
    )

    complaints_df = pd.read_csv(
        r"C:\Users\chitt\complaint-auto-routing-system\data\complaints.csv"
    )

    complaint_embeddings = np.load(
        r"C:\Users\chitt\complaint-auto-routing-system\data\complaint_embeddings.npy"
    )

    return (
        embedding_model,
        routing_model,
        priority_model,
        eta_model,
        complaints_df,
        complaint_embeddings
    )


(
    embedding_model,
    routing_model,
    priority_model,
    eta_model,
    complaints_df,
    complaint_embeddings
) = load_resources()


st.title("Complaint Auto Routing System")

st.write(
    "Submit complaint using Text, Audio, or Video"
)

text_input = st.text_area(
    "Enter Complaint"
)

uploaded_file = st.file_uploader(
    "Upload Audio / Video",
    type=["wav", "mp3", "mp4"]
)

complaint_text = ""


if st.button("Analyze Complaint"):

    # Text Input
    if text_input.strip():
        complaint_text = text_input

    # Audio / Video Input
    elif uploaded_file is not None:

        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(
                uploaded_file.read()
            )

            file_path = tmp.name

        # Audio
        if suffix.lower() in [".wav", ".mp3"]:

            complaint_text = audio_to_text(
                file_path
            )

        # Video
        elif suffix.lower() == ".mp4":

            audio_path = extract_audio(
                file_path
            )

            complaint_text = audio_to_text(
                audio_path
            )

    else:

        st.warning(
            "Please enter a complaint or upload a file."
        )

        st.stop()

    st.subheader("Extracted Complaint")

    st.info(complaint_text)

    # Generate embedding
    embedding = embedding_model.encode(
        [complaint_text]
    )

    # Predictions
    officer = routing_model.predict(
        embedding
    )[0]

    priority = priority_model.predict(
        embedding
    )[0]

    eta = eta_model.predict(
        embedding
    )[0]

    # Similar complaints using cosine similarity
    similarity_scores = cosine_similarity(
        embedding,
        complaint_embeddings
    )

    top_indices = similarity_scores[0].argsort()[-5:][::-1]

    similar_complaints = complaints_df.iloc[
        top_indices
    ]["complaint_text"].tolist()

    # Results
    st.subheader("Prediction Results")

    st.success(
        f"Assigned Officer: {officer}"
    )

    st.info(
        f"Priority: {priority}"
    )

    st.warning(
        f"Estimated Resolution Time: {round(eta)} Days"
    )

    st.subheader(
        "Top 5 Similar Complaints"
    )

    for i, complaint in enumerate(
        similar_complaints,
        start=1
    ):
        st.write(
            f"{i}. {complaint}"
        )