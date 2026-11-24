import io
import re
from pathlib import Path

import numpy as np
import streamlit as st
import speech_recognition as sr
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "artifacts"

st.set_page_config(page_title="E-Commerce Product Classifier", page_icon="🛍️", layout="centered")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize_text(text: str):
    # Equivalent for this project's cleaned alphabetic text to the notebook's word_tokenize output.
    return clean_text(text).split()

@st.cache_resource
def load_artifacts():
    required = [
        MODEL_DIR / "ann_model.keras",
        MODEL_DIR / "word2vec.model",
        MODEL_DIR / "label_encoder.joblib",
    ]
    missing = [str(p.name) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: " + ", ".join(missing) +
            ". Run `python train_model.py --data ecommerceDataset.csv` first."
        )

    ann = load_model(MODEL_DIR / "ann_model.keras")
    w2v = Word2Vec.load(str(MODEL_DIR / "word2vec.model"))
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    return ann, w2v, encoder

def document_vector(tokens, embedding_model):
    vectors = [
        embedding_model.wv[word]
        for word in tokens
        if word in embedding_model.wv
    ]
    if not vectors:
        return np.zeros(embedding_model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)

def predict_category(text, ann, embedding_model, encoder):
    tokens = tokenize_text(text)
    vector = document_vector(tokens, embedding_model).reshape(1, -1)
    probabilities = ann.predict(vector, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return encoder.inverse_transform([index])[0], float(probabilities[index])

def transcribe_audio(uploaded_audio):
    recognizer = sr.Recognizer()
    audio_bytes = uploaded_audio.getvalue()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

st.title("🛍️ E-Commerce Product Category Classifier")
st.write("Classify product descriptions using **Word2Vec embeddings + an ANN model**.")
st.caption("Enter a description or record your voice. The voice input is transcribed before classification.")

try:
    ann_model, embedding_model, label_encoder = load_artifacts()
except Exception as error:
    st.error(str(error))
    st.info("Model files must be generated and placed in the `artifacts/` folder before running the app.")
    st.stop()

tab_text, tab_voice = st.tabs(["⌨️ Text input", "🎙️ Voice input"])

with tab_text:
    product_text = st.text_area(
        "Product description",
        placeholder="Example: Wireless bluetooth headphones with noise cancellation",
        height=130,
    )
    if st.button("Predict category", type="primary", key="text_predict"):
        if not product_text.strip():
            st.warning("Please enter a product description.")
        else:
            category, confidence = predict_category(
                product_text, ann_model, embedding_model, label_encoder
            )
            st.success(f"Predicted category: **{category}**")
            st.metric("Model confidence", f"{confidence * 100:.2f}%")

with tab_voice:
    st.write("Click the microphone, speak a product description, and stop recording.")
    audio_value = st.audio_input("Record your product description")
    if audio_value is not None:
        st.audio(audio_value)
        if st.button("Transcribe and predict", type="primary", key="voice_predict"):
            try:
                recognized_text = transcribe_audio(audio_value)
                st.write("**Recognized text:**", recognized_text)
                category, confidence = predict_category(
                    recognized_text, ann_model, embedding_model, label_encoder
                )
                st.success(f"Predicted category: **{category}**")
                st.metric("Model confidence", f"{confidence * 100:.2f}%")
            except sr.UnknownValueError:
                st.error("Speech could not be understood. Please record again.")
            except sr.RequestError:
                st.error("Speech recognition service is unavailable. Check your internet connection.")
            except Exception as error:
                st.error(f"Audio processing failed: {error}")

st.divider()
st.caption("Model: Word2Vec (100-dimensional document vectors) + Artificial Neural Network.")
