import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import tempfile

# --- Config ---
st.set_page_config(page_title="🎵 Music Genre Classifier", layout="centered")
st.title("🎧 Music Genre Classification App")
st.markdown("Classify your music into genres using two powerful models: **MFCC + Random Forest** and **CNN + Spectrogram (Transfer Learning)**.")

# --- Load Models ---
@st.cache_resource
def load_tabular_model():
    with open("models/tabular_model.pkl", "rb") as f:
        model, encoder = pickle.load(f)
    return model, encoder

@st.cache_resource
def load_cnn_model():
    return load_model("models/cnn_model.h5")

tabular_model, label_encoder = load_tabular_model()
cnn_model = load_cnn_model()

# --- Audio Preprocessing for MFCC Model ---
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.reshape(1, -1)

# --- Generate Spectrogram Image ---
def create_spectrogram_image(file_path):
    y, sr = librosa.load(file_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(2.24, 2.24))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.axis('off')
    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp_img.name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return tmp_img.name

# --- Predict with CNN ---
def predict_from_spectrogram(img_path):
    # Fix: Convert to RGB to avoid alpha channel issue
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = cnn_model.predict(img_array)
    class_idx = np.argmax(preds)
    return class_idx, preds[0]

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a .wav music file", type=["wav"])

# --- Model Choice ---
model_option = st.selectbox("Choose Model", ["MFCC + Random Forest", "Spectrogram CNN"])

# --- Predict Genre ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if model_option == "MFCC + Random Forest":
        st.subheader("🎶 MFCC Model Prediction")
        features = extract_mfcc(tmp_path)
        prediction = tabular_model.predict(features)[0]
        genre = label_encoder.inverse_transform([prediction])[0]
        st.success(f"Predicted Genre: **{genre}**")

    else:
        st.subheader("🖼️ CNN + Spectrogram Prediction")
        spec_path = create_spectrogram_image(tmp_path)
        class_idx, probs = predict_from_spectrogram(spec_path)

        # Genre labels from folder (sorted)
        genres = sorted(os.listdir("data/genres"))
        predicted_genre = genres[class_idx]

        st.image(spec_path, caption="Generated Spectrogram", width=300)
        st.success(f"Predicted Genre: **{predicted_genre}**")

        # Show top-3 predictions (optional bonus)
        top_indices = np.argsort(probs)[-3:][::-1]
        st.markdown("### 🔢 Top 3 Predictions:")
        for idx in top_indices:
            st.write(f"**{genres[idx]}**: {probs[idx]*100:.2f}%")

        # Full probability chart
        prob_dict = {genres[i]: float(probs[i]) for i in range(len(probs))}
        st.bar_chart(prob_dict)

        # Clean up
        os.remove(spec_path)

    os.remove(tmp_path)
