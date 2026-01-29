import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import tensorflow_hub as hub

# Load models
binary_model = tf.keras.models.load_model("copd_binary_model.h5")
severity_model = tf.keras.models.load_model("copd_severity_model.h5")

with open("scaler_binary.pkl","rb") as f:
    scaler_binary = pickle.load(f)

with open("label_encoder_binary.pkl","rb") as f:
    enc_binary = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler_sev = pickle.load(f)

with open("label_encoder.pkl","rb") as f:
    enc_sev = pickle.load(f)

with open("prakriti_model.pkl","rb") as f:
    prakriti_model = pickle.load(f)

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

st.title("AI COPD & Ayurveda System")

wav = st.file_uploader("Upload Lung Sound (.wav)")

if wav:
    y, sr = librosa.load(wav, sr=16000)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    _, emb, _ = yamnet(waveform)
    features = np.mean(emb.numpy(), axis=0).reshape(1,-1)

    # COPD prediction
    x = scaler_binary.transform(features)
    p = binary_model.predict(x)[0][0]
    copd = "COPD" if p>0.5 else "Healthy"

    st.subheader("COPD Result")
    st.write(copd)

    if copd=="COPD":
        # Severity
        x2 = scaler_sev.transform(features)
        sev = np.argmax(severity_model.predict(x2))
        sev_map = ["Mild","Moderate","Severe"]
        st.subheader("Severity")
        st.write(sev_map[sev])

st.subheader("Prakriti Questions")
answers = []
for q in ["Dry skin","Oily skin","Heavy body","Cold","Hot","Fast movement"]:
    answers.append(st.checkbox(q))

if st.button("Analyze"):
    X = np.array(answers).reshape(1,-1)
    prak = prakriti_model.predict(X)[0]
    prak_map=["Vata","Pitta","Kapha"]
    st.subheader("Prakriti")
    st.write(prak_map[prak])
