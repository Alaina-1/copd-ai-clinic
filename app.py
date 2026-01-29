import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import tensorflow_hub as hub

st.set_page_config(page_title="AYURON", layout="centered")
st.write("Scaler expects:", scaler.n_features_in_)
st.write("Binary model expects:", binary_model.input_shape[1])

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_all():

    binary_model = tf.keras.models.load_model("copd_yamnet_model.h5")
    severity_model = tf.keras.models.load_model("copd_severity_model.h5")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    with open("scaler (1).pkl","rb") as f:
        scaler = pickle.load(f)

    with open("prakriti_model.pkl","rb") as f:
        prakriti_model = pickle.load(f)

    return binary_model, severity_model, yamnet, scaler, prakriti_model

binary_model, severity_model, yamnet, scaler, prakriti_model = load_all()

# -------------------------------
# Ayurveda Logic
# -------------------------------
def ayurveda_recommendation(prakriti, stage):
    rec = {}
    if prakriti == "Kapha":
        rec["Pranayama"] = ["Kapalbhati", "Bhastrika"]
        rec["Medicines"] = ["Pippali Rasayana", "Vyaghri Haritaki"]
    elif prakriti == "Vata":
        rec["Pranayama"] = ["Anulom Vilom", "Slow Breathing"]
        rec["Medicines"] = ["Vasadi Kashaya"]
    else:
        rec["Pranayama"] = ["Cooling Breathing"]
        rec["Medicines"] = ["Guduchi"]

    if stage == None:
        rec["Status"] = "Healthy Lungs"
    elif stage == 0:
        rec["Stage"] = "Mild COPD"
    elif stage == 1:
        rec["Stage"] = "Moderate COPD"
    else:
        rec["Stage"] = "Severe COPD"

    return rec

# -------------------------------
# UI
# -------------------------------
st.title("🌿 AYURON – AI COPD & Ayurveda")

wav = st.file_uploader("Upload lung sound (.wav)")

if wav:
    y, sr = librosa.load(wav, sr=16000)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)

    _, emb, _ = yamnet(waveform)
    features = np.mean(emb.numpy(), axis=0)

    # 🔥 Resize to model input size
    needed = binary_model.input_shape[1]
    features = features[:needed].reshape(1,-1)

    # 🔥 Scale
    X = scaler.transform(features)

    # -------------------------------
    # Binary COPD
    # -------------------------------
    prob = float(binary_model.predict(X)[0][0])

    if prob < 0.5:
        st.success("Healthy Lungs Detected")
        st.write(f"Confidence: {(1-prob)*100:.1f}%")
        stage = None
    else:
        st.error("COPD Detected")
        st.write(f"Confidence: {prob*100:.1f}%")

        # -------------------------------
        # Severity
        # -------------------------------
        sev_pred = severity_model.predict(X)[0]
        stage = int(np.argmax(sev_pred))
        sev_map = ["Mild", "Moderate", "Severe"]
        st.subheader("Severity")
        st.write(sev_map[stage])

    st.session_state["stage"] = stage

# -------------------------------
# Prakriti
# -------------------------------
st.subheader("🌿 Prakriti Questionnaire")

questions = [
    "Dry skin", "Oily skin", "Thick skin",
    "Feels cold", "Feels hot",
    "Light body", "Heavy body",
    "Irregular digestion", "Strong digestion", "Slow digestion",
    "Light sleep", "Deep sleep",
    "Anxious", "Intense", "Calm",
    "Fast movement", "Slow movement"
]

user = [st.checkbox(q) for q in questions]

if st.button("Analyze Prakriti"):
    Xp = np.array(user).astype(int).reshape(1,-1)
    probs = prakriti_model.predict_proba(Xp)[0]
    prakriti = ["Vata","Pitta","Kapha"][np.argmax(probs)]

    st.success(f"Your Prakriti: {prakriti}")
    st.session_state["prakriti"] = prakriti

# -------------------------------
# Ayurveda Output
# -------------------------------
if "prakriti" in st.session_state and "stage" in st.session_state:
    plan = ayurveda_recommendation(st.session_state["prakriti"], st.session_state["stage"])
    st.subheader("🌿 Ayurvedic Plan")
    for k,v in plan.items():
        st.write(f"**{k}:** {v}")
