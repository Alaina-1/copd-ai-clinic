import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import tensorflow_hub as hub

st.set_page_config(page_title="AYURON", layout="centered")

# ===============================
# 🌿 AYURVEDA ENGINE
# ===============================
def ayurveda_recommendation(prakriti, copd_stage):

    rec = {}

    if "Kapha" in prakriti:
        rec["Pranayama"] = ["Kapalbhati", "Bhastrika", "Diaphragmatic breathing"]
    elif "Vata" in prakriti:
        rec["Pranayama"] = ["Anulom Vilom", "Slow deep breathing"]
    else:
        rec["Pranayama"] = ["Diaphragmatic breathing", "Anulom Vilom"]

    medicines = []
    if "Kapha" in prakriti:
        medicines += ["Pippali Rasayana", "Vyaghri Haritaki Avaleha"]
    if "Vata" in prakriti:
        medicines += ["Vasadi Kashaya", "Hareetakyadi Yoga"]

    rec["Medicines"] = medicines

    rec["Must Eat"] = ["Warm food", "Ginger", "Garlic", "Green gram", "Barley"]
    rec["Avoid"] = ["Curd", "Milk", "Cold drinks", "Fried food", "Sweets", "Banana"]

    if copd_stage is None:
        rec["Status"] = "Healthy lungs detected"
    elif copd_stage <= 1:
        rec["Panchakarma"] = ["Snehapana", "Virechana"]
    else:
        rec["Panchakarma"] = ["Not advised in severe COPD"]

    if copd_stage == 2:
        rec["Warning"] = "Severe COPD — consult doctor before therapy"

    return rec

# ===============================
# 🧠 LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    severity_model = tf.keras.models.load_model("copd_severity_model.h5")
    binary_model = tf.keras.models.load_model("copd_yamnet_model.h5")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    with open("scaler (1).pkl","rb") as f:
        scaler = pickle.load(f)

    with open("prakriti_model.pkl","rb") as f:
        prakriti_model = pickle.load(f)

    return severity_model, binary_model, yamnet, scaler, prakriti_model

severity_model, binary_model, yamnet, scaler, prakriti_model = load_models()

# ===============================
# 🏥 UI
# ===============================
st.title("🫁 AYURON – AI COPD & Ayurveda")

wav = st.file_uploader("Upload Lung Sound (.wav)")

# ===============================
# COPD ANALYSIS
# ===============================
if wav is not None:

    y, sr = librosa.load(wav, sr=16000)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    _, emb, _ = yamnet(waveform)
    features = np.mean(emb.numpy(), axis=0).reshape(1,-1)

    # 1️⃣ Binary model
    binary_prob = float(binary_model.predict(features)[0][0])

    st.subheader("🫁 Diagnosis")

    if binary_prob < 0.5:
        st.success("Healthy lungs detected")
        st.write(f"Confidence: {(1-binary_prob)*100:.1f}%")
        st.session_state["copd_stage"] = None

    else:
        st.error("COPD Detected")
        st.write(f"Confidence: {binary_prob*100:.1f}%")

        # 2️⃣ Severity
        X = scaler.transform(features)
        sev_pred = severity_model.predict(X)[0]
        sev = int(np.argmax(sev_pred))
        sev_map = ["Mild", "Moderate", "Severe"]

        st.subheader("📊 Severity")
        st.write(sev_map[sev])
        st.write(f"Confidence: {np.max(sev_pred)*100:.1f}%")

        st.session_state["copd_stage"] = sev

# ===============================
# PRAKRITI
# ===============================
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
    X = np.array(user).astype(int).reshape(1,-1)
    probs = prakriti_model.predict_proba(X)[0]

    st.subheader("🧬 Prakriti Result")
    st.write("Vata:", round(probs[0],2))
    st.write("Pitta:", round(probs[1],2))
    st.write("Kapha:", round(probs[2],2))

    prakriti = ["Vata","Pitta","Kapha"][np.argmax(probs)]
    st.success(f"Your Prakriti: {prakriti}")
    st.session_state["prakriti"] = prakriti

# ===============================
# AYURVEDA PLAN
# ===============================
if "prakriti" in st.session_state and "copd_stage" in st.session_state:

    plan = ayurveda_recommendation(
        st.session_state["prakriti"],
        st.session_state["copd_stage"]
    )

    st.subheader("🌿 Ayurvedic Treatment Plan")

    for k, v in plan.items():
        st.write(f"**{k}:**")
        if isinstance(v, list):
            for item in v:
                st.write("•", item)
        else:
            st.write(v)
