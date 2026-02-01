import streamlit as st
import numpy as np
import joblib
import time

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Disease Prediction",
    page_icon="🩺",
    layout="centered"
)

# -------------------- Load Resources (Cached) --------------------
@st.cache_resource
def load_model():
    return joblib.load("bernoulli_nb_top50.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load("label_encoder_top50.pkl")

@st.cache_resource
def load_symptoms():
    return joblib.load("symptoms_top50.pkl")

model = load_model()
label_encoder = load_encoder()
symptoms = load_symptoms()

# -------------------- UI Header --------------------
st.markdown("## 🩺 Disease Prediction System")
st.caption("Select symptoms to get the **Top-5 most probable diseases**")

st.divider()

# -------------------- Symptom Selection --------------------
with st.container():
    selected_symptoms = st.multiselect(
        "🧾 Choose your symptoms:",
        options=sorted(symptoms),
        placeholder="Start typing symptoms..."
    )

# -------------------- Prediction --------------------
if st.button("🔍 Predict Disease", use_container_width=True):

    if len(selected_symptoms) == 0:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Animation: Spinner
        with st.spinner("Analyzing symptoms..."):
            time.sleep(0.8)

            input_data = np.zeros(len(symptoms), dtype=np.int8)
            for i, symptom in enumerate(symptoms):
                if symptom in selected_symptoms:
                    input_data[i] = 1

            probs = model.predict_proba([input_data])[0]
            top5_idx = probs.argsort()[-5:][::-1]

            top5_diseases = label_encoder.inverse_transform(top5_idx)
            top5_scores = probs[top5_idx]

        # Animation: Progress bar
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress.progress(i + 1)

        st.success("✅ Prediction Complete")

        # -------------------- Results --------------------
        st.subheader("🧠 Top 5 Predicted Diseases")

        for disease, score in zip(top5_diseases, top5_scores):
            st.metric(
                label=f"🦠 {disease}",
                value=f"{score * 100:.2f} %"
            )

# -------------------- Footer --------------------
st.divider()
st.caption("⚠️ **Disclaimer:** This application is for educational purposes only and is **not** a medical diagnosis tool.")
