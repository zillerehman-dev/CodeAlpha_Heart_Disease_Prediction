import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = "heart_disease_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("❤️ Heart Disease Risk Predictor")
st.caption(
    "A machine learning model (tuned Logistic Regression, ~85% test accuracy, "
    "~0.91 ROC-AUC on held-out data) trained on the UCI Cleveland Heart Disease dataset."
)
st.warning(
    "⚠️ **Disclaimer:** This application is for educational and research purposes "
    "only and is **not** a medical diagnostic tool. It should never replace advice "
    "from a qualified healthcare professional."
)

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=54)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox(
        "Chest Pain Type", options=[
            ("Typical angina", 0), ("Atypical angina", 1),
            ("Non-anginal pain", 2), ("Asymptomatic", 3),
        ], format_func=lambda x: x[0])[1]
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=60, max_value=250, value=130)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=700, value=246)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    restecg = st.selectbox(
        "Resting ECG Result", options=[
            ("Normal", 0), ("ST-T wave abnormality", 1), ("Left ventricular hypertrophy", 2),
        ], format_func=lambda x: x[0])[1]

with col2:
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise-Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment", options=[
            ("Upsloping", 0), ("Flat", 1), ("Downsloping", 2),
        ], format_func=lambda x: x[0])[1]
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Thalassemia", options=[
            ("Normal", 1), ("Fixed defect", 2), ("Reversible defect", 3),
        ], format_func=lambda x: x[0])[1]

if st.button("Predict", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.header("Result")
    if prediction == 1:
        st.error("**Prediction: Heart Disease Detected**")
    else:
        st.success("**Prediction: No Heart Disease Detected**")

    st.metric("Disease Probability", f"{probability * 100:.1f}%")
    st.progress(min(max(probability, 0.0), 1.0))

    st.caption(
        "This probability reflects a statistical pattern learned from historical data, "
        "not a medical certainty. Please consult a qualified healthcare professional "
        "for an actual diagnosis."
    )

st.divider()
st.caption(
    "Model: tuned Logistic Regression • Trained on 302 unique patient records "
    "(UCI Cleveland Heart Disease dataset, duplicates removed) • "
    "Test set (60 patients): 85.2% accuracy, 87.5% precision, 84.8% recall, 90.6% ROC-AUC."
)
