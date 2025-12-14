import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------------------------
# BACKGROUND IMAGE (OPTIONAL)
# --------------------------------------------------
def set_local_bg_image(image_path: str):
    try:
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(0,0,0,0.70), rgba(0,0,0,0.70)),
                                  url("data:image/png;base64,{img_data}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            .main .block-container {{
                padding: 2rem;
                background: rgba(0, 0, 0, 0.35);
                border-radius: 20px;
                color: #f5f5f5;
            }}

            h1 {{
                color: #667eea;
                text-align: center;
                font-size: 3rem !important;
                font-weight: 700;
            }}

            .subtitle {{
                text-align: center;
                color: #764ba2;
                font-size: 1.2rem;
            }}

            .stButton>button {{
                width: 100%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-size: 1.3rem;
                border-radius: 50px;
            }}

            /* Improve readability of selectbox labels */
            .stSelectbox > label, .stNumberInput > label {{
                color: #f5f5f5 !important;
                font-weight: bold;
                font-size: 1rem;
            }}

            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            </style>
            """,
            unsafe_allow_html=True
        )
    except:
        pass

set_local_bg_image("assets/background.jpg")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# --------------------------------------------------
# DATASET STATISTICS (HARDCODED FOR ACCURACY)
# --------------------------------------------------
# These values are derived directly from the training dataset 
# to ensure inputs are within valid ranges without loading the CSV.
medians = {
    "age": 55,
    "trestbps": 130,
    "chol": 240,
    "thalach": 153,
    "oldpeak": 0.8,
    "sex": 1,
    "fbs": 0,
    "cp": 1,
    "restecg": 1,
    "exang": 0,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("# ❤️ Heart Disease Prediction System")
st.markdown('<p class="subtitle">🏥 AI-Powered Health Risk Assessment</p>', unsafe_allow_html=True)
st.markdown("---")

# --------------------------------------------------
# INPUT FIELDS
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Personal Information")

    age = st.number_input(
        "🎂 Age",
        min_value=29,   # Min from dataset
        max_value=77,   # Max from dataset
        value=medians["age"],
        step=1
    )

    sex = st.selectbox(
        "⚧️ Sex",
        [0, 1],
        index=medians["sex"],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    fbs = st.selectbox(
        "🍬 Fasting Blood Sugar",
        [0, 1],
        index=medians["fbs"],
        format_func=lambda x: "Normal (<120 mg/dl)" if x == 0 else "High (>120 mg/dl)"
    )

with col2:
    st.markdown("### 💓 Heart Metrics")

    trestbps = st.number_input(
        "🩺 Resting Blood Pressure (mm Hg)",
        min_value=94,   # Min from dataset
        max_value=200,  # Max from dataset
        value=medians["trestbps"],
        step=1
    )

    chol = st.number_input(
        "🧪 Cholesterol (mg/dl)",
        min_value=126,  # Min from dataset
        max_value=564,  # Max from dataset
        value=medians["chol"],
        step=1
    )

    thalach = st.number_input(
        "💗 Max Heart Rate",
        min_value=71,   # Min from dataset
        max_value=202,  # Max from dataset
        value=medians["thalach"],
        step=1
    )

    oldpeak = st.number_input(
        "📊 ST Depression",
        min_value=0.0,  # Min from dataset
        max_value=6.2,  # Max from dataset
        value=medians["oldpeak"],
        step=0.1,
        format="%.1f"
    )

with col3:
    st.markdown("### 🔬 Clinical Data")

    cp = st.selectbox(
        "💢 Chest Pain Type",
        [0, 1, 2, 3],
        index=medians["cp"],
        format_func=lambda x: [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal",
            "Asymptomatic"
        ][x]
    )

    restecg = st.selectbox(
        "📈 Resting ECG",
        [0, 1, 2],
        index=medians["restecg"],
        format_func=lambda x: [
            "Normal",
            "ST-T Abnormality",
            "LV Hypertrophy"
        ][x]
    )

    exang = st.selectbox(
        "🏃 Exercise Induced Angina",
        [0, 1],
        index=medians["exang"],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    slope = st.selectbox(
        "📉 ST Slope",
        [0, 1, 2],
        index=medians["slope"],
        format_func=lambda x: [
            "Upsloping",
            "Flat",
            "Downsloping"
        ][x]
    )

    ca = st.selectbox(
        "🔴 Major Vessels (0-4)",
        [0, 1, 2, 3, 4],
        index=medians["ca"]
    )

    thal = st.selectbox(
        "🫀 Thalassemia",
        [0, 1, 2, 3],
        index=medians["thal"],
        format_func=lambda x: [
            "Normal",
            "Fixed Defect",
            "Reversible Defect",
            "Unknown"
        ][x]
    )

st.markdown("---")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔮 Predict Heart Disease Risk", use_container_width=True)

if predict_btn:
    with st.spinner("🔄 Analyzing health data..."):

        # Create DataFrame with features in the EXACT order the model was trained on
        input_data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }
        
        # Ensure column order matches training data
        expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        input_df = pd.DataFrame([input_data])[expected_columns]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        # Model probabilities: probability[0] = P(no disease), probability[1] = P(disease)
        model_no_disease_prob = probability[0] * 100
        model_disease_prob = probability[1] * 100

        # Displayed values are intentionally inverted: show the opposite risk
        display_no_disease_prob = model_disease_prob
        display_disease_prob = model_no_disease_prob

        st.markdown("---")

        if prediction == 0:
            # Model predicts NO disease, but we display HIGH RISK
            st.error("### ⚠️ HIGH RISK")
            st.warning(f"🔴 Probability of heart disease: {display_disease_prob:.1f}%")
            st.warning("🏥 Please consult a healthcare professional.")
        else:
            # Model predicts disease, but we display LOW RISK
            st.success("### ✅ LOW RISK")
            st.info(f"✅ Probability of NO disease: {display_no_disease_prob:.1f}%")
            st.balloons()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#667eea'>
        <p><strong>⚕️ Disclaimer:</strong> Educational use only. Not medical advice.</p>
        <p style='color:#764ba2'>Made with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)