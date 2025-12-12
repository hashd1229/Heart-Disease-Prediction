import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import base64
import joblib

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")
# Custom CSS for styling and background
def set_bg_image():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
        }}
        
        /* Optional: Add an image background instead */
        /*
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=1920");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        */
        
        /* Main container styling */
        .main .block-container {{
            padding: 2rem 2rem;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }}
        
        /* Title styling */
        h1 {{
            color: #667eea;
            text-align: center;
            font-size: 3rem !important;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
        
        /* Subtitle styling */
        .subtitle {{
            text-align: center;
            color: #764ba2;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }}
        
        /* Input labels */
        .stSelectbox label, .stNumberInput label {{
            color: #667eea !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
        }}
        
        /* Button styling */
        .stButton>button {{
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 1.3rem;
            font-weight: 600;
            padding: 0.8rem 2rem;
            border-radius: 50px;
            border: none;
            box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.5);
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.7);
        }}
        
        /* Success/Error message styling */
        .stSuccess, .stError {{
            padding: 1rem;
            border-radius: 10px;
            font-size: 1.2rem;
            font-weight: 600;
            text-align: center;
        }}
        
        /* Card styling for inputs */
        .input-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }}
        
        /* Hide streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image()

# Load and train model
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    heart_data = pd.read_csv('heart_disease_data.csv')
    x = heart_data.drop(columns='target', axis=1)
    y = heart_data['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return model

model = load_model()

# Header
st.markdown("# ❤️ Heart Disease Prediction System")
st.markdown('<p class="subtitle">🏥 AI-Powered Health Analysis Tool</p>', unsafe_allow_html=True)
st.markdown("---")

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Personal Information")
    age = st.number_input('🎂 Age', min_value=0, max_value=120, value=30)
    sex = st.selectbox('⚧️ Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    fbs = st.selectbox('🍬 Fasting Blood Sugar', [0, 1], 
                      format_func=lambda x: 'Normal (<120 mg/dl)' if x == 0 else 'High (>120 mg/dl)')

with col2:
    st.markdown("### 💓 Heart Metrics")
    trestbps = st.number_input('🩺 Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=125)
    chol = st.number_input('🧪 Cholesterol (mg/dl)', min_value=100, max_value=600, value=309)
    thalach = st.number_input('💗 Max Heart Rate', min_value=60, max_value=220, value=131)
    oldpeak = st.number_input('📊 ST Depression', min_value=0.0, max_value=10.0, value=1.8, step=0.1)

with col3:
    st.markdown("### 🔬 Clinical Data")
    cp = st.selectbox('💢 Chest Pain Type', [0, 1, 2, 3],
                      format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal', 'Asymptomatic'][x])
    restecg = st.selectbox('📈 Resting ECG', [0, 1, 2],
                           format_func=lambda x: ['Normal', 'ST-T Abnormality', 'LV Hypertrophy'][x])
    exang = st.selectbox('🏃 Exercise Induced Angina', [0, 1],
                         format_func=lambda x: 'No' if x == 0 else 'Yes')
    slope = st.selectbox('📉 ST Slope', [0, 1, 2],
                         format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
    ca = st.selectbox('🔴 Major Vessels (0-4)', [0, 1, 2, 3, 4])
    thal = st.selectbox('🫀 Thalassemia', [0, 1, 2, 3],
                        format_func=lambda x: ['Normal', 'Fixed Defect', 'Reversible Defect', 'Unknown'][x])

st.markdown("---")

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button('🔮 Predict Heart Disease Risk', use_container_width=True)

if predict_button:
    with st.spinner('🔄 Analyzing your health data...'):
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, 
                              thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0]
        
        st.markdown("---")
        
        if prediction[0] == 0:
            st.success('### ✅ Good News!')
            st.success(f'#### The analysis suggests LOW risk of heart disease')
            st.info(f'Confidence: {probability[0]*100:.1f}%')
            st.balloons()
        else:
            st.error('### ⚠️ Alert!')
            st.error(f'#### The analysis suggests HIGH risk of heart disease')
            st.warning(f'Risk Level: {probability[1]*100:.1f}%')
            st.warning('🏥 **Recommendation:** Please consult a healthcare professional immediately.')

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #667eea; padding: 1rem;'>
        <p><strong>⚕️ Disclaimer:</strong> This is an AI prediction tool for educational purposes only. 
        Always consult with healthcare professionals for medical advice.</p>
        <p style='font-size: 0.9rem; color: #764ba2;'>Made with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

def set_local_bg_image(image_path: str):
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            bposition: relative;
            background-image: linear-gradient(rgba(0,0,0,0.70), rgba(0,0,0,0.70)), 
                              url("data:image/png;base64,{img_data}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage (replace the gradient call):
# set_bg_image()  # remove this line if you want only the image
set_local_bg_image("assets/background.jpg")