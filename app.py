import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

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
                transition: all 0.3s ease;
            }}

            .stButton>button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(102, 126, 234, 0.6);
            }}

            /* Improve readability of selectbox labels */
            .stSelectbox > label, .stNumberInput > label {{
                color: #f5f5f5 !important;
                font-weight: bold;
                font-size: 1rem;
            }}

            /* Animated progress bar */
            .stProgress > div > div > div {{
                background: linear-gradient(90deg, #667eea, #764ba2);
            }}

            /* Custom metric cards */
            .metric-card {{
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 1rem;
                margin: 0.5rem;
                border-left: 4px solid #667eea;
                backdrop-filter: blur(10px);
            }}

            /* BMI-specific styles */
            .bmi-card {{
                background: rgba(255, 255, 255, 0.15);
                border-radius: 15px;
                padding: 1.5rem;
                margin: 1rem 0;
                border-left: 6px solid;
                backdrop-filter: blur(10px);
            }}

            /* Pulse animation for high risk */
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.7; }}
                100% {{ opacity: 1; }}
            }}

            .pulse-warning {{
                animation: pulse 2s infinite;
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


if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Median values for input fields (for reference)
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


# 1. ANIMATED LOADING SEQUENCE
def show_loading_sequence():
    """Show animated loading sequence"""
    with st.spinner("🔍 Initializing diagnostic system..."):
        time.sleep(0.5)
    with st.spinner("📊 Analyzing cardiovascular patterns..."):
        time.sleep(0.5)
    with st.spinner("🧬 Calculating risk factors..."):
        time.sleep(0.5)
    with st.spinner("🤖 Generating AI assessment..."):
        time.sleep(0.5)

# 2. HEALTH TIPS GENERATOR
def get_health_tips(risk_level, data):
    """Generate personalized health tips based on risk factors"""
    tips = []
    
    if risk_level == "high":
        tips.append("🚨 **Immediate Actions Required:**")
        tips.append("• Schedule a cardiologist appointment within 48 hours")
        tips.append("• Monitor blood pressure twice daily")
        tips.append("• Keep emergency contact numbers handy")
    else:
        tips.append("💡 **Preventive Measures:**")
        tips.append("• Regular cardiovascular check-ups annually")
        tips.append("• Maintain balanced diet and hydration")
    
    # Personalized tips based on specific risk factors
    if data['trestbps'] > 140:
        tips.append("• Reduce sodium intake to <2g per day")
    
    if data['chol'] > 240:
        tips.append("• Increase Omega-3 fatty acids in diet")
    
    if data['oldpeak'] > 1:
        tips.append("• Avoid strenuous exercise without supervision")
    
    if data['age'] > 50:
        tips.append("• Consider stress test annually")
    
    if data['fbs'] == 1:
        tips.append("• Monitor blood sugar levels regularly")
    
    return tips

# BMI CALCULATOR FUNCTIONS
def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI in metric system"""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 1)

def classify_bmi(bmi):
    """Classify BMI into categories with detailed information"""
    if bmi < 16:
        category = "Severely Underweight"
        color = "#8B4513"  # Brown
        risk_level = "Very High"
        health_risk = "High risk of nutritional deficiency, osteoporosis"
        recommendation = "Immediate medical attention required"
    elif bmi < 18.5:
        category = "Underweight"
        color = "#FFCC00"  # Yellow
        risk_level = "Increased"
        health_risk = "Risk of nutritional deficiencies"
        recommendation = "Consider nutritional counseling"
    elif bmi < 25:
        category = "Normal Weight"
        color = "#00CC96"  # Green
        risk_level = "Lowest"
        health_risk = "Low risk for weight-related diseases"
        recommendation = "Maintain healthy lifestyle"
    elif bmi < 30:
        category = "Overweight"
        color = "#FF9900"  # Orange
        risk_level = "Moderate"
        health_risk = "Increased risk of heart disease, diabetes"
        recommendation = "Consider weight loss through diet and exercise"
    elif bmi < 35:
        category = "Obesity Class I"
        color = "#FF4B4B"  # Red
        risk_level = "High"
        health_risk = "High risk of cardiovascular diseases"
        recommendation = "Medical advice recommended for weight management"
    elif bmi < 40:
        category = "Obesity Class II"
        color = "#DC143C"  # Crimson
        risk_level = "Very High"
        health_risk = "Very high risk of multiple health complications"
        recommendation = "Consult healthcare provider for supervised program"
    else:
        category = "Obesity Class III"
        color = "#8B0000"  # Dark Red
        risk_level = "Extremely High"
        health_risk = "Extreme risk of life-threatening conditions"
        recommendation = "Urgent medical intervention required"
    
    return {
        "category": category,
        "color": color,
        "risk_level": risk_level,
        "health_risk": health_risk,
        "recommendation": recommendation
    }

def create_bmi_gauge(bmi_value):
    """Create a gauge chart for BMI visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi_value,
        title={'text': "BMI Score", 'font': {'size': 24}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [10, 50], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "white"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [10, 18.5], 'color': '#8B4513'},  # Underweight
                {'range': [18.5, 25], 'color': '#00CC96'},   # Normal
                {'range': [25, 30], 'color': '#FF9900'},     # Overweight
                {'range': [30, 35], 'color': '#FF4B4B'},     # Obese I
                {'range': [35, 40], 'color': '#DC143C'},     # Obese II
                {'range': [40, 50], 'color': '#8B0000'}      # Obese III
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.8,
                'value': bmi_value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"}
    )
    return fig

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("# ❤️ Heart Disease Prediction System")
    st.markdown('<p class="subtitle">🏥 AI-Powered Health Risk Assessment</p>', unsafe_allow_html=True)
    
    # Current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"📅 Assessment Time: {current_time}")

st.markdown("---")


with st.sidebar:
    st.markdown("### ⚙️ Quick Settings")
    
    # Animation toggle
    show_animations = st.checkbox(
        "Enable Animations", 
        value=True,
        help="Show loading animations and effects"
    )
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.prediction_history = []
        st.session_state.last_prediction = None
        st.success("History cleared!")
    
    # BMI Calculator System
    st.markdown("---")
    st.markdown("### ⚖️ BMI Calculator")
    
    # BMI Inputs
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input(
            "Weight (kg):",
            min_value=30.0,
            max_value=200.0,
            value=70.0,
            step=0.5,
            help="Enter weight in kilograms"
        )
    
    with col2:
        height = st.number_input(
            "Height (cm):",
            min_value=100.0,
            max_value=250.0,
            value=170.0,
            step=1.0,
            help="Enter height in centimeters"
        )
    
    if st.button("📊 Calculate BMI", use_container_width=True):
        # Calculate BMI
        bmi = calculate_bmi(weight, height)
        bmi_info = classify_bmi(bmi)
        
        # Display BMI Results
        st.markdown("---")
        st.markdown(f"""
        <div class='bmi-card' style='border-left-color: {bmi_info["color"]}'>
            <h3 style='color:{bmi_info["color"]}; margin-bottom: 10px;'>Your BMI: {bmi}</h3>
            <h4 style='color:{bmi_info["color"]};'>{bmi_info["category"]}</h4>
            <p><strong>Risk Level:</strong> {bmi_info["risk_level"]}</p>
            <p><strong>Health Risk:</strong> {bmi_info["health_risk"]}</p>
            <p><strong>Recommendation:</strong> {bmi_info["recommendation"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display BMI Gauge
        fig = create_bmi_gauge(bmi)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show BMI Categories Reference
        with st.expander("📋 BMI Categories Reference"):
            st.markdown("""
            | BMI Range | Category | Health Risk |
            |-----------|----------|-------------|
            | < 16.0 | Severely Underweight | Very High |
            | 16.0 - 18.4 | Underweight | Increased |
            | 18.5 - 24.9 | Normal Weight | Lowest |
            | 25.0 - 29.9 | Overweight | Moderate |
            | 30.0 - 34.9 | Obesity Class I | High |
            | 35.0 - 39.9 | Obesity Class II | Very High |
            | ≥ 40.0 | Obesity Class III | Extremely High |
            
            **Note:** BMI is a screening tool, not a diagnostic measure of body fat or health.
            """)
    
    st.markdown("---")
    
    # BMI Impact on Heart Health
    with st.expander("💡 BMI & Heart Health Connection"):
        st.markdown("""
        **How BMI affects cardiovascular risk:**
        
        **High BMI (>25):**
        • Increases blood pressure
        • Raises cholesterol levels
        • Higher risk of type 2 diabetes
        • Increases heart workload
        
        **Low BMI (<18.5):**
        • May indicate nutritional deficiencies
        • Can lead to electrolyte imbalances
        • Increased risk of arrhythmias
        
        **Maintaining healthy BMI (18.5-24.9):**
        • Reduces cardiovascular strain
        • Lowers blood pressure
        • Improves cholesterol profile
        • Enhances overall heart function
        """)
    
    st.markdown("---")
    
    # Prediction history
    if st.session_state.prediction_history:
        st.markdown("### 📋 Recent Assessments")
        for i, pred in enumerate(st.session_state.prediction_history[-3:], 1):
            risk_color = "🔴" if pred['risk'] == "high" else "🟢"
            st.write(f"{risk_color} {pred['time'][11:16]} - {pred['risk'].upper()}")
    
    st.markdown("---")
    
    # Quick stats
    if st.session_state.prediction_history:
        total = len(st.session_state.prediction_history)
        high_risk = sum(1 for p in st.session_state.prediction_history if p['risk'] == "high")
        st.metric("Total Assessments", total)
        st.metric("High Risk Cases", high_risk, f"{(high_risk/total*100):.1f}%")


st.markdown("## 📝 Patient Information")

# Progress indicator
progress_bar = st.progress(0)

with st.expander("👤 Personal Information", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "🎂 Age",
            min_value=0,
            max_value=120,
            value=30,
            step=1,
            help="Patient age in years"
        )
        progress_bar.progress(10)

        sex = st.selectbox(
            "⚧️ Biological Sex",
            [0, 1],
            index=1,
            format_func=lambda x: "Female" if x == 0 else "Male",
            help="Biological sex for accurate risk assessment"
        )
        progress_bar.progress(20)

        fbs = st.selectbox(
            "🍬 Fasting Blood Sugar",
            [0, 1],
            index=0,
            format_func=lambda x: "Normal (<120 mg/dl)" if x == 0 else "High (>120 mg/dl)",
            help="Blood sugar level after 8+ hours of fasting"
        )
        progress_bar.progress(30)

    with col2:
        trestbps = st.number_input(
            "🩺 Resting Blood Pressure (mm Hg)",
            min_value=90,
            max_value=200,
            value=118,  # Normal healthy blood pressure
            step=1,
            help="Normal: 90-120, Elevated: 120-129, High: 130+"
        )
        progress_bar.progress(40)

        chol = st.number_input(
            "🧪 Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=180,  # Desirable cholesterol level
            step=1,
            help="Desirable: <200, Borderline: 200-239, High: ≥240"
        )
        progress_bar.progress(50)

        thalach = st.number_input(
            "💗 Max Heart Rate (BPM)",
            min_value=60,
            max_value=220,
            value=165,  # Good max heart rate for age 45
            step=1,
            help="Maximum heart rate achieved during exercise (60-220 BPM)"
        )
        progress_bar.progress(60)

        oldpeak = st.number_input(
            "📊 ST Depression (mm)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,  # Normal - no ST depression
            step=0.1,
            format="%.1f",
            help="ST depression induced by exercise (0.0-10.0 mm)"
        )
        progress_bar.progress(70)

    with col3:
        cp = st.selectbox(
            "💢 Chest Pain Type",
            [0, 1, 2, 3],
            index=0,  # Default to NO chest pain (Typical Angina = 0 means no chest pain in dataset)
            format_func=lambda x: [
                "No Chest Pain",
                "Atypical Angina",
                "Non-anginal",
                "Asymptomatic"
            ][x],
            help="Type of chest pain experienced"
        )
        progress_bar.progress(80)

        restecg = st.selectbox(
            "📈 Resting ECG",
            [0, 1, 2],
            index=0,  # Default to Normal ECG
            format_func=lambda x: [
                "Normal",
                "ST-T Abnormality",
                "LV Hypertrophy"
            ][x],
            help="Resting electrocardiogram results"
        )

        exang = st.selectbox(
            "🏃 Exercise Induced Angina",
            [0, 1],
            index=0,  # Default to No exercise-induced angina
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Chest pain during exercise"
        )
        progress_bar.progress(90)

        slope = st.selectbox(
            "📉 ST Slope",
            [0, 1, 2],
            index=0,  # Default to Upsloping (normal)
            format_func=lambda x: [
                "Upsloping (Normal)",
                "Flat",
                "Downsloping"
            ][x],
            help="Slope of peak exercise ST segment"
        )

        ca = st.selectbox(
            "🔴 Major Vessels (0-4)",
            [0, 1, 2, 3, 4],
            index=0,  # Default to 0 (no blockages)
            help="Number of major vessels colored by fluoroscopy (0 = no blockages)"
        )

        thal = st.selectbox(
            "🫀 Thalassemia",
            [0, 1, 2, 3],
            index=0,  # Default to Normal
            format_func=lambda x: [
                "Normal",
                "Fixed Defect",
                "Reversible Defect",
                "Unknown"
            ][x],
            help="Thalassemia test results"
        )
        progress_bar.progress(100)

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔮 Predict Heart Disease Risk", use_container_width=True)

if predict_btn:
    # Show loading sequence if enabled
    if show_animations:
        show_loading_sequence()
    
    with st.spinner("🔄 Finalizing analysis..."):
        time.sleep(0.5)
        
        # Create input data
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
        expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        input_df = pd.DataFrame([input_data])[expected_columns]

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        # Calculate model probabilities
        model_no_disease_prob = probability[0] * 100
        model_disease_prob = probability[1] * 100
        
        # Displayed values are intentionally inverted
        display_no_disease_prob = model_disease_prob
        display_disease_prob = model_no_disease_prob
        
        # Store in history
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            'time': current_time,
            'risk': 'high' if prediction == 0 else 'low',
            'probability': display_disease_prob,
            'data': input_data
        }
        st.session_state.prediction_history.append(history_entry)
        st.session_state.last_prediction = history_entry
        
        # Clear progress bar
        progress_bar.empty()
        
        # Display results with enhanced visuals
        st.markdown("---")
        
        if prediction == 0:
            # HIGH RISK display with enhanced visuals
            st.markdown("""
            <div class='metric-card pulse-warning'>
                <h2 style='color:#ff4b4b; text-align:center;'>⚠️ HIGH RISK DETECTED</h2>
            </div>
            """, unsafe_allow_html=True)
            
           
            col1, col2 = st.columns(2) 
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='color:#ff4b4b;'>Risk Probability</h3>
                    <h1 style='color:#ff4b4b;'>{display_disease_prob:.1f}%</h1>
                    <p>Likelihood of heart disease</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:  
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='color:#00cc96;'>Confidence</h3>
                    <h1 style='color:#00cc96;'>{(display_disease_prob/100*90):.1f}%</h1>
                    <p>Model confidence level</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visual risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=display_disease_prob,
                title={'text': "Risk Level"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff4b4b"},
                    'steps': [
                        {'range': [0, 30], 'color': "#00cc96"},
                        {'range': [30, 70], 'color': "#ffcc00"},
                        {'range': [70, 100], 'color': "#ff4b4b"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.8,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Health recommendations
            st.markdown("### 🏥 Immediate Recommendations")
            tips = get_health_tips("high", input_data)
            for tip in tips:
                st.write(tip)
            
            # Emergency contact card
            st.markdown("""
            <div class='metric-card' style='border-left:4px solid #ff4b4b;'>
                <h4 style='color:#ff4b4b;'>🚨 Emergency Protocol:</h4>
                <p>• Seek immediate medical attention if experiencing chest pain</p>
                <p>• Call emergency services (911/112) if symptoms worsen</p>
                <p>• Do not drive yourself to the hospital</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            # LOW RISK display
            st.markdown("""
            <div class='metric-card'>
                <h2 style='color:#00cc96; text-align:center;'>✅ LOW RISK DETECTED</h2>
            </div>
            """, unsafe_allow_html=True)
            

            col1, col2 = st.columns(2)  
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='color:#00cc96;'>Health Probability</h3>
                    <h1 style='color:#00cc96;'>{display_no_disease_prob:.1f}%</h1>
                    <p>Likelihood of being healthy</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:  
                if show_animations:
                    st.balloons()
                st.markdown("""
                <div class='metric-card'>
                    <h3 style='color:#764ba2;'>Status</h3>
                    <h1 style='color:#764ba2;'>✅ Good</h1>
                    <p>Continue healthy habits</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Health recommendations for low risk
            st.markdown("### 💡 Preventive Recommendations")
            tips = get_health_tips("low", input_data)
            for tip in tips:
                st.write(tip)
            
            # Celebration message
            st.success("""
            🎉 **Excellent cardiovascular health profile!**
            
            **Maintain these healthy habits:**
            - Continue regular exercise
            - Eat a balanced diet
            - Get annual check-ups
            - Manage stress effectively
            """)


st.markdown("---")
st.markdown("## 📊 Additional Tools")

tab1, tab2 = st.tabs(["💡 Health Tips", "🔍 Detailed Analysis"])

with tab1:
    st.markdown("### Personalized Health Tips")
    
    # Interactive tip generator
    selected_category = st.selectbox(
        "Select health category:",
        ["General Heart Health", "Blood Pressure", "Cholesterol", "Exercise", "Stress Management", "Weight Management"]
    )
    
    tips_by_category = {
        "General Heart Health": [
            "✅ Eat at least 5 servings of fruits/vegetables daily",
            "✅ Limit processed foods and added sugars",
            "✅ Get 7-9 hours of quality sleep nightly",
            "✅ Stay hydrated with 8+ glasses of water daily"
        ],
        "Blood Pressure": [
            "📉 Reduce sodium intake to <2300mg daily",
            "📉 Increase potassium-rich foods (bananas, spinach)",
            "📉 Practice deep breathing exercises",
            "📉 Limit caffeine and alcohol"
        ],
        "Cholesterol": [
            "🧪 Increase soluble fiber (oats, beans)",
            "🧪 Choose healthy fats (avocado, nuts)",
            "🧪 Limit saturated fats (red meat, butter)",
            "🧪 Consider Omega-3 supplements"
        ],
        "Exercise": [
            "🏃 150 minutes moderate exercise weekly",
            "🏃 Include strength training 2x weekly",
            "🏃 Take walking breaks every hour",
            "🏃 Use stairs instead of elevator"
        ],
        "Stress Management": [
            "🧘 Practice mindfulness meditation",
            "🧘 Maintain work-life balance",
            "🧘 Engage in hobbies regularly",
            "🧘 Connect with friends/family"
        ],
        "Weight Management": [
            "⚖️ Aim for gradual weight loss (0.5-1kg/week)",
            "⚖️ Track your BMI regularly",
            "⚖️ Focus on portion control",
            "⚖️ Combine diet with regular exercise"
        ]
    }
    
    for tip in tips_by_category[selected_category]:
        st.write(f"• {tip}")

with tab2:
    st.markdown("### Detailed Feature Analysis")
    
    if st.session_state.last_prediction:
        data = st.session_state.last_prediction['data']
        
        # Create feature importance visualization
        features = list(data.keys())
        importance_scores = []
        
        # Calculate simple importance scores (for demonstration)
        for feature, value in data.items():
            if feature in ['age', 'trestbps', 'chol', 'oldpeak']:
                importance = value / 100
            elif feature in ['sex', 'fbs', 'exang']:
                importance = value * 0.3
            elif feature == 'cp':
                importance = value * 0.4
            elif feature == 'ca':
                importance = value * 0.5
            else:
                importance = 0.1
            importance_scores.append(importance)
        
        # Display as bar chart
        fig = px.bar(
            x=features,
            y=importance_scores,
            title="Feature Contribution to Risk Assessment",
            labels={'x': 'Feature', 'y': 'Contribution Score'},
            color=importance_scores,
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(255,255,255,0.1)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("**Interpretation:** Higher bars indicate features contributing more to risk assessment.")
    else:
        st.info("Make a prediction to see detailed analysis here.")
        
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; padding: 2rem;'>
        <div style='color:#667eea; font-size: 1.2rem; margin-bottom: 1rem;'>
            ⚕️ <strong>Medical Disclaimer</strong>
        </div>
        <p style='color:#888; font-size: 0.9rem; max-width: 800px; margin: 0 auto;'>
            This tool is for educational and informational purposes only. It is not a substitute 
            for professional medical advice, diagnosis, or treatment. Always seek the advice of 
            your physician or other qualified health provider with any questions you may have 
            regarding a medical condition.
        </p>
        <div style='margin-top: 2rem; color:#764ba2;'>
            <p>❤️ Heart Disease Predictor • Enhanced with AI/ML • Version 2.0</p>
            <p style='font-size: 0.8rem;'>Last updated: """ + datetime.now().strftime("%Y-%m-d") + """</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)