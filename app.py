import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Student Dropout Risk Early-Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and modern typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .risk-card {
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .low-risk {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .mod-risk {
        background: linear-gradient(135deg, #f857a6 0%, #ff5858 100%);
    }
    .high-risk {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2a5298;
        margin-bottom: 15px;
    }
    .recommendation-box {
        background-color: #eef2f7;
        border-left: 6px solid #1e3c72;
        padding: 18px;
        border-radius: 8px;
        font-size: 15px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Load model and artifacts
@st.cache_resource
def load_prediction_artifacts():
    model_path = os.path.join("models", "logistic_regression_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    features_path = os.path.join("models", "feature_names.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        st.error("Model artifacts missing! Please run 04_model_training.py first.")
        st.stop()
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    
    return model, scaler, feature_names

model, scaler, feature_names = load_prediction_artifacts()

# Header Banner
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 28px;">🎓 Student Dropout Early-Warning & Risk Prediction System</h1>
    <p style="margin-top: 8px; font-size: 15px; opacity: 0.9;">
        Logistic Regression Predictive Intelligence for Early Retention & Academic Counseling
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Preset Selectors (Multiple Test Cases)
st.sidebar.header("🧪 Test Case Scenarios & Presets")
st.sidebar.write("Load a pre-configured student profile to test model predictions:")

preset_option = st.sidebar.radio(
    "Select Student Profile:",
    ["Custom Input", "Scenario A: High Risk Student", "Scenario B: Moderate / Borderline Student", "Scenario C: Low Risk / High Performing Student"]
)

# Preset feature values
if preset_option == "Scenario A: High Risk Student":
    default_tuition = 0 # Arrears
    default_scholarship = 0 # No
    default_debtor = 1 # Yes
    default_c1_app = 1
    default_c2_app = 0
    default_c1_grade = 9.5
    default_c2_grade = 0.0
    default_age = 26
    default_adm_grade = 110.0
elif preset_option == "Scenario B: Moderate / Borderline Student":
    default_tuition = 1 # Up to date
    default_scholarship = 0 # No
    default_debtor = 0 # No
    default_c1_app = 3
    default_c2_app = 2
    default_c1_grade = 11.5
    default_c2_grade = 10.5
    default_age = 21
    default_adm_grade = 130.0
elif preset_option == "Scenario C: Low Risk / High Performing Student":
    default_tuition = 1 # Up to date
    default_scholarship = 1 # Yes
    default_debtor = 0 # No
    default_c1_app = 6
    default_c2_app = 6
    default_c1_grade = 16.5
    default_c2_grade = 17.0
    default_age = 19
    default_adm_grade = 165.0
else:
    default_tuition = 1
    default_scholarship = 0
    default_debtor = 0
    default_c1_app = 5
    default_c2_app = 5
    default_c1_grade = 13.5
    default_c2_grade = 13.0
    default_age = 20
    default_adm_grade = 140.0

# Input Layout (Columns)
st.subheader("📋 Student Profile & Attributes")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📚 Academic Performance")
    c1_app = st.slider("1st Sem Approved Units", 0, 10, int(default_c1_app), help="Number of course units completed in semester 1")
    c2_app = st.slider("2nd Sem Approved Units", 0, 10, int(default_c2_app), help="Number of course units completed in semester 2")
    c1_grade = st.number_input("1st Sem Average Grade (0-20)", 0.0, 20.0, float(default_c1_grade), step=0.5)
    c2_grade = st.number_input("2nd Sem Average Grade (0-20)", 0.0, 20.0, float(default_c2_grade), step=0.5)
    adm_grade = st.number_input("Admission Grade (0-200)", 0.0, 200.0, float(default_adm_grade), step=1.0)

with col2:
    st.markdown("#### 💳 Financial & Socioeconomic")
    tuition_status = st.selectbox("Tuition Fees Payment Status", ["Up to date (1)", "In Arrears / Overdue (0)"], index=0 if default_tuition==1 else 1)
    scholarship = st.selectbox("Scholarship Holder", ["No (0)", "Yes (1)"], index=1 if default_scholarship==1 else 0)
    debtor = st.selectbox("Student Debtor Status", ["No (0)", "Yes (1)"], index=1 if default_debtor==1 else 0)
    mother_qual = st.slider("Mother's Qualification Code", 1, 30, 13)
    father_qual = st.slider("Father's Qualification Code", 1, 30, 14)

with col3:
    st.markdown("#### 👤 Demographic & Enrollment")
    age = st.slider("Age at Enrollment", 17, 70, int(default_age))
    gender = st.selectbox("Gender", ["Female (0)", "Male (1)"], index=1)
    displaced = st.selectbox("Displaced Student (Living Away)", ["No (0)", "Yes (1)"], index=1)
    attendance = st.selectbox("Attendance Mode", ["Daytime (1)", "Evening (0)"], index=0)
    application_order = st.slider("Application Order Choice", 1, 9, 1)

# Build feature dictionary
input_dict = {
    'Marital status': 1,
    'Application mode': 1,
    'Application order': application_order,
    'Course': 9000,
    'Daytime/evening attendance\t': 1 if 'Daytime' in attendance else 0,
    'Previous qualification': 1,
    'Previous qualification (grade)': adm_grade,
    'Nacionality': 1,
    'Mother\'s qualification': mother_qual,
    'Father\'s qualification': father_qual,
    'Mother\'s occupation': 5,
    'Father\'s occupation': 5,
    'Admission grade': adm_grade,
    'Displaced': 1 if 'Yes' in displaced else 0,
    'Educational special needs': 0,
    'Debtor': 1 if 'Yes' in debtor else 0,
    'Tuition fees up to date': 1 if 'Up to date' in tuition_status else 0,
    'Gender': 1 if 'Male' in gender else 0,
    'Scholarship holder': 1 if 'Yes' in scholarship else 0,
    'Age at enrollment': age,
    'International': 0,
    'Curricular units 1st sem (credited)': 0,
    'Curricular units 1st sem (enrolled)': max(c1_app, 5),
    'Curricular units 1st sem (evaluations)': max(c1_app + 2, 5),
    'Curricular units 1st sem (approved)': c1_app,
    'Curricular units 1st sem (grade)': c1_grade,
    'Curricular units 1st sem (without evaluations)': 0,
    'Curricular units 2nd sem (credited)': 0,
    'Curricular units 2nd sem (enrolled)': max(c2_app, 5),
    'Curricular units 2nd sem (evaluations)': max(c2_app + 2, 5),
    'Curricular units 2nd sem (approved)': c2_app,
    'Curricular units 2nd sem (grade)': c2_grade,
    'Curricular units 2nd sem (without evaluations)': 0,
    'Unemployment rate': 10.8,
    'Inflation rate': 1.4,
    'GDP': 1.74
}

# Align inputs with model features
input_df = pd.DataFrame([input_dict])
for f in feature_names:
    if f not in input_df.columns:
        input_df[f] = 0.0
input_df = input_df[feature_names]

# Scaled prediction
scaled_input = scaler.transform(input_df)
dropout_prob = float(model.predict_proba(scaled_input)[0][1])

st.markdown("---")
st.subheader("🎯 Model Prediction & Early-Warning Risk Assessment")

res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    pct_prob = dropout_prob * 100
    if dropout_prob > 0.65:
        risk_class = "high-risk"
        risk_label = "🚨 HIGH DROPOUT RISK"
        badge_color = "#ef473a"
    elif dropout_prob >= 0.30:
        risk_class = "mod-risk"
        risk_label = "⚠️ MODERATE DROPOUT RISK"
        badge_color = "#f857a6"
    else:
        risk_class = "low-risk"
        risk_label = "✅ LOW RISK / HIGH RETENTION"
        badge_color = "#38ef7d"
        
    st.markdown(f"""
    <div class="risk-card {risk_class}">
        <h2 style="margin:0; font-size:24px;">{risk_label}</h2>
        <h1 style="font-size:48px; margin:10px 0;">{pct_prob:.1f}%</h1>
        <p style="margin:0; font-size:14px;">Predicted Dropout Probability</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(min(max(dropout_prob, 0.0), 1.0))

with res_col2:
    st.markdown("#### 💡 Early-Warning Counseling Recommendation")
    if dropout_prob > 0.65:
        st.error("""
        **Urgent Intervention Priority:**
        - **Academic Support:** Student exhibits low course completion rates. Immediately assign a peer tutor for 1st/2nd sem core subjects.
        - **Financial Counseling:** Verify tuition fee payment status and explore emergency student grants or payment plans.
        - **Advising Schedule:** Mandatory weekly check-in with the academic retention office.
        """)
    elif dropout_prob >= 0.30:
        st.warning("""
        **Moderate Risk Monitoring:**
        - **Mid-term Review:** Schedule a mid-semester evaluation meeting to review 2nd semester approved unit trajectory.
        - **Study Workshop:** Recommend time management and study skill development workshops.
        - **Attendance Check:** Monitor weekly lecture and lab attendance logs.
        """)
    else:
        st.success("""
        **Standard Academic Progress:**
        - **Positive Indicators:** Strong academic course completion and up-to-date financial status.
        - **Action:** No emergency intervention needed. Continue standard periodic advisor reviews.
        """)

    # Top Influential Drivers for this specific student
    st.markdown("#### 🔍 Primary Predictive Risk Drivers")
    coefs = model.coef_[0]
    scaled_feats = scaled_input[0]
    contributions = coefs * scaled_feats
    
    contrib_df = pd.DataFrame({
        'Feature': feature_names,
        'Log-Odds Impact': contributions
    }).sort_values(by='Log-Odds Impact', ascending=False)
    
    top_pos = contrib_df[contrib_df['Log-Odds Impact'] > 0].head(3)
    top_neg = contrib_df[contrib_df['Log-Odds Impact'] < 0].tail(3)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.write("**Factors Increasing Risk:**")
        for idx, row in top_pos.iterrows():
            st.write(f"• `{row['Feature']}` (+{row['Log-Odds Impact']:.2f})")
    with col_d2:
        st.write("**Protective Factors:**")
        for idx, row in top_neg.iterrows():
            st.write(f"• `{row['Feature']}` ({row['Log-Odds Impact']:.2f})")

st.markdown("---")
st.caption("Student Dropout Early-Warning System | Powered by Logistic Regression & Streamlit")
