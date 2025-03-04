import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_excel("bankruptcy-prevention.xlsx")

# Convert categorical columns to numeric
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":  
        df[col] = le.fit_transform(df[col])  # Convert text to numbers

# Split dataset
X = df.drop(columns=[" class"])
y = df[" class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train models
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train, y_train)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save SVM model
with open("svc.pkl", "wb") as model_file:
    pickle.dump(svc, model_file)

# Load trained model
with open("svc.pkl", "rb") as model_file:
    svc_model = pickle.load(model_file)

# Streamlit App
st.title("🔍 Bankruptcy Prediction System")

# Apply dynamic dark mode styling with custom slider color
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    div[data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
    }
    div[data-testid="stAppViewContainer"] {
        background-color: black !important;
    }
    div[data-testid="stSlider"] .st-dc {
        background: green !important;  /* Custom slider color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Navigation")

st.markdown(
    """
    <style>
    /* Make sidebar title bigger */
    div[data-testid="stSidebar"] h1 {
        font-size: 28px !important;
        font-weight: bold;
        color: #00FF00; /* Change to your preferred color */
        text-align: center;
    }

    /* Make sidebar radio options bigger */
    div[data-testid="stSidebar"] label {
        font-size: 20px !important;
        font-weight: bold;
        color: white;
    }

    /* Add hover effect for radio buttons */
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        color: #FFD700 !important; /* Gold color on hover */
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

option = st.sidebar.radio("Choose an option:", ["Overview", "Analysis", "Prediction"])

# Overview Section
if option == "Overview":
    st.header("📌 Project Overview")
    st.write("""
    This project predicts whether a company is likely to go bankrupt based on financial risk factors.
    It uses a **Support Vector Machine (SVM)** along with **Logistic Regression** and **Random Forest** models.
    
    **Features Used:**
    - Industrial Risk
    - Management Risk
    - Financial Flexibility
    - Credibility
    - Competitiveness
    - Operating Risk

    Navigate to **Prediction** to test the bankruptcy risk prediction.
    """)

# Analysis Section
elif option == "Analysis":
    st.header("📊 Exploratory Data Analysis (EDA)")

    # Sample Data
    st.subheader("🔹 Sample Data")
    st.write(df.head())

    # Statistical Overview
    st.subheader("🔹 Statistical Overview")
    st.write(df.describe())

    # Bankruptcy Distribution (Plotly)
    st.subheader("🔹 Bankruptcy Distribution")
    fig1 = px.histogram(df, x=" class", title="Bankruptcy Distribution", labels={" class": "Company Status"},
                        color=" class", color_discrete_map={0: "red", 1: "green"})
    st.plotly_chart(fig1)

    # Compare Risk Factors (Plotly)
    st.subheader("🔹 Risk Factor Comparisons")
    selected_feature = st.selectbox("Choose a feature to compare:", X.columns)
    fig2 = px.box(df, x=" class", y=selected_feature, color=" class", 
                  title=f"Comparison of {selected_feature} Across Bankruptcy Classes",
                  labels={" class": "Company Status"})
    st.plotly_chart(fig2)

    # Correlation Matrix (Plotly)
    st.subheader("🔹 Correlation Heatmap")
    corr_matrix = df.corr()
    fig3 = px.imshow(corr_matrix, text_auto=True, title="Feature Correlation Matrix", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig3)

    # Machine Learning Model Performance
    st.header("📊 Model Performance Evaluation")

    # Predictions
    svc_pred = svc_model.predict(X_test)
    log_reg_pred = log_reg.predict(X_test)
    rf_pred = rf.predict(X_test)

    # Accuracy Comparison
    svc_acc = accuracy_score(y_test, svc_pred)
    log_reg_acc = accuracy_score(y_test, log_reg_pred)
    rf_acc = accuracy_score(y_test, rf_pred)

    st.subheader("🔹 Accuracy Comparison")
    st.write(f"✅ **SVM Accuracy:** {svc_acc:.4f}")
    st.write(f"✅ **Logistic Regression Accuracy:** {log_reg_acc:.4f}")
    st.write(f"✅ **Random Forest Accuracy:** {rf_acc:.4f}")

    # Classification Report for SVM
    st.subheader("🔹 SVM Model Classification Report")
    st.text(classification_report(y_test, svc_pred))

# Prediction Section
elif option == "Prediction":
    st.header("🎯 Make a Prediction")

    with st.container():
        st.subheader("🔹 Input Features")

        with st.form("user_inputs"):
            industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.5)
            management_risk = st.slider("Management Risk", 0.0, 1.0, 1.0)
            financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.0)
            credibility = st.slider("Credibility", 0.0, 1.0, 0.0)
            competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.0)
            operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.5)

            submit_button = st.form_submit_button("🔍 Predict")

    # Convert inputs to array
    features = np.array([industrial_risk, management_risk, financial_flexibility, credibility, competitiveness,
                         operating_risk]).reshape(1, -1)

    # Run prediction when button is clicked
    if submit_button:
        prediction = svc_model.predict(features)
        result = "💥 **Bankrupt**" if prediction[0] == 0 else "✅ **Non-Bankrupt**"
        st.success(f"Prediction: {result}")
