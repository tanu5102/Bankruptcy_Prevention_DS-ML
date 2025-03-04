# Bankruptcy_Prevention_DS-ML
## 📌 Project Overview
This project aims to predict the likelihood of a company going bankrupt using Machine Learning (ML) models. By analyzing key risk factors such as industrial risk, financial flexibility, and management risk, this system helps businesses, investors, and policymakers make informed decisions to mitigate financial failure.

## 📝 Dataset
The dataset includes various financial and operational risk factors that impact a company’s bankruptcy status.

### Feature	Description
industrial_risk	0 = Low, 0.5 = Medium, 1 = High
management_risk	0 = Low, 0.5 = Medium, 1 = High
financial_flexibility	0 = Low, 0.5 = Medium, 1 = High
credibility	0 = Low, 0.5 = Medium, 1 = High
competitiveness	0 = Low, 0.5 = Medium, 1 = High
operating_risk	0 = Low, 0.5 = Medium, 1 = High
class (Target)	0 = Bankrupt, 1 = Non-Bankrupt

## 📊 Exploratory Data Analysis (EDA)
Data Cleaning: Checked for missing values and removed irrelevant columns.
Outlier Detection: Used Isolation Forest to identify anomalies.
Feature Correlation: Analyzed financial risk factors affecting bankruptcy.
Data Transformation: Used Label Encoding for categorical variables.

## 🤖 Machine Learning Models Used
Model	Purpose
Logistic Regression	Baseline binary classification model
Support Vector Machine (SVM)	Handles complex decision boundaries
Decision Tree	Rule-based classification
Gradient Boosting	Sequential learning for better accuracy
Random Forest	Ensemble model for robust decision-making
Final model selected: Support Vector Machine (SVM) due to its better handling of high-dimensional data and complex decision boundaries.

## ⚙️ How to Run the Project
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/bankruptcy-prevention.git
cd bankruptcy-prevention
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Streamlit App
bash
Copy
Edit
streamlit run main.py
4️⃣ Model Prediction Example
python
Copy
Edit
import pickle
import numpy as np

### Load trained model
with open("svc.pkl", "rb") as model_file:
    model = pickle.load(model_file)

### Sample input: [industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]
sample_data = np.array([[0.5, 1.0, 0.0, 0.0, 0.0, 0.5]])

### Make prediction
prediction = model.predict(sample_data)
print("Bankruptcy Risk:", "High" if prediction[0] == 0 else "Low")
## 🌐 Deployment
This project is deployed using Streamlit, providing an interactive web-based interface for users to analyze and predict bankruptcy risk.

## 📌 Challenges Faced
✔ Overfitting Issues → Used train-test split & cross-validation.
✔ Choosing the Right Model → Compared Random Forest vs. SVM, selected the best performer.
✔ Handling Missing Data → Used imputation techniques for inconsistent values.
