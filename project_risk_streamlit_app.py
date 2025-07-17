import streamlit as st
import joblib
import pandas as pd

st.title("🔍 Employee Churn Risk Predictor")
st.markdown("Upload employee details to predict attrition risk and get a recommendation.")

# Load model
model = joblib.load("project_risk_model.pkl")

# Input fields
Age = st.slider("Age", 18, 60, 30)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=5000)
DistanceFromHome = st.slider("Distance From Home (km)", 1, 50, 10)
JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
OverTime = st.selectbox("OverTime", ["Yes", "No"])
JobRole = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

# Predict button
if st.button("Predict Risk"):
    input_df = pd.DataFrame([{
        "Age": Age,
        "MonthlyIncome": MonthlyIncome,
        "DistanceFromHome": DistanceFromHome,
        "JobSatisfaction": JobSatisfaction,
        "OverTime": OverTime,
        "JobRole": JobRole,
        "Department": Department,
        "MaritalStatus": MaritalStatus
    }])
    
    # Make prediction
    risk_score = model.predict(input_df)[0]
    st.write(f"🧠 **Predicted Churn Risk Score**: {risk_score:.2f}")
    
    # Recommendation logic
    if risk_score > 0.7:
        st.error("⚠️ High risk. Recommend assigning a mentor and reducing workload.")
    elif risk_score > 0.4:
        st.warning("🟠 Medium risk. Schedule 1-on-1 discussion.")
    else:
        st.success("🟢 Low risk. No immediate action needed.")
