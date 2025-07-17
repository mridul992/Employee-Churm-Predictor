
# 🔍 Project Risk Prediction & Actionable Agent

This project uses Machine Learning to predict employee attrition (churn risk) and recommends actions for HR/Project Managers using an AI-powered agent logic.

---

## 📊 Problem Statement

Organizations often face delays or disruptions when employees are likely to leave. This project predicts **who is at risk of leaving** and provides **actionable recommendations** to retain them.

---

## 🧠 ML Models Used

Trained and evaluated the following algorithms:

- Decision Tree Regressor 🌳  
- Linear Regression 📈  
- Random Forest Regressor 🌲  
- Support Vector Machine (SVR) 💡  
- XGBoost Regressor ⚡  

Best model was selected based on R² score and used for final deployment.

---

## 📁 Dataset

**Source:** IBM HR Analytics Dataset  
**Filename:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`  
**Target Column:** `Attrition` (Yes/No → 1/0)  
**Key Features Used:**
- Age, Monthly Income, Distance From Home
- Job Satisfaction, Department, Job Role
- OverTime, Marital Status, etc.

---

## ⚙️ Project Structure

```
project-risk-agent/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── models/
│   └── project_risk_model.pkl
│
├── notebooks/
│   └── project_risk_prediction.ipynb
│
├── streamlit_app/
│   └── project_risk_streamlit_app.py
│
├── README.md
```

---

## 🔄 Agent Logic (AI Decision Loop)

> Predict → Interpret → Recommend → Notify

### ✔️ Example Flow:
1. **Predicts risk score** from employee profile  
2. **Interprets risk level** (low, medium, high)  
3. **Recommends actions** like:
   - Assign mentor
   - Reduce workload
   - Schedule 1-on-1
4. **(Optional)**: Can send Slack/Email notification to HR

---

## 🖥️ Run the App (Streamlit)

Install dependencies:
```bash
pip install -r requirements.txt
```

Run app:
```bash
streamlit run streamlit_app/project_risk_streamlit_app.py
```

---

## 📦 Model Export

Final selected model is saved as:
```bash
models/project_risk_model.pkl
```
You can reuse it in the Streamlit app or any automation pipeline.

---

## ✅ Predictions Example

| Test | Risk Score | Recommendation                          |
|------|------------|------------------------------------------|
| 1    | 0.78       | Assign a mentor and reduce workload      |
| 2    | 0.65       | Schedule 1-on-1                          |
| 3    | 0.33       | No immediate action needed               |
| 4    | 0.89       | High risk - escalate to management       |

---

## ✨ Future Improvements

- Integrate with Slack/Email notification  
- Add SHAP interpretability  
- Deploy on cloud (e.g., Heroku, Streamlit Cloud)  
- Extend to project delay prediction (team-level)

---

## 🙌 Made  by Mridul Chopra
