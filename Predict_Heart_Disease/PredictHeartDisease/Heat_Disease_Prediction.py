import streamlit as st
import joblib
import numpy as np

model = joblib.load('heart_disease_rf_model.pkl')

st.title("Heart Disease Prediction")

age = st.number_input("Age", min_value=20, max_value=100)
sex = st.selectbox("Sex", ["Male", "Female"])
cigs_per_day = st.number_input("Cigarettes per Day", min_value=0, max_value=100)
tot_chol = st.number_input("Total Cholesterol", min_value=100, max_value=500)
sys_bp = st.number_input("Systolic BP", min_value=90, max_value=200)
dia_bp = st.number_input("Diastolic BP", min_value=60, max_value=150)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=150)
glucose = st.number_input("Glucose", min_value=50, max_value=300)
current_smoker = st.selectbox("Current Smoker", [0, 1])
bp_meds = st.selectbox("On BP Medication", [0, 1])
prevalent_stroke = st.selectbox("History of Stroke", [0, 1])
prevalent_hyp = st.selectbox("Hypertension", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])

user_data = np.array([[age, 1 if sex == "Male" else 0, cigs_per_day, tot_chol, sys_bp, dia_bp, bmi, heart_rate, glucose, current_smoker, bp_meds, prevalent_stroke, prevalent_hyp, diabetes]])

prediction = model.predict(user_data)
risk = "Yes" if prediction[0] == 1 else "No"

st.subheader("10-year CHD Risk")
st.write("Based on the input data, the risk of CHD in the next 10 years is:", risk)

