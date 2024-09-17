import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('best_income_model.pkl')

# Streamlit app title
st.title('Household Income Prediction')

# Collect input features from the user
age = st.number_input('Age', min_value=18, max_value=70, value=30)
education = st.selectbox('Education Level', ['High School', "Bachelor's", "Master's", 'Doctorate'])
occupation = st.selectbox('Occupation', ['Healthcare', 'Education', 'Technology', 'Finance', 'Others'])
dependents = st.number_input('Number of Dependents', min_value=0, max_value=5, value=1)
location = st.selectbox('Location', ['Urban', 'Suburban', 'Rural'])
work_experience = st.number_input('Work Experience (years)', min_value=0, max_value=50, value=10)
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
employment_status = st.selectbox('Employment Status', ['Full-time', 'Part-time', 'Self-employed'])
household_size = st.number_input('Household Size', min_value=1, max_value=7, value=3)
homeownership_status = st.selectbox('Homeownership Status', ['Own', 'Rent'])
housing_type = st.selectbox('Type of Housing', ['Apartment', 'Single-family home', 'Townhouse'])
gender = st.selectbox('Gender', ['Male', 'Female'])
transportation = st.selectbox('Primary Mode of Transportation', ['Car', 'Public transit', 'Biking', 'Walking'])

# Collect all the inputs into a DataFrame (single row)
input_data = pd.DataFrame({
    'Age': [age],
    'Education_Level': [education],
    'Occupation': [occupation],
    'Number_of_Dependents': [dependents],
    'Location': [location],
    'Work_Experience': [work_experience],
    'Marital_Status': [marital_status],
    'Employment_Status': [employment_status],
    'Household_Size': [household_size],
    'Homeownership_Status': [homeownership_status],
    'Type_of_Housing': [housing_type],
    'Gender': [gender],
    'Primary_Mode_of_Transportation': [transportation]
})

# Directly make the prediction (assuming the model can handle the raw input data)
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'Predicted Annual Household Income: ${prediction[0]:,.2f}')
