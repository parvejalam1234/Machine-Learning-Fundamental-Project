import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('best_model.pkl', 'rb'))

st.title('Bank Customer Churn Prediction')

# Input fields for user input
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.number_input('Tenure (Years)', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance', min_value=0.0, value=10000.0)
num_products = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
has_crcard = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)

# Encode categorical variables as done during training
geography_map = {'France': 0, 'Spain': 1, 'Germany': 2}
gender_map = {'Male': 0, 'Female': 1}

geography_encoded = geography_map[geography]
gender_encoded = gender_map[gender]

# Preprocessing input
input_data = np.array([[credit_score, geography_encoded, gender_encoded, age, tenure, balance, num_products, has_crcard, is_active_member, estimated_salary]])

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction == 1:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')
