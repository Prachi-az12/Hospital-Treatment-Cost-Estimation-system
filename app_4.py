import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🏥 Hospital Treatment Cost Estimation")

# User inputs
age = st.number_input("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
children = st.number_input("Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Convert inputs to dataframe
input_data = pd.DataFrame({
    "age":[age],
    "sex":[sex],
    "bmi":[bmi],
    "children":[children],
    "smoker":[smoker],
    "region":[region]
})

# Encode categorical features same as training
input_data = pd.get_dummies(input_data, drop_first=True)

# Align columns with training data
missing_cols = set(model.feature_names_in_) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0
input_data = input_data[model.feature_names_in_]

# Prediction
if st.button("Predict Cost"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Treatment Cost: ₹ {prediction:,.2f}")
