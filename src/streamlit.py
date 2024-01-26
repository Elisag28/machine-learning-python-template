import streamlit as st
import pickle
import pandas as pd

# model
with open('../models/LinearRegression_.sav', 'rb') as file:
    model = pickle.load(file)

# scaler
with open('../models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# mapping for region, sex and smoker
region_mapping = {
    'southwest' : 0, 
    'southeast': 1, 
    'northwest': 2, 
    'northeast': 3
    }

smoker_mapping = {
    'Yes': 0, 
    'No': 1
    }

sex_mapping = {
    'Female': 0, 
    'Male' : 1
    }

# user input
age = st.slider(
    'Plase, enter the age:',
    min_value=18,
    max_value=70,
    step = 1
    )

sex_n = st.selectbox(
    'Plase, enter the sex:',
    ['Female', 'Male']
    )

bmi = st.slider(
    'Body mass index (BMI):',
    min_value=15.0,
    max_value=53.0, 
    step = 0.1
    )

children = st.slider(
    'Number of children:', 
    min_value=0,
    max_value=7, 
    step = 1
    )

smoker_n = st.selectbox(
    'Are you a smoker?', 
    ['Yes', 'No'] 
    )

region_n = st.selectbox(
    'Region:',
    ['southwest', 'southeast', 'northwest', 'northeast'],
    )

if st.button("Predict"):
    row = [
        age,
        sex_mapping[sex_n],
        smoker_mapping[smoker_n],
        children,
        bmi,
        region_mapping[region_n]
        ]

    scal_data = scaler.transform([row])
    y_pred = model.predict([row])

    st.write(f'The charge for your insurance is: ${y_pred[0]:,.2f}')