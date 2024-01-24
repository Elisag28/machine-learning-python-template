import streamlit as st
import pickle
import pandas as pd

# model
with open('../models/LinearRegression_.sav', 'rb') as file:
    modelo_regresion = pickle.load(file)

# scaler
with open('/workspace/machine-learning-streamlit/models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# mapping for region, sex and smoker
region_mapping = {
    0: 'southwest', 
    1: 'southeast', 
    2: 'northwest', 
    3: 'northeast'
    }

smoker_mapping = {
    0: 'yes', 
    1: 'no'
    }

sex_mapping = {
    0: 'female', 
    1: 'male'
    }

# user input
age = st.slider(
    'Plase, enter the age:',
    min_value=18,
    max_value=70,
    step = 1
    )

sex_factorized = st.selectbox(
    'Plase, enter the sex:',
    ['Female', 'Male'],
    format_func=lambda x: sex_mapping[x]
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

smoker_factorized = st.selectbox(
    'Are you a smoker?', 
    ['Yes', 'No'],
    format_func=lambda x: smoker_mapping[x]
    )

region_factorized = st.selectbox(
    'Region:',
    ['southwest', 'southeast', 'northwest', 'northeast'],
    )

# create a dataframe with the data entered by the user
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex_factorized],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker_factorized],
    'region': [region_factorized.lower()]
})

# conver categorical variables to dummy
input_data = pd.get_dummies(input_data, columns=['sex', 'smoker', 'region'], drop_first=True)

# normalice the data
input_data_scaled = scaler.transform(input_data)

# make prediction
prediction = modelo_regresion.predict(input_data_scaled)

# Mostrar resultados
st.write(f'The charge for your insurance is: : ${prediction[0]:,.2f}')