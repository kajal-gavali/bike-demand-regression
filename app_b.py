# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:30:04 2024

@author: Kajal
"""

# Create a Streamlit app
#%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from category_encoders import OneHotEncoder

# Load the trained model
model = pickle.load(open('temp_model.pkl', 'rb'))
bike_rental_model = pickle.load(open('bike_rental_model.pkl', 'rb'))
# Define the app
def main():
    st.title("Bike Rental Prediction")

    # Input features
    season = st.selectbox('Season', ['Fall', 'Spring', 'Summer', 'Winter'])
    #year = st.number_input('Year', min_value=2011, max_value=2012, value=0)
    hour = st.selectbox('Hour', range(0, 24))
    month = st.selectbox('Month', range(1, 13))
    holiday = st.selectbox('Holiday', ['Yes', 'No'])
    weekday = st.selectbox('Weekday', range(7))
    workingday = st.selectbox('Working Day', ['No Work', 'Working Day'])
    weather = st.selectbox('Weather Situation', ['Clear', 'Heavy Rain/Snow', 'Light Snow/Rain', 'Mist + Cloudy'])
    temp = st.number_input('Temperature (Normalized)', min_value=0.0, max_value=1.0, value=0.5)
    #atemp = st.number_input('Feeling Temperature (Normalized)', min_value=0.0, max_value=1.0, value=0.5)
    hum = st.number_input('Humidity (Normalized)', min_value=0.0, max_value=1.0, value=0.5)
    windspeed = st.number_input('Windspeed (Normalized)', min_value=0.0, max_value=1.0, value=0.5)

    season_number = []
    holiday_number = []
    workingday_number = []
    weather_number = []
    
    if(season == 'Fall'):
        season_number = [1, 0, 0, 0]
    elif(season == 'Spring'):
        season_number = [0, 1, 0, 0]
    elif(season == 'Summer'):
        season_number = [0, 0, 1, 0]
    else:
        season_number = [0, 0, 0, 1]
        
    if(holiday == 'No'):
        holiday_number = [1, 0]
    else:
        holiday_number = [0, 1]
        
    if(workingday == 'No Work'):
        workingday_number = [1, 0]
    else:
        workingday_number = [0, 1]
        
    if(weather == 'Clear'):
        weather_number = [1, 0, 0, 0]
    elif(weather == 'Heavy Rain/Snow'):
        weather_number = [0, 1, 0, 0]
    elif(weather == 'Light Snow/Rain'):
        weather_number = [0, 0, 1, 0]
    else:
        weather_number = [0, 0, 0, 1]


    # Preprocess input data
    input_data = pd.DataFrame({
        #'season': [season],
        #'yr': [year],
        'month': [month],
        'hour': [hour],
        'weekday': [weekday],
        'temp': [temp],
        'humidity': [hum],
        'windspeed': [windspeed],
        'season_fall': [season_number[0]],
        'season_springer': [season_number[1]],
        'season_summer': [season_number[2]],
        'season_winter': [season_number[3]],
        'holiday_No': [holiday_number[0]],
        'holiday_Yes': [holiday_number[1]],
        'workingday_No work': [workingday_number[0]],
        'workingday_Working Day': [workingday_number[1]],
        'weather_Clear': [weather_number[0]],
        'weather_Heavy Rain': [weather_number[1]],
        'weather_Light Snow': [weather_number[2]],
        'weather_Mist': [weather_number[3]]
    })
    
    print(input_data)

    # One-hot encode categorical features
    #enc = OneHotEncoder(handle_unknown='ignore')
    #input_data = enc.fit_transform(input_data)
    print(input_data)

    # Make prediction
    if st.button('Predict'):
        prediction = model.predict(input_data)
        st.success(f'Predicted Bike Rentals: {int(prediction[0])}')
if __name__ == '__main__':
    main()
