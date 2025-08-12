import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

# --- Load the saved model, encoder, and scaler ---
try:
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please run the training script first.")
    st.stop()

# --- Helper function to display AQI info ---


def get_aqi_info(aqi):
    if aqi <= 50:
        return "Good", "Minimal impact", "#2ECC71"  # Green
    elif aqi <= 100:
        return "Satisfactory", "Minor breathing discomfort to sensitive people", "#F1C40F"  # Yellow
    elif aqi <= 200:
        return "Moderate", "Breathing discomfort to people with lung disease", "#E67E22"  # Orange
    elif aqi <= 300:
        return "Poor", "Breathing discomfort to most people on prolonged exposure", "#E74C3C"  # Red
    elif aqi <= 400:
        return "Very Poor", "Respiratory illness on prolonged exposure", "#9B59B6"  # Purple
    else:
        return "Severe", "Affects healthy people and seriously impacts those with existing diseases", "#78281F"  # Maroon


# --- Streamlit User Interface ---
st.set_page_config(page_title="India AQI Predictor", layout="wide")
st.title("🌬️ Air Quality Index (AQI) Predictor for Indian Cities")
st.write("Enter the current environmental conditions to predict the next day's AQI.")

# --- User Inputs in Sidebar ---
st.sidebar.header("Input Features")

# List of cities the model was trained on
city_list = ['Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal',
             'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam',
             'Gurugram', 'Guwahati', 'Hyderabad', 'Jaipur', 'Jorapokhar', 'Kochi',
             'Kolkata', 'Lucknow', 'Mumbai', 'Patna', 'Shillong', 'Talcher',
             'Thiruvananthapuram', 'Visakhapatnam']

city = st.sidebar.selectbox("Select City", sorted(city_list))

# Create two columns for inputs
col1, col2 = st.sidebar.columns(2)

# Pollutant inputs
pm25 = col1.number_input("PM2.5", min_value=0.0, value=55.5)
pm10 = col2.number_input("PM10", min_value=0.0, value=115.5)
no = col1.number_input("NO", min_value=0.0, value=15.5)
no2 = col2.number_input("NO2", min_value=0.0, value=35.5)
nox = col1.number_input("NOx", min_value=0.0, value=30.5)
nh3 = col2.number_input("NH3", min_value=0.0, value=25.5)
co = col1.number_input("CO", min_value=0.0, value=1.5)
so2 = col2.number_input("SO2", min_value=0.0, value=12.5)
o3 = col1.number_input("O3", min_value=0.0, value=30.5)

# Engineered feature inputs
st.sidebar.markdown("---")
st.sidebar.subheader("Historical & Time Features")
today = datetime.now()
month = today.month
year = today.year
aqi_lag_1 = st.sidebar.slider("Previous Day's AQI", 0, 500, 150)
aqi_rolling_7 = st.sidebar.slider("Last 7-Day Average AQI", 0, 500, 160)
pm25_rolling_7 = st.sidebar.slider(
    "Last 7-Day Average PM2.5", 0.0, 300.0, 60.0)


# --- Prediction Logic ---
if st.sidebar.button("Predict Next Day's AQI", use_container_width=True):

    # Create a DataFrame from user inputs
    # The order must match the training features
    input_df = pd.DataFrame({
        'City': [city], 'PM2.5': [pm25], 'PM10': [pm10], 'NO': [no], 'NO2': [no2],
        'NOx': [nox], 'NH3': [nh3], 'CO': [co], 'SO2': [so2], 'O3': [o3],
        'Month': [month], 'Year': [year], 'AQI_lag_1': [aqi_lag_1],
        'AQI_rolling_7': [aqi_rolling_7], 'PM2.5_rolling_7': [pm25_rolling_7]
    })

    st.subheader("User Input Summary")
    st.write(input_df)

    # --- Preprocessing ---
    # 1. Encode the City
    input_df['City'] = encoder.transform(input_df[['City']])

    # 2. Scale all features
    input_scaled = scaler.transform(input_df)

    # --- Prediction ---
    prediction = model.predict(input_scaled)
    predicted_aqi = int(prediction[0])

    # --- Display Results ---
    st.subheader("Prediction Result")
    category, health_impact, color = get_aqi_info(predicted_aqi)

    st.metric(label="Predicted AQI for Tomorrow", value=predicted_aqi)

    st.markdown(f"""
    <div style="padding: 15px; border-radius: 10px; background-color: {color}; color: white;">
        <h4 style="color: white;">Category: {category}</h4>
        <p><b>Health Impact:</b> {health_impact}</p>
    </div>
    """, unsafe_allow_html=True)
