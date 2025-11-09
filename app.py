import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Urban Sustainability Intelligence",
    page_icon="🏙️",
    layout="wide",
)

# --- Caching Functions ---

@st.cache_data
def generate_data_in_memory():
    """
    Generates the simulated data directly in memory.
    This bypasses the SQLite/OneDrive file issue.
    """

    timestamps = pd.date_range(start='2024-01-01', end='2024-12-31 23:59:59', freq='h')
    n = len(timestamps)
    
    # Simulate Air Quality Index (AQI)
    base_aqi = 40
    aqi_weekly_cycle = 15 * (1 + np.sin(2 * np.pi * timestamps.dayofweek / 7))
    aqi_noise = np.random.normal(0, 5, n)
    aqi = base_aqi + aqi_weekly_cycle + aqi_noise
    
    # Simulate Traffic Load
    hour = timestamps.hour
    base_traffic = np.random.uniform(10, 20, n)
    morning_rush = 70 * np.exp(-((hour - 8.5)**2) / (2 * 1.5**2))
    evening_rush = 60 * np.exp(-((hour - 17.5)**2) / (2 * 2**2))
    traffic_load = base_traffic + morning_rush + evening_rush + np.random.normal(0, 3, n)
    
    # Simulate Energy Consumption
    base_energy = 500
    daytime_business_use = 150 * (1 + np.sin(2 * np.pi * (hour - 8) / 24))
    evening_residential_peak = 100 * np.exp(-((hour - 19)**2) / (2 * 2**2))
    energy_consumption = base_energy + daytime_business_use + evening_residential_peak + np.random.normal(0, 15, n)
    
    # Combine into a DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'aqi': aqi,
        'traffic_load': traffic_load,
        'energy_consumption': energy_consumption
    })
    
    # Clean up
    df['aqi'] = df['aqi'].clip(lower=0)
    df['traffic_load'] = df['traffic_load'].clip(lower=0)
    df['energy_consumption'] = df['energy_consumption'].clip(lower=0)
    df = df.round(2)
    
    # Process
    df.set_index('timestamp', inplace=True)
    st.success("✅ Data generated successfully!")
    return df

@st.cache_resource
def load_all_models(models_dir):
    """Loads all trained models from disk."""
    models = {}
    models['rf_model'] = joblib.load(os.path.join(models_dir, 'random_forest_regressor.joblib'))
    models['lstm_model'] = load_model(os.path.join(models_dir, 'lstm_forecaster.keras'))
    models['autoencoder'] = load_model(os.path.join(models_dir, 'autoencoder_anomaly.keras'))
    models['lstm_scaler'] = joblib.load(os.path.join(models_dir, 'lstm_main_scaler.joblib'))
    models['lstm_aqi_scaler'] = joblib.load(os.path.join(models_dir, 'lstm_aqi_scaler.joblib'))
    models['ae_scaler'] = joblib.load(os.path.join(models_dir, 'ae_scaler.joblib'))
    return models

# --- Load Data and Models ---
try:
    MODELS_DIR = 'models'
    df = generate_data_in_memory() 
    models = load_all_models(MODELS_DIR)
except Exception as e:
    st.error(f"Error loading model assets. Did you run the notebook to save the models? Error: {e}")
    st.stop()

# --- Helper Functions (from notebook) ---
def get_ml_features(data):
    """Creates time-based features."""
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek
    return data

df_ml = get_ml_features(df.copy())

# --- Main App Title ---
st.title("🏙️ Urban Sustainability Intelligence Platform")
st.markdown("Analyzing and predicting AQI, traffic, and energy in a simulated urban environment.")

# --- Tabbed Interface ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Exploratory Data Analysis (EDA)",
    "🤖 ML Prediction (Random Forest)",
    "🧠 DL Forecast (LSTM)",
    "🚨 Anomaly Detection (Autoencoder)"
])

# ==============================================================================
# TAB 1: EDA
# ==============================================================================
with tab1:
    st.header("Exploratory Data Analysis")
    st.write("A look at the raw data and its relationships.")
    
    # --- Time Series Plot ---
    st.subheader("Metrics Over Time (Full Dataset)")
    st.line_chart(df[['aqi', 'traffic_load', 'energy_consumption']])

    # --- Correlation Heatmap ---
    st.subheader("Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)
    
    st.write("**Insight:** The heatmap shows the relationships. For example, `traffic_load` and `energy_consumption` have a moderate positive correlation.")

# ==============================================================================
# TAB 2: ML PREDICTION (RANDOM FOREST)
# ==============================================================================
with tab2:
    st.header("Live AQI Prediction (Random Forest)")
    st.write("Predict the current AQI based on other live metrics. This model had an R-squared of **~0.79**.")
    
    st.sidebar.header("LIVE Prediction Inputs")
    
    # --- User Inputs ---
    hour_slider = st.sidebar.slider("Hour of Day", 0, 23, int(df_ml['hour'].mean()))
    day_slider = st.sidebar.slider("Day of Week", 0, 6, int(df_ml['dayofweek'].mean()))
    traffic_slider = st.sidebar.slider("Current Traffic Load", 0.0, 100.0, float(df_ml['traffic_load'].mean()))
    energy_slider = st.sidebar.slider("Current Energy Consumption", 400.0, 900.0, float(df_ml['energy_consumption'].mean()))
    
    # --- Make Prediction ---
    input_features = pd.DataFrame({
        'traffic_load': [traffic_slider],
        'energy_consumption': [energy_slider],
        'hour': [hour_slider],
        'dayofweek': [day_slider]
    })
    
    prediction = models['rf_model'].predict(input_features)
    
    st.metric(label="Predicted AQI", value=f"{prediction[0]:.2f}")
    st.write("*Adjust the sliders in the sidebar to see the prediction change in real-time.*")


# TAB 3: DL FORECAST (LSTM)

with tab3:
    st.header("Next-Hour AQI Forecast (LSTM)")
    st.write("Uses the last 24 hours of data to forecast the next hour's AQI. (RMSE: **~5.65**)")

    # --- Get Last 24 Hours ---
    last_24h_data = df_ml.iloc[-24:]
    
    st.subheader("Data used for forecast (Last 24 Hours):")
    st.dataframe(last_24h_data)
    
    # --- Format data for LSTM ---
    scaled_input = models['lstm_scaler'].transform(last_24h_data)
    # Add a "batch" dimension
    scaled_input = np.expand_dims(scaled_input, axis=0) 
    
    # --- Make Forecast ---
    scaled_forecast = models['lstm_model'].predict(scaled_input)
    final_forecast = models['lstm_aqi_scaler'].inverse_transform(scaled_forecast)
    
    st.metric(label="Forecasted AQI for Next Hour", value=f"{final_forecast[0][0]:.2f}")

# ==============================================================================
# TAB 4: ANOMALY DETECTION (AUTOENCODER)
# ==============================================================================
with tab4:
    st.header("Pollution Spike (Anomaly) Detection")
    st.write("This Autoencoder model was trained on 'normal' data. It flags any data points that it can't reconstruct well, indicating an anomaly.")
    
    # --- Calculate Reconstruction Error for all data ---
    all_data_scaled = models['ae_scaler'].transform(df_ml)
    reconstructions = models['autoencoder'].predict(all_data_scaled)
    mse = np.mean(np.power(all_data_scaled - reconstructions, 2), axis=1)
    df_ml['reconstruction_error'] = mse
    
    # --- Get Anomalies ---
    # We use the 99th percentile as the threshold
    error_threshold = np.percentile(mse, 99) 
    anomalies = df_ml[df_ml['reconstruction_error'] > error_threshold]
    
    st.metric("Anomaly Threshold (99th Percentile)", value=f"{error_threshold:.4f}")
    st.metric("Total Anomalies Found", value=len(anomalies))
    
    # --- Plot Anomalies ---
    st.subheader("AQI Over Time with Anomalies Highlighted")
    fig_anom, ax_anom = plt.subplots(figsize=(15, 6))
    ax_anom.plot(df_ml.index, df_ml['aqi'], label='AQI')
    ax_anom.scatter(anomalies.index, anomalies['aqi'], color='red', label='Anomaly Detected')
    ax_anom.set_title('AQI Over Time with Detected Anomalies')
    ax_anom.set_ylabel('AQI')
    ax_anom.legend()
    st.pyplot(fig_anom)
    
    st.subheader("Top 10 Anomalies Detected")
    st.dataframe(anomalies.sort_values('reconstruction_error', ascending=False).head(10))