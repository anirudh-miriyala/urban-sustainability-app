import pandas as pd
import numpy as np
import os # Import os to handle path creation

print("Starting Step 1: Generate Simulated Data...")

# --- Configuration ---
# This path goes up two levels (../..) from src/data_ingestion/
# and then down into data/raw/
OUTPUT_PATH = '../../data/raw/simulated_urban_data.csv'

# --- Create the directory if it doesn't exist ---
# This prevents errors if the folder isn't there
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 1. Create a year's worth of hourly timestamps
timestamps = pd.date_range(start='2024-01-01', end='2024-12-31 23:59:59', freq='h')
n = len(timestamps)

# 2. Simulate Air Quality Index (AQI)
base_aqi = 40
aqi_weekly_cycle = 15 * (1 + np.sin(2 * np.pi * timestamps.dayofweek / 7))
aqi_noise = np.random.normal(0, 5, n)
aqi = base_aqi + aqi_weekly_cycle + aqi_noise

# 3. Simulate Traffic Load
hour = timestamps.hour
base_traffic = np.random.uniform(10, 20, n)
morning_rush = 70 * np.exp(-((hour - 8.5)**2) / (2 * 1.5**2))
evening_rush = 60 * np.exp(-((hour - 17.5)**2) / (2 * 2**2))
traffic_load = base_traffic + morning_rush + evening_rush + np.random.normal(0, 3, n)

# 4. Simulate Energy Consumption
base_energy = 500
daytime_business_use = 150 * (1 + np.sin(2 * np.pi * (hour - 8) / 24))
evening_residential_peak = 100 * np.exp(-((hour - 19)**2) / (2 * 2**2))
energy_consumption = base_energy + daytime_business_use + evening_residential_peak + np.random.normal(0, 15, n)

# 5. Combine into a DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'aqi': aqi,
    'traffic_load': traffic_load,
    'energy_consumption': energy_consumption
})

# Clean up: ensure no negative values
df['aqi'] = df['aqi'].clip(lower=0)
df['traffic_load'] = df['traffic_load'].clip(lower=0)
df['energy_consumption'] = df['energy_consumption'].clip(lower=0)

# Round for cleaner data
df = df.round(2)

# Save to CSV at the correct path
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Successfully created '{OUTPUT_PATH}' with {len(df)} rows.")
print("\nHere's a sample of the data:")
print(df.head())