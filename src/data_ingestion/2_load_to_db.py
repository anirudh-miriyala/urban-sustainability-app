import pandas as pd
import sqlite3
import os

print("Starting Step 2: Load to Database...")

# --- Configuration ---
RAW_DATA_PATH = '../../data/raw/simulated_urban_data.csv'
PROCESSED_DATA_PATH = '../../data/processed/urban_data.db'
TABLE_NAME = 'urban_metrics'

# --- 1. Create the 'processed' directory if it doesn't exist ---
processed_dir = os.path.dirname(PROCESSED_DATA_PATH)
os.makedirs(processed_dir, exist_ok=True)

try:
    # --- 2. Load the CSV data using Pandas ---
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)

    # --- 3. Process the data ---
    # Convert the timestamp column from a string into a real datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print("Data processed. 'timestamp' column converted to datetime.")

    # --- 4. Connect to and write to the SQLite database ---
    print(f"Connecting to database at {PROCESSED_DATA_PATH}...")
    # This command creates the .db file if it's not there
    conn = sqlite3.connect(PROCESSED_DATA_PATH)
    
    # Write the DataFrame to a table in the database
    print(f"Writing data to table '{TABLE_NAME}'...")
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
    
    conn.close()

    print(f"\n✅ Success! Data loaded into '{TABLE_NAME}' in {PROCESSED_DATA_PATH}")

    # --- 5. Verify by reading it back ---
    print("\nVerifying data from database...")
    conn = sqlite3.connect(PROCESSED_DATA_PATH)
    test_df = pd.read_sql(f'SELECT * FROM {TABLE_NAME} LIMIT 5', conn)
    conn.close()
    print("Verification successful. First 5 rows from database:")
    print(test_df)

except FileNotFoundError:
    print(f"❌ ERROR: The file {RAW_DATA_PATH} was not found.")
    print("Please make sure you have run '1_simulate_data.py' successfully.")
except Exception as e:
    print(f"❌ An error occurred: {e}")