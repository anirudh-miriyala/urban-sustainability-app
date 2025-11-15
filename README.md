# 🏙️ Urban Sustainability Intelligence Platform (USIP)

This project is a complete, end-to-end data science application that analyzes and predicts key urban sustainability indicators. It uses a custom-simulated dataset to train multiple machine learning models and presents the results in a 4-tab interactive web dashboard built with **Streamlit**.

This project was developed as a course project for "Data Science using Python" (22CSE554).

## 📸 Application Dashboard

The final application is a multi-page dashboard with four main tabs:

* **📈 Exploratory Data Analysis (EDA):** Visualizes the full dataset's time-series trends and a correlation heatmap to show relationships between metrics.
* **🤖 ML Prediction (Random Forest):** An interactive "what-if" tool. Use the sidebar sliders to set live conditions (like traffic, hour of day) and get a real-time AQI prediction from the Random Forest model.
* **🧠 DL Forecast (LSTM):** Uses the last 24 hours of data to feed into a trained LSTM model and forecast the **next hour's** AQI.
* **🚨 Anomaly Detection (Autoencoder):** Runs the entire dataset through an Autoencoder to find and display "pollution spikes" or other anomalous events that don't fit normal patterns.

---

## 🛠️ Tech Stack

This project was built entirely with Python and the following key libraries:

* **Data Science:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (for Random Forest)
* **Deep Learning:** TensorFlow (Keras) (for LSTM & Autoencoder)
* **Data Visualization:** Matplotlib, Seaborn
* **Web Dashboard:** Streamlit
* **Model Saving:** Joblib

---

## 📁 Project Structure
Urban_Sustainability_Platform/ │ ├── 🚀 app.py # The main Streamlit application script │ ├── 🔬 Project_Analysis.ipynb # Jupyter notebook for all data gen, analysis, and model training │ ├── 📦 models/ # Contains all saved .keras and .joblib model files │ ├── 📄 requirements.txt # All Python dependencies needed to run the app │ └── 📖 README.md # You are here!


---

## 🚀 How to Run Locally

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd Urban_Sustainability_Platform
    ```

2.  **Install dependencies:**
    Make sure you have all the required libraries by installing them from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    In your terminal, run the following command:
    ```bash
    streamlit run app.py
    ```
    *If the command above fails, try running it as a Python module:*
    ```bash
    python -m streamlit run app.py
    ```

---

## 🧠 Models & Analysis

All analysis and model training was performed in the `Project_Analysis.ipynb` notebook.

* **Data:** A custom dataset was simulated with realistic patterns (weekly cycles, daily rush hours) for **AQI**, **Traffic Load**, and **Energy Consumption**.
* **Models Trained:**
    * **Linear Regression:** Used as a baseline. (R²: 0.432)
    * **Random Forest (Regression):** The primary prediction model. **(Best Performance: R² = 0.789)**
    * **LSTM (Forecasting):** Used to predict the next hour's AQI. **(RMSE = 5.65)**
    * **Autoencoder (Anomaly Detection):** Used to identify outl
