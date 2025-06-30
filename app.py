import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="wide")

@st.cache_resource
def load_model(model_dir='model_files'):
    model_path = os.path.join(model_dir, 'best_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model or scaler file not found. Train and save the model first.")
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_fraud(features, model, scaler):
    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[0][1]
    return probability

def main():
    st.title("💳 Credit Card Fraud Detection")
    st.write("Enter transaction details to predict fraud probability.")
    
    model, scaler = load_model()
    if model is None or scaler is None:
        return
    
    with st.form("prediction_form"):
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
        latitude = st.number_input("Customer Latitude", min_value=-90.0, max_value=90.0, value=40.0)
        longitude = st.number_input("Customer Longitude", min_value=-180.0, max_value=180.0, value=-74.0)
        city_pop = st.number_input("City Population", min_value=0, value=100000)
        unix_time = st.number_input("Unix Timestamp", min_value=0, value=1577836800)
        merch_lat = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, value=40.0)
        merch_long = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, value=-74.0)
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            if amount > 35000:
                st.warning("⚠️ High transaction amount! ₹35,000+ is considered risky.")

            features = pd.DataFrame({
                'amt': [amount], 'lat': [latitude], 'long': [longitude],
                'city_pop': [city_pop], 'unix_time': [unix_time],
                'merch_lat': [merch_lat], 'merch_long': [merch_long]
            })

            probability = predict_fraud(features, model, scaler)
            prediction = "🚨 FRAUDULENT" if probability > 0.5 else "✅ LEGITIMATE"
            risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"

            st.write("### Prediction Results")
            st.metric("Fraud Probability", f"{probability:.2%}")
            st.metric("Transaction Status", prediction)
            st.metric("Risk Level", risk_level)
            st.write("### Transaction Details")
            st.dataframe(features)

if __name__ == "__main__":
    main()
