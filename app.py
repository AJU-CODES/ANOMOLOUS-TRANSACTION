import streamlit as st
import pandas as pd
import numpy as np 
import joblib
import os

st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="wide")

@st.cache_resource
def load_model(model_dir='model_files'):
    model_path = os.path.join(model_dir, 'best_model.pkl')
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        st.error("Model or preprocessor file not found. Train and save the model first.")
        return None, None
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def predict_fraud(features, model, preprocessor):
    try:
        transformed = preprocessor.transform(features)
        prob = model.predict_proba(transformed)[0][1] if hasattr(model, 'predict_proba') else model.predict(transformed)[0]
        return prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    st.title("💳 Credit Card Fraud Detection")
    st.write("Enter transaction details to predict fraud probability.")

    model, preprocessor = load_model()
    if model is None or preprocessor is None:
        return

    with st.form("prediction_form"):
        # Numeric Inputs
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
        latitude = st.number_input("Customer Latitude", min_value=-90.0, max_value=90.0, value=40.0)
        longitude = st.number_input("Customer Longitude", min_value=-180.0, max_value=180.0, value=-74.0)
        city_pop = st.number_input("City Population", min_value=0, value=100000)
        unix_time = st.number_input("Unix Timestamp", min_value=0, value=1577836800)
        merch_lat = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, value=40.0)
        merch_long = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, value=-74.0)

        # Category
        category = st.selectbox("Merchant Category", [
            'gas_transport', 'grocery_pos', 'home', 'shopping_pos', 'kids_pets',
            'shopping_net', 'entertainment', 'personal_care', 'food_dining',
            'health_fitness', 'misc_pos', 'misc_net', 'grocery_net', 'travel'
        ])

        # Gender
        gender = st.selectbox("Gender", ['M', 'F'])

        # Merchant
        top_merchants = [
            'fraud_Kilback LLC', 'fraud_Cormier LLC', 'fraud_Schumm PLC',
            'fraud_Kuhn LLC', 'fraud_Dickinson Ltd', 'fraud_Boyer PLC',
            'fraud_Emard Inc', 'fraud_Parisian and Sons'
        ]
        merchant_input = st.selectbox("Merchant Name", top_merchants + ['Other'])
        merchant = merchant_input if merchant_input in top_merchants else 'Other'

        # Job
        top_jobs = [
            'Film/video editor', 'Exhibition designer', 'Surveyor, land/geomatics',
            'Naval architect', 'Designer, ceramics/pottery', 'Materials engineer',
            'Environmental consultant', 'Financial adviser', 'IT trainer', 'Systems developer'
        ]
        job_input = st.selectbox("Occupation", top_jobs + ['Other'])
        job = job_input if job_input in top_jobs else 'Other'

        # States
        states = [
            'TX','NY','PA','CA','OH','MI','IL','FL','AL','MO','MN','AR','NC','SC','VA','KY','WI','IN','IA',
            'OK','GA','MD','WV','NJ','NE','KS','LA','MS','WY','WA','OR','TN','NM','ME','ND','CO','SD','MA',
            'MT','VT','UT','AZ','NH','CT','ID','NV','DC','HI','AK','RI'
        ]
        state_input = st.selectbox("Customer State", states + ['Other'])
        state = state_input if state_input in states else 'Other'

        submitted = st.form_submit_button("Predict")

        if submitted:
            if amount > 35000:
                st.warning("⚠️ High transaction amount! $35,000+ is considered risky.")

            input_df = pd.DataFrame({
                'amt': [amount],
                'lat': [latitude],
                'long': [longitude],
                'city_pop': [city_pop],
                'unix_time': [unix_time],
                'merch_lat': [merch_lat],
                'merch_long': [merch_long],
                'category': [category],
                'gender': [gender],
                'state': [state],
                'job': [job],
                'merchant': [merchant]
            })

            fraud_prob = predict_fraud(input_df, model, preprocessor)
            if fraud_prob is not None:
                status = "🚨 FRAUDULENT" if fraud_prob > 0.5 else "✅ LEGITIMATE"
                risk = "High" if fraud_prob > 0.7 else "Medium" if fraud_prob > 0.3 else "Low"

                st.write("### 🧾 Prediction Result")
                st.metric("Fraud Probability", f"{fraud_prob:.2%}")
                st.metric("Transaction Status", status)
                st.metric("Risk Level", risk)

                st.write("### 🔍 Input Summary")
                st.dataframe(input_df)

if __name__ == "__main__":
    main()
