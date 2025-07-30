# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 21:00:24 2025

@author: SSD
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Load the saved model and the feature names
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
loaded_model = model_data["model"]
feature_names = model_data["features_names"]

# Function for Prediction
def customer_churn_prediction(input_data):
    # Create DataFrame with correct feature names
    input_data_df = pd.DataFrame([input_data], columns=feature_names)

    # Load saved encoders
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    # Encode categorical features
    for column, encoder in encoders.items():
        if column in input_data_df.columns:
            input_data_df[column] = encoder.transform(input_data_df[column])
        else:
            print(f"Warning: '{column}' not found in input data")

    # Make prediction
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)  # fix: it’s predict_proba, not predict_prob

    # Return results
    return {
        "Prediction": "Churn" if prediction[0] == 1 else "No Churn",
        "Model Prediction Probability": float(pred_prob[0][1])
    }

# Streamlit UI
def main():
    st.title('Customer Churn Prediction Web App By Ayman')

    # Get user inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents (children or elderly)", ["Yes", "No"])
    tenure = st.number_input('Enter your Tenure (in months)', min_value=0)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
    InternetService = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input('Enter Monthly Charges', min_value=0.0)
    TotalCharges = st.number_input('Enter Total Charges', min_value=0.0)

    # Button and prediction
    if st.button('Churn Test Result'):
        input_data = [
            gender, SeniorCitizen, Partner, Dependents, tenure,
            PhoneService, MultipleLines, InternetService, OnlineSecurity,
            OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
            StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
            MonthlyCharges, TotalCharges
        ]

        result = customer_churn_prediction(input_data)
        st.success(f"Prediction: {result['Prediction']}")
        st.info(f"Churn Probability: {round(result['Model Prediction Probability']*100, 2)}%")

if __name__ == '__main__':
    main()