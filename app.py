import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load trained models and encoders
xgb_model = joblib.load("xgb_model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Function to make a prediction
def predict_gesture(input_data):
    input_df = pd.DataFrame([input_data])  # Convert to DataFrame
    input_scaled = scaler.transform(input_df)  # Scale data
    prediction = xgb_model.predict(input_scaled)  # Predict
    
    try:
        predicted_label = label_encoder.inverse_transform([prediction[0]])[0]  # Convert to label
    except:
        predicted_label = "Bad"  # If an error occurs, return "Bad"
    
    # Replace "Null" or empty predictions with "Bad"
    if predicted_label == "Null" or not predicted_label:
        predicted_label = "Bad"

    return predicted_label

st.title("🖐 Hand Gesture Recognition")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.subheader("Enter Features for a Single Prediction")

    user_input = []
    for i in range(14): 
        user_input.append(st.number_input(f"Feature {i+1}", value=0.0))

    if st.button("Predict Gesture"):
        predicted_gesture = predict_gesture(user_input)
        st.success(f"🖖 Predicted Gesture: **{predicted_gesture}**")

with tab2:
    st.subheader("Upload CSV for Batch Prediction")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Drop unnecessary columns like "Unnamed: 0" if they exist
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        
        try:
            df_scaled = scaler.transform(df)
            predictions = xgb_model.predict(df_scaled)
            df["Predicted Gesture"] = label_encoder.inverse_transform(predictions)

            # Replace "Null" or empty values with "Bad"
            df["Predicted Gesture"] = df["Predicted Gesture"].replace("Null", "Bad").fillna("Bad")

            st.dataframe(df)

            output_file = "predictions.csv"
            df.to_csv(output_file, index=False)
            st.download_button("📥 Download Predictions", data=open(output_file, "rb"), file_name="predictions.csv")

        except Exception as e:
            st.error(f"Error: {e}")
