import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Model
def load_model():
    with open("gradient_boosting_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def main():
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(135deg, #1E1E2F, #2D3E50);
                color: white;
            }
            .stButton>button {
                background-color: #008CBA;
                color: white;
                font-size: 18px;
                border-radius: 10px;
                padding: 12px;
                font-weight: bold;
                width: 100%;
            }
            .stNumberInput>div>div>input {
                background-color: #222;
                color: white;
                border-radius: 5px;
                border: 1px solid #555;
                padding: 8px;
            }
            h1 {
                text-align: center;
                color: #4CAF50;
                font-size: 36px;
                font-weight: bold;
            }
            h3 {
                text-align: left;
                color: #DDD;
                font-size: 24px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1>📊 Churn Prediction using Gradient Boost</h1>", unsafe_allow_html=True)
    model = load_model()
    
    st.markdown("<h3>Enter Customer Data for Prediction</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", value=18, step=1, min_value=0)
        no_of_days_subscribed = st.number_input("No of Days Subscribed", value=0, step=1, min_value=0)
        weekly_mins_watched = st.number_input("Weekly Minutes Watched", value=0.0, step=0.1, min_value=0.0)
        minimum_daily_mins = st.number_input("Minimum Daily Minutes", value=0.0, step=0.1, min_value=0.0)
    
    with col2:
        maximum_daily_mins = st.number_input("Maximum Daily Minutes", value=0.0, step=0.1, min_value=0.0)
        weekly_max_night_mins = st.number_input("Weekly Max Night Minutes", value=0.0, step=0.1, min_value=0.0)
        videos_watched = st.number_input("Videos Watched", value=0, step=1, min_value=0)
        maximum_days_inactive = st.number_input("Maximum Days Inactive", value=0, step=1, min_value=0)
        customer_support_calls = st.number_input("Customer Support Calls", value=0, step=1, min_value=0)
    
    if st.button("🚀 Predict Churn"):
        input_data = np.array([[age, no_of_days_subscribed, weekly_mins_watched, minimum_daily_mins, maximum_daily_mins,
                                weekly_max_night_mins, videos_watched, maximum_days_inactive, customer_support_calls]])
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            st.error("⚠️ Customer is likely to churn.")
        else:
            st.success("✅ Customer is not likely to churn.")

if __name__ == "__main__":
    main()