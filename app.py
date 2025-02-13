import streamlit as st
import pandas as pd
import numpy as np
import pickle

def load_model():
    with open("gradient_boosting_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def predict_churn(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title("Churn Prediction using Gradient Boost")
    model = load_model()
    
    st.write("### Enter Customer Data for Prediction")
    
    # Creating input fields for user input
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
    
    # Creating a prediction button
    if st.button("Predict Churn"):
        input_data = np.array([[age, no_of_days_subscribed, weekly_mins_watched, minimum_daily_mins, maximum_daily_mins,
                                weekly_max_night_mins, videos_watched, maximum_days_inactive, customer_support_calls]])
        prediction = predict_churn(model, input_data)
        
        if prediction[0] == 1:
            st.error("Customer is likely to churn.")
        else:
            st.success("Customer is not likely to churn.")

if __name__ == "__main__":
    main()
