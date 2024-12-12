import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as plt
import plotly.express as px
import io
import numpy as np
import time

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# Add tabs for single and batch prediction
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# Set the base URL for the backend API
API_BASE_URL = "https://proact-v1-backend.onrender.com"  # Replace with your backend URL

def make_prediction(input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=input_data, timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < max_retries - 1:
                st.warning(f"Connection attempt {attempt + 1} failed. Retrying...")
                time.sleep(2)
            else:
                st.error(f"Failed to connect to prediction server: {str(e)}")
                raise

def make_batch_prediction(file, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Ensure the file is at the beginning
            file.seek(0)
            
            # Prepare files for upload
            files = {'file': ('data.csv', file, 'text/csv')}
            
            # Make the prediction request
            response = requests.post(f"{API_BASE_URL}/batch-predict", files=files, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < max_retries - 1:
                st.warning(f"Batch prediction attempt {attempt + 1} failed. Retrying...")
                time.sleep(2)
            else:
                st.error(f"Failed to connect to prediction server: {str(e)}")
                raise

def generate_feature_importance(results_df):
    """
    Generate feature importance and correlation insights
    """
    # Select numerical features
    features = ['air_temperature', 'process_temperature', 
                'rotational_speed', 'torque', 'tool_wear']
    
    # Correlation with breakdown probability
    correlations = results_df[features + ['breakdown_probability']].corr()['breakdown_probability'][:-1]
    
    # Visualize feature importance
    fig_importance = px.bar(
        x=correlations.index, 
        y=abs(correlations.values),
        title='Feature Importance for Breakdown Probability',
        labels={'x': 'Features', 'y': 'Absolute Correlation Strength'}
    )
    return fig_importance

def batch_prediction_page():
    st.header("Batch Prediction")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV for Batch Prediction", 
        type=['csv'], 
        help="Upload a CSV file with machine data for batch prediction"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Display uploaded data preview
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head())
            
            # Prediction button
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing batch prediction..."):
                    # Send request to batch prediction endpoint
                    response = make_batch_prediction(uploaded_file)
                    
                    # Check response
                    if response:
                        prediction_results = response
                        results_df = pd.DataFrame(prediction_results['results'])
                        summary = prediction_results['summary']
                        
                        # Summary Metrics
                        st.subheader("Prediction Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", summary['total_samples'])
                        with col2:
                            st.metric("Breakdown Predictions", summary['breakdown_predictions'])
                        with col3:
                            st.metric("Avg Breakdown Probability", 
                                      f"{summary['avg_breakdown_probability']:.2f}")
                        
                        # Tabs for different views
                        breakdown_tab, feature_impact_tab = st.tabs([
                            "Breakdown Analysis",
                            "Feature Impact"
                        ])
                        
                        with breakdown_tab:
                            st.subheader("Breakdown Probability Distribution")
                            fig_breakdown = px.histogram(
                                results_df,
                                x='breakdown_probability',
                                title='Breakdown Probability Histogram',
                                nbins=20
                            )
                            st.plotly_chart(fig_breakdown)
                        
                        with feature_impact_tab:
                            st.subheader("Feature Importance Analysis")
                            fig_importance = generate_feature_importance(results_df)
                            st.plotly_chart(fig_importance)
                    else:
                        st.error("Failed to get prediction results.")
        
        except Exception as e:
            st.error(f"Error processing batch prediction: {str(e)}")

with tab1:
    st.title("🔧 Predictive Maintenance Dashboard")

    # Sidebar inputs
    st.sidebar.header("Machine Parameters")

    # Input parameters
    air_temp = st.sidebar.slider("Air Temperature [K]", 290.0, 320.0, 298.0, 0.1)
    process_temp = st.sidebar.slider("Process Temperature [K]", 300.0, 350.0, 308.0, 0.1)
    rot_speed = st.sidebar.slider("Rotational Speed [rpm]", 1000, 2500, 1420)
    torque = st.sidebar.slider("Torque [Nm]", 3.0, 90.0, 45.0, 0.1)
    tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 300, 120)
    machine_type = st.sidebar.selectbox("Machine Type", [0, 1, 2], format_func=lambda x: f"Type {x}")

    # Create input data
    input_data = {
        "air_temperature": air_temp,
        "process_temperature": process_temp,
        "rotational_speed": rot_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "type": machine_type
    }

    # Make prediction when user clicks the button
    if st.sidebar.button("Predict"):
        try:
            result = make_prediction(input_data)
            
            st.metric("Breakdown Probability", f"{result['breakdown_probability'] * 100:.2f}%")
            st.metric("Breakdown Prediction", "Yes" if result['breakdown_prediction'] == 1 else "No")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

with tab2:
    batch_prediction_page()

st.markdown("---")
st.markdown("### About")
st.write("""
This dashboard provides real-time predictive maintenance analysis for industrial equipment.
It uses machine learning to predict potential breakdowns and their causes based on operational parameters.
""")
