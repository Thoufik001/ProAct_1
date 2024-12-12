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

def make_prediction(input_data, max_retries=3):
    base_url = "http://localhost:8000"
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{base_url}/predict", json=input_data, timeout=10)
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
    base_url = "http://localhost:8000"
    for attempt in range(max_retries):
        try:
            # Ensure the file is at the beginning
            file.seek(0)
            
            # Prepare files for upload
            files = {'file': ('data.csv', file, 'text/csv')}
            
            # Make the prediction request
            response = requests.post(f"{base_url}/batch-predict", files=files, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < max_retries - 1:
                st.warning(f"Batch prediction attempt {attempt + 1} failed. Retrying...")
                time.sleep(2)
            else:
                st.error(f"Failed to connect to prediction server: {str(e)}")
                raise

API_BASE_URL = "http://localhost:8000"

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

def generate_breakdown_insights(results_df):
    """
    Generate insights and recommendations based on batch prediction results
    """
    # Breakdown probability distribution
    fig_breakdown_dist = px.line(
        results_df.sort_values('breakdown_probability'), 
        x=results_df.index, 
        y='breakdown_probability', 
        title='Breakdown Probability Distribution',
        labels={'x': 'Sample Index', 'breakdown_probability': 'Breakdown Probability'}
    )
    
    # Add horizontal line for high-risk threshold
    fig_breakdown_dist.add_hline(
        y=0.7, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="High Risk Threshold (0.7)"
    )
    
    return fig_breakdown_dist

def generate_maintenance_recommendations(results_df):
    """
    Generate maintenance recommendations based on prediction results
    """
    # High-risk samples analysis
    high_risk_samples = results_df[results_df['breakdown_probability'] > 0.7]
    
    # Insights generation
    insights = []
    
    # Temperature insights
    temp_diff = results_df['process_temperature'] - results_df['air_temperature']
    if temp_diff.mean() > 10:
        insights.append("🌡️ High temperature differential detected. Consider improving cooling systems.")
    
    # Rotational speed insights
    if results_df['rotational_speed'].mean() > 1800:
        insights.append("⚙️ High average rotational speed. Recommend regular lubrication and bearing checks.")
    
    # Tool wear insights
    if results_df['tool_wear'].mean() > 100:
        insights.append("🛠️ Significant tool wear observed. Consider more frequent tool replacements.")
    
    # Breakdown probability insights
    breakdown_rate = len(high_risk_samples) / len(results_df) * 100
    if breakdown_rate > 20:
        insights.append(f"⚠️ High breakdown risk: {breakdown_rate:.2f}% of machines show elevated failure probability.")
    
    return insights, high_risk_samples

def generate_feature_line_graphs(results_df):
    """
    Generate line graphs for key features and predictions
    """
    # Ensure UDI is available, if not create a sequential index
    if 'UDI' not in results_df.columns:
        results_df['UDI'] = range(len(results_df))
    
    # Prepare line graphs for different features
    feature_graphs = []
    
    # 1. Breakdown and Probability Line Graph
    fig_breakdown = px.line(
        results_df, 
        x='UDI', 
        y=['breakdown_prediction', 'breakdown_probability'],
        title='Breakdown Prediction and Probability by UDI',
        labels={'value': 'Value', 'variable': 'Metric'},
        color_discrete_map={
            'breakdown_prediction': 'red', 
            'breakdown_probability': 'blue'
        }
    )
    feature_graphs.append(fig_breakdown)
    
    # 2. Temperature Line Graphs
    fig_temp = px.line(
        results_df, 
        x='UDI', 
        y=['air_temperature', 'process_temperature'],
        title='Temperature Variations by UDI',
        labels={'value': 'Temperature (K)', 'variable': 'Temperature Type'},
        color_discrete_map={
            'air_temperature': 'green', 
            'process_temperature': 'orange'
        }
    )
    feature_graphs.append(fig_temp)
    
    # 3. Rotational Speed and Torque Line Graph
    fig_speed_torque = px.line(
        results_df, 
        x='UDI', 
        y=['rotational_speed', 'torque'],
        title='Rotational Speed and Torque by UDI',
        labels={'value': 'Value', 'variable': 'Metric'},
        color_discrete_map={
            'rotational_speed': 'purple', 
            'torque': 'brown'
        }
    )
    feature_graphs.append(fig_speed_torque)
    
    # 4. Tool Wear Line Graph
    fig_tool_wear = px.line(
        results_df, 
        x='UDI', 
        y='tool_wear',
        title='Tool Wear by UDI',
        labels={'tool_wear': 'Tool Wear (min)'},
        color_discrete_sequence=['magenta']
    )
    feature_graphs.append(fig_tool_wear)
    
    return feature_graphs

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
                    # Prepare file for upload
                    files = {'file': uploaded_file.getvalue()}
                    
                    # Send request to batch prediction endpoint
                    response = requests.post(
                        f"{API_BASE_URL}/batch-predict", 
                        files=files
                    )
                    
                    # Check response
                    if response.status_code == 200:
                        prediction_results = response.json()
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
                        breakdown_tab, line_graphs_tab, feature_impact_tab, insights_tab, details_tab = st.tabs([
                            "Breakdown Analysis",
                            "Feature Line Graphs", 
                            "Feature Impact",
                            "Insights & Recommendations", 
                            "Detailed Results"
                        ])
                        
                        with breakdown_tab:
                            # Breakdown and Probability Line Graph
                            st.subheader("Breakdown Prediction Analysis")
                            fig_breakdown = generate_feature_line_graphs(results_df)[0]  # First graph is breakdown
                            st.plotly_chart(fig_breakdown)
                        
                        with line_graphs_tab:
                            # Generate and display line graphs (skipping the first breakdown graph)
                            feature_line_graphs = generate_feature_line_graphs(results_df)[1:]
                            for fig in feature_line_graphs:
                                st.plotly_chart(fig)
                        
                        with feature_impact_tab:
                            # Feature Importance
                            st.subheader("Feature Impact Analysis")
                            fig_importance = generate_feature_importance(results_df)
                            st.plotly_chart(fig_importance)
                        
                        with insights_tab:
                            # Generate Maintenance Recommendations
                            recommendations, high_risk_samples = generate_maintenance_recommendations(results_df)
                            
                            # Display Insights
                            st.subheader("Key Insights")
                            for insight in recommendations:
                                st.info(insight)
                            
                            # High-Risk Samples
                            st.subheader("High-Risk Samples")
                            if not high_risk_samples.empty:
                                st.dataframe(high_risk_samples)
                            else:
                                st.success("No high-risk samples detected!")
                        
                        with details_tab:
                            # Detailed Results
                            st.subheader("Detailed Predictions")
                            st.dataframe(results_df)
                            
                            # Export results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Prediction Results", 
                                data=csv, 
                                file_name='batch_prediction_results.csv',
                                mime='text/csv'
                            )
                    
                    else:
                        st.error(f"Prediction Error: {response.text}")
        
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
            
            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            # Breakdown Probability Gauge
            with col1:
                fig = plt.Figure(data=[plt.Indicator(
                    mode="gauge+number",
                    value=result['breakdown_probability'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Breakdown Probability"},
                    gauge={'axis': {'range': [0, 100]},
                          'bar': {'color': "darkblue"},
                          'steps': [
                              {'range': [0, 40], 'color': "lightgreen"},
                              {'range': [40, 70], 'color': "yellow"},
                              {'range': [70, 100], 'color': "red"}
                          ]})])
                st.plotly_chart(fig)

            # Breakdown Prediction
            with col2:
                st.metric(
                    "Breakdown Prediction",
                    "Yes" if result['breakdown_prediction'] == 1 else "No",
                    delta="High Risk" if result['breakdown_prediction'] == 1 else "Low Risk",
                    delta_color="inverse"
                )

            # Confidence Level
            with col3:
                st.metric(
                    "Confidence Level",
                    result['confidence_level'],
                    delta=None
                )

            # Additional Information
            st.subheader("Detailed Analysis")
            
            # Create a DataFrame for parameter analysis
            params_df = pd.DataFrame({
                'Parameter': ['Air Temperature', 'Process Temperature', 'Rotational Speed', 
                             'Torque', 'Tool Wear'],
                'Value': [air_temp, process_temp, rot_speed, torque, tool_wear],
                'Status': ['Normal'] * 5  # Default status
            })

            # Update status based on thresholds
            if process_temp - air_temp > 25:
                params_df.loc[params_df['Parameter'].isin(['Air Temperature', 'Process Temperature']), 'Status'] = 'Warning'
            if rot_speed > 2000:
                params_df.loc[params_df['Parameter'] == 'Rotational Speed', 'Status'] = 'Warning'
            if torque > 70:
                params_df.loc[params_df['Parameter'] == 'Torque', 'Status'] = 'Warning'
            if tool_wear > 200:
                params_df.loc[params_df['Parameter'] == 'Tool Wear', 'Status'] = 'Warning'

            # Create parameter status chart
            fig = px.bar(params_df, x='Parameter', y='Value', color='Status',
                        color_discrete_map={'Normal': 'green', 'Warning': 'orange'},
                        title='Parameter Status Analysis')
            st.plotly_chart(fig)

            # Predicted Cause
            st.info(f"Predicted Major Cause: {result['major_cause_prediction']}")

            # Recommendations
            st.subheader("Recommendations")
            if result['breakdown_prediction'] == 1:
                st.error("⚠️ Immediate Action Required:")
                
                # Base recommendations based on parameter values
                recommendations = []
                if result['parameters']['temp_difference'] > 25:
                    recommendations.append("• Check cooling system efficiency")
                if input_data['rotational_speed'] > 2000:
                    recommendations.append("• Reduce operational speed")
                if input_data['torque'] > 70:
                    recommendations.append("• Inspect load conditions")
                if input_data['tool_wear'] > 200:
                    recommendations.append("• Schedule tool replacement")
                
                # Add cause-specific recommendations
                cause_specific_recs = {
                    1: [  # Power Failure
                        "• Inspect electrical connections and power supply",
                        "• Check for voltage fluctuations",
                        "• Verify backup power systems"
                    ],
                    2: [  # Tool Wear Failure
                        "• Replace cutting tools immediately",
                        "• Review tool material compatibility",
                        "• Adjust cutting parameters"
                    ],
                    3: [  # Overstrain Failure
                        "• Reduce operational load",
                        "• Check for mechanical misalignment",
                        "• Inspect bearing conditions"
                    ],
                    4: [  # Random Failures
                        "• Perform comprehensive system diagnostic",
                        "• Review maintenance history",
                        "• Check for environmental factors"
                    ],
                    5: [  # Other
                        "• Conduct detailed system inspection",
                        "• Review operational parameters",
                        "• Schedule maintenance check"
                    ]
                }
                
                # Add recommendations specific to the predicted cause
                if result['major_cause_code'] in cause_specific_recs:
                    st.write(f"Based on predicted cause ({result['major_cause_prediction']}):")
                    for rec in cause_specific_recs[result['major_cause_code']]:
                        recommendations.append(rec)
                
                # Display all recommendations
                for rec in recommendations:
                    st.write(rec)
                    
                # Display probability distribution of causes
                st.subheader("Failure Cause Analysis")
                cause_probs = pd.DataFrame({
                    'Cause': list(result['cause_probabilities'].keys()),
                    'Probability': list(result['cause_probabilities'].values())
                })
                cause_probs = cause_probs.sort_values('Probability', ascending=False)
                
                fig = px.bar(cause_probs, 
                            x='Cause', 
                            y='Probability',
                            title='Probability Distribution of Failure Causes',
                            color='Probability',
                            color_continuous_scale='Viridis')
                st.plotly_chart(fig)
                
            else:
                st.success("✅ System operating within normal parameters")
                st.write("• Continue regular maintenance schedule")
                if any(v > 0.2 for v in result['cause_probabilities'].values()):
                    st.warning("Note: Although no immediate breakdown is predicted, monitor the system closely.")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    else:
        st.info("👈 Adjust the parameters in the sidebar and click 'Predict' to get maintenance predictions.")

with tab2:
    batch_prediction_page()

# Footer
st.markdown("---")
st.markdown("### About")
st.write("""
This dashboard provides real-time predictive maintenance analysis for industrial equipment.
It uses machine learning to predict potential breakdowns and their causes based on operational parameters.
""")
