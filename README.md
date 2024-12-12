# ProAct - Predictive Maintenence Project

## Overview
ProAct is a predictive maintenance application designed to reduce downtime and optimize equipment usage by forecasting potential failures. By leveraging machine learning and operational data, ProAct enables proactive decision-making, helping industries improve efficiency and reduce costs.

## Features
Data Processing: Efficiently handles large-scale datasets for predictive maintenance analysis.
Machine Learning: Trains predictive models to forecast equipment breakdowns and identify root causes.
Streamlit Web Interface: Provides an interactive, user-friendly dashboard for data visualization and prediction results.
Custom Retraining: Easily retrain models with updated datasets for better performance.

## Requirements
Ensure you have the following Python packages installed:
pandas
numpy
scikit-learn
imbalanced-learn
requests
streamlit
matplotlib
seaborn
python-multipart
All dependencies can be installed using the requirements.txt file provided.

## Installation
1. Clone the Repository:
git clone https://github.com/Thoufik001/ProAct_1.git
cd ProAct_1
2. Install the Required Dependencies:
pip install -r requirements.txt

## Running the Application
1. Run the Backend:
Start the main backend application for handling predictions and model training:
python main.py
2. Launch the Streamlit Web App:
Start the interactive Streamlit dashboard to access the predictive maintenance features:
streamlit run streamlit_app.py
3. Access the Web Interface:
Open your browser and visit: http://localhost:8501

## Usage
Batch Predictions: Upload a dataset to receive batch equipment failure predictions.
Visualization: View trends and patterns in operational data.
Retraining: Upload new data to retrain the predictive model for improved accuracy.

## Project Structure
ProAct_1/
├── data/                   # Sample datasets for testing
├── models/                 # Pre-trained models and encoders
├── main.py                 # Backend API for predictions
├── retrain.py              # Script to retrain models with new data
├── streamlit_app.py        # Streamlit web app interface
├── train_model.py          # Machine learning model training script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

## License
This software is the proprietary property of ERPROOTS Pvt Ltd. Unauthorized copying, modification, or distribution is strictly prohibited.

## For inquiries, please contact:
Thoufik Abdullah - https://www.linkedin.com/in/thoufik-abdullah-m/
