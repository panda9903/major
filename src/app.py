import streamlit as st
import pandas as pd
import numpy as np
from utils.data_preprocessor import DataPreprocessor
from models.clustering import ClusteringModels
from visualization.visualizer import Visualizer
import plotly.graph_objects as go
import plotly.express as px
import os

@st.cache_resource
def get_data():
    """Load and preprocess data once, then cache it."""
    data_path = 'merged_bidmc_1s.csv'
    preprocessor = DataPreprocessor(data_path)
    data = preprocessor.load_data()
    X, feature_names = preprocessor.preprocess_signals()
    return data, X, feature_names, preprocessor

def create_input_form():
    """Create a form for inputting new patient data."""
    with st.expander("Input New Patient Data", expanded=True):
        col1, col2 = st.columns(2)
        
        # ECG Signals
        with col1:
            st.subheader("ECG Signals")
            ii = st.number_input("Lead II", value=0.45, format="%.3f", 
                               help="ECG Lead II signal value (typical range: 0.35-0.55)")
            v = st.number_input("Lead V", value=0.45, format="%.3f",
                              help="ECG Lead V signal value (typical range: 0.35-0.55)")
            avr = st.number_input("Lead AVR", value=0.45, format="%.3f",
                                help="ECG Lead AVR signal value (typical range: 0.35-0.55)")
        
        # PPG and Vital Signs
        with col2:
            st.subheader("PPG & Vital Signs")
            pleth = st.number_input("PLETH", value=1.8, format="%.3f",
                                  help="PPG/Plethysmogram signal value (typical range: 1.5-2.2)")
            hr = st.number_input("Heart Rate", value=85, format="%d",
                               help="Heart rate in beats per minute (typical range: 60-100)")
            pulse = st.number_input("Pulse", value=85, format="%d",
                                  help="Pulse rate (typical range: 60-100)")
            resp = st.number_input("Respiration Rate", value=20, format="%d",
                                 help="Respiration rate in breaths per minute (typical range: 12-20)")
            spo2 = st.number_input("SpO2", value=98, format="%d",
                                 help="Blood oxygen saturation percentage (typical range: 95-100)")
            
        return np.array([[ii, v, avr, pleth, hr, pulse, resp, spo2]])

def display_model_metrics(clustering_models):
    """Display clustering evaluation metrics."""
    st.subheader("Model Evaluation Metrics")
    
    metrics_df = pd.DataFrame()
    for model_name, metrics in clustering_models.metrics.items():
        metrics_df[model_name] = pd.Series(metrics)
    
    st.table(metrics_df.round(3))

def plot_cluster_distributions(clustering_models, X, feature_names):
    """Plot cluster distributions for each model."""
    st.subheader("Cluster Distributions")
    
    # Get predictions for all models at once
    predictions = clustering_models.predict(X)
    
    for model_name in ['kmeans', 'gmm', 'hierarchical', 'som']:
        st.write(f"**{model_name.upper()} Clustering**")
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': X[:, 0],
            'y': X[:, 1],
            'cluster': predictions[model_name].astype(str)
        })
        
        # Create scatter plot using first two features
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='cluster',
            labels={'x': feature_names[0], 'y': feature_names[1]},
            title=f"{model_name.upper()} Clustering Results"
        )
        st.plotly_chart(fig)

def main():
    st.title('Cardiovascular Risk Prediction System')
    st.write('Analyzing ECG and PPG signals for cardiovascular risk assessment')
    
    # Initialize session state
    if 'clustering' not in st.session_state:
        st.session_state.clustering = ClusteringModels()
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
        
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Load and preprocess data (cached)
    data, X, feature_names, preprocessor = get_data()
    
    # Training section
    st.header('Model Training')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Train Models'):
            with st.spinner('Training clustering models...'):
                try:
                    st.session_state.clustering.fit(X)
                    st.session_state.models_trained = True
                    st.success('Models trained successfully!')
                except Exception as e:
                    st.error(f'Error during training: {str(e)}')
    
    with col2:
        if st.button('Save Models'):
            if st.session_state.models_trained:
                try:
                    st.session_state.clustering.save_models('models')
                    st.success('Models saved successfully!')
                except Exception as e:
                    st.error(f'Error saving models: {str(e)}')
            else:
                st.error('Please train the models first.')
    
    # Display model metrics if models are trained
    if st.session_state.models_trained:
        display_model_metrics(st.session_state.clustering)
        plot_cluster_distributions(st.session_state.clustering, X, feature_names)
    
    # New Patient Classification Section
    st.header('New Patient Classification')
    if not st.session_state.models_trained:
        st.warning('Please train the models first before classifying new patients.')
    else:
        # Create input form
        new_data = create_input_form()
        
        if st.button('Classify Patient'):
            with st.spinner('Analyzing patient data...'):
                try:
                    # Preprocess the new data point
                    new_data_scaled = preprocessor.scaler.transform(new_data)
                    
                    # Get predictions from all models
                    predictions = st.session_state.clustering.predict(new_data_scaled)
                    
                    # Get final prediction using majority voting
                    final_prediction = st.session_state.clustering.majority_vote(predictions)
                    
                    # Display results
                    st.subheader('Classification Results')
                    
                    # Show overall risk assessment
                    risk_status = "High Risk" if final_prediction[0] == 1 else "Low Risk"
                    risk_color = "🔴" if final_prediction[0] == 1 else "🟢"
                    st.markdown(f"### Overall Risk Assessment: {risk_color} {risk_status}")
                    
                    # Show individual model predictions
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Individual Model Predictions:")
                        for model_name, preds in predictions.items():
                            model_prediction = "High Risk" if preds[0] == 1 else "Low Risk"
                            st.write(f"- {model_name.upper()}: {model_prediction}")
                    
                    # Show prediction confidence
                    with col2:
                        n_high_risk = sum(1 for preds in predictions.values() if preds[0] == 1)
                        confidence = max(n_high_risk, len(predictions) - n_high_risk) / len(predictions)
                        st.metric("Prediction Confidence", f"{confidence*100:.1f}%")
                    
                except Exception as e:
                    st.error(f'Error during classification: {str(e)}')
    
    # Signal Visualization section
    st.header('Signal Visualization')
    
    # Add tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["ECG Signals", "PPG Signals", "Vital Signs"])
    
    with tab1:
        ecg_fig = visualizer.plot_ecg_signals(data)
        st.plotly_chart(ecg_fig)
    
    with tab2:
        ppg_fig = visualizer.plot_ppg_signals(data)
        st.plotly_chart(ppg_fig)
    
    with tab3:
        vital_signs_fig = visualizer.plot_vital_signs(data)
        st.plotly_chart(vital_signs_fig)

if __name__ == '__main__':
    main() 