import streamlit as st
import pandas as pd
import numpy as np
from utils.data_preprocessor import DataPreprocessor
from models.clustering import ClusteringModels
from visualization.visualizer import Visualizer
import os

@st.cache_resource
def get_data():
    """Load and preprocess data once, then cache it."""
    data_path = 'merged_bidmc_1s.csv'
    preprocessor = DataPreprocessor(data_path)
    data = preprocessor.load_data()
    X, feature_names = preprocessor.preprocess_signals()
    X_train, X_test = preprocessor.split_data(X)
    return data, X_train, X_test, feature_names, preprocessor

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
    data, X_train, X_test, feature_names, preprocessor = get_data()
    
    # Training section
    st.header('Model Training')
    if st.button('Train Models'):
        with st.spinner('Training clustering models...'):
            try:
                # Train initial clustering with K-means
                kmeans_labels = st.session_state.clustering.train_kmeans(X_train)
                
                # Train other models using K-means labels
                st.session_state.clustering.train_knn(X_train, kmeans_labels)
                st.session_state.clustering.train_gmm(X_train)
                st.session_state.clustering.train_som(X_train, kmeans_labels)
                
                st.session_state.models_trained = True
                st.success('Models trained successfully!')
            except Exception as e:
                st.error(f'Error during training: {str(e)}')
    
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
                    # Preprocess the new data point using the same scaler
                    new_data_scaled = preprocessor.scaler.transform(new_data)
                    
                    # Get predictions from all models
                    predictions = st.session_state.clustering.predict_all(new_data_scaled)
                    
                    # Get final prediction using majority voting
                    final_prediction = st.session_state.clustering.majority_vote(predictions)
                    
                    # Display results
                    st.subheader('Classification Results')
                    
                    # Show overall risk assessment
                    risk_status = "High Risk" if final_prediction[0] == 1 else "Low Risk"
                    risk_color = "🔴" if final_prediction[0] == 1 else "🟢"
                    st.markdown(f"### Overall Risk Assessment: {risk_color} {risk_status}")
                    
                    # Show individual model predictions
                    st.write("Individual Model Predictions:")
                    for model_name, preds in predictions.items():
                        if preds is not None:
                            model_prediction = "High Risk" if preds[0] == 1 else "Low Risk"
                            st.write(f"- {model_name}: {model_prediction}")
                            
                    # Add visualization of the new point in feature space
                    if st.checkbox('Show Feature Space Visualization'):
                        fig = visualizer.plot_new_point_in_context(
                            X_train, 
                            st.session_state.clustering.kmeans.labels_,
                            new_data_scaled[0],
                            feature_names[:2]  # Using first two features
                        )
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f'Error during classification: {str(e)}')
    
    # Visualization section
    st.header('Signal Visualization')
    st.subheader('ECG and PPG Signals')
    signals_fig = visualizer.plot_signals(data)
    st.pyplot(signals_fig)
    
    st.subheader('Vital Signs')
    vital_signs_fig = visualizer.plot_vital_signs(data)
    st.plotly_chart(vital_signs_fig)
    
    # Risk Assessment section
    st.header('Risk Assessment')
    if st.button('Analyze Risk'):
        if not st.session_state.models_trained:
            st.error('Please train the models first by clicking the "Train Models" button above.')
        else:
            with st.spinner('Analyzing risk...'):
                try:
                    # Get predictions from all models
                    predictions = st.session_state.clustering.predict_all(X_test)
                    
                    # Perform majority voting
                    final_predictions = st.session_state.clustering.majority_vote(predictions)
                    
                    # Plot risk distribution
                    risk_fig = visualizer.plot_risk_distribution(final_predictions)
                    st.pyplot(risk_fig)
                    
                    # Display summary statistics
                    st.subheader('Risk Assessment Summary')
                    total_samples = len(final_predictions)
                    high_risk = np.sum(final_predictions)
                    low_risk = total_samples - high_risk
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric('High Risk Patients', f'{high_risk} ({high_risk/total_samples*100:.1f}%)')
                    with col2:
                        st.metric('Low Risk Patients', f'{low_risk} ({low_risk/total_samples*100:.1f}%)')
                except Exception as e:
                    st.error(f'Error during prediction: {str(e)}')
    
    # Model comparison section
    st.header('Model Comparison')
    show_predictions = st.checkbox('Show Model Predictions')
    if show_predictions:
        if not st.session_state.models_trained:
            st.error('Please train the models first by clicking the "Train Models" button above.')
        else:
            try:
                predictions = st.session_state.clustering.predict_all(X_test)
                for model_name, preds in predictions.items():
                    if preds is not None:
                        st.write(f'{model_name} predictions:')
                        risk_fig = visualizer.plot_risk_distribution(preds)
                        st.pyplot(risk_fig)
            except Exception as e:
                st.error(f'Error showing model predictions: {str(e)}')

if __name__ == '__main__':
    main() 