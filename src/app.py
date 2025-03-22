import streamlit as st
import pandas as pd
import numpy as np
from utils.data_preprocessor import DataPreprocessor
from models.clustering import ClusteringModels
from models.supervised import SupervisedModels
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

def display_model_metrics(model, model_type="clustering"):
    """Display model evaluation metrics."""
    st.subheader(f"{model_type.title()} Model Evaluation Metrics")
    
    if model_type == "clustering":
        metrics_df = pd.DataFrame()
        for model_name, metrics in model.metrics.items():
            metrics_df[model_name] = pd.Series(metrics)
        st.table(metrics_df.round(3))
    else:
        metrics_df = pd.DataFrame()
        for model_name, metrics in model.metrics.items():
            metrics_df[model_name] = pd.Series({
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1']
            })
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
    if 'supervised' not in st.session_state:
        st.session_state.supervised = SupervisedModels()
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
        
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Load and preprocess data (cached)
    data, X, feature_names, preprocessor = get_data()
    
    # Training section
    st.header('Model Training')
    
    if st.button('Train All Models'):
        try:
            # Step 1: Train clustering models
            with st.spinner('Training clustering models...'):
                st.session_state.clustering.fit(X)
                st.success('✅ Clustering models trained successfully!')
            
            # Step 2: Generate labels
            with st.spinner('Generating labels from clustering...'):
                predictions = st.session_state.clustering.predict(X)
                labels = st.session_state.clustering.majority_vote(predictions)
                st.success('✅ Labels generated successfully!')
            
            # Step 3: Train supervised models
            with st.spinner('Training supervised models...'):
                st.session_state.supervised.fit(X, labels)
                st.session_state.models_trained = True
                st.success('✅ Supervised models trained successfully!')
            
            st.success('🎉 All models trained successfully!')
            
        except Exception as e:
            st.error(f'Error during training: {str(e)}')

    # Move Patient Classification section here
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
                    
                    # Get predictions from supervised models
                    predictions = st.session_state.supervised.predict(new_data_scaled)
                    
                    # Display results
                    st.subheader('Classification Results')
                    
                    # Show predictions from each model
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Model Predictions:")
                        for model_name, preds in predictions.items():
                            risk_status = "High Risk" if preds[0] == 1 else "Low Risk"
                            st.write(f"- {model_name.upper()}: {risk_status}")
                    
                    # Show prediction confidence
                    with col2:
                        n_high_risk = sum(1 for preds in predictions.values() if preds[0] == 1)
                        confidence = max(n_high_risk, len(predictions) - n_high_risk) / len(predictions)
                        st.metric("Prediction Confidence", f"{confidence*100:.1f}%")
                    
                except Exception as e:
                    st.error(f'Error during classification: {str(e)}')

    # Display metrics if models are trained
    if st.session_state.models_trained:
        # Show supervised metrics first
        display_model_metrics(st.session_state.supervised, "supervised")
        
        # Then show clustering metrics and plots
        display_model_metrics(st.session_state.clustering, "clustering")
        plot_cluster_distributions(st.session_state.clustering, X, feature_names)
        
        # Add confusion matrices visualization
        st.header('Classification Confusion Matrices')
        st.write("Confusion matrices show the performance of each model in terms of true positives, false positives, true negatives, and false negatives.")
        
        # Get predictions for all models
        predictions = st.session_state.supervised.predict(st.session_state.supervised.X_test)
        
        # Create a grid of confusion matrices
        for model_name, preds in predictions.items():
            cm_fig = visualizer.plot_confusion_matrix(
                st.session_state.supervised.y_test,
                preds,
                model_name.upper()
            )
            st.plotly_chart(cm_fig)
    
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