import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

class Visualizer:
    def __init__(self):
        """Initialize the visualizer with default style settings."""
        plt.style.use('default')  # Using default matplotlib style instead of seaborn
        self.colors = ['#2ecc71', '#e74c3c']  # Custom color palette for binary classification
    
    def plot_signals(self, data, time_column='Time [s]', signals=['II', 'V', 'AVR', 'PLETH']):
        """Plot ECG and PPG signals.
        
        Args:
            data (pd.DataFrame): DataFrame containing the signals
            time_column (str): Name of the time column
            signals (list): List of signal names to plot
        """
        fig, axes = plt.subplots(len(signals), 1, figsize=(15, 10), sharex=True)
        fig.suptitle('ECG and PPG Signals', fontsize=16)
        
        for ax, signal in zip(axes, signals):
            ax.plot(data[time_column], data[signal], label=signal)
            ax.set_ylabel(signal)
            ax.legend()
            
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        return fig
    
    def plot_clusters(self, X, labels, feature_names):
        """Plot clustering results using the first two features.
        
        Args:
            X (np.ndarray): Feature matrix
            labels (np.ndarray): Cluster assignments
            feature_names (list): Names of the features
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title('Clustering Results')
        plt.colorbar(scatter)
        return fig
    
    def plot_ecg_signals(self, data):
        """Plot ECG signals (Leads II, V, and AVR).
        
        Args:
            data (pd.DataFrame): DataFrame containing ECG signal data
            
        Returns:
            go.Figure: Plotly figure object containing ECG plots
        """
        # Create figure with secondary y-axis
        fig = make_subplots(rows=3, cols=1,
                          subplot_titles=('Lead II', 'Lead V', 'Lead AVR'))
        
        # Add traces for each ECG lead
        fig.add_trace(
            go.Scatter(y=data['II'].values[:1000], name="Lead II"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(y=data['V'].values[:1000], name="Lead V"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(y=data['AVR'].values[:1000], name="Lead AVR"),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="ECG Signal Visualization",
            showlegend=True
        )
        
        return fig
    
    def plot_ppg_signals(self, data):
        """Plot PPG signals.
        
        Args:
            data (pd.DataFrame): DataFrame containing PPG signal data
            
        Returns:
            go.Figure: Plotly figure object containing PPG plot
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(y=data['PLETH'].values[:1000], name="PPG")
        )
        
        fig.update_layout(
            title="PPG Signal Visualization",
            yaxis_title="Amplitude",
            xaxis_title="Time",
            showlegend=True
        )
        
        return fig
    
    def plot_vital_signs(self, data):
        """Plot vital signs (HR, PULSE, RESP, SpO2).
        
        Args:
            data (pd.DataFrame): DataFrame containing vital signs data
            
        Returns:
            go.Figure: Plotly figure object containing vital signs plots
        """
        fig = make_subplots(rows=2, cols=2,
                          subplot_titles=('Heart Rate', 'Pulse', 
                                        'Respiration', 'SpO2'))
        
        # Heart Rate
        fig.add_trace(
            go.Scatter(y=data['HR'].values[:1000], name="HR"),
            row=1, col=1
        )
        
        # Pulse
        fig.add_trace(
            go.Scatter(y=data['PULSE'].values[:1000], name="Pulse"),
            row=1, col=2
        )
        
        # Respiration
        fig.add_trace(
            go.Scatter(y=data['RESP_x'].values[:1000], name="RESP"),
            row=2, col=1
        )
        
        # SpO2
        fig.add_trace(
            go.Scatter(y=data['SpO2'].values[:1000], name="SpO2"),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Vital Signs Visualization",
            showlegend=True
        )
        
        return fig
    
    def plot_risk_distribution(self, predictions):
        """Plot distribution of risk predictions.
        
        Args:
            predictions (np.ndarray): Array of risk predictions
            
        Returns:
            go.Figure: Plotly figure with risk distribution
        """
        fig = go.Figure()
        
        # Count risk categories
        unique, counts = np.unique(predictions, return_counts=True)
        labels = ['Low Risk' if x == 0 else 'High Risk' for x in unique]
        
        fig.add_trace(
            go.Bar(
                x=labels,
                y=counts,
                marker_color=['green', 'red']
            )
        )
        
        fig.update_layout(
            title="Risk Distribution",
            xaxis_title="Risk Category",
            yaxis_title="Count",
            showlegend=False
        )
        
        return fig
    
    def plot_new_point_in_context(self, X_train, labels, new_point, feature_names):
        """Plot new data point in context of training data.
        
        Args:
            X_train (np.ndarray): Training data
            labels (np.ndarray): Cluster labels for training data
            new_point (np.ndarray): New data point to visualize
            feature_names (list): Names of features being plotted
            
        Returns:
            go.Figure: Plotly figure with scatter plot
        """
        fig = go.Figure()
        
        # Plot training data
        fig.add_trace(
            go.Scatter(
                x=X_train[:, 0],
                y=X_train[:, 1],
                mode='markers',
                marker=dict(
                    color=labels,
                    colorscale='Viridis',
                ),
                name='Training Data'
            )
        )
        
        # Plot new point
        fig.add_trace(
            go.Scatter(
                x=[new_point[0]],
                y=[new_point[1]],
                mode='markers',
                marker=dict(
                    color='red',
                    size=15,
                    symbol='star'
                ),
                name='New Patient'
            )
        )
        
        fig.update_layout(
            title="New Patient Data in Context",
            xaxis_title=feature_names[0],
            yaxis_title=feature_names[1],
            showlegend=True
        )
        
        return fig 