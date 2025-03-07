import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
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
    
    def plot_vital_signs(self, data):
        """Create an interactive plot of vital signs.
        
        Args:
            data (pd.DataFrame): DataFrame containing vital signs
            
        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        fig = go.Figure()
        
        # Add traces for HR and SpO2
        fig.add_trace(go.Scatter(x=data['Time [s]'], y=data['HR'],
                                mode='lines', name='Heart Rate'))
        fig.add_trace(go.Scatter(x=data['Time [s]'], y=data['SpO2'],
                                mode='lines', name='SpO2'))
        
        # Update layout
        fig.update_layout(
            title='Vital Signs Monitoring',
            xaxis_title='Time (s)',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_risk_distribution(self, predictions):
        """Plot the distribution of risk predictions.
        
        Args:
            predictions (np.ndarray): Array of risk predictions (0 or 1)
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x=predictions, ax=ax)
        ax.set_xlabel('Risk Level')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Risk Predictions')
        ax.set_xticklabels(['Low Risk', 'High Risk'])
        return fig
        
    def plot_new_point_in_context(self, X, labels, new_point, feature_names):
        """Plot a new data point in the context of existing clusters.
        
        Args:
            X (np.ndarray): Existing data points
            labels (np.ndarray): Cluster labels for existing points
            new_point (np.ndarray): New data point to visualize
            feature_names (list): Names of the features to plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot existing points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                           alpha=0.6, label='Existing Points')
        
        # Plot new point
        ax.scatter(new_point[0], new_point[1], color='red', marker='*', 
                  s=200, label='New Patient')
        
        # Add labels and title
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title('New Patient in Feature Space')
        ax.legend()
        
        # Add colorbar
        plt.colorbar(scatter, label='Risk Level')
        
        return fig 