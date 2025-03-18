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
        self.colors = {
            'train': '#2ecc71',  # Green for training data
            'test': '#e74c3c',   # Red for test data
            'cluster0': '#3498db',  # Blue for cluster 0
            'cluster1': '#e67e22'   # Orange for cluster 1
        }
    
    def plot_signals(self, data, time_column='Time [s]', signals=['II', 'V', 'AVR', 'PLETH'],
                    train_indices=None, test_indices=None):
        """Plot ECG and PPG signals.
        
        Args:
            data (pd.DataFrame): DataFrame containing the signals
            time_column (str): Name of the time column
            signals (list): List of signal names to plot
            train_indices (array): Indices of training data
            test_indices (array): Indices of test data
        """
        fig, axes = plt.subplots(len(signals), 1, figsize=(15, 10), sharex=True)
        fig.suptitle('ECG and PPG Signals', fontsize=16)
        
        for ax, signal in zip(axes, signals):
            if train_indices is not None and test_indices is not None:
                # Plot training data
                ax.plot(data[time_column].iloc[train_indices], 
                       data[signal].iloc[train_indices], 
                       label=f'{signal} (Train)',
                       color=self.colors['train'],
                       alpha=0.7)
                # Plot test data
                ax.plot(data[time_column].iloc[test_indices], 
                       data[signal].iloc[test_indices], 
                       label=f'{signal} (Test)',
                       color=self.colors['test'],
                       alpha=0.7)
            else:
                ax.plot(data[time_column], data[signal], label=signal)
            ax.set_ylabel(signal)
            ax.legend()
            
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        return fig
    
    def plot_clusters(self, X, labels, feature_names, train_mask=None):
        """Plot clustering results using the first two features.
        
        Args:
            X (np.ndarray): Feature matrix
            labels (np.ndarray): Cluster assignments
            feature_names (list): Names of the features
            train_mask (array): Boolean mask for training data
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if train_mask is not None:
            # Plot training data
            scatter_train = ax.scatter(X[train_mask, 0], X[train_mask, 1], 
                                     c=labels[train_mask], cmap='viridis',
                                     marker='o', label='Train')
            # Plot test data
            scatter_test = ax.scatter(X[~train_mask, 0], X[~train_mask, 1], 
                                    c=labels[~train_mask], cmap='viridis',
                                    marker='x', label='Test')
            plt.legend()
        else:
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
            
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title('Clustering Results')
        plt.colorbar(scatter)
        return fig
    
    def plot_ecg_signals(self, data, train_indices=None, test_indices=None):
        """Plot ECG signals (Leads II, V, and AVR).
        
        Args:
            data (pd.DataFrame): DataFrame containing ECG signal data
            train_indices (array): Indices of training data
            test_indices (array): Indices of test data
            
        Returns:
            go.Figure: Plotly figure object containing ECG plots
        """
        fig = make_subplots(rows=3, cols=1,
                          subplot_titles=('Lead II', 'Lead V', 'Lead AVR'))
        
        leads = ['II', 'V', 'AVR']
        for i, lead in enumerate(leads, 1):
            if train_indices is not None and test_indices is not None:
                # Plot training data
                fig.add_trace(
                    go.Scatter(
                        y=data[lead].iloc[train_indices].values[:1000],
                        name=f"{lead} (Train)",
                        line=dict(color=self.colors['train'])
                    ),
                    row=i, col=1
                )
                # Plot test data
                fig.add_trace(
                    go.Scatter(
                        y=data[lead].iloc[test_indices].values[:1000],
                        name=f"{lead} (Test)",
                        line=dict(color=self.colors['test'])
                    ),
                    row=i, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(y=data[lead].values[:1000], name=lead),
                    row=i, col=1
                )
        
        fig.update_layout(
            height=800,
            title_text="ECG Signal Visualization",
            showlegend=True
        )
        
        return fig
    
    def plot_ppg_signals(self, data, train_indices=None, test_indices=None):
        """Plot PPG signals.
        
        Args:
            data (pd.DataFrame): DataFrame containing PPG signal data
            train_indices (array): Indices of training data
            test_indices (array): Indices of test data
            
        Returns:
            go.Figure: Plotly figure object containing PPG plot
        """
        fig = go.Figure()
        
        if train_indices is not None and test_indices is not None:
            # Plot training data
            fig.add_trace(
                go.Scatter(
                    y=data['PLETH'].iloc[train_indices].values[:1000],
                    name="PPG (Train)",
                    line=dict(color=self.colors['train'])
                )
            )
            # Plot test data
            fig.add_trace(
                go.Scatter(
                    y=data['PLETH'].iloc[test_indices].values[:1000],
                    name="PPG (Test)",
                    line=dict(color=self.colors['test'])
                )
            )
        else:
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
    
    def plot_vital_signs(self, data, train_indices=None, test_indices=None):
        """Plot vital signs (HR, PULSE, RESP, SpO2).
        
        Args:
            data (pd.DataFrame): DataFrame containing vital signs data
            train_indices (array): Indices of training data
            test_indices (array): Indices of test data
            
        Returns:
            go.Figure: Plotly figure object containing vital signs plots
        """
        fig = make_subplots(rows=2, cols=2,
                          subplot_titles=('Heart Rate', 'Pulse', 
                                        'Respiration', 'SpO2'))
        
        vital_signs = [('HR', 1, 1), ('PULSE', 1, 2),
                      ('RESP_x', 2, 1), ('SpO2', 2, 2)]
        
        for vital, row, col in vital_signs:
            if train_indices is not None and test_indices is not None:
                # Plot training data
                fig.add_trace(
                    go.Scatter(
                        y=data[vital].iloc[train_indices].values[:1000],
                        name=f"{vital} (Train)",
                        line=dict(color=self.colors['train'])
                    ),
                    row=row, col=col
                )
                # Plot test data
                fig.add_trace(
                    go.Scatter(
                        y=data[vital].iloc[test_indices].values[:1000],
                        name=f"{vital} (Test)",
                        line=dict(color=self.colors['test'])
                    ),
                    row=row, col=col
                )
            else:
                fig.add_trace(
                    go.Scatter(y=data[vital].values[:1000], name=vital),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=800,
            title_text="Vital Signs Visualization",
            showlegend=True
        )
        
        return fig
    
    def plot_risk_distribution(self, predictions, train_mask=None):
        """Plot distribution of risk predictions.
        
        Args:
            predictions (np.ndarray): Array of risk predictions
            train_mask (array): Boolean mask for training data
            
        Returns:
            go.Figure: Plotly figure with risk distribution
        """
        fig = go.Figure()
        
        if train_mask is not None:
            # Plot training data distribution
            train_unique, train_counts = np.unique(predictions[train_mask], return_counts=True)
            train_labels = ['Low Risk (Train)' if x == 0 else 'High Risk (Train)' for x in train_unique]
            
            fig.add_trace(
                go.Bar(
                    x=train_labels,
                    y=train_counts,
                    name='Training Data',
                    marker_color=[self.colors['train'], self.colors['train']]
                )
            )
            
            # Plot test data distribution
            test_unique, test_counts = np.unique(predictions[~train_mask], return_counts=True)
            test_labels = ['Low Risk (Test)' if x == 0 else 'High Risk (Test)' for x in test_unique]
            
            fig.add_trace(
                go.Bar(
                    x=test_labels,
                    y=test_counts,
                    name='Test Data',
                    marker_color=[self.colors['test'], self.colors['test']]
                )
            )
        else:
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
            showlegend=True if train_mask is not None else False
        )
        
        return fig
    
    def plot_new_point_in_context(self, X_train, X_test, labels_train, labels_test, new_point, feature_names):
        """Plot new data point in context of training and test data.
        
        Args:
            X_train (np.ndarray): Training data
            X_test (np.ndarray): Test data
            labels_train (np.ndarray): Training labels
            labels_test (np.ndarray): Test labels
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
                    color=labels_train,
                    colorscale='Viridis',
                    symbol='circle'
                ),
                name='Training Data'
            )
        )
        
        # Plot test data
        fig.add_trace(
            go.Scatter(
                x=X_test[:, 0],
                y=X_test[:, 1],
                mode='markers',
                marker=dict(
                    color=labels_test,
                    colorscale='Viridis',
                    symbol='x'
                ),
                name='Test Data'
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