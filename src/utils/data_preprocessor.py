import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_path):
        """Initialize the DataPreprocessor with the path to the data file.
        
        Args:
            data_path (str): Path to the CSV data file
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.train_indices = None
        self.test_indices = None
        
    def load_data(self):
        """Load the data from CSV file."""
        # Read CSV and strip whitespace from column names
        self.data = pd.read_csv(self.data_path)
        self.data.columns = self.data.columns.str.strip()
        return self.data
    
    def preprocess_signals(self):
        """Preprocess the ECG and PPG signals.
        
        Returns:
            tuple: (X, feature_names) where X is the preprocessed features and 
                  feature_names is the list of feature names
        """
        # Select relevant features (ECG and PPG signals)
        signal_features = ['II', 'V', 'AVR', 'PLETH']
        vital_signs = ['HR', 'PULSE', 'RESP_x', 'SpO2']
        
        # Combine all features
        feature_names = signal_features + vital_signs
        X = self.data[feature_names].values
        
        # Standardize the features
        X = self.scaler.fit_transform(X)
        
        return X, feature_names
    
    def split_data(self, X, test_size=0.2, random_state=42):
        """Split the data into training and testing sets.
        
        Args:
            X (np.ndarray): Feature matrix
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, train_indices, test_indices)
        """
        # Split indices
        indices = np.arange(len(X))
        X_train, X_test, train_indices, test_indices = train_test_split(
            X, indices, test_size=test_size, random_state=random_state
        )
        
        # Store indices for later use
        self.train_indices = train_indices
        self.test_indices = test_indices
        
        return X_train, X_test, train_indices, test_indices
    
    def get_train_test_data(self):
        """Get the training and testing data based on stored indices.
        
        Returns:
            tuple: (train_data, test_data) DataFrames
        """
        if self.train_indices is None or self.test_indices is None:
            raise ValueError("Data has not been split yet. Call split_data first.")
        
        train_data = self.data.iloc[self.train_indices]
        test_data = self.data.iloc[self.test_indices]
        
        return train_data, test_data 