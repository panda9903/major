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
        
    def load_data(self):
        """Load the data from CSV file."""
        # Read CSV and strip whitespace from column names
        self.data = pd.read_csv(self.data_path)
        self.data.columns = self.data.columns.str.strip()
        print(self.data.head())
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
            tuple: (X_train, X_test) training and testing sets
        """
        X_train, X_test = train_test_split(X, test_size=test_size, 
                                         random_state=random_state)
        return X_train, X_test 