import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

class ClusteringModels:
    def __init__(self, n_clusters=2, random_state=42):
        """Initialize clustering models.
        
        Args:
            n_clusters (int): Number of clusters (2 for binary risk classification)
            random_state (int): Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Initialize models
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        self.som = None  # Will be initialized during training
        
    def train_kmeans(self, X):
        """Train K-Means clustering model.
        
        Args:
            X (np.ndarray): Training data
            
        Returns:
            np.ndarray: Cluster assignments
        """
        return self.kmeans.fit_predict(X)
    
    def train_knn(self, X, y):
        """Train K-Nearest Neighbors model.
        
        Args:
            X (np.ndarray): Training data
            y (np.ndarray): Training labels (from another clustering)
            
        Returns:
            np.ndarray: Cluster assignments
        """
        self.knn.fit(X, y)
        return self.knn.predict(X)
    
    def train_gmm(self, X):
        """Train Gaussian Mixture Model.
        
        Args:
            X (np.ndarray): Training data
            
        Returns:
            np.ndarray: Cluster assignments
        """
        return self.gmm.fit_predict(X)
    
    def create_som(self, input_dim):
        """Create and compile Self-Organizing Map model.
        
        Args:
            input_dim (int): Number of input features
        """
        self.som = Sequential([
            Dense(100, activation='relu', input_shape=(input_dim,)),
            Dense(50, activation='relu'),
            Dense(self.n_clusters, activation='softmax')
        ])
        
        self.som.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    
    def train_som(self, X, y, epochs=50, batch_size=32):
        """Train Self-Organizing Map model.
        
        Args:
            X (np.ndarray): Training data
            y (np.ndarray): Training labels (from another clustering)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            np.ndarray: Cluster assignments
        """
        if self.som is None:
            self.create_som(X.shape[1])
            
        self.som.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        predictions = self.som.predict(X)
        return np.argmax(predictions, axis=1)
    
    def predict_all(self, X):
        """Get predictions from all trained models.
        
        Args:
            X (np.ndarray): Data to predict
            
        Returns:
            dict: Dictionary containing predictions from each model
        """
        predictions = {
            'kmeans': self.kmeans.predict(X),
            'knn': self.knn.predict(X),
            'gmm': self.gmm.predict(X),
            'som': np.argmax(self.som.predict(X), axis=1) if self.som else None
        }
        return predictions
    
    def majority_vote(self, predictions):
        """Perform majority voting on predictions from all models.
        
        Args:
            predictions (dict): Dictionary containing predictions from each model
            
        Returns:
            np.ndarray: Final predictions based on majority voting. In case of ties,
            classifies as high risk (class 1).
        """
        # Stack predictions from all models
        pred_array = np.vstack([
            pred for pred in predictions.values() if pred is not None
        ]).T
        
        # Custom voting function that returns 1 (high risk) in case of ties
        def vote_with_tie_handling(x):
            counts = np.bincount(x)
            if len(counts) > 1 and counts[0] == counts[1]:
                return 1  # Return high risk in case of tie
            return counts.argmax()
        
        # Get majority vote for each sample
        final_predictions = np.apply_along_axis(
            vote_with_tie_handling,
            axis=1, 
            arr=pred_array
        )
        
        return final_predictions 