import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from minisom import MiniSom
from scipy.spatial.distance import cdist
import joblib
import os

class ClusteringModels:
    def __init__(self, n_clusters=2, random_state=42):
        """Initialize clustering models for cardiovascular risk prediction.
        
        Args:
            n_clusters (int): Number of clusters (2 for binary risk classification)
            random_state (int): Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize clustering models
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state
        )
        
        self.gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state
        )
        
        self.hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        self.hierarchical_centroids = None  # Will store centroids for prediction
        
        # Initialize SOM with a 2x2 grid for binary clustering
        self.som = None
        
        # Store cluster risk mappings
        self.risk_mappings = {
            'kmeans': None,
            'gmm': None,
            'hierarchical': None,
            'som': None
        }
        
        # Store evaluation metrics
        self.metrics = {}

    def create_som(self, input_dim):
        """Create Self-Organizing Map."""
        self.som = MiniSom(
            2, 2, input_dim,
            sigma=1.0,
            learning_rate=0.5,
            random_seed=self.random_state
        )

    def fit(self, X):
        """Fit all clustering models and compute evaluation metrics.
        
        Args:
            X (np.ndarray): Training data
        """
        # Train K-Means
        kmeans_labels = self.kmeans.fit_predict(X)
        
        # Train GMM
        gmm_labels = self.gmm.fit_predict(X)
        
        # Train Hierarchical Clustering
        hierarchical_labels = self.hierarchical.fit_predict(X)
        # Store centroids for prediction
        self._store_hierarchical_centroids(X, hierarchical_labels)
        
        # Train SOM
        if self.som is None:
            self.create_som(X.shape[1])
        som_labels = self._train_som(X)
        
        # Compute and store evaluation metrics
        self._compute_metrics(X, {
            'kmeans': kmeans_labels,
            'gmm': gmm_labels,
            'hierarchical': hierarchical_labels,
            'som': som_labels
        })
        
        # Map clusters to risk labels
        self._map_risk_labels(X, {
            'kmeans': kmeans_labels,
            'gmm': gmm_labels,
            'hierarchical': hierarchical_labels,
            'som': som_labels
        })

    def _store_hierarchical_centroids(self, X, labels):
        """Store centroids for hierarchical clustering prediction."""
        self.hierarchical_centroids = np.array([
            np.mean(X[labels == i], axis=0)
            for i in range(self.n_clusters)
        ])

    def _train_som(self, X, epochs=1000):
        """Train SOM and return cluster assignments."""
        self.som.random_weights_init(X)
        self.som.train(X, epochs, verbose=False)
        
        # Get cluster assignments
        som_labels = np.array([
            self.som.winner(x)[0] * 2 + self.som.winner(x)[1]
            for x in X
        ])
        
        return som_labels

    def _compute_metrics(self, X, labels_dict):
        """Compute clustering evaluation metrics."""
        for model_name, labels in labels_dict.items():
            self.metrics[model_name] = {
                'silhouette': silhouette_score(X, labels),
                'davies_bouldin': davies_bouldin_score(X, labels),
                'calinski_harabasz': calinski_harabasz_score(X, labels)
            }

    def _map_risk_labels(self, X, labels_dict):
        """Map clusters to risk labels based on medical analysis."""
        for model_name, labels in labels_dict.items():
            # For each cluster, compute the mean distance from center
            center = np.mean(X, axis=0)
            cluster_centers = {
                c: np.mean(X[labels == c], axis=0)
                for c in np.unique(labels)
            }
            
            # Compute distances from overall center
            cluster_distances = {
                c: np.linalg.norm(cent - center)
                for c, cent in cluster_centers.items()
            }
            
            # Map to risk labels: further clusters are considered higher risk
            self.risk_mappings[model_name] = {
                c: int(d > np.median(list(cluster_distances.values())))
                for c, d in cluster_distances.items()
            }

    def predict(self, X):
        """Predict risk labels using all models.
        
        Args:
            X (np.ndarray): Data to predict
            
        Returns:
            dict: Predictions from each model mapped to risk labels
        """
        predictions = {}
        
        # K-Means prediction
        kmeans_clusters = self.kmeans.predict(X)
        predictions['kmeans'] = np.array([
            self.risk_mappings['kmeans'][c] for c in kmeans_clusters
        ])
        
        # GMM prediction
        gmm_clusters = self.gmm.predict(X)
        predictions['gmm'] = np.array([
            self.risk_mappings['gmm'][c] for c in gmm_clusters
        ])
        
        # Hierarchical prediction
        hierarchical_clusters = self._predict_hierarchical(X)
        predictions['hierarchical'] = np.array([
            self.risk_mappings['hierarchical'][c] for c in hierarchical_clusters
        ])
        
        # SOM prediction
        if self.som is not None:
            som_clusters = np.array([
                self.som.winner(x)[0] * 2 + self.som.winner(x)[1]
                for x in X
            ])
            predictions['som'] = np.array([
                self.risk_mappings['som'][c] for c in som_clusters
            ])
        else:
            predictions['som'] = np.zeros(len(X))  # Default prediction if SOM not trained
        
        return predictions

    def _predict_hierarchical(self, X):
        """Predict clusters for hierarchical clustering using stored centroids."""
        distances = cdist(X, self.hierarchical_centroids)
        return np.argmin(distances, axis=1)

    def majority_vote(self, predictions):
        """Perform majority voting with bias towards high risk.
        
        Args:
            predictions (dict): Predictions from each model
            
        Returns:
            np.ndarray: Final risk predictions
        """
        # Stack all predictions
        pred_array = np.vstack([
            pred for pred in predictions.values()
        ]).T
        
        # Count votes for each class
        votes = np.sum(pred_array, axis=1)
        n_models = pred_array.shape[1]
        
        # If votes are tied or majority is for high risk (1),
        # classify as high risk
        return (votes >= n_models/2).astype(int)

    def save_models(self, directory):
        """Save trained models and mappings."""
        os.makedirs(directory, exist_ok=True)
        
        # Save sklearn models
        joblib.dump(self.kmeans, os.path.join(directory, 'kmeans.pkl'))
        joblib.dump(self.gmm, os.path.join(directory, 'gmm.pkl'))
        joblib.dump(self.hierarchical, os.path.join(directory, 'hierarchical.pkl'))
        
        # Save SOM weights
        if self.som is not None:
            self.som.save(os.path.join(directory, 'som.pkl'))
        
        # Save risk mappings and metrics
        np.save(os.path.join(directory, 'risk_mappings.npy'), self.risk_mappings)
        np.save(os.path.join(directory, 'metrics.npy'), self.metrics)

    def load_models(self, directory):
        """Load trained models and mappings."""
        self.kmeans = joblib.load(os.path.join(directory, 'kmeans.pkl'))
        self.gmm = joblib.load(os.path.join(directory, 'gmm.pkl'))
        self.hierarchical = joblib.load(os.path.join(directory, 'hierarchical.pkl'))
        
        # Load SOM if it exists
        som_path = os.path.join(directory, 'som.pkl')
        if os.path.exists(som_path):
            self.som = MiniSom.load(som_path)
        
        # Load risk mappings and metrics
        self.risk_mappings = np.load(os.path.join(directory, 'risk_mappings.npy'),
                                   allow_pickle=True).item()
        self.metrics = np.load(os.path.join(directory, 'metrics.npy'),
                             allow_pickle=True).item() 