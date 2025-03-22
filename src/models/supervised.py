import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import os

class CNN1D(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(CNN1D, self).__init__()
        # Calculate output sizes for each layer
        L_in = input_shape[0]
        
        # First conv layer
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)  # Added padding
        L_out1 = L_in  # Same padding preserves size
        L_pool1 = L_out1 // 2
        
        # Second conv layer
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # Added padding
        L_out2 = L_pool1
        L_pool2 = L_out2 // 2
        
        # Third conv layer
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)  # Added padding
        L_out3 = L_pool2
        
        # Calculate final feature size
        self.feature_size = 64 * L_out3
        
        # Layers
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.feature_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, n_classes)
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Second block
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Third block
        x = self.conv3(x)
        x = torch.relu(x)
        
        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SupervisedModels:
    def __init__(self, random_state=42, test_size=0.2):
        """Initialize supervised learning models."""
        self.random_state = random_state
        self.test_size = test_size
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Initialize models
        self.xgb_model = xgb.XGBClassifier(
            random_state=random_state,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,  # Prevent overfitting
            subsample=0.8,  # Prevent overfitting
            colsample_bytree=0.8  # Prevent overfitting
        )
        
        self.svm_model = SVC(
            random_state=random_state,
            kernel='rbf',
            probability=True,
            C=1.0,  # Regularization parameter
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.cnn_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = {}
        self.X_test = None
        self.y_test = None
        
    def create_cnn(self, input_shape, n_classes):
        """Create CNN model architecture."""
        self.cnn_model = CNN1D(input_shape, n_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=None)  # Can add class weights if needed
        self.optimizer = optim.Adam(self.cnn_model.parameters(), lr=0.001, weight_decay=1e-5)  # Added L2 regularization
    
    def fit(self, X, y):
        """Train all models with proper validation."""
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Store test set for later evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Train XGBoost
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Train SVM
        self.svm_model.fit(X_train, y_train)
        
        # Train CNN
        if self.cnn_model is None:
            self.create_cnn((X.shape[1],), len(np.unique(y)))
        
        # Prepare data for CNN
        X_train_cnn = torch.FloatTensor(X_train.reshape(-1, 1, X_train.shape[1])).to(self.device)
        y_train_cnn = torch.LongTensor(y_train).to(self.device)
        X_test_cnn = torch.FloatTensor(X_test.reshape(-1, 1, X_test.shape[1])).to(self.device)
        y_test_cnn = torch.LongTensor(y_test).to(self.device)
        
        train_dataset = TensorDataset(X_train_cnn, y_train_cnn)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Train CNN with validation
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        self.cnn_model.train()
        for epoch in range(50):
            # Training
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.cnn_model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            
            # Validation
            self.cnn_model.eval()
            with torch.no_grad():
                val_outputs = self.cnn_model(X_test_cnn)
                val_loss = self.criterion(val_outputs, y_test_cnn)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Evaluate all models on test set
        self.evaluate()
    
    def predict(self, X):
        """Get predictions from all models."""
        predictions = {}
        
        # XGBoost predictions
        predictions['xgb'] = self.xgb_model.predict(X)
        
        # SVM predictions
        predictions['svm'] = self.svm_model.predict(X)
        
        # CNN predictions
        self.cnn_model.eval()
        with torch.no_grad():
            X_cnn = torch.FloatTensor(X.reshape(-1, 1, X.shape[1])).to(self.device)
            outputs = self.cnn_model(X_cnn)
            predictions['cnn'] = outputs.argmax(dim=1).cpu().numpy()
        
        return predictions
    
    def evaluate(self):
        """Evaluate all models on the test set."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("No test data available. Please run fit() first.")
            
        predictions = self.predict(self.X_test)
        
        for model_name, preds in predictions.items():
            self.metrics[model_name] = {
                'accuracy': accuracy_score(self.y_test, preds),
                'precision': precision_score(self.y_test, preds, average='weighted'),
                'recall': recall_score(self.y_test, preds, average='weighted'),
                'f1': f1_score(self.y_test, preds, average='weighted'),
                'confusion_matrix': confusion_matrix(self.y_test, preds)
            }
    
    def save_models(self, directory):
        """Save trained models."""
        os.makedirs(directory, exist_ok=True)
        
        # Save XGBoost
        self.xgb_model.save_model(os.path.join(directory, 'xgb_model.json'))
        
        # Save SVM
        joblib.dump(self.svm_model, os.path.join(directory, 'svm_model.pkl'))
        
        # Save CNN
        torch.save(self.cnn_model.state_dict(), os.path.join(directory, 'cnn_model.pt'))
        
        # Save metrics
        np.save(os.path.join(directory, 'metrics.npy'), self.metrics)
    
    def load_models(self, directory):
        """Load trained models."""
        # Load XGBoost
        self.xgb_model.load_model(os.path.join(directory, 'xgb_model.json'))
        
        # Load SVM
        self.svm_model = joblib.load(os.path.join(directory, 'svm_model.pkl'))
        
        # Load CNN
        self.cnn_model.load_state_dict(torch.load(os.path.join(directory, 'cnn_model.pt')))
        
        # Load metrics
        self.metrics = np.load(os.path.join(directory, 'metrics.npy'), allow_pickle=True).item() 