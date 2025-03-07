# Cardiovascular Risk Prediction System

This project implements a machine learning system for predicting cardiovascular risk by analyzing ECG (Electrocardiogram) and PPG (Photoplethysmogram) signals. The system uses multiple clustering algorithms and a voting mechanism to classify patients into risk categories.

## Features

- Data preprocessing of ECG and PPG signals
- Implementation of multiple clustering algorithms:
  - K-Means
  - K-Nearest Neighbors (KNN)
  - Gaussian Mixture Model (GMM)
  - Self-Organizing Maps (SOM)
- Majority voting mechanism for final risk classification
- Interactive visualization of signals and results
- User-friendly web interface using Streamlit

## Project Structure

```
.
├── src/
│   ├── models/
│   │   └── clustering.py
│   ├── utils/
│   │   └── data_preprocessor.py
│   ├── visualization/
│   │   └── visualizer.py
│   └── app.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd cardiovascular-risk-prediction
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:

```bash
cd src
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Use the interface to:
   - Train the clustering models
   - Visualize ECG and PPG signals
   - Analyze cardiovascular risk
   - Compare model predictions

## Data

The system uses the following signals from the dataset:

- ECG signals (II, V, AVR)
- PPG signal (PLETH)
- Vital signs (HR, PULSE, RESP, SpO2)

## Model Details

### Clustering Algorithms

1. **K-Means**

   - Initial clustering to establish baseline labels
   - Binary classification (2 clusters)

2. **K-Nearest Neighbors (KNN)**

   - Uses K-means labels for training
   - k=5 neighbors

3. **Gaussian Mixture Model (GMM)**

   - Probabilistic clustering
   - 2 components for binary classification

4. **Self-Organizing Maps (SOM)**
   - Neural network-based clustering
   - Preserves topological properties of the input space

### Voting Mechanism

The final risk classification is determined by majority voting across all models:

- Each model assigns a binary label (0: Low Risk, 1: High Risk)
- The most common prediction is selected as the final risk assessment

## License

This project is licensed under the MIT License - see the LICENSE file for details.
