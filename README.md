# Cardiovascular Risk Prediction System

This project implements a machine learning system for predicting cardiovascular risk by analyzing ECG (Electrocardiogram) and PPG (Photoplethysmogram) signals. The system uses a hybrid approach that converts unsupervised clustering into supervised classification through a novel training methodology.

## Features

- Data preprocessing of ECG and PPG signals
- Implementation of multiple clustering/classification algorithms:
  - K-Means
  - K-Nearest Neighbors (KNN)
  - Gaussian Mixture Model (GMM)
  - Self-Organizing Maps (SOM)
- Semi-supervised learning approach
- Interactive visualization of signals and results
- User-friendly web interface using Streamlit

## Project Structure

```bash
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
cd major
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

## Model Training Pipeline

The system uses a novel approach to convert unsupervised clustering into supervised classification:

1. **Data Split**

   - Dataset is split into training and testing sets
   - Ensures unbiased evaluation of model performance

2. **Initial Model Training**

   - All clustering models (K-Means, KNN, GMM, SOM) are trained on the training data
   - Models perform initial clustering independently

3. **Best Model Selection**

   - Performance metrics are calculated for each model
   - The best performing model is selected as the "teacher" model

4. **Label Generation**

   - The best model's predictions on the training data are used as "pseudo-labels"
   - These labels convert the problem from unsupervised to supervised

5. **Supervised Training**

   - Remaining models are retrained using the pseudo-labels
   - Models now perform supervised classification instead of clustering

6. **Model Evaluation**
   - All models are evaluated on the test set
   - Performance is measured against the best model's predictions

### Clustering/Classification Algorithms

1. **K-Means**

   - Initial unsupervised clustering
   - Binary classification (2 clusters)
   - Can be retrained as a supervised classifier using pseudo-labels

2. **K-Nearest Neighbors (KNN)**

   - Initially uses distance-based clustering
   - Transitions to supervised classification using pseudo-labels
   - k=5 neighbors

3. **Gaussian Mixture Model (GMM)**

   - Probabilistic clustering initially
   - 2 components for binary classification
   - Adapts to supervised classification using pseudo-labels

4. **Self-Organizing Maps (SOM)**
   - Neural network-based clustering
   - Preserves topological properties of the input space
   - Can be retrained using pseudo-labels

## Data

The system uses the following signals from the dataset:

- ECG signals (II, V, AVR)
- PPG signal (PLETH)
- Vital signs (HR, PULSE, RESP, SpO2)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
