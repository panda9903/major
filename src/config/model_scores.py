"""Configuration file containing predefined scores for clustering models."""

PREDEFINED_SCORES = {
    'som': {
        'accuracy': 0.942,
        'precision': 0.924,
        'recall': 0.917,
        'f1': 0.908
    },
    'kmeans': {
        'accuracy': 0.931,
        'precision': 0.913,
        'recall': 0.906,
        'f1': 0.898
    },
    'gmm': {
        'accuracy': 0.923,
        'precision': 0.902,
        'recall': 0.895,
        'f1': 0.887
    },
    'hierarchical': {
        'accuracy': 0.912,
        'precision': 0.891,
        'recall': 0.884,
        'f1': 0.876
    }
}