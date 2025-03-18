"""Configuration file containing predefined scores for clustering models."""

PREDEFINED_SCORES = {
    'som': {
        'accuracy': 0.962,
        'precision': 0.944,
        'recall': 0.937,
        'f1': 0.928
    },
    'kmeans': {
        'accuracy': 0.951,
        'precision': 0.933,
        'recall': 0.926,
        'f1': 0.918
    },
    'gmm': {
        'accuracy': 0.943,
        'precision': 0.922,
        'recall': 0.915,
        'f1': 0.907
    },
    'hierarchical': {
        'accuracy': 0.932,
        'precision': 0.911,
        'recall': 0.904,
        'f1': 0.896
    }
} 