"""Top‑level package for the MMA win prediction project.

This package exposes convenient functions and classes for simulating
data, computing ELO ratings, training machine learning models and
building interactive dashboards. See individual modules for detailed
documentation.
"""

from .data_processing import simulate_fighters, simulate_fights, prepare_training_data, encode_features, scale_features
from .elo_system import EloRatingSystem
from .model_training import train_model, evaluate_model

__all__ = [
    'simulate_fighters',
    'simulate_fights',
    'prepare_training_data',
    'encode_features',
    'scale_features',
    'EloRatingSystem',
    'train_model',
    'evaluate_model',
]
