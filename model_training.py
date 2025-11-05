"""Model training and evaluation functions for MMA win prediction.

This module contains helper functions to train machine learning models
on the prepared feature matrix and evaluate their performance using
cross‑validation. Tree‑based models (decision trees, random forests)
are preferred because they handle nonlinear relationships and mixed
feature types with minimal preprocessing.

Functions
---------
train_model(X: pd.DataFrame, y: pd.Series, model_type: str, random_state: int)
    Instantiate and fit a classifier given a model type.

evaluate_model(model, X: pd.DataFrame, y: pd.Series, cv: int) -> dict
    Perform cross‑validated evaluation using accuracy, F1 and ROC‑AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'random_forest',
    random_state: int = 42
) -> object:
    """Train a classification model on the provided data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix after encoding/scaling.
    y : pd.Series
        Target vector where 1 indicates fighter A wins.
    model_type : {'random_forest', 'decision_tree', 'gradient_boosting'}, default='random_forest'
        Choice of algorithm. Random forests generally provide strong
        baseline performance for tabular data. Gradient boosting is
        optional for comparison.
    random_state : int, default=42
        Seed for reproducible model initialization.

    Returns
    -------
    model : object
        Fitted scikit‑learn model ready for prediction.
    """
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight='balanced')
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=random_state, class_weight='balanced')
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X, y)
    return model


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5
) -> Dict[str, float]:
    """Evaluate a trained model using cross‑validation.

    Three metrics are computed: accuracy, F1 score (macro), and ROC‑AUC.
    Because the target is binary, ROC‑AUC reflects the model's ability
    to rank fighters correctly. This function returns the mean score
    across the specified number of folds.

    Parameters
    ----------
    model : object
        Fitted classifier following scikit‑learn API.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    cv : int, default=5
        Number of cross‑validation folds.

    Returns
    -------
    Dict[str, float]
        Dictionary containing averaged metrics (accuracy, f1, roc_auc).
    """
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    metrics = {metric: np.mean(cv_results[f'test_{metric}']) for metric in scoring}
    return metrics
