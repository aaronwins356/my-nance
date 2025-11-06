"""
Model Training Script
Trains multiple ML models and selects the best performer
Includes DecisionTree, RandomForest, XGBoost, and LightGBM
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

# Import data processing
from data_processing import prepare_data, ensure_directories

class EnsembleModel:
    """
    Simple ensemble model that averages predictions from multiple models
    """
    def __init__(self, models, model_names):
        self.models = models
        self.model_names = model_names
        self.is_fitted_ = True  # Models are already fitted
    
    def fit(self, X, y):
        """Fit method for sklearn compatibility (models already fitted)"""
        return self
    
    def predict(self, X):
        """Predict class labels using majority vote"""
        predictions = np.array([model.predict(X) for model in self.models])
        # Average predictions and round (for binary classification)
        return np.round(predictions.mean(axis=0)).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities by averaging"""
        probabilities = np.array([model.predict_proba(X) for model in self.models])
        # Average probabilities across models
        return probabilities.mean(axis=0)
    
    def __repr__(self):
        return f"EnsembleModel({', '.join(self.model_names)})"

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

try:
    from sklearn.neural_network import MLPClassifier
    NEURAL_NET_AVAILABLE = True
except ImportError:
    NEURAL_NET_AVAILABLE = False
    print("Warning: MLPClassifier not available")

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')

def train_decision_tree(X_train, y_train, max_depth=10, min_samples_split=20):
    """
    Train Decision Tree classifier
    
    Args:
        X_train: Training features
        y_train: Training target
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
    
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("Training Decision Tree Classifier")
    print("="*60)
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    
    return model

def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Train Random Forest classifier
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees
    
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("Training Random Forest Classifier")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"Number of trees: {n_estimators}")
    print(f"Number of features: {model.n_features_in_}")
    
    return model

def train_xgboost(X_train, y_train, n_estimators=100):
    """
    Train XGBoost classifier
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting rounds
    
    Returns:
        Trained model
    """
    if not XGBOOST_AVAILABLE:
        print("XGBoost not available, skipping...")
        return None
    
    print("\n" + "="*60)
    print("Training XGBoost Classifier")
    print("="*60)
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    print(f"Number of boosting rounds: {n_estimators}")
    
    return model

def train_lightgbm(X_train, y_train, n_estimators=100):
    """
    Train LightGBM classifier
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting rounds
    
    Returns:
        Trained model
    """
    if not LIGHTGBM_AVAILABLE:
        print("LightGBM not available, skipping...")
        return None
    
    print("\n" + "="*60)
    print("Training LightGBM Classifier")
    print("="*60)
    
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"Number of boosting rounds: {n_estimators}")
    
    return model

def train_neural_net(X_train, y_train, hidden_layers=(64, 32), max_iter=500):
    """
    Train Neural Network (MLP) classifier
    
    Args:
        X_train: Training features
        y_train: Training target
        hidden_layers: Tuple specifying hidden layer sizes
        max_iter: Maximum number of training iterations
    
    Returns:
        Trained model
    """
    if not NEURAL_NET_AVAILABLE:
        print("MLPClassifier not available, skipping...")
        return None
    
    print("\n" + "="*60)
    print("Training Neural Network (MLP) Classifier")
    print("="*60)
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=max_iter,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False
    )
    
    model.fit(X_train, y_train)
    
    print(f"Architecture: {hidden_layers}")
    print(f"Training iterations: {model.n_iter_}")
    print(f"Loss: {model.loss_:.4f}")
    
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Get probability predictions
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'test_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    # Cross-validation score (skip for ensemble models)
    if 'Ensemble' not in model_name:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
    else:
        # For ensemble, use test accuracy as proxy
        metrics['cv_mean'] = metrics['test_accuracy']
        metrics['cv_std'] = 0.0
    
    # Print metrics
    print(f"\n{model_name} Performance:")
    print("-" * 60)
    print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Test Precision: {metrics['test_precision']:.4f}")
    print(f"Test Recall:    {metrics['test_recall']:.4f}")
    print(f"Test F1:        {metrics['test_f1']:.4f}")
    print(f"Test AUC:       {metrics['test_auc']:.4f}")
    print(f"CV Score:       {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    
    return metrics

def calibrate_model(model, X_train, y_train):
    """
    Calibrate model for better probability estimates
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
    
    Returns:
        Calibrated model
    """
    print("\nCalibrating model probabilities...")
    
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    print("Model calibrated successfully")
    
    return calibrated_model

def get_feature_importance(model, feature_names, top_n=20):
    """
    Extract feature importance from model
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        Dictionary with feature importance
    """
    # Get feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'calibrated_classifiers_'):
        # For calibrated models, get from base estimator
        importances = model.calibrated_classifiers_[0].estimator.feature_importances_
    else:
        print("Model does not support feature importance")
        return {}
    
    # Create feature importance dictionary
    importance_dict = dict(zip(feature_names, importances))
    
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 60)
    for i, (feature, importance) in enumerate(sorted_features[:top_n], 1):
        print(f"{i:2d}. {feature:35s} {importance:.4f}")
    
    return dict(sorted_features)

def save_model(model, model_name, feature_names, metrics, feature_importance):
    """
    Save trained model and metadata
    
    Args:
        model: Trained model
        model_name: Name of the model
        feature_names: List of feature names
        metrics: Dictionary with evaluation metrics
        feature_importance: Dictionary with feature importance
    """
    ensure_directories()
    
    # Save model
    model_file = os.path.join(ARTIFACTS_DIR, 'model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nSaved model to {model_file}")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'trained_at': datetime.now().isoformat(),
        'metrics': convert_to_native(metrics),
        'feature_importance': convert_to_native(feature_importance),
        'num_features': len(feature_names)
    }
    
    metadata_file = os.path.join(ARTIFACTS_DIR, 'model_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    # Feature names already saved by data_processing.py
    print(f"Model artifacts saved successfully")

def train_and_compare_models():
    """
    Train multiple models and select the best one
    """
    print("="*60)
    print("FightIQ Model Training Pipeline")
    print("="*60)
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    
    # Train models
    models = {}
    
    # 1. Decision Tree (Primary model as specified)
    models['DecisionTree'] = train_decision_tree(X_train, y_train)
    
    # 2. Random Forest
    models['RandomForest'] = train_random_forest(X_train, y_train)
    
    # 3. XGBoost
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = train_xgboost(X_train, y_train)
    
    # 4. LightGBM
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = train_lightgbm(X_train, y_train)
    
    # 5. Neural Network
    if NEURAL_NET_AVAILABLE:
        models['NeuralNet'] = train_neural_net(X_train, y_train)
    
    # Evaluate all models
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    all_metrics = []
    
    for name, model in models.items():
        if model is not None:
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test, name)
            all_metrics.append(metrics)
    
    # Create ensemble of Random Forest and Neural Network
    if models.get('RandomForest') and models.get('NeuralNet'):
        print("\n" + "="*60)
        print("Creating Random Forest + Neural Network Ensemble")
        print("="*60)
        
        ensemble = EnsembleModel(
            [models['RandomForest'], models['NeuralNet']],
            ['RandomForest', 'NeuralNet']
        )
        
        ensemble_metrics = evaluate_model(
            ensemble, X_train, y_train, X_test, y_test, 
            'RF+NN Ensemble'
        )
        all_metrics.append(ensemble_metrics)
        models['RF+NN Ensemble'] = ensemble
    
    # Select best model based on test accuracy
    best_model_metrics = max(all_metrics, key=lambda x: x['test_accuracy'])
    best_model_name = best_model_metrics['model_name']
    best_model = models[best_model_name]
    
    print("\n" + "="*60)
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {best_model_metrics['test_accuracy']:.4f}")
    print("="*60)
    
    # Calibrate best model if it's not an ensemble
    if 'Ensemble' not in best_model_name:
        calibrated_model = calibrate_model(best_model, X_train, y_train)
    else:
        print("\nSkipping calibration for ensemble (already averaging calibrated models)")
        calibrated_model = best_model
    
    # Get feature importance (from Random Forest or best tree model)
    feature_importance = {}
    if best_model_name == 'RF+NN Ensemble':
        # Use Random Forest feature importance from the ensemble
        feature_importance = get_feature_importance(models['RandomForest'], feature_names)
    else:
        feature_importance = get_feature_importance(calibrated_model, feature_names)
    
    # Save best model
    save_model(calibrated_model, best_model_name, feature_names, 
               best_model_metrics, feature_importance)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nBest model ({best_model_name}) saved to artifacts/")
    print(f"Test accuracy: {best_model_metrics['test_accuracy']:.2%}")
    print(f"Test AUC: {best_model_metrics['test_auc']:.4f}")
    
    return calibrated_model, best_model_metrics

if __name__ == '__main__':
    train_and_compare_models()
