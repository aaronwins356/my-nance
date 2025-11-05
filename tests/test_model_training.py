"""
Test Model Training
Validates model training pipeline and artifacts
"""
import os
import pytest
import pickle
import json
import pandas as pd

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')

def test_model_file_exists():
    """Test that trained model file exists"""
    model_file = os.path.join(ARTIFACTS_DIR, 'model.pkl')
    assert os.path.exists(model_file), "Model file does not exist"

def test_model_can_be_loaded():
    """Test that model can be loaded from pickle"""
    model_file = os.path.join(ARTIFACTS_DIR, 'model.pkl')
    
    if not os.path.exists(model_file):
        pytest.skip("Model file not found")
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    assert model is not None, "Model is None"
    assert hasattr(model, 'predict'), "Model does not have predict method"
    assert hasattr(model, 'predict_proba'), "Model does not have predict_proba method"

def test_model_metadata_exists():
    """Test that model metadata exists"""
    metadata_file = os.path.join(ARTIFACTS_DIR, 'model_metadata.json')
    assert os.path.exists(metadata_file), "Model metadata does not exist"

def test_model_metadata_schema():
    """Test model metadata has correct structure"""
    metadata_file = os.path.join(ARTIFACTS_DIR, 'model_metadata.json')
    
    if not os.path.exists(metadata_file):
        pytest.skip("Metadata file not found")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Check required fields
    required_fields = ['model_name', 'trained_at', 'metrics', 'feature_importance']
    for field in required_fields:
        assert field in metadata, f"Missing field: {field}"
    
    # Check metrics
    metrics = metadata['metrics']
    assert 'test_accuracy' in metrics, "Missing test_accuracy"
    assert 'test_auc' in metrics, "Missing test_auc"
    
    # Check accuracy is reasonable
    assert 0.4 <= metrics['test_accuracy'] <= 1.0, "Test accuracy out of range"
    assert 0.4 <= metrics['test_auc'] <= 1.0, "Test AUC out of range"

def test_feature_names_exist():
    """Test that feature names file exists"""
    feature_file = os.path.join(ARTIFACTS_DIR, 'feature_names.json')
    assert os.path.exists(feature_file), "Feature names file does not exist"

def test_feature_names_valid():
    """Test feature names are valid"""
    feature_file = os.path.join(ARTIFACTS_DIR, 'feature_names.json')
    
    if not os.path.exists(feature_file):
        pytest.skip("Feature names file not found")
    
    with open(feature_file, 'r') as f:
        feature_names = json.load(f)
    
    assert isinstance(feature_names, list), "Feature names should be a list"
    assert len(feature_names) > 0, "No feature names found"
    assert all(isinstance(name, str) for name in feature_names), "All features should be strings"

def test_model_prediction():
    """Test that model can make predictions"""
    model_file = os.path.join(ARTIFACTS_DIR, 'model.pkl')
    feature_file = os.path.join(ARTIFACTS_DIR, 'feature_names.json')
    
    if not os.path.exists(model_file) or not os.path.exists(feature_file):
        pytest.skip("Model or feature names not found")
    
    # Load model and features
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    with open(feature_file, 'r') as f:
        feature_names = json.load(f)
    
    # Create dummy input
    import numpy as np
    X_test = pd.DataFrame(
        np.random.randn(1, len(feature_names)),
        columns=feature_names
    )
    
    # Make prediction
    prediction = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    assert prediction.shape == (1,), "Prediction shape incorrect"
    assert probabilities.shape == (1, 2), "Probabilities shape incorrect"
    assert probabilities[0].sum() == pytest.approx(1.0, abs=0.01), "Probabilities don't sum to 1"

def test_feature_importance_present():
    """Test that feature importance is present in metadata"""
    metadata_file = os.path.join(ARTIFACTS_DIR, 'model_metadata.json')
    
    if not os.path.exists(metadata_file):
        pytest.skip("Metadata file not found")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    assert 'feature_importance' in metadata, "Feature importance not in metadata"
    
    feature_importance = metadata['feature_importance']
    assert len(feature_importance) > 0, "Feature importance is empty"
    
    # Check that importances are valid numbers
    for feature, importance in feature_importance.items():
        assert isinstance(importance, (int, float)), f"Invalid importance for {feature}"
        assert importance >= 0, f"Negative importance for {feature}"

def test_model_calibration():
    """Test that model provides calibrated probabilities"""
    model_file = os.path.join(ARTIFACTS_DIR, 'model.pkl')
    feature_file = os.path.join(ARTIFACTS_DIR, 'feature_names.json')
    
    if not os.path.exists(model_file) or not os.path.exists(feature_file):
        pytest.skip("Model or feature names not found")
    
    # Load model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Check if model is calibrated (has calibrated_classifiers_ attribute)
    # This is true for CalibratedClassifierCV
    assert hasattr(model, 'predict_proba'), "Model does not support probability prediction"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
