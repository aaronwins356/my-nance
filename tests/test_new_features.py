"""
Test New Features: Fighter Styles, Betting Odds, and Neural Network
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def test_fighter_styles_file_exists():
    """Test that fighter styles file exists"""
    styles_file = os.path.join(DATA_DIR, 'fighter_styles.csv')
    assert os.path.exists(styles_file), "Fighter styles file does not exist"

def test_fighter_styles_schema():
    """Test that fighter styles has correct schema"""
    styles_file = os.path.join(DATA_DIR, 'fighter_styles.csv')
    
    if not os.path.exists(styles_file):
        pytest.skip("Fighter styles file not found")
    
    df = pd.read_csv(styles_file)
    
    # Check required columns
    assert 'fighter_name' in df.columns, "Missing fighter_name column"
    assert 'primary_style' in df.columns, "Missing primary_style column"
    
    # Check that styles are valid
    valid_styles = ['Striker', 'Wrestler', 'BJJ', 'Well-Rounded', 'Boxer', 'Unknown']
    invalid_styles = df[~df['primary_style'].isin(valid_styles)]
    assert len(invalid_styles) == 0, f"Invalid styles found: {invalid_styles['primary_style'].unique()}"

def test_fight_odds_file_exists():
    """Test that fight odds file exists"""
    odds_file = os.path.join(DATA_DIR, 'fight_odds.csv')
    assert os.path.exists(odds_file), "Fight odds file does not exist"

def test_fight_odds_schema():
    """Test that fight odds has correct schema"""
    odds_file = os.path.join(DATA_DIR, 'fight_odds.csv')
    
    if not os.path.exists(odds_file):
        pytest.skip("Fight odds file not found")
    
    df = pd.read_csv(odds_file)
    
    # Check required columns
    required_cols = ['fighter1', 'fighter2', 'event_date', 'fighter1_odds', 'fighter2_odds', 'odds_type']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check that odds are numeric
    assert df['fighter1_odds'].dtype in ['int64', 'float64'], "fighter1_odds should be numeric"
    assert df['fighter2_odds'].dtype in ['int64', 'float64'], "fighter2_odds should be numeric"

def test_american_odds_conversion():
    """Test American odds to probability conversion"""
    from utils import american_odds_to_probability
    
    # Test favorite odds
    prob_favorite = american_odds_to_probability(-200)
    assert 0.65 < prob_favorite < 0.70, "Favorite odds conversion incorrect"
    
    # Test underdog odds
    prob_underdog = american_odds_to_probability(+150)
    assert 0.39 < prob_underdog < 0.42, "Underdog odds conversion incorrect"
    
    # Test that probabilities sum to more than 1 (bookmaker margin)
    prob1 = american_odds_to_probability(-150)
    prob2 = american_odds_to_probability(+130)
    assert (prob1 + prob2) > 1.0, "Probabilities should include bookmaker margin"

def test_data_processing_with_styles():
    """Test that data processing handles style features"""
    from data_processing import prepare_single_fight
    
    fighter1_stats = {
        'name': 'Test Fighter 1',
        'wins': 10,
        'losses': 2,
        'wins_by_ko': 5,
        'wins_by_submission': 3,
        'height': 72,
        'reach': 74,
        'age': 28,
        'sig_strikes_per_min': 4.5,
        'sig_strikes_absorbed_per_min': 3.0,
        'takedown_avg_per_15min': 2.0,
        'takedown_defense_pct': 75,
        'submission_avg_per_15min': 0.8,
        'striking_accuracy_pct': 48,
        'striking_defense_pct': 60
    }
    
    fighter2_stats = {
        'name': 'Test Fighter 2',
        'wins': 8,
        'losses': 3,
        'wins_by_ko': 2,
        'wins_by_submission': 4,
        'height': 70,
        'reach': 72,
        'age': 30,
        'sig_strikes_per_min': 3.8,
        'sig_strikes_absorbed_per_min': 3.5,
        'takedown_avg_per_15min': 3.0,
        'takedown_defense_pct': 70,
        'submission_avg_per_15min': 1.2,
        'striking_accuracy_pct': 45,
        'striking_defense_pct': 55
    }
    
    fighter_styles = {
        'Test Fighter 1': 'Striker',
        'Test Fighter 2': 'Wrestler'
    }
    
    odds = {
        'fighter1_odds': -150,
        'fighter2_odds': +130
    }
    
    # Prepare features
    features = prepare_single_fight(
        fighter1_stats, fighter2_stats,
        fighter_styles=fighter_styles,
        odds=odds
    )
    
    # Check that style features are present
    assert 'f1_style_striker' in features.columns, "Missing f1_style_striker"
    assert 'f2_style_wrestler' in features.columns, "Missing f2_style_wrestler"
    assert features['f1_style_striker'].iloc[0] == 1, "Fighter 1 should be striker"
    assert features['f2_style_wrestler'].iloc[0] == 1, "Fighter 2 should be wrestler"
    
    # Check that odds features are present
    assert 'f1_odds_implied_prob' in features.columns, "Missing f1_odds_implied_prob"
    assert 'f2_odds_implied_prob' in features.columns, "Missing f2_odds_implied_prob"
    assert 'odds_diff' in features.columns, "Missing odds_diff"
    
    # Check that odds probabilities sum to 1
    total_prob = features['f1_odds_implied_prob'].iloc[0] + features['f2_odds_implied_prob'].iloc[0]
    assert abs(total_prob - 1.0) < 0.01, "Odds probabilities should sum to 1"

def test_neural_network_import():
    """Test that neural network can be imported"""
    try:
        from sklearn.neural_network import MLPClassifier
        assert True, "MLPClassifier imported successfully"
    except ImportError:
        pytest.skip("MLPClassifier not available")

def test_ensemble_model():
    """Test that ensemble model works correctly"""
    from train import EnsembleModel
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    # Create dummy data
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(10, 10)
    
    # Train simple models
    model1 = DecisionTreeClassifier(random_state=42)
    model1.fit(X_train, y_train)
    
    model2 = RandomForestClassifier(n_estimators=10, random_state=42)
    model2.fit(X_train, y_train)
    
    # Create ensemble
    ensemble = EnsembleModel([model1, model2], ['DT', 'RF'])
    
    # Test predictions
    predictions = ensemble.predict(X_test)
    assert predictions.shape == (10,), "Prediction shape incorrect"
    assert all(pred in [0, 1] for pred in predictions), "Predictions should be binary"
    
    # Test probabilities
    probabilities = ensemble.predict_proba(X_test)
    assert probabilities.shape == (10, 2), "Probability shape incorrect"
    assert all(abs(prob.sum() - 1.0) < 0.01 for prob in probabilities), "Probabilities should sum to 1"

def test_style_matchup_features():
    """Test that style matchup features are created correctly"""
    from scripts.build_dataset import load_fighter_styles
    
    styles = load_fighter_styles()
    
    if len(styles) == 0:
        pytest.skip("No fighter styles available")
    
    # Check that at least some fighters have styles
    assert len(styles) > 0, "No fighter styles found"
    
    # Check that styles are valid
    valid_styles = ['Striker', 'Wrestler', 'BJJ', 'Well-Rounded', 'Boxer', 'Unknown']
    for style in styles.values():
        assert style in valid_styles, f"Invalid style: {style}"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
