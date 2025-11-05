"""
Test CSV Schema
Validates the structure and content of generated CSV files
"""
import os
import pytest
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def test_rankings_csv_exists():
    """Test that rankings CSV file exists"""
    rankings_file = os.path.join(DATA_DIR, 'ufc_rankings.csv')
    assert os.path.exists(rankings_file), "Rankings CSV file does not exist"

def test_rankings_csv_schema():
    """Test rankings CSV has correct schema"""
    rankings_file = os.path.join(DATA_DIR, 'ufc_rankings.csv')
    
    if not os.path.exists(rankings_file):
        pytest.skip("Rankings file not found")
    
    df = pd.read_csv(rankings_file)
    
    # Check required columns
    required_columns = ['weight_class', 'rank', 'name']
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check data types
    assert df['name'].dtype == object
    assert len(df) > 0, "Rankings file is empty"

def test_fighter_stats_csv_exists():
    """Test that fighter stats CSV exists"""
    stats_file = os.path.join(DATA_DIR, 'fighter_stats.csv')
    assert os.path.exists(stats_file), "Fighter stats CSV file does not exist"

def test_fighter_stats_csv_schema():
    """Test fighter stats CSV has correct schema"""
    stats_file = os.path.join(DATA_DIR, 'fighter_stats.csv')
    
    if not os.path.exists(stats_file):
        pytest.skip("Fighter stats file not found")
    
    df = pd.read_csv(stats_file)
    
    # Check required columns
    required_columns = [
        'name', 'wins', 'losses', 'height', 'reach', 'age',
        'sig_strikes_per_min', 'takedown_avg_per_15min'
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check no missing values in critical columns
    assert not df['name'].isnull().any(), "Name column has null values"
    assert not df['wins'].isnull().any(), "Wins column has null values"
    
    # Check data ranges
    assert (df['wins'] >= 0).all(), "Negative wins found"
    assert (df['losses'] >= 0).all(), "Negative losses found"
    assert (df['age'] >= 18).all() and (df['age'] <= 50).all(), "Age out of reasonable range"

def test_fight_results_csv_exists():
    """Test that fight results CSV exists"""
    results_file = os.path.join(DATA_DIR, 'fight_results.csv')
    assert os.path.exists(results_file), "Fight results CSV file does not exist"

def test_fight_results_csv_schema():
    """Test fight results CSV has correct schema"""
    results_file = os.path.join(DATA_DIR, 'fight_results.csv')
    
    if not os.path.exists(results_file):
        pytest.skip("Fight results file not found")
    
    df = pd.read_csv(results_file)
    
    # Check required columns
    required_columns = [
        'event_id', 'event_date', 'fighter1', 'fighter2',
        'winner', 'loser', 'method'
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check winner is either fighter1 or fighter2
    assert df.apply(
        lambda row: row['winner'] in [row['fighter1'], row['fighter2']], 
        axis=1
    ).all(), "Winner is not one of the fighters"
    
    # Check method values
    valid_methods = ['KO/TKO', 'Submission', 'Decision']
    assert df['method'].isin(valid_methods).all(), "Invalid fight method found"

def test_training_dataset_csv_exists():
    """Test that training dataset exists"""
    dataset_file = os.path.join(DATA_DIR, 'fighters_top10_men.csv')
    assert os.path.exists(dataset_file), "Training dataset CSV file does not exist"

def test_training_dataset_csv_schema():
    """Test training dataset has correct schema"""
    dataset_file = os.path.join(DATA_DIR, 'fighters_top10_men.csv')
    
    if not os.path.exists(dataset_file):
        pytest.skip("Training dataset not found")
    
    df = pd.read_csv(dataset_file)
    
    # Check target variable
    assert 'fighter1_won' in df.columns, "Missing target variable"
    assert df['fighter1_won'].isin([0, 1]).all(), "Invalid target values"
    
    # Check for fighter names
    assert 'fighter1_name' in df.columns, "Missing fighter1_name"
    assert 'fighter2_name' in df.columns, "Missing fighter2_name"
    
    # Check for key features
    key_features = ['f1_wins', 'f2_wins', 'height_diff', 'reach_diff', 'elo_diff']
    for feature in key_features:
        assert feature in df.columns, f"Missing key feature: {feature}"
    
    # Check no NaN in numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    assert not df[numeric_cols].isnull().any().any(), "NaN values found in numeric columns"

def test_elo_ratings_csv_exists():
    """Test that ELO ratings CSV exists"""
    elo_file = os.path.join(DATA_DIR, 'elo_ratings.csv')
    assert os.path.exists(elo_file), "ELO ratings CSV file does not exist"

def test_elo_ratings_csv_schema():
    """Test ELO ratings CSV has correct schema"""
    elo_file = os.path.join(DATA_DIR, 'elo_ratings.csv')
    
    if not os.path.exists(elo_file):
        pytest.skip("ELO ratings file not found")
    
    df = pd.read_csv(elo_file)
    
    # Check required columns
    required_columns = ['fighter', 'elo_rating', 'fights']
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check ELO range (typical range is 1000-2000)
    assert (df['elo_rating'] >= 1000).all(), "ELO rating too low"
    assert (df['elo_rating'] <= 2500).all(), "ELO rating too high"
    
    # Check fights is positive
    assert (df['fights'] > 0).all(), "Fighter with zero fights"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
